# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
# Wraps tue-mps/eomt (Encoder-only Mask Transformer, CVPR'25, instance seg with a
# DINOv3 backbone). Mirrors yolo_any_tune.py CLI.
# USAGE: python3 eomt_any_tune.py --data /path/to/dataset --epochs 12 --batch_sz 4 --model eomt-large
# DINOv3 weights: pass --dinov3_dir /path/to/hf_style_dir   (config.json + safetensors).

import os, sys, yaml, argparse, shutil, hashlib, csv, subprocess, json, zipfile, re
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rfdetr_any_tune import yolo_to_coco_layout  # noqa: E402
from bench_schema import zero_metrics  # noqa: E402

REPO_URL = "https://github.com/tue-mps/eomt.git"
REPO_BASE = Path(os.environ.get("BENCH_ANY", str(Path.home() / "bench_any")))
REPO_CACHE = REPO_BASE / "eomt"

BACKBONES = {
    # Direct DINOv3 variant tags (paper Table 2 / Table 11) — recognised inside
    # the user's --model string, e.g. `eomt-vitl16` or `eomt-vit7b16`.
    "vits16":        "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vits16plus":    "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "vitb16":        "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl16":        "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "vith16plus":    "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "vit7b16":       "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "convnext_tiny":  "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "convnext_small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "convnext_base":  "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "convnext_large": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
    # Friendly size aliases (legacy).
    "smallplus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "small":  "facebook/dinov3-vits16-pretrain-lvd1689m",
    "base":   "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large":  "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "huge":   "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "giant":  "facebook/dinov3-vit7b16-pretrain-lvd1689m",
}


def _ensure_imports(modules, label=""):
    missing = [spec for imp, spec in modules
               if subprocess.run([sys.executable, "-c", f"import {imp}"], capture_output=True).returncode]
    if not missing: return
    print(f"📦 [{label}] pip install {' '.join(missing)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing]); return
    except subprocess.CalledProcessError:
        for s in missing:
            try: subprocess.check_call([sys.executable, "-m", "pip", "install", s])
            except subprocess.CalledProcessError as e: print(f"⚠️ pip install {s} rc={e.returncode}")


def _has_native_lightning():
    return subprocess.run(
        [sys.executable, "-c", "import lightning, lightning.pytorch.loggers.csv_logs"],
        capture_output=True).returncode == 0


def _patch_upstream_main(repo_dir):
    """Patch small upstream main.py issues/noise for the benchmark harness."""
    p = Path(repo_dir) / "main.py"
    if not p.is_file(): return
    txt = p.read_text()
    changed = False
    if "# bench_any: gitignore_parser softened" not in txt:
        txt = re.sub(
            r"from\s+gitignore_parser\s+import\s+([^\n]+)",
            lambda m: ("try:\n"
                       f"    from gitignore_parser import {m.group(1).strip()}\n"
                       "except ImportError:\n"
                       "    parse_gitignore = lambda *a, **k: (lambda _p: False)"
                       "  # bench_any: gitignore_parser softened"),
            txt, count=1)
        changed = True
    if "callbacks are set via config.yaml" not in txt:
        # Original upstream uses ModelSummary + LRMonitor.  Replace with
        # empty list — Lightning's Trainer auto-adds a default
        # ModelCheckpoint(save_top_k=1) when no user checkpoint exists,
        # which saves epoch=N-step=M.ckpt.  We rely on that default.
        # Do NOT inject a custom ModelCheckpoint here or in config.yaml:
        # it suppresses the default and, with save_top_k=0/monitor=None,
        # has been observed to not write any file (Lightning 2.6.1).
        # See potential_bugs.md BUG 3.
        for old in (
            '"callbacks": [\n'
            '                ModelSummary(max_depth=3),\n'
            '                LearningRateMonitor(logging_interval="epoch"),\n'
            '            ],',
            '"callbacks": [\n'
            '                ModelCheckpoint(save_last=True, save_top_k=0),  # bench_any: quiet callbacks\n'
            '            ],',
            '"callbacks": [],  # bench_any: quiet callbacks',
            '"callbacks": [],  # bench_any: quiet callbacks (populated by config.yaml)',
        ):
            txt = txt.replace(old,
                '"callbacks": [],  # bench_any: quiet callbacks (Lightning default checkpoint)'

            )
        # Clean up stale ModelCheckpoint / ModelSummary / LRMonitor imports.
        for stale_import in (
            'from lightning.pytorch.callbacks import ModelCheckpoint  # bench_any: for save_last\n',
            'from lightning.pytorch.callbacks import ModelSummary, LearningRateMonitor',
        ):
            txt = txt.replace(stale_import, '')
        changed = True
    # Repair corruption from a prior buggy patch iteration: the comment
    # suffix "(populated by config.yaml)" leaked onto its own lines outside
    # the string literal, producing a SyntaxError at import time.
    for leak in ("\n (populated by config.yaml)", "\n(populated by config.yaml)",
                 "\n (Lightning default checkpoint)", "\n(Lightning default checkpoint)"):
        while leak in txt:
            txt = txt.replace(leak, "")
            changed = True
    if changed:
        p.write_text(txt)
        print("🩹 Patched eomt/main.py (bench compatibility)")


def _patch_upstream_dataset(repo_dir):
    """Upstream `eomt/datasets/dataset.py` does `torch.stack(masks)` unconditionally;
    with images that have zero annotations (common in real-world val sets) this raises
    `RuntimeError: stack expects a non-empty TensorList` and the entire fit aborts on
    the sanity-check pass. Patch to return a (0, H, W) empty tensor in that case.

    Also patches `eomt/datasets/coco_instance.py` to use an identity mapping for
    CLASS_MAPPING. Upstream has it hard-coded to COCO's 1-indexed 80 classes, which
    skips all annotations in our 0-indexed benchmark COCO JSONs. Module-level
    overrides in the launcher are lost in DataLoader forkserver workers."""
    p = Path(repo_dir) / "datasets" / "dataset.py"
    if p.is_file():
        txt = p.read_text()
        MARKER = "# bench_any: empty-masks guard v2"
        if MARKER not in txt:
            # Roll back v1 marker so the replace targets match the original lines.
            txt = re.sub(r"\s*# bench_any: empty-masks guard(?! v2)", "", txt)
            new_txt = txt.replace(
                '"masks": tv_tensors.Mask(torch.stack(masks)),',
                ('"masks": tv_tensors.Mask(torch.stack(masks) if len(masks) > 0 '
                 'else torch.zeros((0, img.shape[-2], img.shape[-1]), dtype=torch.long)),  ' + MARKER),
            )
            new_txt = new_txt.replace(
                '"labels": torch.tensor(labels),',
                '"labels": torch.tensor(labels, dtype=torch.long),  ' + MARKER)
            new_txt = new_txt.replace(
                '"is_crowd": torch.tensor(is_crowd),',
                '"is_crowd": torch.tensor(is_crowd, dtype=torch.bool),  ' + MARKER)
            if new_txt != txt:
                p.write_text(new_txt); print("🩹 Patched eomt/datasets/dataset.py (empty-masks guard)")

    p_coco = Path(repo_dir) / "datasets" / "coco_instance.py"
    if p_coco.is_file():
        txt = p_coco.read_text()
        MARKER = "# bench_any: identity mapping"
        if MARKER not in txt:
            # Replace the huge hardcoded dictionary with a permissive range.
            # Using 1000 to cover any likely benchmark class count.
            new_txt = re.sub(r"CLASS_MAPPING = \{.*?\}",
                             f"CLASS_MAPPING = {{i: i for i in range(1000)}}  {MARKER}",
                             txt, flags=re.DOTALL)
            if new_txt != txt:
                p_coco.write_text(new_txt); print("🩹 Patched eomt/datasets/coco_instance.py (identity mapping)")


def _patch_upstream_transforms(repo_dir):
    """Replace upstream's unbounded recurse-until-valid path in `transforms.py::forward`
    with a bounded retry loop (max 16 attempts) plus a sane fallback: on the final
    attempt, return the padded full image with all-`is_crowd=False` masks intact (no
    random crop). Without this, datasets with small/edge-only objects routinely hit
    Python's recursion limit because every random_crop misses the GT mask and the
    upstream code recurses forever. The previous guard was wrong (it dropped the
    mask whenever the original target was non-empty but a crop happened to miss it,
    making mask/dice loss = 0 throughout training)."""
    p = Path(repo_dir) / "datasets" / "transforms.py"
    if not p.is_file(): return
    txt = p.read_text()
    # Roll back the older v1 guard if present so the search target matches.
    if "# bench_any: empty-target recursion guard" in txt:
        txt = re.sub(
            r'        if not valid\.any\(\):\n'
            r'            if len\(target_orig\["masks"\]\) == 0:  # bench_any: empty-target recursion guard\n'
            r'                return img, self\._filter\(target, valid\)\n'
            r'            return self\(img_orig, target_orig\)\n',
            '        if not valid.any():\n'
            '            return self(img_orig, target_orig)\n',
            txt, count=1)
    MARKER = "# bench_any: bounded crop-retry v3"
    if MARKER in txt: return
    OLD = ('    def forward(\n'
           '        self, img: Tensor, target: dict[str, Union[Tensor, TVTensor]]\n'
           '    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:\n'
           '        img_orig, target_orig = img, target\n'
           '\n'
           '        target = self._filter(target, ~target["is_crowd"])\n'
           '\n'
           '        img = self.color_jitter(img)\n'
           '        img, target = self.random_horizontal_flip(img, target)\n'
           '        img, target = self.scale_jitter(img, target)\n'
           '        img, target = self.pad(img, target)\n'
           '        img, target = self.random_crop(img, target)\n'
           '\n'
           '        valid = target["masks"].flatten(1).any(1)\n'
           '        if not valid.any():\n'
           '            return self(img_orig, target_orig)\n'
           '\n'
           '        target = self._filter(target, valid)\n'
           '\n'
           '        return img, target\n')
    NEW = ('    def forward(\n'
           '        self, img: Tensor, target: dict[str, Union[Tensor, TVTensor]]\n'
           '    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:\n'
           '        img_orig, target_orig = img, target\n'
           '        target_filt = self._filter(target_orig, ~target_orig["is_crowd"])\n'
           '        # Bounded retry loop instead of unbounded recursion. ' + MARKER + '\n'
           '        for _attempt in range(16):\n'
           '            t = {k: v for k, v in target_filt.items()}\n'
           '            i = self.color_jitter(img_orig)\n'
           '            i, t = self.random_horizontal_flip(i, t)\n'
           '            i, t = self.scale_jitter(i, t)\n'
           '            i, t = self.pad(i, t)\n'
           '            i, t = self.random_crop(i, t)\n'
           '            valid = t["masks"].flatten(1).any(1)\n'
           '            if valid.any():\n'
           '                return i, self._filter(t, valid)\n'
           '        # Fallback: pad the original image to img_size and use unfiltered\n'
           '        # masks so the sample still trains the class head; mask/dice loss\n'
           '        # will be small but well-defined.\n'
           '        i, t = self.pad(img_orig, {k: v for k, v in target_filt.items()})\n'
           '        valid = t["masks"].flatten(1).any(1)\n'
           '        if valid.any():\n'
           '            return i, self._filter(t, valid)\n'
           '        return i, t\n')
    new_txt = txt.replace(OLD, NEW)
    if new_txt == txt: return
    p.write_text(new_txt); print("🩹 Patched eomt/datasets/transforms.py (bounded crop retry)")


def _patch_upstream_vit(repo_dir):
    """DINOv3ViTModel exposes blocks under `.model.layer`, not `.layer`. Probe across
    layouts and harden the matching `del backbone.layer` so DINOv3 doesn't AttributeError."""
    p = Path(repo_dir) / "models" / "vit.py"
    if not p.is_file(): return
    txt = p.read_text()
    MARKER = "# bench_any: dinov3 backbone-blocks probe v2"
    if MARKER in txt: return
    if "# bench_any: dinov3 .encoder.layer fallback" in txt:
        txt = re.sub(
            r"backbone\.blocks\s*=\s*getattr\(backbone, 'layer', None\) or backbone\.encoder\.layer\s*# bench_any: dinov3 \.encoder\.layer fallback",
            "backbone.blocks = backbone.layer", txt, count=1)
    new_txt = re.sub(
        r"backbone\.blocks\s*=\s*backbone\.layer\b",
        ("backbone.blocks = (getattr(backbone, 'layer', None) "
         "or getattr(getattr(backbone, 'encoder', None), 'layer', None) "
         "or getattr(getattr(backbone, 'model', None), 'layer', None) "
         "or getattr(backbone, 'layers', None) "
         "or getattr(backbone, 'blocks', None))  " + MARKER),
        txt, count=1)
    if new_txt == txt: return
    def _harden_del(m):
        indent = m.group(1)
        targets = [t.strip() for t in m.group(2).split(",") if t.strip()]
        out = [f"{indent}# bench_any: harden del for dinov3 missing-attr"]
        for t in targets:
            out += [f"{indent}try: del {t}", f"{indent}except AttributeError: pass"]
        return "\n".join(out)
    new_txt = re.sub(r"^([ \t]+)del\s*\(\s*([^)]+?)\s*\)", _harden_del, new_txt, count=1, flags=re.MULTILINE)
    p.write_text(new_txt); print("🩹 Patched eomt/models/vit.py (DINOv3 blocks probe + del guard)")


def _patch_upstream_instance_eval(repo_dir):
    """Upstream `eomt/training/mask_classification_instance.py` has a critical bug:
    in `eval_step`, it builds a list of dicts `targets_` for evaluation, but then
    incorrectly passes the raw batch dictionary `targets` to `update_metrics_instance`.
    This causes `MeanAveragePrecision` to receive invalid GT, resulting in -1.0 mAP.
    Patch to pass the correct `targets_` variable."""
    p = Path(repo_dir) / "training" / "mask_classification_instance.py"
    if not p.is_file(): return
    txt = p.read_text()
    MARKER = "# bench_any: fix targets_ var bug"
    if MARKER in txt: return
    new_txt = txt.replace(
        "self.update_metrics_instance(preds, targets, i)",
        f"self.update_metrics_instance(preds, targets_, i)  {MARKER}")
    if new_txt == txt: return
    p.write_text(new_txt); print("🩹 Patched eomt/training/mask_classification_instance.py (targets_ bug)")


def _patch_upstream_max_detections(repo_dir):
    """Raise the per-image detection cap on EoMT's torchmetrics
    `MeanAveragePrecision`. The penguin datasets can have several hundred
    instances per image, but torchmetrics defaults to
    `max_detection_thresholds=[1, 10, 100]`, silently dropping all but the 100
    highest-scoring detections (and spamming a ">100 detections" warning). Bump
    the top cap to 1000.

    The upstream call is wrapped in a list comprehension:
        [MeanAveragePrecision(iou_type="segm") for _ in range(num_blocks)]
    so a naive str.replace with max_detection_thresholds=[1,10,1000] causes a
    syntax error because the ] inside the kwarg list closes the outer [ ... for].
    We instead wrap each MAP instance after construction by patching the method
    to inject .max_detection_thresholds = [1, 10, 1000] on each module."""
    p = Path(repo_dir) / "training" / "lightning_module.py"
    if not p.is_file(): return
    txt = p.read_text()
    MARKER_V1 = "# bench_any: raise max detections"
    MARKER_V2 = "# bench_any: raise max detections (v2 post-init loop)"
    # Self-repair: v1 of this patch (prior to 2025-06-20) did a naive
    # str.replace that put `max_detection_thresholds=[1,10,1000]` INSIDE
    # the list comprehension, which broke the bracket balance. If we detect
    # the v1 marker, revert to pristine and re-patch with the v2 approach.
    if MARKER_V1 in txt and MARKER_V2 not in txt:
        txt = txt.replace(
            'MeanAveragePrecision(iou_type="segm", '
            'max_detection_thresholds=[1, 10, 1000])  '
            f'{MARKER_V1}',
            'MeanAveragePrecision(iou_type="segm")')
    if MARKER_V2 in txt: return
    new_txt = txt.replace(
        'self.metrics = nn.ModuleList(\n'
        '            [MeanAveragePrecision(iou_type="segm") for _ in range(num_blocks)]\n'
        '        )',
        'self.metrics = nn.ModuleList(\n'
        '            [MeanAveragePrecision(iou_type="segm") for _ in range(num_blocks)]\n'
        '        )\n'
        '        for _m in self.metrics:\n'
        f'            _m.max_detection_thresholds = [1, 10, 1000]  {MARKER_V2}')
    if new_txt == txt: return
    p.write_text(new_txt); print("🩹 Patched eomt/training/lightning_module.py (max detections -> 1000)")


def ensure_repo():
    REPO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not (REPO_CACHE / ".git").exists():
        print(f"📥 Cloning {REPO_URL} → {REPO_CACHE}")
        subprocess.check_call(["git", "clone", "--depth=1", REPO_URL, str(REPO_CACHE)])
    _ensure_imports([
        ("torchmetrics",        "torchmetrics"),
        ("jsonargparse",        "jsonargparse[signatures]"),
        ("transformers",        "transformers"),
        ("pycocotools",         "pycocotools"),
        ("PIL",                 "Pillow"),
        ("gitignore_parser",    "gitignore_parser"),
        ("pytorch_lightning",   "pytorch-lightning>=2.0"),
        ("lightning_utilities", "lightning-utilities"),
        # Required by upstream eomt/models/scale_block.py (LayerNorm2d). Not in our
        # whitelist before -> ModuleNotFoundError at LightningCLI import time.
        ("timm",                "timm"),
        # Real `lightning` package (re-exports lightning.pytorch + lightning.fabric).
        # Upstream eomt/main.py does `from lightning.pytorch import cli`; without
        # native `lightning` installed we'd fall back to a sys.modules alias hack
        # in `_launch.py`, but that hack only lives in the parent process —
        # DataLoader's forkserver workers spawn fresh Python and re-import main.py
        # directly, hitting `ModuleNotFoundError: No module named 'lightning'`.
        # Installing the real package makes both paths work identically.
        ("lightning",           "lightning"),
    ], label="eomt")
    _patch_upstream_main(REPO_CACHE)
    _patch_upstream_dataset(REPO_CACHE)
    _patch_upstream_transforms(REPO_CACHE)
    _patch_upstream_instance_eval(REPO_CACHE)
    _patch_upstream_max_detections(REPO_CACHE)
    _patch_upstream_vit(REPO_CACHE)
    has_native = _has_native_lightning()
    if not has_native: print("ℹ  using `pytorch_lightning` aliased as `lightning`")
    return REPO_CACHE, has_native


def pick_backbone(name):
    n = name.lower()
    # Explicit DINOv3 variant tags take priority over size aliases. Longer keys
    # first so 'vits16plus' wins over 'vits16', 'vith16plus' over 'vith16', etc.
    for tag in ("convnext_large", "convnext_small", "convnext_base", "convnext_tiny",
                "vits16plus", "vith16plus", "vit7b16",
                "vits16", "vitb16", "vitl16", "vith16"):
        if tag in n: return BACKBONES[tag]
    for size in ("smallplus", "giant", "huge", "large", "base", "small"):
        if size in n and size in BACKBONES: return BACKBONES[size]
    return BACKBONES["large"]


def resolve_local_hf_backbone_dir(path_like):
    """Validate an EoMT local DINOv3 directory.

    EoMT uses Transformers AutoConfig/AutoModel, so a local override must be an
    HF-style model directory with config.json and weight shards/safetensors. A
    folder of facebookresearch .pth weights is valid for dinov3-semseg, but not
    for this wrapper.
    """
    if not path_like:
        return "", ""
    p = Path(path_like).expanduser()
    if not p.is_dir():
        return "", f"⚠️ Ignoring --dinov3_dir={p}; directory does not exist for EoMT."
    if not (p / "config.json").is_file():
        return "", (
            f"⚠️ Ignoring --dinov3_dir={p} for EoMT; it is not an HF-style DINOv3 "
            "directory with config.json. Plain .pth DINOv3 weights are only usable "
            "by dinov3-semseg in this bench."
        )
    weight_globs = ("*.safetensors", "*.bin", "*.pth")
    if not any(next(p.glob(g), None) is not None for g in weight_globs):
        return "", (
            f"⚠️ Ignoring --dinov3_dir={p} for EoMT; config.json exists but no "
            "model weight file was found."
        )
    return str(p.resolve()), ""


def coco_to_eomt_zips(coco_dir, out_dir):
    """Pack per-split COCO json + flat image dir into the three zips EoMT expects:
       train2017.zip, val2017.zip, annotations_trainval2017.zip.

    Drops zero-annotation images only from training. Eval keeps negative images;
    otherwise detector false positives on empty frames would disappear from the
    benchmark data. The empty-mask dataset patch above handles eval empties."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    coco_dir = Path(coco_dir)
    splits = {"train": "train", "test" if (coco_dir / "test").is_dir() else "valid": "val"}
    cat_count = 0
    for src_split, dst_split in splits.items():
        src = coco_dir / src_split
        if not src.is_dir():
            print(f"⚠️ split '{src_split}' missing; skip"); continue
        ann_src = src / "_annotations.coco.json"
        keep_names = None  # None = keep all (no annotations file)
        if ann_src.is_file():
            j = json.loads(ann_src.read_text())
            cat_count = max(cat_count, len(j.get("categories", [])))
            if dst_split == "train":
                ann_img_ids = {a["image_id"] for a in j.get("annotations", [])}
                kept_imgs = [im for im in j.get("images", []) if im["id"] in ann_img_ids]
                n_dropped = len(j.get("images", [])) - len(kept_imgs)
                if n_dropped:
                    print(f"   ↪ {dst_split}: dropped {n_dropped} zero-annotation training image(s)")
                j["images"] = kept_imgs
                keep_names = {im["file_name"] for im in kept_imgs}
            else:
                keep_names = {im["file_name"] for im in j.get("images", [])}
            (out_dir / f"_inst_{dst_split}.json").write_text(json.dumps(j))
        with zipfile.ZipFile(out_dir / f"{dst_split}2017.zip", "w", zipfile.ZIP_STORED) as z:
            for p in sorted(src.iterdir()):
                if p.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'): continue
                if keep_names is not None and p.name not in keep_names: continue
                z.write(str(p.resolve()), arcname=f"{dst_split}2017/{p.name}")
    with zipfile.ZipFile(out_dir / "annotations_trainval2017.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for dst in splits.values():
            f = out_dir / f"_inst_{dst}.json"
            if f.is_file(): z.write(str(f), arcname=f"annotations/instances_{dst}2017.json")
    return out_dir, cat_count


_BENCH_TRAINED_CKPT = "_bench_trained.ckpt"


def _preserve_trained_ckpt(run_dir):
    """Inference-only (BUG 3B): copy the latest trained *.ckpt to
    run_dir/_bench_trained.ckpt BEFORE write_yaml's scrub wipes version_*/. The
    trained checkpoint lives under <run_name>/version_0/checkpoints/ which the
    scrub deletes; without preservation, the inference-only `validate` command
    finds no checkpoint and scores an uninitialized model → all-zero metrics.

    The preserved file sits at the run-dir top level (NOT under version_*), so
    the scrub's version_* rmtree leaves it; write_yaml's top-level *.ckpt sweep
    is taught to skip _bench_trained.ckpt (see PRESERVE_SKIP below). After the
    scrub, a clean version_0 starts for the validate run so harvest_metrics
    picks the inference metrics (no version-collision false positive). Returns
    the absolute path, or None if no trained checkpoint exists."""
    import shutil
    ckpts = sorted(Path(run_dir).rglob("*.ckpt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        return None
    dst = Path(run_dir) / _BENCH_TRAINED_CKPT
    shutil.copy2(ckpts[0], dst)
    print(f"🔁 inference-only: preserved trained checkpoint {ckpts[0].name} -> {_BENCH_TRAINED_CKPT}")
    return str(dst)


def write_yaml(path, data_path, num_classes, backbone_id, max_epochs, imgsz, batch_sz, lr, num_workers, run_name, run_dir, grad_accum_steps=1, patience=15):
    # --- CRITICAL: scrub stale Lightning artefacts before writing config ---
    # Without this, Lightning's CSVLogger auto-increments version_0 -> version_N
    # and EoMT's post-eval may load a previous run's checkpoint trained with
    # num_classes=80 (COCO default) instead of this run's num_classes=1 head,
    # producing 0 mAP/AR for every val/test image.
    import shutil
    run_dir_p = Path(run_dir)
    if run_dir_p.exists():
        # remove every version_*/ subdir (each holds its own checkpoints/ + metrics.csv)
        for sub in run_dir_p.rglob("version_*"):
            if sub.is_dir():
                shutil.rmtree(sub, ignore_errors=True)
        # remove any stray *.ckpt left at the top level, BUT keep the
        # inference-only preserved trained checkpoint (_bench_trained.ckpt) — it
        # was copied here by _preserve_trained_ckpt before the scrub so the
        # `validate` subcommand can load it via --ckpt_path (BUG 3B).
        for ck in run_dir_p.rglob("*.ckpt"):
            if ck.name == _BENCH_TRAINED_CKPT:
                continue
            try: ck.unlink()
            except Exception: pass
    run_dir_p.mkdir(parents=True, exist_ok=True)

    cfg = {
        "trainer": {
            "max_epochs": int(max_epochs),
            "check_val_every_n_epoch": max(1, int(max_epochs) // 6 or 1),
            "default_root_dir": str(run_dir),
            "enable_model_summary": False,
            "log_every_n_steps": 1,
            # bench_any: inject an EXPLICIT ModelCheckpoint (BUG 3A). Lightning's
            # auto-default (save_top_k=1, monitor=None, save_last=None) only saves
            # in on_validation_end and is skipped when validation doesn't complete
            # on the final epoch (OOM / early-stop / val-cadence) — leaving
            # inference-only with no checkpoint to load. save_last=True guarantees
            # last.ckpt on train_end AND every validation end; save_top_k=1 is
            # load-bearing (NOT 0): _save_topk_checkpoint early-returns at 0 so
            # _last_global_step_saved never updates and on_validation_end's
            # save_last branch skips — verified in Lightning 2.6.1 source.
            # save_top_k=1 keeps one top-k ckpt (overwritten) + one last.ckpt ≈
            # 2 files, not 125 (each ~617 MB for eomt-large). every_n_epochs=1
            # saves every epoch so a mid-epoch-val crash leaves the prior
            # epoch's last.ckpt. See potential_bugs.md BUG 3.
            "callbacks": [
                {
                    "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "init_args": {
                        "save_last": True,
                        "save_top_k": 1,
                        "every_n_epochs": 1,
                        "monitor": None,
                        "filename": "epoch{epoch}-step{step}",
                    },
                },
                # bench_any: a SECOND, monitored checkpoint. The one above has
                # monitor=None, so its save_top_k=1 selects nothing — it is a
                # rolling last, which is why EoMT's reported mAP was whatever
                # the final evaluated epoch happened to be while yolo and rtdetr
                # report their best.pt. This one selects on the same mask AP the
                # bench publishes (column verified present in metrics.csv), and
                # _bench_best.ckpt is what the post-train validate pass scores.
                {
                    "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "init_args": {
                        "save_last": False,
                        "save_top_k": 1,
                        "monitor": "metrics/val_ap_all",
                        "mode": "max",
                        "filename": "_bench_best",
                        "enable_version_counter": False,
                    },
                },
                # bench_any: wire --patience (was a dead arg). Lightning
                # EarlyStopping fires on metrics/val_ap_all, the same monitor as
                # the _bench_best checkpoint above. EoMT validates ~6×/150ep, so
                # patience 15 is a safety net that never trips in practice (needs
                # 15 non-improving validations); it makes the CLI honest without
                # changing the full-budget run that the 0.831 headline needs.
                {
                    "class_path": "lightning.pytorch.callbacks.EarlyStopping",
                    "init_args": {
                        "monitor": "metrics/val_ap_all",
                        "mode": "max",
                        "patience": int(patience),
                        "check_finite": True,
                    },
                },
            ],
            # Preserves effective batch when micro-batch is forced low to fit memory:
            # effective_batch = batch_sz * grad_accum_steps. EoMT-large + 200 queries +
            # multi-decoder mask logits is ~5x heavier than YOLO11x per sample, hence
            # the small default micro-bs.
            "accumulate_grad_batches": max(1, int(grad_accum_steps)),
            "logger": {"class_path": "lightning.pytorch.loggers.csv_logs.CSVLogger",
                       "init_args": {"save_dir": str(run_dir), "name": run_name}},
        },
        "model": {
            "class_path": "training.mask_classification_instance.MaskClassificationInstance",
            "init_args": {
                "lr": float(lr), "num_classes": int(num_classes),
                "img_size": [int(imgsz), int(imgsz)],
                "attn_mask_annealing_enabled": False,
                "network": {"class_path": "models.eomt.EoMT",
                            "init_args": {"num_q": 200,
                                          "encoder": {"class_path": "models.vit.ViT",
                                                      "init_args": {"backbone_name": backbone_id}}}},
            },
        },
        # CRITICAL: upstream main.py does
        #   parser.link_arguments("data.init_args.num_classes",
        #                         "model.init_args.num_classes")
        # so the *data* side is the source of truth. If we only set num_classes on
        # the model block, jsonargparse silently overwrites it with COCOInstance's
        # default (80) and the class head ends up sized for 80 classes while our
        # GT labels are 0..N-1 — yielding mAP=0/NaN across every val epoch (this
        # was the "EoMT gets 0 mAP regardless of dataset size" bug).
        "data": {"class_path": "datasets.coco_instance.COCOInstance",
                 "init_args": {"path": str(data_path), "img_size": [int(imgsz), int(imgsz)],
                               "num_workers": int(num_workers), "batch_size": int(batch_sz),
                               "num_classes": int(num_classes)}},
    }
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))


def write_launcher(launcher_path, repo_dir, num_classes, has_native_lightning, aug_policy="native"):
    """Tiny shim that overrides EoMT's hard-coded COCO 80-class mapping with identity
    (id→id) over the actual class set, then runs upstream main.py via runpy."""
    alias_block = "" if has_native_lightning else """
import importlib
def _alias_lightning():
    pl = importlib.import_module("pytorch_lightning")
    sys.modules["lightning"] = pl
    sys.modules["lightning.pytorch"] = pl
    pl.pytorch = pl
    for sub in ("loggers", "loggers.csv_logs", "callbacks", "trainer", "core", "utilities", "cli"):
        try:
            m = importlib.import_module(f"pytorch_lightning.{sub}")
            sys.modules[f"lightning.pytorch.{sub}"] = m
            sys.modules[f"lightning.{sub}"] = m
        except Exception: pass
    try:
        fab = importlib.import_module("lightning_fabric")
        sys.modules["lightning.fabric"] = fab; pl.fabric = fab
    except Exception: pass
_alias_lightning()
"""
    # Switch DataLoader worker IPC off /dev/shm — mask tensors otherwise blow the
    # tmpfs quota mid-epoch (`Disk quota exceeded (122)`).
    shm_block = ("import torch.multiprocessing as _mp\n"
                 "try: _mp.set_sharing_strategy('file_system')\n"
                 "except Exception: pass\n")
    # ---- bench_any: dump per-image COCO predictions during eval ----------
    # EoMT writes no test predictions to disk by default, so our post-hoc
    # `coco_eval_full` had nothing to evaluate and seg_P/R/F1 / S/M/L / AR_100
    # were stuck at zero. We monkey-patch (a) the dataset to attach the source
    # filename to each target, and (b) the LightningModule's eval_step to
    # append RLE-encoded predictions to <RUN_DIR>/predictions.json on every
    # validation epoch (the file is overwritten each epoch — the LAST epoch
    # wins, which matches what `metrics.csv` reports).
    pred_json = str(Path(launcher_path).parent / "predictions.json")
    fps_json  = str(Path(launcher_path).parent / "_bench_fps.json")
    patch_block = f"""
# ---- bench_any: COCO predictions dump --------------------------------------
def _bench_install_pred_dump():
    import json as _json, threading as _thr, time as _time
    from pathlib import Path as _P
    import torch as _t
    try:
        from datasets.dataset import Dataset as _DS
        from training.mask_classification_instance import (
            MaskClassificationInstance as _MCI,
        )
    except Exception as _e:
        print(f"[bench_any] pred-dump install skipped: {{_e}}")
        return
    try:
        from pycocotools import mask as _mu
    except Exception as _e:
        print(f"[bench_any] pycocotools missing, no RLE dump: {{_e}}")
        return
    _orig_getitem = _DS.__getitem__
    # bench_any: shared cross-framework Albumentations policy (train-only).
    _BENCH_AUG_POLICY = {aug_policy!r}
    _bench_aug_pipeline = None
    if _BENCH_AUG_POLICY in ("off", "fair"):
        try:
            import sys as _sys, os as _os
            _be_dir = _os.environ.get("BENCH_EVAL_DIR")
            if _be_dir and _be_dir not in _sys.path:
                _sys.path.insert(0, _be_dir)
            from bench_aug import shared_albumentations_policy as _sap
            _bench_aug_pipeline = _sap(_BENCH_AUG_POLICY, 640)
            print(f"[bench_any] eomt shared aug pipeline installed (policy={{_BENCH_AUG_POLICY}})")
        except Exception as _e:
            print(f"[bench_any] eomt shared aug pipeline unavailable: {{_e}}")
            _bench_aug_pipeline = None

    def _patched_getitem(self, index):
        out = _orig_getitem(self, index)
        try:
            img, tgt = out
            if isinstance(tgt, dict):
                tgt["_bench_file_name"] = _P(self.imgs[index]).name
            # Apply shared Albumentations on (image, masks) for train split only.
            # Train vs val is inferred from the zip path filename.
            if _bench_aug_pipeline:
                _zp = str(getattr(self, "zip_path", "") or getattr(self, "target_zip_path", ""))
                if "train" in _P(_zp).name.lower():
                    try:
                        import albumentations as _A
                        import numpy as _np
                        import torch as _t2
                        from torchvision import tv_tensors as _tvt
                        # img is tv_tensors.Image (C,H,W) tensor uint8 or float.
                        _arr = img.detach().cpu().numpy() if hasattr(img, "detach") else _np.array(img)
                        if _arr.ndim == 3 and _arr.shape[0] in (1, 3):
                            _arr = _arr.transpose(1, 2, 0)
                        if _arr.dtype != _np.uint8:
                            _arr = _np.clip(_arr, 0, 255).astype(_np.uint8)
                        _masks_t = tgt.get("masks") if isinstance(tgt, dict) else None
                        _masks_np = None
                        if _masks_t is not None and len(_masks_t) > 0:
                            _masks_np = [m.detach().cpu().numpy().astype(_np.uint8) for m in _masks_t]
                        _kwargs = {{"image": _arr}}
                        if _masks_np:
                            _kwargs["masks"] = _masks_np
                        _tform = _A.Compose(_bench_aug_pipeline)
                        _aug = _tform(**_kwargs)
                        _new_arr = _aug["image"]
                        if _new_arr.ndim == 3:
                            _new_arr = _new_arr.transpose(2, 0, 1)
                        img = _tvt.Image(_t2.from_numpy(_new_arr.copy()))
                        if _masks_np and "masks" in _aug and _aug["masks"]:
                            _stk = _np.stack([_m.astype(_np.int64) for _m in _aug["masks"]], axis=0)
                            tgt["masks"] = _tvt.Mask(_t2.from_numpy(_stk))
                        out = (img, tgt)
                    except Exception as _e2:
                        # On any failure, fall through with the untouched output.
                        pass
        except Exception:
            pass
        return out
    _DS.__getitem__ = _patched_getitem

    _PRED_PATH = _P({pred_json!r})
    _FPS_PATH  = _P({fps_json!r})
    _store = {{"items": [], "lock": _thr.Lock(), "epoch": -1,
              "tot_ms": 0.0, "tot_imgs": 0}}

    _orig_eval = _MCI.eval_step
    def _patched_eval(self, batch, batch_idx=None, log_prefix=None):
        # Reset accumulator on the first batch of every val epoch so the
        # final file only contains predictions from the last completed epoch.
        try:
            cur_ep = int(getattr(self.trainer, "current_epoch", 0))
        except Exception:
            cur_ep = 0
        with _store["lock"]:
            if cur_ep != _store["epoch"]:
                _store["items"] = []
                _store["tot_ms"] = 0.0
                _store["tot_imgs"] = 0
                _store["epoch"] = cur_ep
        # Run the original logic for metric updates etc. Keep this resilient to
        # upstream signature drift; prediction dumping must not disappear just
        # because EoMT renamed/dropped one optional eval_step kwarg.
        try:
            _orig_eval(self, batch, batch_idx=batch_idx, log_prefix=log_prefix)
        except TypeError:
            try:
                _orig_eval(self, batch, batch_idx)
            except TypeError:
                _orig_eval(self, batch)

        # Mirror the prediction code path to also dump JSON; time the
        # forward+post-process so we can emit fps_eval at the end. We use
        # torch.cuda.synchronize() before & after to count GPU work, not
        # just Python overhead — otherwise a CPU-side timer underreports.
        import torch.nn.functional as _F
        imgs, targets = batch
        _is_cuda = bool(getattr(self, "device", None) and str(self.device).startswith("cuda"))
        if _is_cuda:
            try: _t.cuda.synchronize()
            except Exception: pass
        _t0 = _time.perf_counter()
        img_sizes = [im.shape[-2:] for im in imgs]
        transformed_imgs = self.resize_and_pad_imgs_instance_panoptic(imgs)
        mask_logits_per_layer, class_logits_per_layer = self(transformed_imgs)
        # Use the LAST layer's predictions (final, post-refinement).
        mask_logits = mask_logits_per_layer[-1]
        class_logits = class_logits_per_layer[-1]
        mask_logits = _F.interpolate(mask_logits, self.img_size, mode="bilinear")
        mask_logits = self.revert_resize_and_pad_logits_instance_panoptic(
            mask_logits, img_sizes
        )
        batch_rows = []
        for j in range(len(mask_logits)):
            scores = class_logits[j].softmax(dim=-1)[:, :-1]
            labels = (
                _t.arange(scores.shape[-1], device=self.device)
                .unsqueeze(0).repeat(scores.shape[0], 1).flatten(0, 1)
            )
            topk_scores, topk_indices = scores.flatten(0, 1).topk(
                self.eval_top_k_instances, sorted=False
            )
            labels = labels[topk_indices]
            topk_indices = topk_indices // scores.shape[-1]
            ml = mask_logits[j][topk_indices]
            masks = ml > 0
            mask_scores = (
                ml.sigmoid().flatten(1) * masks.flatten(1)
            ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
            scores = topk_scores * mask_scores
            # Resolve file name from the val dataset's image list using ordinal
            # indexing (val DataLoader uses SequentialSampler).  This avoids
            # relying on _bench_file_name monkey-patches that do not survive
            # forkserver DataLoader workers.  Falls back to the target dict if
            # the dataset is unreachable (e.g. trainer sanity-check step).
            file_name = f"unknown_{{batch_idx or 0}}_{{j}}.jpg"
            try:
                _ds = self.trainer.datamodule.val_dataset
                _idx = int(batch_idx or 0) * len(mask_logits) + j
                if _idx < len(_ds.imgs):
                    file_name = _P(_ds.imgs[_idx]).name
            except Exception:
                if isinstance(targets, dict) and "_bench_file_name" in targets:
                    _fn_col = targets["_bench_file_name"]
                    file_name = _fn_col[j] if j < len(_fn_col) else file_name
                elif isinstance(targets, (list, tuple)) and j < len(targets) and isinstance(targets[j], dict):
                    file_name = targets[j].get("_bench_file_name", file_name)
            # Keep only predictions with a sane score; COCOeval needs a score.
            keep = (scores > 1e-3).nonzero(as_tuple=True)[0].tolist()
            for k in keep:
                m = masks[k].detach().cpu().numpy().astype("uint8")
                if m.sum() == 0: continue
                import numpy as _np
                rle = _mu.encode(_np.asfortranarray(m))
                if isinstance(rle.get("counts"), bytes):
                    rle["counts"] = rle["counts"].decode("ascii")
                ys, xs = _np.where(m > 0)
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                batch_rows.append({{
                    "_file_name": file_name,
                    # category_id is 1-indexed to match our GT (one_indexed_cats=True).
                    "category_id": int(labels[k].item()) + 1,
                    "score": float(scores[k].item()),
                    "bbox": [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
                    "segmentation": rle,
                }})
        # End the timed region: sync GPU then accumulate ms & image count.
        if _is_cuda:
            try: _t.cuda.synchronize()
            except Exception: pass
        _dt_ms = (_time.perf_counter() - _t0) * 1000.0
        with _store["lock"]:
            _store["items"].extend(batch_rows)
            _store["tot_ms"] += float(_dt_ms)
            _store["tot_imgs"] += int(len(mask_logits))
            # Atomically write the current accumulator to disk every step;
            # cheap for benchmark-sized test sets and ensures we always have
            # the latest snapshot even if the run is interrupted.
            try:
                tmp = _PRED_PATH.with_suffix(".json.tmp")
                tmp.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp, "w") as _fh:
                    _json.dump(_store["items"], _fh)
                tmp.replace(_PRED_PATH)
            except Exception as _e:
                print(f"[bench_any] pred dump write failed: {{_e}}")
            # Persist a rolling FPS estimate (images per second over the
            # accumulated wall-time of this epoch's eval forward passes).
            try:
                _fps = (_store["tot_imgs"] * 1000.0 / _store["tot_ms"]) if _store["tot_ms"] > 0 else 0.0
                _ftmp = _FPS_PATH.with_suffix(".json.tmp")
                with open(_ftmp, "w") as _ffh:
                    _json.dump({{"fps_eval": float(_fps),
                                "n_imgs": int(_store["tot_imgs"]),
                                "ms_total": float(_store["tot_ms"])}}, _ffh)
                _ftmp.replace(_FPS_PATH)
            except Exception:
                pass
    _MCI.eval_step = _patched_eval
    print(f"[bench_any] EoMT prediction dump enabled -> {{_PRED_PATH}}")

_bench_install_pred_dump()
# ---------------------------------------------------------------------------
"""
    log_block = (
        "import logging, warnings\n"
        "logging.getLogger('lightning.pytorch').setLevel(logging.WARNING)\n"
        "logging.getLogger('lightning').setLevel(logging.WARNING)\n"
        "warnings.filterwarnings('ignore', message=r'.*isinstance\\(treespec, LeafSpec\\).*')\n"
        "try:\n"
        "    import lightning.fabric.utilities.rank_zero as _brz\n"
        "    import lightning.pytorch.utilities.rank_zero as _prz\n"
        "    _brz.rank_zero_info = lambda *a, **k: None\n"
        "    _prz.rank_zero_info = lambda *a, **k: None\n"
        "except Exception: pass\n"
    )
    src = (f"import sys, runpy, os\n{log_block}{shm_block}sys.path.insert(0, {str(repo_dir)!r}){alias_block}\n"
           f"{patch_block}\n"
           f"if __name__ == '__main__':\n"
           f"    _script = {str(Path(repo_dir) / 'main.py')!r}\n"
           f"    sys.argv[0] = _script\n"
           f"    runpy.run_path(_script, run_name='__main__')\n")
    Path(launcher_path).write_text(src)


def harvest_metrics(run_dir):
    """Read latest Lightning `metrics.csv`. EoMT logs torchmetrics MAP under the
    `metrics/val_ap_*` namespace (see training/lightning_module.py). Values of
    -1.0 are torchmetrics' "no GT" sentinel and are clamped to 0.0 here so the
    bench summary doesn't show negatives."""
    out = zero_metrics()
    # Pick the highest-numbered version_*/metrics.csv (the run we just finished).
    # Globbing without sorting picks an arbitrary one on some filesystems and
    # was causing EoMT to harvest from a stale version_0 with num_classes=80.
    candidates = sorted(
        Path(run_dir).rglob("metrics.csv"),
        key=lambda p: int(re.search(r"version_(\d+)", str(p)).group(1))
                       if re.search(r"version_(\d+)", str(p)) else -1,
    )
    if not candidates:
        print(f"⚠️ no metrics.csv under {run_dir}; metrics = 0"); return out
    csvs = [candidates[-1]]  # Use only the latest version
    rows = list(csv.DictReader(open(csvs[0])))
    if not rows: return out
    def _g(row, *ks):
        for k in ks:
            v = row.get(k)
            if v not in (None, ""):
                try: return max(0.0, float(v))   # clamp -1 sentinel to 0
                except ValueError: pass
        return 0.0
    # Walk back-to-front for the last row that has a real val_ap_all value.
    for row in reversed(rows):
        raw = row.get("metrics/val_ap_all") or row.get("val/map") or row.get("val_ap_all") or ""
        if raw.strip() == "": continue
        out["seg_mAP50_95"] = _g(row, "metrics/val_ap_all", "val/map", "val_map", "val_ap_all")
        out["seg_mAP50"]    = _g(row, "metrics/val_ap_50_all", "val/map_50", "val_map_50", "val_ap_50_all")
        out["seg_mAP75"]    = _g(row, "metrics/val_ap_75_all", "val/map_75", "val_map_75", "val_ap_75_all")
        out["seg_AR_100"]   = _g(row, "metrics/val_mar_100_all", "val/mar_100", "val_mar_100", "val_mar_100_all")
        # Size-bucketed mask AP from torchmetrics MeanAveragePrecision. EoMT's
        # lightning_module logs each MAP key under the same 'metrics/val_ap_*'
        # namespace as the rest; tag spelling fallbacks cover possible variants.
        out["seg_mAP_S"]    = _g(row, "metrics/val_ap_small_all",  "val/map_small",  "val_map_small",  "val_ap_small_all")
        out["seg_mAP_M"]    = _g(row, "metrics/val_ap_medium_all", "val/map_medium", "val_map_medium", "val_ap_medium_all")
        out["seg_mAP_L"]    = _g(row, "metrics/val_ap_large_all",  "val/map_large",  "val_map_large",  "val_ap_large_all")
        break
    return out


if __name__ == "__main__":
    import multiprocessing
    if sys.platform != "win32":
        try: multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError: pass

    # ---------------------------------------------------------------- Args ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="eomt-large", help="eomt-{small,base,large,giant}")
    ap.add_argument("--task", default="segment", choices=["detect", "segment"])
    ap.add_argument("--epochs", default=12, type=int)
    ap.add_argument("--patience", default=15, type=int)
    ap.add_argument("--batch_sz", default=0, type=float,
                    help="Micro-batch size. 0 (default) or float<1 = auto, sized to the EoMT variant"
                         " so it fits 16GB (giant=1, large=1, base=2, small=4). Combine with"
                         " --grad_accum_steps to keep effective batch high.")
    ap.add_argument("--grad_accum_steps", default=0, type=int,
                    help="Lightning accumulate_grad_batches. 0 = pick automatically so that"
                         " micro_bs * grad_accum >= --batch_effective (default 8), the floor"
                         " shared across all tuners in this benchmark suite.")
    ap.add_argument("--batch_effective", default=8, type=int,
                    help="Target effective batch size; auto grad_accum = ceil(N / micro_bs).")
    ap.add_argument("--compile", dest="compile", action="store_true",
                    help="Enable torch.compile/inductor. OFF by default because its recompile + workspace"
                         " caches grow VRAM monotonically across steps on EoMT (start ~1GB -> OOM in <1"
                         " epoch on 16GB). Leaving compile off is the single biggest stability win here.")
    ap.add_argument("--no_compile", dest="compile", action="store_false",
                    help="(default) keep torch.compile disabled.")
    ap.set_defaults(compile=False)
    ap.add_argument("--imgsz", default=640, type=int)
    ap.add_argument("--lr", default=1e-4, type=float)
    ap.add_argument("--threads", default=4, type=int)
    ap.add_argument("--resume", action='store_true')
    ap.add_argument("--inference_only", action='store_true')
    ap.add_argument("--tune", action='store_true')
    ap.add_argument("--tune_iter", default=5, type=int)
    ap.add_argument("--repo_dir", default="")
    ap.add_argument("--dinov3_dir", default=os.environ.get("DINOV3_DIR", ""),
                    help="Local HF-style DINOv3 dir (config.json + weights). Used as backbone_name.")
    ap.add_argument("--dinov3_weights", default=os.environ.get("DINOV3_WEIGHTS", ""),
                    help="Accepted for parity; if no --dinov3_dir is given, a .pth alone is NOT loaded.")
    ap.add_argument("--hub_token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--aug_policy", choices=["native", "off", "fair"], default="native",
                    help="When != native, inject the shared Albumentations pipeline into the "
                         "templated launcher's _patched_getitem so the train-split EoMT dataset's "
                         "(image, masks) pair is augmented BEFORE upstream transforms run.")
    ap.add_argument("--seg_via_sam2", action='store_true',
                    help="After eval, prompt SAM2 with predicted bboxes to compute sam2_* mAP.")
    ap.add_argument("--sam2_ckpt", default=os.environ.get("SAM2_CKPT", ""))
    ap.add_argument("--sam2_config", default=os.environ.get("SAM2_CONFIG", ""))
    args = ap.parse_args()
    # R.6: reject fractional --batch_sz loudly. The Ultralytics-style fractional
    # auto-batch (0<bs<1) does NOT apply to EoMT; a fraction silently becomes the
    # variant auto-size (int(0.98)=0 would zero-batch), so refuse it upfront
    # rather than let a miscaled run train on empty batches. 0 (the default) is
    # the explicit "auto-size to the EoMT variant" sentinel and stays valid.
    if 0 < args.batch_sz < 1:
        print(f"❌ --batch_sz must be 0 (auto) or >= 1 (got {args.batch_sz}); "
              "fractional batch_sz is Ultralytics-only and would silently mis-size EoMT")
        sys.exit(1)
    if args.repo_dir: REPO_CACHE = Path(args.repo_dir).expanduser().resolve()
    if args.hub_token: os.environ["HF_TOKEN"] = args.hub_token

    ORIG_DATA = Path(args.data).resolve()
    if not ORIG_DATA.is_dir(): print("❌ Data directory not found"); sys.exit(1)
    DATA_YAML = ORIG_DATA / 'data.yaml'
    if not DATA_YAML.exists(): print("❌ data.yaml not found"); sys.exit(1)
    DATA_NAME = ORIG_DATA.name

    # Bench contract: evaluate test when present. EoMT only evaluates Lightning
    # validation, so coco_to_eomt_zips maps generated COCO test -> val2017.
    import yaml as _yaml
    with open(DATA_YAML) as _yf:
        _ycfg = _yaml.safe_load(_yf) or {}
    EVAL_SPLIT = "test" if "test" in _ycfg else ("valid" if "valid" in _ycfg else "val")

    print("🔄 Converting YOLO → COCO …")
    COCO_DIR, CLASSES = yolo_to_coco_layout(DATA_YAML, DATA_NAME)
    print(f"✅ COCO ready (classes: {CLASSES}; eval_split={EVAL_SPLIT})")

    REPO_DIR, HAS_NATIVE_LIGHTNING = ensure_repo()
    backbone_id = pick_backbone(args.model)
    local_backbone, local_warn = resolve_local_hf_backbone_dir(args.dinov3_dir)
    if local_warn:
        print(local_warn)
    if local_backbone:
        backbone_id = local_backbone
        print(f"🔗 Using local HF DINOv3 dir: {backbone_id}")
    elif args.dinov3_weights:
        print(f"⚠️ --dinov3_weights without --dinov3_dir is not supported in this slim wrapper; "
              f"falling back to HF id {backbone_id}. Pass --dinov3_dir <hf_dir> with config.json.")

    MODEL_STEM = Path(args.model).stem
    _env_run = os.environ.get("BENCH_RUN_NAME")
    RUN_NAME = _env_run if _env_run else f"{DATA_NAME[:15]}_{MODEL_STEM}"
    TASK_DIR = "segment"
    # RUN_DIR on disk (not /dev/shm) — Lightning checkpoints fill tmpfs on long runs.
    # Routed under ~/bench_any/ so nothing lands in the project / dataset directory.
    RUN_DIR = Path.home() / 'bench_any' / 'runs' / TASK_DIR / RUN_NAME
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ZIPS_DIR = RUN_DIR / "_zips"
    os.environ.setdefault('WANDB_PROJECT', "EoMT-segment")
    os.environ.setdefault('WANDB_DIR', str(RUN_DIR))

    print("🗜️  Packing zips for COCOInstance …")
    ZIPS_DIR, _ = coco_to_eomt_zips(COCO_DIR, ZIPS_DIR)
    NUM_CLASSES = len(CLASSES)

    # Variant-aware micro-batch defaults for a 16GB GPU. EoMT-large is 314M params and
    # emits mask logits per encoder block (Mask2Former-style); per-sample activation cost
    # is ~5x a YOLO11x sample, so we cannot use the same bs=8 the user passes for YOLO.
    def _eomt_auto_bs(name):
        n = name.lower()
        if "giant" in n: return 1
        if "large" in n: return 1
        if "base"  in n: return 2
        if "small" in n: return 4
        return 1
    SAFE_MICRO_BS = _eomt_auto_bs(args.model)
    if args.batch_sz >= 1:
        requested_micro_bs = max(1, int(args.batch_sz))
        MICRO_BS = min(requested_micro_bs, SAFE_MICRO_BS)
        if MICRO_BS != requested_micro_bs:
            print(f"⚠️ EoMT {args.model}: capping micro_bs {requested_micro_bs} → {MICRO_BS} to avoid OOM; "
                  "effective batch is preserved with grad accumulation.")
    else:
        MICRO_BS = SAFE_MICRO_BS
    # Auto grad-accum so the effective batch is ALWAYS >= 8. This matches the floor used by
    # every other tuner in the bench (Ultralytics nbs=64, RF-DETR auto target=16, etc.) and
    # keeps gradient noise comparable across models on large datasets.
    EFFECTIVE_BS_FLOOR = max(1, int(args.batch_effective))
    GRAD_ACCUM = (args.grad_accum_steps if args.grad_accum_steps > 0
                  else max(1, (EFFECTIVE_BS_FLOOR + MICRO_BS - 1) // MICRO_BS))
    print(f"📐 micro_bs={MICRO_BS}  grad_accum={GRAD_ACCUM}  effective_bs={MICRO_BS * GRAD_ACCUM}"
          f"  compile={'on' if args.compile else 'off'}")

    YAML_PATH = RUN_DIR / "config.yaml"
    # BUG 3B: in inference-only mode, preserve the trained checkpoint BEFORE
    # write_yaml scrubs version_*/ (which deletes the trained
    # <run_name>/version_0/checkpoints/*.ckpt). The preserved copy survives the
    # scrub and is passed to `validate` via --ckpt_path below.
    trained_ckpt = _preserve_trained_ckpt(RUN_DIR) if args.inference_only else None
    write_yaml(YAML_PATH, ZIPS_DIR, NUM_CLASSES, backbone_id,
               max_epochs=args.epochs, imgsz=args.imgsz,
               batch_sz=MICRO_BS, lr=args.lr,
               num_workers=(0 if args.threads < 0
                            else (args.threads or min(8, os.cpu_count() or 8))),
               run_name=RUN_NAME, run_dir=RUN_DIR,
               grad_accum_steps=GRAD_ACCUM,
               patience=args.patience)

    LAUNCHER = RUN_DIR / "_launch.py"
    write_launcher(LAUNCHER, REPO_DIR, NUM_CLASSES, HAS_NATIVE_LIGHTNING,
                   aug_policy=args.aug_policy)
    _eomt_aug_status = "applied" if args.aug_policy in ("off", "fair") else "native"
    print(f"[bench] aug_policy={args.aug_policy}: eomt -> {_eomt_aug_status}")

    subcmd = "validate" if args.inference_only else "fit"
    cmd = [sys.executable, str(LAUNCHER), subcmd, "-c", str(YAML_PATH)]
    if not args.compile: cmd.append("--compile_disabled")
    if args.inference_only or args.resume:
        # BUG 3B: prefer the preserved trained checkpoint (copied by
        # _preserve_trained_ckpt before the scrub so it survives); fall back to
        # the latest *.ckpt by mtime for the --resume path (dir may not have
        # been scrubbed, e.g. a direct resume into an existing trained dir).
        # Broadened from `args.resume` to `args.inference_only or args.resume`
        # so inference-only loads the trained checkpoint even when --resume is
        # not co-passed. Lightning's _parse_ckpt_path accepts an arbitrary
        # absolute file path, so the top-level _bench_trained.ckpt works.
        last = trained_ckpt
        if last is None or not Path(last).is_file():
            ckpts = sorted(RUN_DIR.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
            last = str(ckpts[0]) if ckpts else None
        if last:
            cmd += ["--ckpt_path", str(last)]
            print(f"🔁 {subcmd}: loading trained checkpoint {Path(last).name}")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_DIR}:{env.get('PYTHONPATH', '')}"
    # Fight allocator fragmentation: EoMT's per-block mask logits + variable-length matcher
    # tensors leave the caching allocator full of medium holes; expandable_segments lets the
    # allocator return memory to the GPU instead of growing residency every step.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"🚂 EoMT {subcmd}")
    print(f"   Command: {' '.join(cmd)}")
    print(f"   CWD: {REPO_DIR}")
    print(f"   Config: {YAML_PATH}")
    print(f"   Zips: {ZIPS_DIR}")
    print(f"   Run dir: {RUN_DIR}")
    sys.stdout.flush()
    # Stream child output, filtering only Lightning's promotional tip lines.
    # Tracebacks, OOMs, metrics, and bench diagnostics still pass through.
    quiet_parts = (
        "💡 Tip: For seamless cloud logging",
        "💡 Tip: For seamless cloud uploads",
    )
    def _run_child(child_cmd):
        proc = subprocess.Popen(
            child_cmd, cwd=str(REPO_DIR), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if any(part in line for part in quiet_parts):
                continue
            print(line, end="")
        return proc.wait()

    rc = _run_child(cmd)
    if rc != 0: print(f"⚠️ EoMT rc={rc}; harvesting whatever's available")
    else: print(f"✓ EoMT subprocess completed successfully")

    # ---- Score the SELECTED checkpoint, not the last epoch ----------------
    # Lightning calls the eval hook from inside the training loop, so a `fit`
    # run leaves metrics.csv ending on whatever epoch was evaluated last. yolo
    # and rtdetr both publish their best.pt, so publishing EoMT's last epoch
    # made the bench comparison unfair. The monitored ModelCheckpoint injected
    # in write_config selects on metrics/val_ap_all; re-validating from it here
    # writes a fresh version_N/metrics.csv (which harvest_metrics already
    # prefers) and re-fires the pred-dump patch, so predictions.json comes from
    # the same weights as the published number.
    if subcmd == "fit" and rc == 0:
        _best_ckpts = sorted(RUN_DIR.rglob("_bench_best*.ckpt"),
                             key=lambda p: p.stat().st_mtime, reverse=True)
        if _best_ckpts:
            _best = _best_ckpts[0]
            print(f"🏅 EoMT: re-validating the selected checkpoint {_best.name}")
            _vcmd = [sys.executable, str(LAUNCHER), "validate", "-c", str(YAML_PATH)]
            if not args.compile: _vcmd.append("--compile_disabled")
            _vcmd += ["--ckpt_path", str(_best)]
            _vrc = _run_child(_vcmd)
            if _vrc != 0:
                print(f"⚠️ EoMT best-checkpoint validate rc={_vrc}; the harvested "
                      f"metrics fall back to the LAST evaluated epoch")
        else:
            print("⚠️ EoMT: no monitored _bench_best*.ckpt found; the harvested "
                  "metrics are the LAST evaluated epoch, not the selected checkpoint")

    metrics = harvest_metrics(RUN_DIR)

    # ---- Read fps_eval persisted by the monkey-patched eval_step ----------
    # EoMT can't run a standalone fps probe (would require reinstantiating
    # the full LightningModule from the config); instead the patched
    # eval_step measures forward-pass wall-time per image and writes a tiny
    # JSON next to predictions.json. Read it here so the bench CSV gets a
    # populated fps_eval column.
    try:
        fps_file = RUN_DIR / "_bench_fps.json"
        if fps_file.is_file():
            _fps_data = json.loads(fps_file.read_text())
            _fps = float(_fps_data.get("fps_eval", 0.0))
            if _fps > 0:
                metrics["fps_eval"] = _fps
                print(f"   ⏱  fps_eval = {_fps:.2f} (over {_fps_data.get('n_imgs', '?')} test imgs)")
    except Exception as _e:
        print(f"   ⚠️ fps_eval read failed: {_e}")

    # ---- Derive box metrics from segmentation predictions ----
    # EoMT only computes segm mAP. Derive bbox mAP from the same predictions.
    # Both names are read again by the SAM2 block below, so bind them before the
    # try: a training crash leaves no predictions.json and the old code then died
    # on an unbound bbox_json, masking the real failure.
    pred_json = None
    bbox_json = None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import bench_eval as _be
        # EoMT/Lightning writes predictions somewhere - find them
        for cand in [RUN_DIR / "predictions.json",
                     RUN_DIR / "test_predictions.json",
                     RUN_DIR / RUN_DIR.name / "predictions.json"]:
            if cand.is_file():
                pred_json = cand
                break
        # Try version_* dirs
        if not pred_json:
            ver_dirs = sorted(RUN_DIR.rglob("version_*"),
                            key=lambda p: int(re.search(r"version_(\d+)", str(p)).group(1))
                                         if re.search(r"version_(\d+)", str(p)) else -1)
            for vd in reversed(ver_dirs):
                if (vd / "predictions.json").is_file():
                    pred_json = vd / "predictions.json"
                    break

        if pred_json and pred_json.is_file():
            gt_json = RUN_DIR / "_bench_gt_coco.json"
            # Use the same split we detected earlier
            _be.yolo_split_to_coco_gt(str(Path(args.data) / "data.yaml"), EVAL_SPLIT, str(gt_json),
                                     one_indexed_cats=True)  # EoMT uses 1-indexed
            # Our bench_any monkey-patch dumps predictions with `_file_name`
            # (because the EoMT dataset doesn't carry COCO image_ids).
            # Translate those to GT image_ids before COCOeval can consume them.
            try:
                with open(gt_json) as _gfh: _gt_d = json.load(_gfh)
                _images = _gt_d.get("images", [])
                _gt_ids = {int(im["id"]) for im in _images}
                _name_to_id = {im["file_name"]: im["id"] for im in _images}
                _stem_to_id = {Path(im["file_name"]).stem: im["id"] for im in _images}

                with open(pred_json) as _pfh: _preds = json.load(_pfh)
                if _preds and isinstance(_preds[0], dict) and "_file_name" in _preds[0]:
                    _resolved = []
                    _warned = 0
                    for _p in _preds:
                        if "image_id" in _p:
                            try:
                                if int(_p["image_id"]) in _gt_ids:
                                    _resolved.append({k: v for k, v in _p.items() if k != "_file_name"})
                                    continue
                            except Exception:
                                pass
                        _fn = _p.get("_file_name")
                        if not _fn: continue
                        _iid = _name_to_id.get(_fn)
                        if _iid is None:
                            _iid = _stem_to_id.get(Path(_fn).stem)
                        if _iid is None and str(_fn).startswith("unknown_"):
                            try:
                                _idx = int(str(_fn).split("_")[1])
                                if 0 <= _idx < len(_images):
                                    _iid = _images[_idx]["id"]
                            except Exception:
                                pass
                        if _iid is None:
                            if _warned < 10:
                                print(f"   ⚠️ Dropping prediction with unknown EoMT file name: {_fn}")
                            _warned += 1
                            continue

                        _p2 = {k: v for k, v in _p.items() if k != "_file_name"}
                        _p2["image_id"] = int(_iid)
                        _resolved.append(_p2)
                    _resolved_path = RUN_DIR / "_bench_predictions_resolved.json"
                    with open(_resolved_path, "w") as _rfh: json.dump(_resolved, _rfh)
                    print(f"   🔧 Resolved {len(_resolved)}/{len(_preds)} predictions via _file_name/stem → image_id")
                    if _warned > 10:
                        print(f"   ⚠️ Suppressed {_warned - 10} additional EoMT unknown-file warnings")
                    if _resolved:
                        pred_json = _resolved_path
                    else:
                        print(f"   ⚠️ EoMT prediction file had rows but none matched GT image names: {pred_json}")
            except Exception as _e_res:
                print(f"   ⚠️ _file_name→image_id resolution failed: {_e_res}")
            # Fix image_id mismatch if needed
            fixed_pred = RUN_DIR / "_bench_predictions_fixed.json"
            n_fixed = _be.fix_yolo_pred_image_ids(str(pred_json), str(gt_json), str(fixed_pred))
            if n_fixed > 0:
                print(f"   🔧 Fixed {n_fixed} prediction image_ids to match GT")
                pred_json = fixed_pred
            # Derive boxes from masks; also evaluate the original segm preds
            # so seg_P/R/F1 + seg S/M/L + AR_100 get populated from the same
            # COCOeval pass that already powers the published mask mAP.
            bbox_json = RUN_DIR / "_bench_bbox_from_masks.json"
            n = _be.masks_to_box_preds(str(pred_json), str(bbox_json))
            if n > 0:
                extra = _be.coco_eval_full(str(gt_json),
                                           str(bbox_json), str(pred_json))
                # Always take bbox metrics from COCOeval (EoMT has no native
                # box head). For seg, only fill cells the trainer harvest
                # left at 0 — keep the official seg_mAP* from the trainer's
                # logger when present.
                for k, v in extra.items():
                    if k.startswith("box_"):
                        metrics[k] = v
                    elif k.startswith("seg_"):
                        if float(metrics.get(k, 0.0) or 0.0) == 0.0:
                            metrics[k] = v
                print(f"   📦 Derived {n} box predictions from masks, computed box+seg mAP / P/R/F1")
            else:
                try:
                    with open(pred_json) as _pfh: _dbg_preds = json.load(_pfh)
                    _dbg_n = len(_dbg_preds) if isinstance(_dbg_preds, list) else -1
                    _dbg_seg = sum(1 for _p in _dbg_preds if isinstance(_p, dict) and _p.get("segmentation")) if isinstance(_dbg_preds, list) else 0
                    _dbg_bbox = sum(1 for _p in _dbg_preds if isinstance(_p, dict) and _p.get("bbox")) if isinstance(_dbg_preds, list) else 0
                    print(f"   ⚠️ No valid masks found in {pred_json} (rows={_dbg_n}, segmentation_rows={_dbg_seg}, bbox_rows={_dbg_bbox})")
                except Exception as _dbg_e:
                    print(f"   ⚠️ No valid masks found in {pred_json}; debug read failed: {_dbg_e}")
        else:
            print(f"   ⚠️ No predictions.json found for EoMT box derivation")
    except Exception as e:
        print(f"⚠️ Box metrics derivation failed: {e}")

    # Use the derived bbox-only predictions file (masks_to_box_preds output)
    # — it has clean bbox entries which sam2_seg_from_boxes expects.
    _sam2_bbox_src = Path(bbox_json) if bbox_json else Path(pred_json) if pred_json else None
    if args.seg_via_sam2 and args.sam2_ckpt and args.sam2_config:
        try:
            _cfg_yaml = yaml.safe_load(open(str(Path(args.data) / "data.yaml")))
            _root = Path(_cfg_yaml.get("path", Path(args.data)))
            if not _root.is_absolute():
                _root = (Path(args.data) / _root).resolve()
            _split = _cfg_yaml.get("test") or _cfg_yaml.get("val")
            _imgs = (_root / _split).resolve() if _split and not Path(_split).is_absolute() else Path(_split) if _split else None
        except Exception:
            _imgs = None
        if _imgs and _sam2_bbox_src and _sam2_bbox_src.is_file():
            sam_out = RUN_DIR / "_bench_sam2_seg.json"
            sam_metrics = _be.sam2_seg_from_boxes(
                str(_imgs), str(_sam2_bbox_src),
                args.sam2_ckpt, args.sam2_config,
                str(gt_json), str(sam_out))
            for k, v in sam_metrics.items():
                metrics[k] = v
            if any(v > 0 for v in sam_metrics.values()):
                print(f"   🅂 SAM2 prompted seg: {sum(1 for v in sam_metrics.values() if v > 0)} non-zero sam2_* metrics")
        else:
            print("⚠️ --seg_via_sam2 set but --sam2_ckpt/--sam2_config missing or images dir not resolvable; skipping.")

    # ---- v2: count params from the saved Lightning checkpoint -----------
    # We don't reinstantiate the model here (would re-download backbones,
    # cost VRAM, and risk env quirks). Summing tensor numels in the ckpt's
    # state_dict gives the exact deployed parameter count.
    try:
        import torch as _t
        ckpts = sorted(Path(RUN_DIR).rglob("*.ckpt"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        ckpt = next((c for c in ckpts if "last" not in c.name), ckpts[0] if ckpts else None)
        if ckpt is not None:
            sd = _t.load(str(ckpt), map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd) if isinstance(sd, dict) else {}
            n = sum(int(v.numel()) for v in sd.values()
                    if hasattr(v, "numel"))
            metrics["params_M"] = n / 1e6
    except Exception as _e:
        print(f"⚠️ params probe failed: {_e}")
    nz_parts = [f"{k}={v:.4f}" for k, v in metrics.items() if v > 0]
    if nz_parts:
        print(f"📊 {'  '.join(nz_parts)}")
    else:
        print(f"📊 all metrics = 0.0000 (harness working, training learned nothing)")

    # Report the actual micro-batch EoMT ran (e.g. capped to 1 for -large) and
    # the realised effective batch (micro_bs * grad_accum) so the bench CSV shows
    # the true per-model batch instead of the orchestrator's resolved argument.
    metrics["batch_sz"] = int(MICRO_BS)
    metrics["batch_effective"] = int(MICRO_BS * GRAD_ACCUM)

    try:
        # RUN_DIR is already under ~/bench_any/, so no separate mirror is needed.
        with open(RUN_DIR / 'results.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(metrics.keys()))
            w.writeheader(); w.writerow(metrics)
        print(f"✅ Results: {RUN_DIR}")
    except Exception as e:
        print(f"⚠️ Could not save results: {e}")

    shutil.rmtree(ZIPS_DIR, ignore_errors=True)
    print("🎉 Complete!")
    if rc != 0:
        sys.exit(rc)

"""Declarative model routing for bench_any.py.

Add new benchmark families here instead of editing the orchestrator loop. The
first matching entry wins, so put more-specific patterns before broad fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Callable, Dict, List, Sequence, Tuple


SEG_TOKENS = {"seg", "segment"}


def model_key(name: str) -> str:
    return Path(str(name)).name.lower()


def model_stem(name: str) -> str:
    return Path(str(name)).stem.lower()


def seg_or_detect(name: str) -> str:
    tokens = set(re.split(r"[-_.]+", model_stem(name)))
    return "segment" if tokens & SEG_TOKENS else "detect"


TaskResolver = Callable[[str], str]
ProjectResolver = Callable[[str], str]


def _literal_task(task: str) -> TaskResolver:
    return lambda _name: task


def _literal_project(project: str) -> ProjectResolver:
    return lambda _name: project


def _prefixed_project(prefix: str, task_fn: TaskResolver = seg_or_detect) -> ProjectResolver:
    return lambda name: prefix + task_fn(name)


@dataclass(frozen=True)
class ModelRegistryEntry:
    framework: str
    script: str
    include_all: Tuple[str, ...] = ()
    include_any: Tuple[str, ...] = ()
    exclude_any: Tuple[str, ...] = ()
    task_fn: TaskResolver = seg_or_detect
    project_fn: ProjectResolver = _prefixed_project("bench-")
    examples: Tuple[str, ...] = ()
    notes: str = ""

    def matches(self, name: str) -> bool:
        key = model_key(name)
        if self.exclude_any and any(tok in key for tok in self.exclude_any):
            return False
        if self.include_all and not all(tok in key for tok in self.include_all):
            return False
        if self.include_any and not any(tok in key for tok in self.include_any):
            return False
        return bool(self.include_all or self.include_any)

    def route_tuple(self) -> Tuple[str, Callable[[str], bool], str, TaskResolver, ProjectResolver]:
        return (self.framework, self.matches, self.script, self.task_fn, self.project_fn)


REGISTRY: Tuple[ModelRegistryEntry, ...] = (
    ModelRegistryEntry(
        framework="rtdetr_external",
        include_any=(
            "rtdetrv2", "rt-detrv2", "rtdetr-v2", "rtdetr-2", "rt-detr-2",
            "rtdetrv3", "rt-detrv3", "rtdetr-v3", "rtdetr-3", "rt-detr-3",
            "rtdetrv4", "rt-detrv4", "rtdetr-v4", "rtdetr-4", "rt-detr-4",
        ),
        script="rtdetr_external_any_tune.py",
        task_fn=_literal_task("detect"),
        project_fn=_literal_project("RTDETR-external-detect"),
        examples=(
            "rtdetr-2.pt", "rtdetrv2_r50vd_6x_coco_ema.pth",
            "rtdetr-3.pt", "rtdetr-3-r34.pt", "rtdetr-3-r50.pt", "rtdetr-4.pt",
        ),
        notes="External RT-DETR route. rtdetr-2.pt runs through Transformers; local RT-DETRv2 .pth and RT-DETRv4 .pth run through the RT-DETRv4 PyTorch repo. RT-DETRv3 upstream is PaddleDetection; in this Python 3.14 venv it runs only as a clearly marked PyTorch compatibility route using converted .pdparams tensors, not the exact DINOv3Head/o2m official architecture.",
    ),
    ModelRegistryEntry(
        framework="rfdetr",
        include_any=("rfdetr", "rf-detr"),
        script="rfdetr_any_tune.py",
        task_fn=seg_or_detect,
        project_fn=_prefixed_project("RFDETR-"),
        examples=("rf-detr-seg-large", "rf-detr-seg-xlarge.pt", "rf-detr-base-det"),
        notes="Native RF-DETR detector/segmenter wrapper.",
    ),
    ModelRegistryEntry(
        framework="rtdetr",
        include_any=("rtdetr", "rt-detr"),
        script="rtdetr_any_tune.py",
        task_fn=_literal_task("detect"),
        project_fn=_literal_project("RTDETR-detect"),
        examples=("rtdetr-x.pt", "rtdetr-l.pt"),
        notes="Ultralytics RT-DETR route. Use only weights the Ultralytics RTDETR class can load.",
    ),
    ModelRegistryEntry(
        framework="dinov3_m2f",
        include_all=("dino",),
        include_any=("m2f", "mask2former"),
        script="dinov3_m2f_any_tune.py",
        task_fn=_literal_task("segment"),
        project_fn=_literal_project("DINOv3-M2F-segment"),
        examples=("dinov3-m2f-smallplus", "dinov3-m2f-vitl16", "dinov3-m2f-vith16plus"),
        notes="DINOv3 + Mask2Former instance segmentation.",
    ),
    ModelRegistryEntry(
        framework="eomt",
        include_any=("eomt",),
        script="eomt_any_tune.py",
        task_fn=_literal_task("segment"),
        project_fn=_literal_project("EoMT-segment"),
        examples=("eomt-smallplus", "eomt-base", "eomt-large", "eomt-vith16plus"),
        notes="EoMT instance segmentation with DINOv3 backbone aliases.",
    ),
    ModelRegistryEntry(
        framework="dinov3_semseg",
        include_all=("dino",),
        include_any=("semseg", "sem-seg", "seghead"),
        script="dinov3_semseg_any_tune.py",
        task_fn=_literal_task("segment"),
        project_fn=_literal_project("DINOv3-SemSeg"),
        examples=("dinov3-semseg-vits16plus", "dinov3-semseg-vits16", "dinov3-semseg-vitb16"),
        notes="Frozen DINOv3 + lightweight semantic head. Use foreground metrics for paper tables.",
    ),
    ModelRegistryEntry(
        framework="yolo",
        include_any=("",),
        script="yolo_any_tune.py",
        task_fn=seg_or_detect,
        project_fn=_prefixed_project("YOLO-"),
        examples=("yolo11x-seg.pt", "yolo26x-seg.pt", "yolo11x.pt"),
        notes="Fallback route for YOLO-style Ultralytics weights.",
    ),
)


MODEL_SETS: Dict[str, Tuple[str, ...]] = {
    "smoke_core": (
        "dinov3-m2f-smallplus",
        "eomt-large",
        "dinov3-semseg-vits16plus",
    ),
    "dinov3_paper_ready": (
        "dinov3-m2f-vitl16",
        "dinov3-m2f-vith16plus",
        "eomt-smallplus",
        "eomt-vitl16",
        "dinov3-semseg-vitb16",
    ),
    "sam_prompt_table": (
        "rtdetr-x.pt",
        "yolo11x.pt",
    ),
}


@dataclass(frozen=True)
class ModelCatalogItem:
    name: str
    framework: str
    description: str
    local_hint: str = ""


MODEL_CATALOG: Tuple[ModelCatalogItem, ...] = (
    ModelCatalogItem("yolo11x-seg.pt", "yolo", "YOLO11x segmentation baseline; resolves to ~/yolo_general/models/yolo11x-seg.pt when present.", "yolo11x-seg.pt"),
    ModelCatalogItem("yolo26x-seg.pt", "yolo", "YOLO26x segmentation baseline; resolves to ~/yolo_general/models/yolo26x-seg.pt when present.", "yolo26x-seg.pt"),
    ModelCatalogItem("rtdetr-x.pt", "rtdetr", "Ultralytics RT-DETR-X detector baseline; resolves to ~/bench_any/detr_weights/rtdetr-x.pt when present.", "rtdetr-x.pt"),
    ModelCatalogItem("rtdetr-2.pt", "rtdetr_external", "RT-DETRv2 R50vd via Transformers. Downloads HF weights to ~/bench_any/detr_weights/rtdetr-2_hf and runs train/eval.", "rtdetr-2_hf/model.safetensors"),
    ModelCatalogItem("rtdetrv2_r50vd_6x_coco_ema.pth", "rtdetr_external", "Local RT-DETRv2 R50vd EMA checkpoint. Runs through the RT-DETRv4 repo's RT-DETRv2 config, not Transformers.", "rtdetrv2_r50vd_6x_coco_ema.pth"),
    ModelCatalogItem("rtdetr-3.pt", "rtdetr_external", "RT-DETRv3 R18 compatibility alias. Uses upstream rtdetrv3_r18vd_6x_coco.yml metadata and converted Paddle weights, but falls back to PyTorch compatibility because Paddle is unavailable in this venv.", "rtdetrv3_r18vd_6x_coco.pdparams"),
    ModelCatalogItem("rtdetr-3-r18.pt", "rtdetr_external", "RT-DETRv3 R18 compatibility alias; lowest VRAM v3 path. Not exact official Paddle runtime in Python 3.14.", "rtdetrv3_r18vd_6x_coco.pdparams"),
    ModelCatalogItem("rtdetrv3_r18vd_6x.pdparams", "rtdetr_external", "Direct local RT-DETRv3 R18 Paddle checkpoint; compatibility-converted without Paddle at runtime.", "rtdetrv3_r18vd_6x.pdparams"),
    ModelCatalogItem("rtdetr-3-r34.pt", "rtdetr_external", "RT-DETRv3 R34 compatibility alias. Uses local rtdetrv3_r34vd_6x.pdparams; not exact official Paddle runtime in Python 3.14.", "rtdetrv3_r34vd_6x.pdparams"),
    ModelCatalogItem("rtdetrv3_r34vd_6x.pdparams", "rtdetr_external", "Direct local RT-DETRv3 R34 Paddle checkpoint; compatibility-converted without Paddle at runtime.", "rtdetrv3_r34vd_6x.pdparams"),
    ModelCatalogItem("rtdetr-3-r50.pt", "rtdetr_external", "RT-DETRv3 R50 compatibility alias. Heaviest local v3 route; not exact official Paddle runtime in Python 3.14.", "rtdetrv3_r50vd_6x.pdparams"),
    ModelCatalogItem("rtdetrv3_r50vd_6x.pdparams", "rtdetr_external", "Direct local RT-DETRv3 R50 Paddle checkpoint; compatibility-converted without Paddle at runtime.", "rtdetrv3_r50vd_6x.pdparams"),
    ModelCatalogItem("rtdetr-4.pt", "rtdetr_external", "RT-DETRv4-S official checkpoint. Runs through ~/bench_any/runtimes/RT-DETRv4.", "rtdetrv4_s_coco.pth"),
    ModelCatalogItem("rtdetrv4_s_coco.pth", "rtdetr_external", "Direct local RT-DETRv4-S checkpoint. Same runtime as rtdetr-4.pt.", "rtdetrv4_s_coco.pth"),
    ModelCatalogItem("rf-detr-base-det", "rfdetr", "RF-DETR Base detector; patch-14 DINOv2 path. Use with --seg_head sam2 for detector+prompted-mask tables."),
    ModelCatalogItem("rf-detr-medium.pth", "rfdetr", "RF-DETR detection medium checkpoint; detector route, useful for detector+SAM comparison. Uses RF-DETR's published patch-16 checkpoint, so DINOv2 patch-14 hub-init warnings are expected and suppressed by the wrapper.", "rf-detr-medium.pth"),
    ModelCatalogItem("rf-detr-seg-small.pt", "rfdetr", "RF-DETR segmentation small; fastest local RF-DETR segmentation checkpoint. Seg RF-DETR checkpoints are patch-12, not vanilla DINOv2 patch-14.", "rf-detr-seg-small.pt"),
    ModelCatalogItem("rf-detr-seg-medium.pt", "rfdetr", "RF-DETR segmentation medium; middle local RF-DETR segmentation checkpoint. Seg RF-DETR checkpoints are patch-12, not vanilla DINOv2 patch-14.", "rf-detr-seg-medium.pt"),
    ModelCatalogItem("rf-detr-seg-large.pt", "rfdetr", "RF-DETR segmentation large; resolves to ~/bench_any/detr_weights/rf-detr-seg-large.pt when present. Keeps the patch-12 published RF-DETR architecture so the checkpoint loads.", "rf-detr-seg-large.pt"),
    ModelCatalogItem("rf-detr-seg-xlarge.pt", "rfdetr", "RF-DETR segmentation xlarge; stronger/heavier than large; resolves locally if downloaded. Keeps the patch-12 published RF-DETR architecture so the checkpoint loads.", "rf-detr-seg-xlarge.pt"),
    ModelCatalogItem("dinov3-m2f-smallplus", "dinov3_m2f", "DINOv3 ViT-S/16+ Mask2Former. Fastest DINOv3 instance-seg sanity model."),
    ModelCatalogItem("dinov3-m2f-vitl16", "dinov3_m2f", "DINOv3 ViT-L/16 Mask2Former. Stronger dense backbone, much heavier than smallplus."),
    ModelCatalogItem("dinov3-m2f-vith16plus", "dinov3_m2f", "DINOv3 ViT-H/16+ Mask2Former. Paper-ready stronger probe when VRAM allows."),
    ModelCatalogItem("eomt-smallplus", "eomt", "EoMT with DINOv3 ViT-S/16+ backbone; practical EoMT choice for 16 GB GPUs."),
    ModelCatalogItem("eomt-small", "eomt", "EoMT with DINOv3 ViT-S/16 backbone; smallest EoMT route."),
    ModelCatalogItem("eomt-base", "eomt", "EoMT with DINOv3 ViT-B/16 backbone; middle EoMT route for limited VRAM."),
    ModelCatalogItem("eomt-large", "eomt", "EoMT large alias using DINOv3 ViT-L/16 backbone in this wrapper; heavy on 16 GB."),
    ModelCatalogItem("eomt-vitl16", "eomt", "EoMT with explicit DINOv3 ViT-L/16 backbone; heavy on 16 GB."),
    ModelCatalogItem("eomt-vith16plus", "eomt", "EoMT with DINOv3 ViT-H/16+ backbone; observed OOM on a 15.4 GB GPU even at micro_bs=1."),
    ModelCatalogItem("eomt-vit7b16", "eomt", "EoMT with DINOv3 ViT-7B/16 backbone; not realistic for the current GPU."),
    ModelCatalogItem("dinov3-semseg-vits16plus", "dinov3_semseg", "Frozen DINOv3 ViT-S/16+ plus lightweight semantic head; read foreground metrics, not pixel_acc alone.", "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"),
    ModelCatalogItem("dinov3-semseg-vits16", "dinov3_semseg", "Frozen DINOv3 ViT-S/16 semantic head; local weights are present in ~/bench_any/dinov3_weights.", "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
    ModelCatalogItem("dinov3-semseg-vitb16", "dinov3_semseg", "Frozen DINOv3 ViT-B/16 semantic head; largest local semseg backbone currently present without downloading more weights.", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
    ModelCatalogItem("dinov3-semseg-vitl16", "dinov3_semseg", "Frozen DINOv3 ViT-L/16 semantic head; stronger local semseg backbone when matching weights are present.", "dinov3_vitl16*.pth"),
    ModelCatalogItem("dinov3-semseg-vith16plus", "dinov3_semseg", "Frozen DINOv3 ViT-H/16+ semantic head; strongest/heaviest semseg backbone listed here.", "dinov3_vith16plus*.pth"),
)


def model_search_dirs() -> List[Path]:
    raw = os.environ.get("BENCH_MODEL_DIRS", "")
    dirs = [Path(p).expanduser() for p in raw.split(os.pathsep) if p.strip()]
    dirs += [
        Path.home() / "bench_any" / "detr_weights",
        Path.home() / "bench_any" / "dinov3_weights",
        Path.home() / "yolo_general" / "models",
    ]
    out: List[Path] = []
    seen = set()
    for d in dirs:
        key = str(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _candidate_weight_names(entry: str) -> List[str]:
    key = model_key(entry)
    stem = model_stem(entry)
    names = [key]
    if Path(key).suffix.lower() not in (".pt", ".pth", ".safetensors"):
        names += [f"{key}.pt", f"{key}.pth", f"{key}.safetensors"]
    variants = []
    for n in names:
        variants.append(n)
        variants.append(n.replace("rfdetr", "rf-detr"))
        variants.append(n.replace("rf-detr", "rfdetr"))
    if stem in {"rtdetr-2", "rtdetr-v2", "rt-detr-2", "rt-detr-v2"}:
        variants.append("rtdetrv2_r50vd_6x_coco_ema.pth")
    return list(dict.fromkeys(variants))


def resolve_local_model(entry: str) -> Tuple[str, str]:
    """Resolve bare local weight names to known local model folders.

    Returns (resolved_entry, source). Non-weight DINO/EoMT aliases are returned
    unchanged with source='alias'. Existing paths are returned as source='path'.
    """
    expanded = Path(entry).expanduser()
    if expanded.is_file():
        return str(expanded.resolve()), "path"
    if "/" in entry or "\\" in entry:
        return entry, "missing_path"

    fw, _, _, _, _ = pick_framework(entry)
    if fw == "rtdetr_external":
        return entry, "alias"
    # Detector RF-DETR variants (a literal `det` token, e.g. rf-detr-base-det)
    # load by name; seg variants resolve to local weights. NB: a substring test
    # `"-det" in ...` is wrong — it also matches the "det" inside "rf-de(tr)",
    # so it aliased *every* RF-DETR name and never found local seg weights.
    if fw == "rfdetr" and "det" in re.split(r"[-_.]+", model_stem(entry)) and not Path(entry).suffix:
        return entry, "alias"
    if fw in ("dinov3_m2f", "eomt", "dinov3_semseg"):
        return entry, "alias"

    candidates = _candidate_weight_names(entry)
    for d in model_search_dirs():
        if not d.is_dir():
            continue
        for cand in candidates:
            p = d / cand
            if p.is_file():
                return str(p.resolve()), f"local:{d}"
            if "/" in cand and p.is_file():
                return str(p.resolve()), f"local:{d}"
        for cand in candidates:
            matches = sorted(d.rglob(cand))
            if matches:
                return str(matches[0].resolve()), f"local:{d}"
    return entry, "unresolved"


def pick_framework(name: str):
    for entry in REGISTRY:
        if entry.matches(name):
            return entry.route_tuple()
    return REGISTRY[-1].route_tuple()


def expand_model_sets(model_sets: Sequence[str]) -> List[str]:
    out: List[str] = []
    for name in model_sets:
        if name not in MODEL_SETS:
            valid = ", ".join(sorted(MODEL_SETS))
            raise KeyError(f"unknown model set '{name}' (valid: {valid})")
        out.extend(MODEL_SETS[name])
    return out


def registry_lines() -> List[str]:
    lines = ["Usable --models entries. Bare .pt/.pth names are resolved in BENCH_MODEL_DIRS, ~/bench_any/detr_weights, ~/bench_any/dinov3_weights, then ~/yolo_general/models."]
    lines.append("Path-based usage still works and is safest, e.g. --models ~/yolo_general/models/yolo11x-seg.pt ~/bench_any/detr_weights/rtdetr-x.pt")
    lines.append("")
    lines.append("catalog:")
    for item in MODEL_CATALOG:
        resolved = ""
        if item.local_hint:
            found = None
            hint = Path(item.local_hint)
            for d in model_search_dirs():
                p = d / hint
                if p.is_file():
                    found = p.resolve()
                    break
                hits = sorted(d.rglob(item.local_hint))
                if hits:
                    found = hits[0].resolve()
                    break
            resolved = f"  local={found}" if found else "  local=not found"
        lines.append(f"  {item.name:<30} {item.framework:<16} {item.description}{resolved}")
    lines.append("")
    lines.append("routes:")
    for entry in REGISTRY:
        examples = ", ".join(entry.examples) if entry.examples else "-"
        script = entry.script or "<blocked>"
        lines.append(f"{entry.framework:<14} {script:<28} examples: {examples}")
        if entry.notes:
            lines.append(f"{'':<14} {'':<28} {entry.notes}")
    return lines


def model_set_lines() -> List[str]:
    return [f"{name:<20} {' '.join(models)}" for name, models in sorted(MODEL_SETS.items())]

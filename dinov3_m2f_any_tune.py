# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
# Wraps Carti-97/DINOv3-Mask2former (HF Mask2Former + DINOv3 backbone, instance seg).
# Mirrors yolo_any_tune.py CLI; reuses rfdetr_any_tune.py's YOLO→COCO converter.
# USAGE: python3 dinov3_m2f_any_tune.py --data /path/to/dataset --epochs 50 --batch_sz 4 --model dinov3-m2f-smallplus

import os, sys, argparse, shutil, hashlib, csv, subprocess, re, json
import yaml
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rfdetr_any_tune import yolo_to_coco_layout  # noqa: E402
from bench_schema import zero_metrics  # noqa: E402

REPO_URL = "https://github.com/Carti-97/DINOv3-Mask2former.git"
REPO_BASE = Path(os.environ.get("BENCH_ANY", str(Path.home() / "bench_any")))
REPO_CACHE = REPO_BASE / "DINOv3-Mask2former"

### Order matters — matcher does `if k in name`, so longer/more specific keys
### must come first. Carti-97 ships only two M2F config files (smallplus + large);
### bigger variants (vith16plus, vit7b16) reuse the `large` config and swap the
### HF backbone via the DINOV3_M2F_MODEL override + DINO_HF_MODELS table below.
### ConvNeXt backbones are not supported by upstream M2F configs — they map to
### `large` only as a last-resort fallback (will reuse the ViT-L config; for a
### real ConvNeXt M2F run use `dinov3_semseg` or `eomt` instead).
MODEL_FILES = {
    "vits16plus": "models/mask2former_dinov3_vitsmallplus.py",
    "vits16":     "models/mask2former_dinov3_vitsmallplus.py",
    "vitb16":     "models/mask2former_dinov3_vitsmallplus.py",   # closest config
    "vitl16":     "models/mask2former_dinov3_vitlarge.py",
    "vith16plus": "models/mask2former_dinov3_vitlarge.py",        # uses large cfg + bigger HF backbone
    "vith16":     "models/mask2former_dinov3_vitlarge.py",
    "vit7b16":    "models/mask2former_dinov3_vitlarge.py",
    "convnext_tiny":  "models/mask2former_dinov3_vitsmallplus.py",
    "convnext_small": "models/mask2former_dinov3_vitsmallplus.py",
    "convnext_base":  "models/mask2former_dinov3_vitlarge.py",
    "convnext_large": "models/mask2former_dinov3_vitlarge.py",
    "smallplus": "models/mask2former_dinov3_vitsmallplus.py",
    "small":     "models/mask2former_dinov3_vitsmallplus.py",
    "base":      "models/mask2former_dinov3_vitsmallplus.py",
    "large":     "models/mask2former_dinov3_vitlarge.py",
    "huge":      "models/mask2former_dinov3_vitlarge.py",
    "giant":     "models/mask2former_dinov3_vitlarge.py",
}
DINO_HF_MODELS = {
    "vits16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "vits16":     "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vitb16":     "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl16":     "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "vith16":     "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "vit7b16":    "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "convnext_tiny":  "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "convnext_small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "convnext_base":  "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "convnext_large": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
    "smallplus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "small":     "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "base":      "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large":     "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "huge":      "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "giant":     "facebook/dinov3-vit7b16-pretrain-lvd1689m",
}


def _ensure_imports(modules, label=""):
    """Probe-then-install. No sentinels, idempotent."""
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


def _patch_upstream_trainer(repo_dir):
    """Make Carti-97 trainer compatible with transformers>=4.50:
       1) Neutralise removed `send_example_telemetry`.
       2) Quiet HF logging.
       3) Inject background label `0` into instance_id_to_semantic_id at processor calls.
       4) Add bbox mAP computation alongside mask mAP."""
    p = Path(repo_dir) / "mask2former_dinov3_no_trainer_coco.py"
    if not p.is_file(): return
    txt = p.read_text()
    BG = "# bench_any: include bg id 0 (kwarg-rewrite v2)"
    BBOX_MARKER = "# bench_any: bbox mAP added"
    RESUME_MARKER = "# bench_any: robust resume epoch/step parse"
    PRUNE_STATE_MARKER = "# bench_any: keep only latest resumable state"
    WARMSTART_MARKER = "# bench_any: warm-start from weights-only dump"
    EPOCH_PRUNE_MARKER = "# bench_any: prune old weights-only epoch dirs"
    PATIENCE_MARKER = "# bench_any: early stopping on patience"
    # speed port (cvBench): bf16/TF32 + model-resolution rasterise. Both env-gated
    # (BENCH_M2F_PRECISION=no restores fp32 for an A/B); the raster hunk fires on
    # the train split only, eval keeps the native-resolution path.
    AMP_MARKER = "# bench_any: bf16 mixed precision + TF32"
    RASTER_MARKER = "# bench_any: rasterise at model resolution"
    if all(m in txt for m in ("# bench_any: send_example_telemetry neutralised",
                              "# bench_any: keep transformers quiet", BG, BBOX_MARKER,
                              RESUME_MARKER, PRUNE_STATE_MARKER,
                              WARMSTART_MARKER, EPOCH_PRUNE_MARKER, PATIENCE_MARKER,
                              AMP_MARKER, RASTER_MARKER)):
        return
    txt = re.sub(
        r"^[ \t]*instance_id_to_semantic_id\s*=\s*\{0:\s*0,\s*\*\*instance_id_to_semantic_id\}.*?# bench_any: include background id 0 in mapping.*?\n",
        "", txt, flags=re.MULTILINE)
    # Guarded on its own marker, not on the all-markers early return above: any
    # later patch adding a new marker re-enters this function, and the regex
    # matches its own output (the trailing comment is part of `[^\n]+`), so an
    # unguarded sub appends one more copy per launch. The live upstream file had
    # accumulated 83 of them by 2026-08-22.
    if "# bench_any: send_example_telemetry neutralised" not in txt:
        txt = re.sub(
            r"from\s+transformers\.utils\s+import\s+([^\n]+)",
            lambda m: "from transformers.utils import " + ", ".join(
                s.strip() for s in m.group(1).split(",")
                if s.strip() and s.strip() != "send_example_telemetry"
            ) + "  # bench_any: send_example_telemetry neutralised",
            txt, count=1)
    if "send_example_telemetry(" in txt and "send_example_telemetry = lambda" not in txt:
        txt = "send_example_telemetry = lambda *a, **k: None  # bench_any shim\n" + txt
    txt = txt.replace("transformers.utils.logging.set_verbosity_info()",
                      "transformers.utils.logging.set_verbosity_error()  # bench_any: keep transformers quiet")
    txt = re.sub(
        r"instance_id_to_semantic_id\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*,",
        lambda m: (f"instance_id_to_semantic_id={{0: 0, **{m.group(1)}}},  {BG}"
                   if m.group(1) not in ("True", "False", "None") else m.group(0)),
        txt)
    if BG not in txt: txt = f"# {BG}\n" + txt

    # Add bbox mAP computation by instantiating a second MeanAveragePrecision with iou_type="bbox"
    # The existing one computes segm mAP. We derive boxes from masks and compute bbox mAP too.
    if BBOX_MARKER not in txt and "MeanAveragePrecision" in txt:
        # Raise the per-image detection cap and silence the ">100 detections"
        # warning: crowded penguin scenes routinely exceed 100 instances, and the
        # torchmetrics default max_detection_thresholds=[1,10,100] truncates them.
        txt = txt.replace(
            'self.metric = MeanAveragePrecision(iou_type="segm"',
            'self.metric = MeanAveragePrecision(iou_type="segm", '
            'max_detection_thresholds=[1, 10, 1000]'
        )
        # Add bbox metric alongside segm metric
        txt = txt.replace(
            'logger.info(f"Test metrics: {all_metrics}")',
            f'''# {BBOX_MARKER}
        # Also compute bbox mAP from the masks
        try:
            from pycocotools import mask as maskUtils
            bbox_preds, bbox_targets = [], []
            for pred, tgt in zip(predictions, targets):
                # Derive boxes from masks
                if "masks" in pred and len(pred["masks"]) > 0:
                    boxes_from_masks = []
                    for mask in pred["masks"]:
                        if mask.sum() == 0: continue
                        rle = maskUtils.encode(np.asfortranarray(mask.cpu().numpy().astype(np.uint8)))
                        bbox = maskUtils.toBbox(rle)  # [x, y, w, h]
                        boxes_from_masks.append(bbox)
                    if boxes_from_masks:
                        bbox_preds.append({{"boxes": torch.tensor(boxes_from_masks, device=pred["masks"].device),
                                          "scores": pred["scores"], "labels": pred["labels"]}})
                if "masks" in tgt:
                    tgt_boxes = []
                    for mask in tgt["masks"]:
                        if mask.sum() == 0: continue
                        rle = maskUtils.encode(np.asfortranarray(mask.cpu().numpy().astype(np.uint8)))
                        bbox = maskUtils.toBbox(rle)
                        tgt_boxes.append(bbox)
                    if tgt_boxes:
                        bbox_targets.append({{"boxes": torch.tensor(tgt_boxes, device=tgt["masks"].device),
                                            "labels": tgt["labels"]}})
            if bbox_preds and bbox_targets:
                bbox_metric = MeanAveragePrecision(iou_type="bbox", max_detection_thresholds=[1, 10, 1000])
                bbox_metric.update(bbox_preds, bbox_targets)
                bbox_res = bbox_metric.compute()
                for k, v in bbox_res.items():
                    if "map" in str(k):
                        all_metrics[str(k) + "_bbox"] = float(v)
        except Exception as e:
            logger.warning(f"bbox mAP computation failed: {{e}}")
        logger.info(f"Test metrics: {{all_metrics}}")'''
        )

    # --- Resume robustness -------------------------------------------------
    # Upstream resumes by parsing the checkpoint basename, but the epoch-state
    # dirs it writes are named `checkpoint_epoch_N`; its parser does
    # int(basename.replace("epoch_", "")) which becomes int("checkpoint_N") and
    # crashes. Parse the trailing integer robustly instead so
    # --resume_from_checkpoint <run>/checkpoint_epoch_N actually resumes.
    if RESUME_MARKER not in txt:
        txt = txt.replace(
            '        if "epoch" in training_difference:\n'
            '            starting_epoch = int(training_difference.replace("epoch_", "")) + 1\n',
            '        if "epoch" in training_difference:\n'
            '            import re as _re_bench  ' + RESUME_MARKER + '\n'
            '            _m_bench = _re_bench.search(r"(\\d+)$", training_difference)\n'
            '            starting_epoch = (int(_m_bench.group(1)) + 1) if _m_bench else 0\n')

    # --- Disk hygiene for resumable state ----------------------------------
    # With --checkpointing_steps epoch the trainer calls accelerator.save_state
    # once per epoch into a fresh `checkpoint_epoch_N` dir and never prunes the
    # old ones (each carries optimizer+model = ~2x weights). Over a 125-epoch
    # run that exhausts the disk, so keep only the newest resumable state.
    if PRUNE_STATE_MARKER not in txt:
        txt = txt.replace(
            "            accelerator.save_state(checkpoint_dir)\n",
            "            accelerator.save_state(checkpoint_dir)\n"
            "            " + PRUNE_STATE_MARKER + "\n"
            "            if accelerator.is_main_process and args.output_dir is not None:\n"
            "                import glob as _glob_bench, shutil as _shutil_bench\n"
            "                _keep_bench = os.path.abspath(checkpoint_dir)\n"
            "                for _old_bench in _glob_bench.glob(os.path.join(args.output_dir, 'checkpoint_epoch_*')):\n"
            "                    if os.path.abspath(_old_bench) != _keep_bench and os.path.isdir(_old_bench):\n"
            "                        _shutil_bench.rmtree(_old_bench, ignore_errors=True)\n")

    # --- Warm-start from a weights-only dump -------------------------------
    # A run that crashed BEFORE this resume machinery existed (or after a clean
    # finish that pruned the resumable state) leaves only weights-only epoch_N/
    # + best_model/ dirs, which carry no optimizer/epoch state and so cannot feed
    # --resume_from_checkpoint. Rather than throw those learned weights away and
    # restart from random M2F-head init, load them into the freshly-built model
    # when the launcher points BENCH_M2F_WARMSTART at a saved model.safetensors.
    # This recovers e.g. the 45 already-trained epochs of vith16plus (optimizer
    # momentum is lost, but the network is not). Skipped automatically when a
    # real accelerate resume is in effect (launcher leaves the env var unset).
    if WARMSTART_MARKER not in txt:
        txt = txt.replace(
            '    logger.info("DINOv3-Mask2Former model ready to use.")\n',
            '    logger.info("DINOv3-Mask2Former model ready to use.")\n'
            '    ' + WARMSTART_MARKER + '\n'
            '    _ws_bench = os.environ.get("BENCH_M2F_WARMSTART", "")\n'
            '    if _ws_bench and os.path.isfile(_ws_bench):\n'
            '        try:\n'
            '            from safetensors.torch import load_file as _load_sft_bench\n'
            '            _sd_bench = _load_sft_bench(_ws_bench)\n'
            '            # Shape-aware load: a warm-start dump may come from a run\n'
            '            # whose backbone differed (e.g. a pre-fix run that silently\n'
            '            # used the default vitl16=1024 instead of vith16plus=1280).\n'
            '            # strict=False still RAISES on size mismatches for matching\n'
            '            # keys, which used to spam a multi-page traceback and drop\n'
            '            # the entire warm-start. Filter to shape-compatible tensors\n'
            '            # so we salvage the matching head/decoder weights and skip\n'
            '            # the mismatched ones with a concise summary.\n'
            '            _msd_bench = model.state_dict()\n'
            '            _ok_bench = {}\n'
            '            _shape_skip_bench = 0\n'
            '            for _k_ws, _v_ws in _sd_bench.items():\n'
            '                _cur_ws = _msd_bench.get(_k_ws)\n'
            '                if _cur_ws is not None and tuple(_cur_ws.shape) == tuple(_v_ws.shape):\n'
            '                    _ok_bench[_k_ws] = _v_ws\n'
            '                elif _cur_ws is not None:\n'
            '                    _shape_skip_bench += 1\n'
            '            _res_bench = model.load_state_dict(_ok_bench, strict=False)\n'
            '            _miss_bench = len(getattr(_res_bench, "missing_keys", []) or [])\n'
            '            _unexp_bench = len(getattr(_res_bench, "unexpected_keys", []) or [])\n'
            '            if _shape_skip_bench:\n'
            '                logger.warning(f"bench_any warm-start: skipped {_shape_skip_bench} "\n'
            '                               f"shape-mismatched tensors from {_ws_bench} "\n'
            '                               f"(likely a different backbone variant); "\n'
            '                               f"loaded {len(_ok_bench)} compatible tensors.")\n'
            '            logger.info(f"bench_any warm-start: loaded weights from {_ws_bench} "\n'
            '                        f"(loaded={len(_ok_bench)} skipped_shape={_shape_skip_bench} "\n'
            '                        f"missing={_miss_bench} unexpected={_unexp_bench})")\n'
            '        except Exception as _e_ws_bench:\n'
            '            logger.warning(f"bench_any warm-start failed for {_ws_bench}: {_e_ws_bench}")\n')

    # --- Disk hygiene for weights-only epoch dumps -------------------------
    # Upstream writes a full HF model dump into a fresh epoch_N/ every epoch and
    # NEVER prunes them mid-run (the wrapper only prunes after a clean finish).
    # A crash mid-run therefore leaves every epoch on disk — e.g. 46 × 3.3 GB =
    # 152 GB for vith16plus. Prune in-place each epoch, keeping only the newest
    # BENCH_M2F_KEEP_EPOCHS (default 1); best_model/ is named separately and is
    # never matched by the epoch_* glob, so it is always preserved.
    if EPOCH_PRUNE_MARKER not in txt:
        txt = txt.replace(
            "                logger.info(f\"모델과 메트릭이 {epoch_output_dir}에 저장되었습니다.\")\n",
            "                logger.info(f\"모델과 메트릭이 {epoch_output_dir}에 저장되었습니다.\")\n"
            "                " + EPOCH_PRUNE_MARKER + "\n"
            "                import glob as _glob_ep_bench, shutil as _shutil_ep_bench\n"
            "                try:\n"
            "                    _keep_ep_bench = int(os.environ.get('BENCH_M2F_KEEP_EPOCHS', '1'))\n"
            "                except Exception:\n"
            "                    _keep_ep_bench = 1\n"
            "                if _keep_ep_bench > 0:\n"
            "                    _eps_bench = sorted(\n"
            "                        _glob_ep_bench.glob(os.path.join(args.output_dir, 'epoch_*')),\n"
            "                        key=lambda d: int(os.path.basename(d).split('_')[-1])\n"
            "                                      if os.path.basename(d).split('_')[-1].isdigit() else -1)\n"
            "                    _arch_ep_bench = os.path.join(args.output_dir, '_epoch_metrics')\n"
            "                    os.makedirs(_arch_ep_bench, exist_ok=True)\n"
            "                    for _v_ep_bench in _eps_bench[:-_keep_ep_bench]:\n"
            "                        if os.path.isdir(_v_ep_bench):\n"
            "                            for _mj_bench in _glob_ep_bench.glob(\n"
            "                                    os.path.join(_v_ep_bench, 'epoch_*_metrics.json')):\n"
            "                                try: _shutil_ep_bench.copy2(_mj_bench, _arch_ep_bench)\n"
            "                                except Exception: pass\n"
            "                            _shutil_ep_bench.rmtree(_v_ep_bench, ignore_errors=True)\n")

    # --- Early stopping ----------------------------------------------------
    # Upstream tracks best_metric/is_best_epoch but never breaks, so every run
    # spent its whole budget and the wrapper's --patience was dead. The stall
    # counter is bumped where the best epoch is decided; the break happens at the
    # TOP of the following epoch so the stopping epoch still writes best_model/
    # and its epoch dump before the loop exits. BENCH_M2F_PATIENCE=0 disables.
    if PATIENCE_MARKER not in txt:
        txt = txt.replace(
            "        is_best_epoch = current_metric > best_metric\n"
            "        if is_best_epoch:\n",
            "        " + PATIENCE_MARKER + "\n"
            "        try:\n"
            "            _pat_bench = int(os.environ.get('BENCH_M2F_PATIENCE', '0'))\n"
            "        except Exception:\n"
            "            _pat_bench = 0\n"
            "        is_best_epoch = current_metric > best_metric\n"
            "        _stall_bench = 0 if is_best_epoch else (globals().get('_stall_bench', 0) + 1)\n"
            "        globals()['_stall_bench'] = _stall_bench\n"
            "        if _pat_bench > 0:\n"
            "            if _stall_bench >= _pat_bench:\n"
            "                globals()['_stop_bench'] = True\n"
            "                logger.info(f'⏹ bench_any early stop: {_stall_bench} epochs without "
            "improvement >= patience {_pat_bench} (best epoch {best_epoch}, {best_metric:.4f})')\n"
            "            else:\n"
            "                logger.info(f'⏳ bench_any patience: {_stall_bench}/{_pat_bench} "
            "epochs without improvement')\n"
            "        if is_best_epoch:\n", 1)
        txt = txt.replace(
            "    for epoch in range(starting_epoch, args.num_train_epochs):\n"
            "        model.train()\n",
            "    for epoch in range(starting_epoch, args.num_train_epochs):\n"
            "        if globals().get('_stop_bench', False):\n"
            "            logger.info(f'⏹ bench_any early stop honoured at epoch {epoch}; "
            "best epoch was {best_epoch}.')\n"
            "            break\n"
            "        model.train()\n", 1)

    # ---- speed: bf16 mixed precision + TF32 (cvBench port) -------------------
    # M2F was the only fp32 model in the bench (YOLO/RF-DETR/EoMT all run mixed
    # precision already). bf16 needs no GradScaler and cannot overflow the
    # mask/dice losses. BENCH_M2F_PRECISION=no restores fp32 for an A/B.
    if AMP_MARKER not in txt:
        _acc_call = ("    accelerator = Accelerator(gradient_accumulation_steps="
                     "args.gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs])")
        txt = txt.replace(_acc_call,
            f"    {AMP_MARKER}\n"
            "    torch.set_float32_matmul_precision('high')\n"
            "    torch.backends.cudnn.benchmark = True\n"
            "    accelerator = Accelerator(\n"
            "        gradient_accumulation_steps=args.gradient_accumulation_steps,\n"
            "        mixed_precision=os.environ.get('BENCH_M2F_PRECISION', 'bf16'),\n"
            "        kwargs_handlers=[ddp_kwargs])")

    # ---- speed: rasterise instance masks at model resolution (train only) ---
    # Upstream decodes every annotation at native resolution, ORs it into a
    # full-res int32 canvas, augments at full res, then throws it all away in the
    # image_processor resize. py-spy on the live vith16plus run: 18.7% of wall
    # time in __getitem__, dominated by that per-annotation write and the
    # np.unique over the full-res canvas. Eval keeps the native path because
    # `original_size` drives post-processing back to source resolution.
    if RASTER_MARKER not in txt:
        txt = txt.replace(
            "import numpy as np\n",
            "import numpy as np\n"
            "import cv2 as _bench_cv2  # bench_any\n"
            "from pycocotools import mask as _bench_mu  # bench_any\n", 1)
        _orig_loop = (
            "        for i, ann in enumerate(ann_data['anns'], 1):\n"
            "            if 'segmentation' in ann:\n"
            "                if self.coco:\n"
            "                    mask = self.coco.annToMask(ann)\n"
            "                else:\n"
            "                    mask = self._poly_to_mask(ann['segmentation'], h, w)\n"
            "                instance_mask[mask > 0] = i\n"
            "                instance_to_semantic[i] = ann['category_id']\n")
        txt = txt.replace(_orig_loop,
            f"        {RASTER_MARKER}\n"
            "        _tgt = None\n"
            "        if self.split == 'train':\n"
            "            _sz = getattr(self.image_processor, 'size', None) or {}\n"
            "            _th, _tw = int(_sz.get('height', 0) or 0), int(_sz.get('width', 0) or 0)\n"
            "            if _th > 0 and _tw > 0 and (_th, _tw) != (h, w):\n"
            "                _tgt = (_th, _tw)\n"
            "        if _tgt is not None:\n"
            "            _th, _tw = _tgt\n"
            "            _sx, _sy = _tw / w, _th / h\n"
            "            image = image.resize((_tw, _th), Image.BILINEAR)\n"
            "            instance_mask = np.zeros((_th, _tw), dtype=np.int32)\n"
            "        for i, ann in enumerate(ann_data['anns'], 1):\n"
            "            if 'segmentation' in ann:\n"
            "                mask = None\n"
            "                if _tgt is not None:\n"
            "                    try:\n"
            "                        _seg = ann['segmentation']\n"
            "                        if isinstance(_seg, list):\n"
            "                            _polys = []\n"
            "                            for _pg in _seg:\n"
            "                                _q = np.asarray(_pg, dtype=np.float64).copy()\n"
            "                                _q[0::2] *= _sx\n"
            "                                _q[1::2] *= _sy\n"
            "                                _polys.append(_q.tolist())\n"
            "                            mask = _bench_mu.decode(_bench_mu.merge(\n"
            "                                _bench_mu.frPyObjects(_polys, _th, _tw)))\n"
            "                        else:\n"
            "                            _m = (self.coco.annToMask(ann) if self.coco\n"
            "                                  else self._poly_to_mask(_seg, h, w))\n"
            "                            mask = _bench_cv2.resize(_m, (_tw, _th),\n"
            "                                                     interpolation=_bench_cv2.INTER_NEAREST)\n"
            "                    except Exception:\n"
            "                        mask = None\n"
            "                if mask is None:\n"
            "                    _m = (self.coco.annToMask(ann) if self.coco\n"
            "                          else self._poly_to_mask(ann['segmentation'], h, w))\n"
            "                    mask = (_bench_cv2.resize(_m, (_tw, _th),\n"
            "                                              interpolation=_bench_cv2.INTER_NEAREST)\n"
            "                            if _tgt is not None else _m)\n"
            "                instance_mask[mask > 0] = i\n"
            "                instance_to_semantic[i] = ann['category_id']\n")

    p.write_text(txt)
    print("🩹 Patched upstream trainer (transformers>=4.50 + bg-label + bbox mAP + resume "
          "+ warm-start + early stopping + bf16/TF32 + model-resolution rasterise)")


def _patch_upstream_model_files(repo_dir):
    """Pass `hub_token` through to the inner DINOv3 AutoModel.from_pretrained, and
    honour DINOV3_M2F_MODEL env override for a local backbone.

    Idempotent and tolerant of the wording differences between the two upstream
    config files: the `large` config comments the backbone block as
    "DINOv3-Large backbone" while `smallplus` says "DINOv3 backbone". The old
    literal `.replace()` only matched smallplus, so the override never landed on
    `large` — meaning vith16plus/vith16/vitl16/vit7b16 silently trained the
    default vitl16 backbone, and a non-idempotent `import os` insert ran on every
    invocation (it had accreted 53 duplicate import lines)."""
    OVERRIDE = "# bench_any: DINOv3 backbone token/local override"
    for rel in sorted(set(MODEL_FILES.values())):
        p = Path(repo_dir) / rel
        if not p.is_file(): continue
        txt = orig = p.read_text()

        # Self-heal duplicate `import os` lines from earlier non-idempotent runs,
        # then ensure exactly one is present.
        txt = re.sub(r"(?:^import os\n)+", "import os\n", txt, flags=re.MULTILINE)
        if re.search(r"^import os$", txt, flags=re.MULTILINE) is None:
            txt = txt.replace("import logging\n", "import logging\nimport os\n", 1)

        # Inject the DINOV3_M2F_MODEL override + hub_token on the backbone call,
        # matching the assignment regardless of the preceding comment wording.
        if OVERRIDE not in txt:
            txt = re.sub(
                r"^([ \t]*)custom_backbone = DinoV3WithAdapterBackbone\(\s*"
                r"dinov3_model_name,\s*expected_channels\s*\)",
                lambda m: (f"{m.group(1)}dinov3_model_name = os.environ.get("
                           f"'DINOV3_M2F_MODEL', dinov3_model_name)  {OVERRIDE}\n"
                           f"{m.group(1)}custom_backbone = DinoV3WithAdapterBackbone("
                           f"dinov3_model_name, expected_channels, hub_token=hub_token)"),
                txt, count=1, flags=re.MULTILINE)

        # Thread hub_token through the backbone class for gated DINOv3 downloads.
        txt = txt.replace(
            "def __init__(self, model_name: str, out_channels: List[int]):",
            "def __init__(self, model_name: str, out_channels: List[int], hub_token: str = None):", 1)
        txt = txt.replace(
            "self.model = AutoModel.from_pretrained(model_name)",
            "self.model = AutoModel.from_pretrained(model_name, token=hub_token)", 1)

        if txt != orig:
            p.write_text(txt)
            print(f"🩹 Patched {rel} for gated DINOv3 backbone + DINOV3_M2F_MODEL override")


def ensure_repo():
    REPO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not (REPO_CACHE / ".git").exists():
        print(f"📥 Cloning {REPO_URL} → {REPO_CACHE}")
        subprocess.check_call(["git", "clone", "--depth=1", REPO_URL, str(REPO_CACHE)])
    _ensure_imports([("datasets", "datasets"), ("transformers", "transformers"),
                     ("accelerate", "accelerate"), ("torchmetrics", "torchmetrics"),
                     ("pycocotools", "pycocotools"), ("albumentations", "albumentations"),
                     ("evaluate", "evaluate")], label="dinov3-m2f")
    _patch_upstream_trainer(REPO_CACHE)
    _patch_upstream_model_files(REPO_CACHE)
    return REPO_CACHE


def pick_model_file(name):
    n = name.lower()
    for k, rel in MODEL_FILES.items():
        if k in n: return rel
    return MODEL_FILES["smallplus"]


def pick_dinov3_hf_model(name):
    n = name.lower()
    for k, hf in DINO_HF_MODELS.items():
        if k in n: return hf
    return DINO_HF_MODELS["smallplus"]


def warn_if_dinov3_access_missing(model_name, hub_token):
    override = os.environ.get("DINOV3_M2F_MODEL", "").strip()
    if override and (Path(override).expanduser().exists() or not override.startswith("facebook/")):
        return
    hf_model = override or pick_dinov3_hf_model(model_name)
    try:
        from transformers import AutoConfig, logging as hf_logging
        hf_logging.set_verbosity_error()
        AutoConfig.from_pretrained(hf_model, token=(hub_token or None))
    except Exception as e:
        print(f"⚠️ DINOv3 backbone gated/unavailable: {hf_model}: {e}")
        print("    Run `huggingface-cli login` or pass --hub_token / set DINOV3_M2F_MODEL=/local/dir.")


def harvest_metrics(run_dir, eval_split="test"):
    """Read mAP from <run_dir>/best_model/best_model_info.json (Carti-97 writes
    instance-mask metrics there). Anything missing stays at 0.0.

    best_model_info.json holds the metrics of the best-scoring epoch measured on
    the PER-EPOCH eval split, which since the 2026-08-08 real-eval fix is val(173),
    not the reported test split. The final test pass writes its own numbers to
    <run_dir>/all_results.json as test_*. Reading only best_model_info.json
    therefore put validation numbers in every row labelled eval_split=test — the
    corpus-diversity vith16plus run reported 0.6732 (val, epoch 121) where the
    test number is 0.6256. all_results.json wins whenever it carries test_* keys;
    best_model_info.json is the fallback, and the fallback says so out loud."""
    # Ensure we initialize with ALL keys from the bench_any schema so that
    # harvest_results_csv doesn't report 0.0 for missing keys that should be hyphenated.
    out = zero_metrics()
    info = Path(run_dir) / "best_model" / "best_model_info.json"
    if not info.is_file():
        print(f"⚠️ {info} not found; metrics = 0")
        return _overlay_final_eval_metrics(run_dir, out, eval_split)
    try:
        d = json.loads(info.read_text()).get("all_metrics", {}) or {}
        # Segmentation metrics (masks)
        if d.get("map")    is not None: out["seg_mAP50_95"] = float(d["map"])
        if d.get("map_50") is not None: out["seg_mAP50"]    = float(d["map_50"])
        if d.get("map_75") is not None: out["seg_mAP75"]    = float(d["map_75"])
        if d.get("mar_100") is not None:
            out["seg_AR_100"] = float(d["mar_100"])
        # Box metrics (bounding boxes derived from masks by the patched trainer)
        # The patch adds "_bbox" suffix to all bbox metrics
        for test_key in ["test_map_bbox", "map_bbox"]:
            if d.get(test_key) is not None: out["box_mAP50_95"] = float(d[test_key]); break
        for test_key in ["test_map_50_bbox", "map_50_bbox"]:
            if d.get(test_key) is not None: out["box_mAP50"] = float(d[test_key]); break
        for test_key in ["test_map_75_bbox", "map_75_bbox"]:
            if d.get(test_key) is not None: out["box_mAP75"] = float(d[test_key]); break
        for test_key in ["test_mar_100_bbox", "mar_100_bbox"]:
            if d.get(test_key) is not None:
                out["box_R"] = out["box_AR_100"] = float(d[test_key]); break
        # Size-bucketed mask AP keys come straight from torchmetrics MAP.
        # torchmetrics emits -1.0 as a "no GT for this size bucket" sentinel;
        # clamp to 0.0 so the summary doesn't render negatives.
        for src, dst in (("map_small", "seg_mAP_S"),
                         ("map_medium", "seg_mAP_M"),
                         ("map_large",  "seg_mAP_L")):
            if d.get(src) is not None:
                out[dst] = max(0.0, float(d[src]))
        for src, dst in (("map_small_bbox", "box_mAP_S"),
                         ("map_medium_bbox", "box_mAP_M"),
                         ("map_large_bbox",  "box_mAP_L")):
            if d.get(src) is not None:
                out[dst] = max(0.0, float(d[src]))
    except Exception as e:
        print(f"⚠️ failed to parse {info}: {e}")
    return _overlay_final_eval_metrics(run_dir, out, eval_split)


def _overlay_final_eval_metrics(run_dir, out, eval_split="test"):
    """Overlay <run_dir>/all_results.json test_* onto best-epoch metrics.

    The upstream trainer writes all_results.json after its final pass over the
    reported split, so test_* is the only mask metric in the run dir that was
    actually measured on that split. Overlay it and name the source; when it is
    absent, warn that the row is carrying best-epoch validation numbers instead
    of silently mislabelling them."""
    ar = Path(run_dir) / "all_results.json"
    src = {}
    if ar.is_file():
        try:
            src = json.loads(ar.read_text()) or {}
        except Exception as e:
            print(f"⚠️ failed to parse {ar}: {e}")
    pairs = (("test_map", "seg_mAP50_95"), ("test_map_50", "seg_mAP50"),
             ("test_map_75", "seg_mAP75"), ("test_mar_100", "seg_AR_100"),
             ("test_map_small", "seg_mAP_S"), ("test_map_medium", "seg_mAP_M"),
             ("test_map_large", "seg_mAP_L"))
    applied = []
    for k, dst in pairs:
        if src.get(k) is not None:
            out[dst] = max(0.0, float(src[k]))
            applied.append(dst)
    if applied:
        print(f"📐 metric source: {ar.name} test_* (measured on the reported "
              f"{eval_split} split) — overrode {len(applied)} key(s) from "
              f"best_model_info.json")
    else:
        print(f"⚠️ metric source: best_model/best_model_info.json ONLY — these are "
              f"BEST-EPOCH metrics on the per-epoch eval split, which is val, not "
              f"'{eval_split}'. No test_* in all_results.json to override them. "
              f"Do not quote this row as a {eval_split} number without checking.")
    return out


def _m2f_sanity_dump(coco_dir, model_file, classes):
    """Cheap pre-training sanity checks to diagnose the 0.0007 mAP failure mode.

    Two independent signals get printed:
      1) COCO-side: number of categories, id range, and the first annotation's
         (category_id, bbox/segmentation presence). Carti-97's loader expects
         1-indexed category_ids and either polygon or RLE masks; a category_id
         of 0 or missing 'segmentation' silently zeros mask loss.
      2) Model-side: peek at the upstream Carti-97 .pth dict and look for a
         'backbone' / 'model.backbone' key with a recognisable DINOv3 tensor
         (patch_embed.proj.weight) — if absent or all-zero, the backbone is
         being initialised randomly (gated HF download silently fell back).

    Both checks are best-effort and never raise; failures only print warnings."""
    import json as _json
    try:
        train_json = Path(coco_dir) / "train" / "_annotations.coco.json"
        if train_json.is_file():
            j = _json.loads(train_json.read_text())
            cats = j.get("categories", []) or []
            cat_ids = sorted({c.get("id") for c in cats if c.get("id") is not None})
            anns = j.get("annotations", []) or []
            first = anns[0] if anns else {}
            print(f"🩺 [m2f-sanity] COCO categories: {len(cats)} ids={cat_ids[:8]}"
                  f"{'…' if len(cat_ids) > 8 else ''}  yaml_classes={classes}")
            if anns:
                has_seg = bool(first.get("segmentation"))
                has_bbox = bool(first.get("bbox"))
                print(f"🩺 [m2f-sanity] first ann: image_id={first.get('image_id')} "
                      f"category_id={first.get('category_id')}  has_bbox={has_bbox}  has_seg={has_seg} "
                      f"(n_anns={len(anns)})")
                if first.get("category_id") == 0:
                    print("⚠️  [m2f-sanity] category_id starts at 0 — Carti-97/torchmetrics expects "
                          "1-indexed ids; mask loss may collapse to ~0 if the loader doesn't remap.")
            else:
                print("⚠️  [m2f-sanity] no annotations in train/_annotations.coco.json — "
                      "training will produce noise.")
    except Exception as e:
        print(f"⚠️  [m2f-sanity] COCO inspection skipped: {e}")
    try:
        import torch as _torch
        sd = _torch.load(str(model_file), map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        if not isinstance(sd, dict):
            print(f"🩺 [m2f-sanity] model file {model_file.name}: unrecognised checkpoint layout "
                  f"(type={type(sd).__name__}); cannot verify backbone init.")
            return
        keys = list(sd.keys())
        bb_keys = [k for k in keys if "backbone" in k.lower()]
        patch_keys = [k for k in bb_keys if "patch_embed" in k and "proj.weight" in k]
        if not patch_keys:
            print(f"🩺 [m2f-sanity] {model_file.name}: no backbone/patch_embed weight found "
                  f"(n_keys={len(keys)}, n_backbone_keys={len(bb_keys)}). Pretrained backbone "
                  f"will be downloaded at runtime — check HF auth (HF_TOKEN / --hub_token).")
            return
        t = sd[patch_keys[0]]
        try: stats = (float(t.float().abs().mean()), float(t.float().std()))
        except Exception: stats = (None, None)
        looks_random = stats[0] is not None and (stats[0] < 1e-6 or stats[1] < 1e-6)
        tag = "RANDOM-INIT?" if looks_random else "looks-pretrained"
        print(f"🩺 [m2f-sanity] backbone {patch_keys[0]}: shape={tuple(t.shape)} "
              f"|abs|mean={stats[0]:.4g} std={stats[1]:.4g}  → {tag}")
        if looks_random:
            print("⚠️  [m2f-sanity] backbone weights look uninitialised. Run "
                  "`huggingface-cli login`, pass --hub_token, or set DINOV3_M2F_MODEL=/local/dir.")
    except Exception as e:
        print(f"⚠️  [m2f-sanity] model-file inspection skipped: {e}")


def prune_epoch_dirs(run_dir, keep=1):
    """Carti-97 saves a full HF model dump every epoch (~hundreds of MB each), which
    fills up the disk on long runs. Keep only the most recent K and delete the rest;
    `best_model/` is preserved separately."""
    eps = sorted(Path(run_dir).glob("epoch_*"),
                 key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1)
    victims = eps[:-keep] if keep > 0 else eps
    freed = 0
    for v in victims:
        if v.is_dir():
            try: freed += sum(f.stat().st_size for f in v.rglob("*") if f.is_file())
            except OSError: pass
            shutil.rmtree(v, ignore_errors=True)
    if victims:
        print(f"🧹 Pruned {len(victims)} epoch_*/ dirs from {run_dir} (~{freed/1e9:.2f} GB freed)")


if __name__ == "__main__":
    import multiprocessing
    if sys.platform != "win32":
        try: multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError: pass

    # ---------------------------------------------------------------- Args ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="dinov3-m2f-smallplus")
    ap.add_argument("--task", default="segment", choices=["detect", "segment"])
    ap.add_argument("--epochs", default=50, type=int)
    ap.add_argument("--patience", default=15, type=int)
    ap.add_argument("--batch_sz", default=4, type=float)
    ap.add_argument("--imgsz", default=384, type=int)
    ap.add_argument("--lr", default=1e-4, type=float)
    ap.add_argument("--threads", default=4, type=int)
    ap.add_argument("--grad_accum_steps", default=0, type=int,
                    help="HF Trainer gradient_accumulation_steps. 0 = pick automatically so"
                         " micro_bs * grad_accum >= --batch_effective (shared bench floor).")
    ap.add_argument("--batch_effective", default=8, type=int,
                    help="Target effective batch size; auto grad_accum = ceil(N / micro_bs).")
    ap.add_argument("--resume", action='store_true')
    ap.add_argument("--inference_only", action='store_true')
    ap.add_argument("--tune", action='store_true')
    ap.add_argument("--tune_iter", default=5, type=int)
    ap.add_argument("--repo_dir", default="")
    ap.add_argument("--hub_token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--aug_policy", choices=["native", "off", "fair"], default="native",
                    help="When != native, monkey-patch the upstream COCOInstanceDataset so its "
                         "train-split __getitem__ passes through the shared Albumentations "
                         "pipeline (image + masks) before upstream transforms.")
    ap.add_argument("--keep_epochs", default=1, type=int,
                    help="How many epoch_*/ dirs to keep after training (0 = keep all). Saves disk.")
    ap.add_argument("--seg_via_sam2", action='store_true',
                    help="After eval, prompt SAM2 with predicted bboxes to compute sam2_* mAP.")
    ap.add_argument("--sam2_ckpt", default=os.environ.get("SAM2_CKPT", ""))
    ap.add_argument("--sam2_config", default=os.environ.get("SAM2_CONFIG", ""))
    args = ap.parse_args()
    # R.6: reject fractional --batch_sz loudly. int(0.98)=0 silently zero-batches M2F
    # (the Ultralytics-style fractional auto-batch does NOT apply here; M2F truncates).
    if args.batch_sz < 1:
        print(f"❌ --batch_sz must be >= 1 (got {args.batch_sz}); M2F truncates to int(bs)=0")
        sys.exit(1)
    _m2f_aug_status = "applied" if args.aug_policy in ("off", "fair") else "native"
    print(f"[bench] aug_policy={args.aug_policy}: dinov3_m2f -> {_m2f_aug_status}")
    if args.repo_dir:
        REPO_CACHE = Path(args.repo_dir).expanduser().resolve()

    ORIG_DATA = Path(args.data).resolve()
    if not ORIG_DATA.is_dir(): print("❌ Data directory not found"); sys.exit(1)
    DATA_YAML = ORIG_DATA / 'data.yaml'
    if not DATA_YAML.exists(): print("❌ data.yaml not found"); sys.exit(1)
    DATA_NAME = ORIG_DATA.name

    # Bench contract: evaluate test when present. The upstream trainer only
    # evaluates one `valid_dataloader`, so we must control what feeds it.
    # eval_split_as_valid=False keeps YOLO val(173) -> COCO `valid` and YOLO
    # test(1016) -> COCO `test` as separate dirs. The upstream
    # COCOInstanceDataset(..., "valid") then loads 173 images, so the per-epoch
    # eval runs on val, not test (~6x faster). The final eval swaps to a test
    # dataloader (built below) so predictions.json is written from the 1016-image
    # test set, which is what the post-hoc coco_eval_full scores for the paper's
    # seg_mAP. eval_split_as_valid=True collapsed test into valid, making every
    # per-epoch eval iterate the 1016-image test set; that was the bug, not a
    # slow implementation.
    import yaml as _yaml
    with open(DATA_YAML) as _yf:
        _ycfg = _yaml.safe_load(_yf) or {}
    EVAL_SPLIT = "test" if "test" in _ycfg else ("valid" if "valid" in _ycfg else "val")

    print("🔄 Converting YOLO → COCO …")
    COCO_DIR, CLASSES = yolo_to_coco_layout(DATA_YAML, DATA_NAME, eval_split_as_valid=False)
    print(f"✅ COCO ready at {COCO_DIR} (classes: {CLASSES}; eval_split={EVAL_SPLIT})")

    REPO_DIR = ensure_repo()
    MODEL_FILE = REPO_DIR / pick_model_file(args.model)
    if not MODEL_FILE.exists(): print(f"❌ Upstream model file not found: {MODEL_FILE}"); sys.exit(1)
    warn_if_dinov3_access_missing(args.model, args.hub_token)
    # Pre-flight sanity checks — printed once before training. These help
    # distinguish "training collapsed because backbone fell back to random
    # init" (HF gating) from "COCO category-id off-by-one" type failures
    # without needing to wait out a full run of zero-mAP epochs.
    _m2f_sanity_dump(COCO_DIR, MODEL_FILE, CLASSES)

    MODEL_STEM = Path(args.model).stem
    _env_run = os.environ.get("BENCH_RUN_NAME")
    RUN_NAME = _env_run if _env_run else f"{DATA_NAME[:15]}_{MODEL_STEM}"
    TASK_DIR = "segment" if args.task == "segment" else "detect"
    # RUN_DIR is on disk (not /dev/shm) — tmpfs fills up after ~20 epochs of model shards.
    # Routed under ~/bench_any/ so nothing lands in the project / dataset directory.
    RUN_DIR = Path.home() / 'bench_any' / 'runs' / TASK_DIR / RUN_NAME
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH = RUN_DIR / "train.log"
    os.environ.setdefault('WANDB_PROJECT', f"DINOv3-M2F-{TASK_DIR}")
    os.environ.setdefault('WANDB_DIR', str(RUN_DIR))

    bs = max(1, int(args.batch_sz)) if args.batch_sz >= 1 else 1
    epochs = 0 if args.inference_only else args.epochs
    # Auto grad-accum so effective batch >= --batch_effective (default 8), matching the
    # floor used across all bench tuners (Ultralytics nbs=64, RF-DETR auto target=16, etc.).
    GRAD_ACCUM = (args.grad_accum_steps if args.grad_accum_steps > 0
                  else max(1, (max(1, args.batch_effective) + bs - 1) // bs))
    print(f"📐 micro_bs={bs}  grad_accum={GRAD_ACCUM}  effective_bs={bs * GRAD_ACCUM}")

    # Bootstrap: switch torch DataLoader sharing strategy to 'file_system' so worker
    # IPC stops going through /dev/shm (mask tensors easily blow past the tmpfs quota,
    # manifesting as `Disk quota exceeded (122)` at step 0).
    LAUNCHER = RUN_DIR / "_launch.py"
    # ---- bench_any: per-image COCO predictions dump (M2F) ------------------
    # M2F's upstream evaluation_loop only updates a torchmetrics
    # MeanAveragePrecision in-memory and never writes per-image predictions
    # to disk, so our post-hoc `coco_eval_full` had nothing to consume and
    # box_*, seg_P/R/F1, seg S/M/L, seg_AR_100 stayed at 0. We monkey-patch
    # the upstream script (loaded via runpy) BEFORE main() runs:
    #   (a) COCOInstanceDataset.__getitem__ attaches image_id + file_name
    #       to every sample;
    #   (b) collate_fn carries them through the DataLoader;
    #   (c) evaluation_loop additionally dumps RLE-encoded predictions to
    #       <RUN_DIR>/predictions.json at the end of every eval pass (the
    #       LAST eval wins, which matches what HF Trainer logs as test).
    _PRED_JSON = str(RUN_DIR / "predictions.json")
    _FPS_JSON  = str(RUN_DIR / "_bench_fps.json")
    _UPSTREAM = str(REPO_DIR / "mask2former_dinov3_no_trainer_coco.py")
    LAUNCHER.write_text(f"""
import sys, runpy, json as _json, os as _os
import torch.multiprocessing as _mp
try: _mp.set_sharing_strategy('file_system')
except Exception: pass

_script = {_UPSTREAM!r}

# Pre-load the upstream module as a module so we can patch its classes /
# functions before main() runs.
import importlib.util as _ilu
def _load_upstream():
    if "_m2f_upstream" in sys.modules: return sys.modules["_m2f_upstream"]
    _spec = _ilu.spec_from_file_location("_m2f_upstream", _script)
    if _spec is not None and _spec.loader is not None:
        _m = _ilu.module_from_spec(_spec)
        sys.modules["_m2f_upstream"] = _m
        _spec.loader.exec_module(_m)
        return _m
    # Fallback: if spec fails, try to find it via sys.path
    _d = _os.path.dirname(_script)
    if _d not in sys.path: sys.path.insert(0, _d)
    import importlib
    _m = importlib.import_module(_os.path.basename(_script).replace(".py", ""))
    sys.modules["_m2f_upstream"] = _m
    return _m

# Load and patch unconditionally (cvBench speed port): a spawned DataLoader
# worker re-imports this launcher with __name__ != "__main__", and the old guard
# left it with _mod=None, so `_m2f_upstream` was never registered and unpickling
# the patched dataset died with `ModuleNotFoundError: _m2f_upstream`. Loading
# here (idempotent — checks sys.modules first) lets workers resolve the patched
# module, so dataloader_num_workers>0 is legal. This is the only reason that
# count was pinned to 0 below.
_mod = _load_upstream()

if _mod:
    # ---- (a) Dataset: attach image_id + file_name -----------------------------
    try:
        _DS = _mod.COCOInstanceDataset
        _orig_gi = _DS.__getitem__
        # bench_any: inject shared Albumentations pipeline (train split only) BEFORE
        # the upstream image_processor runs. Done by installing a self.transform
        # equal to the shared list (the upstream __getitem__ already applies it).
        _BENCH_AUG_POLICY = {args.aug_policy!r}
        try:
            import sys as _sys, os as _os
            _be_dir = _os.environ.get("BENCH_EVAL_DIR")
            if _be_dir and _be_dir not in _sys.path: _sys.path.insert(0, _be_dir)
            if _BENCH_AUG_POLICY in ("off", "fair"):
                from bench_aug import shared_albumentations_policy as _sap
                import albumentations as _A
                _pipe = _sap(_BENCH_AUG_POLICY, 640)
                _BENCH_TFORM = _A.Compose(_pipe) if _pipe else None
                print(f"[bench_any] dinov3_m2f shared aug installed policy={{_BENCH_AUG_POLICY}}")
            else:
                _BENCH_TFORM = None
        except Exception as _e:
            print(f"[bench_any] dinov3_m2f shared aug unavailable: {{_e}}")
            _BENCH_TFORM = None

        def _patched_gi(self, idx):
            # Apply shared aug on train split only by installing self.transform once.
            if _BENCH_TFORM is not None and getattr(self, "split", "") == "train" \\
                    and getattr(self, "_bench_aug_installed", False) is False:
                self.transform = _BENCH_TFORM
                self._bench_aug_installed = True
            out = _orig_gi(self, idx)
            try:
                img_id = self.image_ids[idx]
                ann = self.annotations[img_id]
                out["_bench_image_id"] = int(img_id)
                out["_bench_file_name"] = str(ann["file_name"])
            except Exception:
                pass
            return out
        _DS.__getitem__ = _patched_gi
    except Exception as _e:
        print(f"[bench_any] M2F dataset patch failed: {{_e}}")

    # ---- (b) collate_fn: carry our two keys through DataLoader ---------------
    try:
        _orig_cf = _mod.collate_fn
        def _patched_cf(examples):
            batch = _orig_cf(examples)
            batch["_bench_image_ids"]  = [int(e.get("_bench_image_id", -1))   for e in examples]
            batch["_bench_file_names"] = [str(e.get("_bench_file_name", "")) for e in examples]
            return batch
        _mod.collate_fn = _patched_cf
    except Exception as _e:
        print(f"[bench_any] M2F collate patch failed: {{_e}}")

    # ---- (c) evaluation_loop: dump predictions.json --------------------------
    try:
        import torch as _t
        import numpy as _np
        import time as _time
        from pycocotools import mask as _mu
        _orig_eval = _mod.evaluation_loop
        _PRED_PATH = {_PRED_JSON!r}
        _FPS_PATH  = {_FPS_JSON!r}
        # bench_any: per-epoch eval runs on val(173), final eval runs on test(1016).
        # eval_split_as_valid=False (set above) keeps the two splits separate in the
        # generated COCO dir, so the upstream trainer's valid_dataloader feeds on
        # val(173) — the per-epoch eval is cheap. The upstream trainer has only one
        # eval dataloader and calls evaluation_loop once per epoch (best-model
        # selection + logging) plus once at the end; the upstream passes valid_dataloader
        # to ALL of them, including the final one. But the final eval must write
        # predictions.json from the 1016-image test set (the post-hoc coco_eval_full
        # scores that file against the test GT for the paper's seg_mAP). So on the
        # final call we swap the dataloader to a test-set dataloader built here from
        # the upstream's own COCOInstanceDataset + collate_fn.
        #
        # Final-call detection: with --checkpointing_steps epoch the upstream trainer
        # calls evaluation_loop exactly once per epoch (no mid-epoch step-checkpoint
        # eval), so call N == epoch N and the final call is _call_n >= total_epochs.
        # Resume is weight-reload-only for M2F (no optimizer state), so the counter
        # restarts at 0 and the call counter is correct across resume.
        #
        # Env knobs:
        #   BENCH_M2F_PEREPOCH_THRESH (default 0.3) score threshold for the per-epoch
        #     val eval. Raising it 6x cuts the mask count (68k -> 11k on PW) and the
        #     RLE post-process with it, at ~1 pt mAP50 cost (verified 0.6247 -> 0.6155).
        #     The final test eval always uses 0.0.
        #   BENCH_M2F_PEREPOCH_STRIDE (default 1) run per-epoch val eval every Nth
        #     epoch instead of every epoch. 1 = every epoch (default, full curve).
        #     5 = epochs 1,5,10,...,120 plus the final test eval — a real learning
        #     curve at ~20% of the eval cost (~30 min vs ~2.5h per m2f cell on PW),
        #     for re-runs where only the curve shape matters. Skipped epochs carry
        #     forward the last sampled metrics (plateaus, not zeros), so the
        #     history.csv curve stays continuous and the upstream best-epoch
        #     tracker (strict >) never fires on a carry-forward. The final test
        #     eval always runs regardless of stride.
        #   BENCH_M2F_KEEP_INMEM_EVAL (unchanged) gate the in-memory MAP.
        _BENCH_TOTAL_EPOCHS = {epochs!r}
        _BENCH_PEREPOCH_THRESH = float(_os.environ.get("BENCH_M2F_PEREPOCH_THRESH", "0.3"))
        _BENCH_PEREPOCH_STRIDE = max(1, int(_os.environ.get("BENCH_M2F_PEREPOCH_STRIDE", "1")))
        _BENCH_COCO_DIR = {str(COCO_DIR)!r}
        _BENCH_EVAL_BS = {bs!r}
        _BENCH_DL_WORKERS = 0
        _bench_eval_call_n = [0]
        _bench_is_final_cached = [None]
        _bench_test_dl = [None]
        _last_sampled_metrics = [None]

        # bench_any: the final test eval must score the SELECTED checkpoint, not
        # whatever the last epoch happened to be. Upstream calls the eval hook
        # from inside the training loop, so without this the reported number is
        # the last epoch on the test split while yolo/rtdetr report their best.
        # Loaded once, in place, on the final call only; the training loop is
        # over by then, so mutating the live weights costs nothing.
        _bench_best_loaded = [False]

        def _bench_load_best_for_final_eval(model, accelerator):
            if _bench_best_loaded[0]:
                return
            _bench_best_loaded[0] = True
            _bd = _os.environ.get("BENCH_M2F_BEST_DIR", "")
            _sft = _os.path.join(_bd, "model.safetensors") if _bd else ""
            if not _sft or not _os.path.isfile(_sft):
                print("[bench_any] M2F final eval: no best_model/model.safetensors "
                      "found; scoring the LAST-epoch weights (report accordingly)")
                return
            try:
                from safetensors.torch import load_file as _load_sft
                _tgt = accelerator.unwrap_model(model)
                _sd = _load_sft(_sft)
                _cur = _tgt.state_dict()
                _ok = {{_k: _v for _k, _v in _sd.items()
                        if _k in _cur and tuple(_cur[_k].shape) == tuple(_v.shape)}}
                # A PARTIAL swap is worse than either endpoint: the scored weights
                # would be neither the selected checkpoint nor the last epoch, and
                # the print below would still read as success. Refuse anything but
                # a complete match and say which weights actually got scored.
                if len(_ok) < len(_sd):
                    print(f"[bench_any] M2F final eval: best-checkpoint keys do NOT "
                          f"match the live model ({{len(_ok)}}/{{len(_sd)}} tensors "
                          f"usable); scoring the LAST-epoch weights instead. The "
                          f"reported number is NOT the selected checkpoint.")
                    return
                _res = _tgt.load_state_dict(_ok, strict=False)
                print(f"[bench_any] M2F final eval: loaded the SELECTED checkpoint "
                      f"from {{_sft}} ({{len(_ok)}}/{{len(_sd)}} tensors, "
                      f"missing={{len(getattr(_res, 'missing_keys', []) or [])}})")
            except Exception as _e_best:
                print(f"[bench_any] M2F final eval: best-checkpoint load FAILED "
                      f"({{_e_best}}); scoring the LAST-epoch weights")

        def _patched_eval(model, image_processor, accelerator, dataloader, id2label):
            from torchmetrics.detection.mean_ap import MeanAveragePrecision
            from tqdm import tqdm as _tqdm
            from accelerate.utils import send_to_device as _send_to_device
            _bench_eval_call_n[0] += 1
            _call_n = _bench_eval_call_n[0]
            _is_final = _call_n >= _BENCH_TOTAL_EPOCHS
            if _is_final:
                _bench_is_final_cached[0] = True
            # Sampling: on a non-final epoch that is not on the stride, skip the
            # full val eval and carry forward the last sampled metrics. This keeps
            # the per-epoch JSON/history populated (plateau, not zero) and avoids
            # the upstream best-epoch tracker firing on a carry-forward (strict >).
            # The final test eval always runs regardless of stride.
            _do_eval = _is_final or (_call_n % _BENCH_PEREPOCH_STRIDE == 0)
            if not _do_eval and _last_sampled_metrics[0] is not None:
                print(f"[bench_any] M2F per-epoch eval {{_call_n}}/{{_BENCH_TOTAL_EPOCHS}} skipped (stride={{_BENCH_PEREPOCH_STRIDE}}); carrying forward last sampled metrics")
                return dict(_last_sampled_metrics[0])
            # On the final call, swap the val dataloader for a test dataloader so
            # predictions.json is written from the 1016-image test set. Built once
            # (cached) from the upstream dataset class + collate_fn.
            if _is_final and _bench_test_dl[0] is None:
                try:
                    import albumentations as _A
                    _test_ds = _mod.COCOInstanceDataset(
                        _BENCH_COCO_DIR, "test", image_processor,
                        transform=_A.Compose([_A.NoOp()]), use_cache=True)
                    _cf = getattr(_mod, "collate_fn", None)
                    _kw = dict(shuffle=False, batch_size=_BENCH_EVAL_BS,
                               num_workers=_BENCH_DL_WORKERS,
                               pin_memory=True)
                    if _cf is not None:
                        _kw["collate_fn"] = _cf
                    _bench_test_dl[0] = _mod.DataLoader(_test_ds, **_kw)
                    print(f"[bench_any] M2F final eval: built test dataloader ({{len(_test_ds)}} imgs) for predictions.json")
                except Exception as _e:
                    print(f"[bench_any] M2F final eval: test dataloader build FAILED ({{_e}}); falling back to passed dataloader")
            if _is_final and _bench_test_dl[0] is not None:
                dataloader = _bench_test_dl[0]
            if _is_final:
                _bench_load_best_for_final_eval(model, accelerator)
            _eval_thresh = 0.0 if _is_final else _BENCH_PEREPOCH_THRESH
            if not _is_final:
                print(f"[bench_any] M2F per-epoch eval {{_call_n}}/{{_BENCH_TOTAL_EPOCHS}} (val, thresh={{_eval_thresh}})")
            else:
                print(f"[bench_any] M2F FINAL test eval {{_call_n}} (test, thresh=0.0)")
            # Raise the per-image detection cap (default [1,10,100]) and silence
            # the ">100 detections" warning: crowded penguin scenes routinely
            # carry hundreds of instances that the default would truncate.
            metric = MeanAveragePrecision(iou_type="segm", class_metrics=True,
                                          max_detection_thresholds=[1, 10, 1000]).to(accelerator.device)
            all_rows = []
            # bench_any: in-memory torchmetrics MAP accumulation is the eval
            # OOM. post_process at threshold=0.0 with return_binary_maps=True
            # makes ~100 full 640x640 bool masks per image on GPU, and
            # metric.update holds them across all 254 val images (~10 GB)
            # until compute() on a 16 GB card. The real per-epoch numbers come
            # from coco_eval_full re-scoring all_rows below, not from
            # metric.compute(), so this accumulation is dead work. Default
            # off; set BENCH_M2F_KEEP_INMEM_EVAL=1 to restore it for debugging.
            _keep_inmem = _os.environ.get("BENCH_M2F_KEEP_INMEM_EVAL", "0") == "1"
            _tot_ms = 0.0; _tot_imgs = 0
            _is_cuda = str(getattr(accelerator, "device", "")).startswith("cuda")
            for inputs in _tqdm(dataloader, total=len(dataloader),
                                disable=not accelerator.is_local_main_process):
                # Pop our bench-only sidecars BEFORE feeding the model.
                _bench_ids   = inputs.pop("_bench_image_ids",  None)
                _bench_names = inputs.pop("_bench_file_names", None)
                original_sizes = inputs.pop("original_sizes", None)

                # The bench-built test dataloader is a plain torch DataLoader, not
                # an accelerate-prepared one, so its batches stay on the CPU while
                # the prepared model sits on the GPU. Under --inference_only
                # (num_train_epochs=0) the FIRST evaluation_loop call is already the
                # final one, and it crashed with "Input type (torch.FloatTensor) and
                # weight type (torch.cuda.FloatTensor) should be the same". Placing
                # the batch here is a no-op when it is already on the device.
                inputs = _send_to_device(inputs, accelerator.device)

                # Time forward + post-process. Sync GPU around the region so we
                # don't undercount when CUDA work overlaps the Python loop.
                if _is_cuda:
                    try: _t.cuda.synchronize()
                    except Exception: pass
                _t0 = _time.perf_counter()
                with _t.no_grad():
                    outputs = model(**inputs)

                if original_sizes is not None:
                    current_target_sizes = original_sizes
                else:
                    current_target_sizes = [m.shape[-2:] for m in inputs["mask_labels"]]

                post = image_processor.post_process_instance_segmentation(
                    outputs, threshold=_eval_thresh, target_sizes=current_target_sizes,
                    return_binary_maps=True,
                )
                # End the timed region: sync GPU then accumulate ms & image count.
                if _is_cuda:
                    try: _t.cuda.synchronize()
                    except Exception: pass
                _tot_ms += (_time.perf_counter() - _t0) * 1000.0
                _tot_imgs += int(len(current_target_sizes))

                post_preds, post_tgts = [], []
                for j, (ipred, tsz) in enumerate(zip(post, current_target_sizes)):
                    if ipred["segments_info"]:
                        masks_t = ipred["segmentation"].to(dtype=_t.bool)
                        labels_t = _t.tensor([x["label_id"] for x in ipred["segments_info"]],
                                             device=accelerator.device)
                        scores_t = _t.tensor([x["score"]    for x in ipred["segments_info"]],
                                             device=accelerator.device)
                    else:
                        masks_t  = _t.zeros([0, *tsz], dtype=_t.bool, device=accelerator.device)
                        labels_t = _t.tensor([], device=accelerator.device)
                        scores_t = _t.tensor([], device=accelerator.device)
                    post_preds.append({{"masks": masks_t, "labels": labels_t, "scores": scores_t}})
                    post_tgts.append({{"masks": inputs["mask_labels"][j].to(dtype=_t.bool),
                                      "labels": inputs["class_labels"][j]}})

                    # Dump to JSON (one row per predicted instance).
                    if (_bench_ids is not None and j < len(_bench_ids)) or (_bench_names is not None and j < len(_bench_names)):
                        img_id = int(_bench_ids[j]) if _bench_ids is not None and j < len(_bench_ids) else -1
                        file_name = str(_bench_names[j]) if _bench_names is not None and j < len(_bench_names) else ""
                        m_np = masks_t.detach().cpu().numpy().astype("uint8")
                        lbls = labels_t.detach().cpu().numpy().tolist()
                        scrs = scores_t.detach().cpu().numpy().tolist()
                        for k in range(m_np.shape[0]):
                            if scrs[k] <= 1e-3: continue
                            m = m_np[k]
                            if m.sum() == 0: continue
                            rle = _mu.encode(_np.asfortranarray(m))
                            if isinstance(rle.get("counts"), bytes):
                                rle["counts"] = rle["counts"].decode("ascii")
                            # bbox via np.where is the dominant per-epoch CPU
                            # cost (one full-mask scan per instance). Only the
                            # final test eval's coco_eval_full bbox path needs
                            # it; per-epoch val eval is segm-only, so skip it.
                            if _is_final:
                                ys, xs = _np.where(m > 0)
                                x1, x2 = int(xs.min()), int(xs.max())
                                y1, y2 = int(ys.min()), int(ys.max())
                                _bbox = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
                            else:
                                _bbox = [0, 0, 0, 0]
                            _row = {{
                                "_file_name": file_name,
                                # category_id stays whatever M2F predicted; it's
                                # already aligned with our 1-indexed GT cats
                                # (the upstream patch maps {{0:0, **inst→sem}}).
                                "category_id": int(lbls[k]),
                                "score": float(scrs[k]),
                                "bbox": _bbox,
                                "segmentation": rle,
                            }}
                            if img_id >= 0:
                                _row["image_id"] = int(img_id)
                            all_rows.append(_row)
                if _keep_inmem:
                    metric.update(post_preds, post_tgts)
                # bench_any: free this batch's full-res binary maps now; the
                # post-process loop above has already flattened them into
                # all_rows, so keeping them lets ~640x640x100 bool tensors pile
                # up across the epoch. Critical when inmem metric is off.
                del post_preds, post_tgts, post
                if _is_cuda:
                    try: _t.cuda.empty_cache()
                    except Exception: pass

            # Snapshot the predictions to disk (only on the main process to avoid
            # multi-rank clobbering). The canonical predictions.json (threshold=0.0,
            # full bbox) is written ONLY on the final test eval so a per-epoch call
            # never overwrites it with a lower-quality (threshold=0.3, stub bbox)
            # dump. Per-epoch calls snapshot to predictions_best.json whenever val
            # mAP sets a new high, so a crash before the final eval still leaves a
            # harvestable prediction set (see the best-tracking block below).
            try:
                if accelerator.is_local_main_process:
                    if _is_final:
                        _tmp = _PRED_PATH + ".tmp"
                        with open(_tmp, "w") as _fh:
                            _json.dump(all_rows, _fh)
                        _os.replace(_tmp, _PRED_PATH)
                        print(f"[bench_any] M2F predictions.json: {{len(all_rows)}} rows -> {{_PRED_PATH}}")
                        # Persist a rolling FPS estimate.
                        _fps = (_tot_imgs * 1000.0 / _tot_ms) if _tot_ms > 0 else 0.0
                        with open(_FPS_PATH, "w") as _ffh:
                            _json.dump({{"fps_eval": float(_fps), "n_imgs": int(_tot_imgs)}}, _ffh)
            except Exception as _e_w:
                print(f"[bench_any] M2F predictions dump write failed: {{_e_w}}")

            # ---- bench_any: REAL per-epoch metrics for logs + graphs ----------
            # The upstream in-memory torchmetrics MeanAveragePrecision is corrupted
            # by the bg-id-0 patch (the whole-image background is injected as an
            # "instance" sharing class id 0 with the foreground), so it reports
            # near-zero mAP every epoch. That same number is what the upstream
            # trainer logs ("epoch N: {{...}}") and writes to epoch_N_metrics.json,
            # which in turn feeds the run's graphs/ — hence the all-near-zero plots.
            # Re-score the dumped predictions with the SAME post-hoc COCO pipeline
            # the CSV trusts (bench_eval.coco_eval_full, which remaps category ids),
            # against the converted GT for this eval split, and return THOSE numbers
            # under torchmetrics-style keys so the log line, the per-epoch JSON and
            # the graphs all show the real metrics. Any failure falls back silently
            # to the original in-memory result so eval never crashes.
            _real_metrics = None
            try:
                if accelerator.is_local_main_process and all_rows:
                    import sys as _sys3, os as _os3
                    # bench_eval.py lives next to the wrapper. In a full bench run
                    # BENCH_EVAL_DIR points there; when the wrapper is launched
                    # directly it may be unset, so fall back to the wrapper's dir
                    # (baked in at launcher-write time) so the import never fails.
                    for _cand3 in (_os3.environ.get("BENCH_EVAL_DIR"), {str(Path(__file__).resolve().parent)!r}):
                        if _cand3 and _cand3 not in _sys3.path:
                            _sys3.path.insert(0, _cand3)
                    import bench_eval as _be3
                    # Locate the COCO GT json for the split this loader iterates.
                    _ds3 = getattr(dataloader, "dataset", None)
                    _ds3 = getattr(_ds3, "dataset", _ds3)  # unwrap random_split Subset
                    _gt3 = None
                    try:
                        _gt3 = str(_ds3.data_dir / _ds3.split / "_annotations.coco.json")
                    except Exception:
                        _gt3 = None
                    if _gt3 and _os3.path.isfile(_gt3):
                        # Score against a COPY so coco_eval_full's in-place category
                        # remap can never mutate the canonical predictions.json that
                        # the wrapper's post-training harvest consumes.
                        _pe = _PRED_PATH + ".epoch_eval.json"
                        with open(_pe, "w") as _pfh3:
                            _json.dump(all_rows, _pfh3)
                        _r3 = _be3.coco_eval_full(_gt3, None, _pe)
                        try: _os3.remove(_pe)
                        except Exception: pass
                        _g3 = lambda k: float(_r3.get(k, 0.0) or 0.0)
                        # Only trust the COCO numbers when they actually scored
                        # something; otherwise keep the in-memory fallback.
                        if any(_g3(k) > 0.0 for k in
                               ("seg_mAP50_95", "seg_mAP50", "seg_mAP75", "seg_AR_100")):
                            _real_metrics = {{
                                "map":        _g3("seg_mAP50_95"),
                                "map_50":     _g3("seg_mAP50"),
                                "map_75":     _g3("seg_mAP75"),
                                "map_small":  _g3("seg_mAP_S"),
                                "map_medium": _g3("seg_mAP_M"),
                                "map_large":  _g3("seg_mAP_L"),
                                "mar_100":    _g3("seg_AR_100"),
                            }}
                            print("[bench_any] M2F real eval (COCO remap): "
                                  f"map={{_real_metrics['map']:.4f}} "
                                  f"map_50={{_real_metrics['map_50']:.4f}} "
                                  f"map_75={{_real_metrics['map_75']:.4f}}")
            except Exception as _e_rm:
                print(f"[bench_any] M2F real per-epoch eval failed ({{_e_rm}}); "
                      "falling back to in-memory torchmetrics")

            # bench_any: snapshot predictions_best.json whenever a per-epoch val
            # eval sets a new high mAP50. If the run dies before the final test
            # eval, this file is the harvestable record of the best epoch's
            # predictions (segm-only, threshold=0.3). The final test eval writes
            # the canonical predictions.json separately above.
            if not _is_final and _real_metrics is not None and accelerator.is_local_main_process:
                try:
                    _best_path = _PRED_PATH + ".best.json"
                    _cur_best = -1.0
                    if _os.path.isfile(_best_path):
                        import json as _bj
                        with open(_best_path) as _bfh:
                            _cur_best = float(_bj.load(_bfh).get("_bench_best_map50", -1.0))
                    if float(_real_metrics["map_50"]) > _cur_best:
                        _tmp_b = _best_path + ".tmp"
                        with open(_tmp_b, "w") as _bfh:
                            _json.dump({{"_bench_best_map50": float(_real_metrics["map_50"]),
                                         "predictions": all_rows}}, _bfh)
                        _os.replace(_tmp_b, _best_path)
                        print(f"[bench_any] M2F predictions_best.json updated: map50={{_real_metrics['map_50']:.4f}} (epoch call {{_call_n}})")
                except Exception as _e_b:
                    print(f"[bench_any] M2F predictions_best snapshot failed: {{_e_b}}")

            if _real_metrics is not None:
                # Cache so a stride-skipped epoch can carry this forward instead
                # of returning a zero/placeholder and breaking the history curve.
                if not _is_final:
                    _last_sampled_metrics[0] = dict(_real_metrics)
                return _real_metrics
            # bench_any: do NOT fall back to metric.compute() here. The in-memory
            # MAP was never updated when _keep_inmem is off, so compute() would
            # raise on an empty state; and when _keep_inmem is on, the bg-id-0
            # patch corrupts it to near-zero (see the comment above). Either way
            # it is the wrong number, and returning it silently would let a
            # failing coco_eval_full look like a successful eval. Raise loudly
            # instead, unless explicitly debugging with the inmem path on.
            if _keep_inmem:
                return metric.compute()
            raise RuntimeError(
                "bench_any M2F eval: coco_eval_full produced no trusted metrics "
                "and the in-memory fallback is disabled (BENCH_M2F_KEEP_INMEM_EVAL "
                "unset). Refusing to return a silent zero. Check the GT path "
                "and bench_eval import in the patched evaluation_loop.")

        _mod.evaluation_loop = _patched_eval
        print(f"[bench_any] M2F prediction dump enabled -> {{_PRED_PATH}}")
    except Exception as _e:
        print(f"[bench_any] M2F evaluation_loop patch failed: {{_e}}")

# Finally invoke main() inside the (now-patched) upstream module.
if __name__ == "__main__":
    if _mod:
        # Shift argv only in main process to maintain compatibility with upstream argparse
        if len(sys.argv) > 1 and sys.argv[1] == _script:
            sys.argv = [_script] + sys.argv[2:]

        if hasattr(_mod, "main"):
            _mod.main()
        else:
            runpy.run_path(_script, run_name='__main__')
""")
    # Clinical: the upstream module is now dynamically loaded at module level as
    # `_m2f_upstream` (cvBench speed port), so forkserver/spawn DataLoader workers
    # CAN import its patched dataset class. `--threads 0` means auto
    # (min(8, cpu_count)); -1 forces single-process.
    dl_workers = 0 if int(args.threads) < 0 else (int(args.threads or 0) or min(8, os.cpu_count() or 8))

    cmd = [sys.executable, str(LAUNCHER),
           str(REPO_DIR / "mask2former_dinov3_no_trainer_coco.py"),
           "--model", str(MODEL_FILE), "--dataset_name", str(COCO_DIR),
           "--image_height", str(args.imgsz), "--image_width", str(args.imgsz),
           "--per_device_train_batch_size", str(bs),
           "--per_device_eval_batch_size", str(max(1, bs)),
           "--dataloader_num_workers", str(dl_workers),
           "--learning_rate", str(args.lr), "--num_train_epochs", str(epochs),
           "--gradient_accumulation_steps", str(GRAD_ACCUM),
           "--output_dir", str(RUN_DIR)]
    if args.hub_token: cmd += ["--hub_token", args.hub_token]
    # Persist a full accelerate state (optimizer/scheduler/RNG/epoch) every epoch
    # so an interrupted run can truly resume mid-training. Upstream only does this
    # when --checkpointing_steps is set; without it the per-epoch epoch_*/ dirs are
    # weights-only (save_pretrained) and cannot restore optimizer/epoch state, so a
    # crash silently restarted training from epoch 0.
    if not args.inference_only and epochs > 0:
        cmd += ["--checkpointing_steps", "epoch"]
    # Resume from the most recent resumable state dir (checkpoint_epoch_N) if one
    # exists. (The old code looked for RUN_DIR/checkpoint-last, which the upstream
    # trainer never creates, so --resume was a silent no-op for M2F.)
    _warmstart_path = None
    if args.resume:
        _state_dirs = sorted(
            RUN_DIR.glob("checkpoint_epoch_*"),
            key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1)
        _resume_ckpt = _state_dirs[-1] if _state_dirs else None
        if _resume_ckpt is not None and _resume_ckpt.is_dir():
            cmd += ["--resume_from_checkpoint", str(_resume_ckpt)]
            print(f"⏯  Resuming M2F from {_resume_ckpt.name}")
        else:
            # No resumable accelerate state. Fall back to a weights-only WARM-START
            # from the newest epoch_N/ (or best_model/) so a run that crashed before
            # the resume machinery existed doesn't throw away its learned weights and
            # restart from random M2F-head init. Optimizer/scheduler momentum is lost
            # (we cannot recover it from a save_pretrained dump), but the network's
            # progress is preserved; resumable checkpoint_epoch_* are written from now
            # on, so any FUTURE crash resumes mid-training proper.
            _ep_dirs = sorted(
                RUN_DIR.glob("epoch_*"),
                key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1)
            for _cand_dir in ([_ep_dirs[-1]] if _ep_dirs else []) + [RUN_DIR / "best_model"]:
                _cand_sft = _cand_dir / "model.safetensors"
                if _cand_sft.is_file():
                    _warmstart_path = _cand_sft
                    break
            if _warmstart_path is not None:
                print(f"⏯  No resumable state; WARM-STARTING weights from "
                      f"{_warmstart_path.parent.name}/model.safetensors "
                      "(optimizer state not recoverable; resumable checkpoints "
                      "written from now on).")
                # Stabilise the warm-start source and reclaim disk. The stale
                # weights-only epoch_*/ dirs (e.g. the 152 GB epoch_0..45 left by
                # the pre-resume crash) must be cleared BEFORE training: this run
                # restarts epoch numbering at 0, so the upstream per-epoch pruner
                # (which keeps the highest-numbered epoch_*) would otherwise keep a
                # stale high epoch_N and delete the fresh ones. Copy the chosen
                # weights to a stable file first so the child can still load them.
                _ws_stable = RUN_DIR / "_bench_warmstart.safetensors"
                try:
                    shutil.copy2(_warmstart_path, _ws_stable)
                    # Epoch numbering restarts at 0 for a warm-started TRAINING run,
                    # so stale epoch_*/ dirs have to go or the upstream pruner keeps
                    # the wrong one. An --inference_only run trains nothing and writes
                    # no epoch dirs, so clearing them there destroys trained weights
                    # for no gain: it already cost the epoch-124 dump of the
                    # corpus-diversity vith16plus run on 2026-08-22.
                    if not args.inference_only:
                        for _old_ep in RUN_DIR.glob("epoch_*"):
                            if _old_ep.is_dir():
                                shutil.rmtree(_old_ep, ignore_errors=True)
                    # Same reason for the final dump: the child re-saves model.safetensors
                    # from the warm-started weights, silently replacing the last-epoch
                    # weights with the best-epoch ones. Park it under an explicit name
                    # first — a rename costs no disk and keeps both checkpoints.
                    _final_sft = RUN_DIR / "model.safetensors"
                    if args.inference_only and _final_sft.is_file():
                        _parked = RUN_DIR / "final_epoch_model.safetensors"
                        if not _parked.exists():
                            _final_sft.rename(_parked)
                            print(f"🔒 --inference_only: parked the last-epoch weights at "
                                  f"{_parked.name} so the eval pass cannot overwrite them.")
                    _warmstart_path = _ws_stable
                    print(f"🧹 {'Kept' if args.inference_only else 'Cleared'} weights-only "
                          f"epoch_*/ dirs; warm-start weights staged at {_ws_stable.name}.")
                except Exception as _e_ws_stage:
                    print(f"⚠️ warm-start staging failed ({_e_ws_stage}); loading "
                          "weights in place without clearing stale epoch dirs.")
            else:
                print("⏯  --resume requested but no checkpoint_epoch_* state dir and "
                      "no weights-only epoch_*/best_model dump found; starting fresh "
                      "(state checkpoints will be written from now on).")

    print(f"🚂 {'Eval' if args.inference_only else 'Training'} (bs={bs}, epochs={epochs})")
    env = os.environ.copy()
    # Select the DINOv3 backbone that matches the requested model. The upstream
    # create_mask2former_dinov3_model() takes no backbone arg, so each config
    # file's hard-coded default is the ONLY backbone unless DINOV3_M2F_MODEL is
    # set — which is exactly how the `large` config serves vith16plus/vit7b16
    # (and others) off one file. Without this, every `large`-routed variant
    # silently trained vitl16 regardless of its name. setdefault preserves a
    # user-provided override (e.g. a local backbone dir).
    _backbone = pick_dinov3_hf_model(args.model)
    env.setdefault("DINOV3_M2F_MODEL", _backbone)
    print(f"🧬 DINOv3 backbone -> {env['DINOV3_M2F_MODEL']}  (model={args.model})")
    # Hand the patched trainer its bench knobs: where to warm-start from (only set
    # when no resumable accelerate state existed) and how many weights-only epoch_*
    # dirs to keep mid-run (prevents the 152 GB blowup observed on the crashed run).
    if _warmstart_path is not None:
        env["BENCH_M2F_WARMSTART"] = str(_warmstart_path)
    else:
        env.pop("BENCH_M2F_WARMSTART", None)
    env["BENCH_M2F_KEEP_EPOCHS"] = str(max(0, int(args.keep_epochs)))
    # --patience was parsed and never forwarded, so upstream — which tracks the
    # best epoch but never breaks — always spent the full budget. An eval-only
    # pass has no loop to stop, so it stays off there.
    env["BENCH_M2F_PATIENCE"] = "0" if args.inference_only else str(max(0, int(args.patience)))
    # Where the child looks for the selected checkpoint to score in the final
    # test eval. Upstream writes best_model/ as it trains, so the directory need
    # not exist yet at launch.
    env["BENCH_M2F_BEST_DIR"] = str(RUN_DIR / "best_model")
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")
    env.setdefault("HF_HUB_VERBOSITY", "error")
    # Throttle tqdm. The upstream trainer + our eval loop pass `disable=` to tqdm
    # EXPLICITLY, which makes tqdm ignore TQDM_DISABLE — but `mininterval` is never
    # passed explicitly, so TQDM_MININTERVAL still applies. At the default 0.1s a
    # 379k-step bar over a multi-day run wrote ~0.7M redraws (a 50 MB train.log);
    # one refresh / 30 s keeps a progress heartbeat while shrinking that ~280x.
    env.setdefault("TQDM_MININTERVAL", "30")
    env["PYTHONPATH"] = f"{REPO_DIR}:{env.get('PYTHONPATH', '')}"
    with open(LOG_PATH, "wb") as logfh:
        proc = subprocess.Popen(cmd, cwd=str(REPO_DIR), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            sys.stdout.buffer.write(line); sys.stdout.flush(); logfh.write(line)
        rc = proc.wait()
    if rc != 0: print(f"⚠️ trainer rc={rc}; harvesting whatever's available")

    # Persist per-epoch history.csv from the epoch_*/epoch_*_metrics.json files
    # BEFORE prune_epoch_dirs removes them. This is what feeds the paper figures.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import bench_plots as _bp
        _rows = _bp._dinov3_m2f_epoch_metrics_to_history(RUN_DIR)
        if _rows:
            _gd = _bp.graphs_dir(RUN_DIR)
            _hf = _gd / "history.csv"
            if _hf.is_file():
                _hf.unlink()
            for _r in _rows:
                _bp.append_history(RUN_DIR, _r)
            print(f"📈 Wrote {len(_rows)} per-epoch history rows to {_hf}")
    except Exception as _e:
        print(f"⚠️ history.csv dump skipped: {_e}")

    # Disk-fix: prune per-epoch model dumps; best_model/ is preserved.
    prune_epoch_dirs(RUN_DIR, keep=args.keep_epochs)
    # On a clean finish the resumable state dirs are no longer needed; reclaim the
    # disk. On a non-zero rc (crash/interrupt) we KEEP the latest checkpoint_epoch_*
    # so the next --resume run can pick up training where it left off.
    if rc == 0:
        for _sd in list(RUN_DIR.glob("checkpoint_epoch_*")) + list(RUN_DIR.glob("step_*_state")):
            if _sd.is_dir():
                shutil.rmtree(_sd, ignore_errors=True)
        # The staged warm-start weights copy is only needed while the child loads
        # it at startup; reclaim it (≈ one full model shard) on a clean finish.
        _ws_stable_done = RUN_DIR / "_bench_warmstart.safetensors"
        if _ws_stable_done.is_file():
            try: _ws_stable_done.unlink()
            except Exception: pass

    metrics = harvest_metrics(RUN_DIR, EVAL_SPLIT)

    # ---- Derive box metrics from the segmentation predictions ----
    # M2F only computes segm mAP. We derive bbox mAP from the same predictions.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import bench_eval as _be
        # M2F trainer writes test predictions somewhere - find them.
        # RUN_DIR/predictions.json is the canonical final-pass dump over the
        # reported split; best_model/predictions.json is a best-epoch snapshot on
        # the per-epoch (val) split. Prefer the canonical one — the old order let
        # a stale val snapshot supply the derived box metrics.
        pred_json = None
        for cand in [RUN_DIR / "predictions.json",
                     RUN_DIR / "best_model" / "predictions.json"]:
            if cand.is_file():
                pred_json = cand
                break
        if not pred_json:
            ep_dirs = sorted(RUN_DIR.glob("epoch_*"),
                           key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1)
            for ep in reversed(ep_dirs):
                if (ep / "predictions.json").is_file():
                    pred_json = ep / "predictions.json"
                    break

        bbox_json = None  # may be set below in box-derivation block
        if pred_json and pred_json.is_file():
            gt_json = RUN_DIR / "_bench_gt_coco.json"
            # Use the same split we detected earlier
            _be.yolo_split_to_coco_gt(str(Path(args.data) / "data.yaml"), EVAL_SPLIT, str(gt_json),
                                     one_indexed_cats=True)  # M2F uses 1-indexed

            # Use _file_name to resolve image_id if possible (robust against sequential ID mismatch)
            try:
                with open(gt_json) as _gfh: _gt_d = json.load(_gfh)
                # Map both exact file_name and stem. Keep valid existing image_id
                # first; this is the most reliable path when the patched loader
                # propagated COCO IDs correctly.
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
                        if _iid is None:
                            if _warned < 10:
                                print(f"   ⚠️ Dropping prediction with unknown M2F file name: {_fn}")
                            _warned += 1
                            continue

                        _p2 = {k: v for k, v in _p.items() if k != "_file_name"}
                        _p2["image_id"] = int(_iid)
                        _resolved.append(_p2)
                    _resolved_path = RUN_DIR / "_bench_predictions_resolved.json"
                    with open(_resolved_path, "w") as _rfh: json.dump(_resolved, _rfh)
                    print(f"   🔧 Resolved {len(_resolved)}/{len(_preds)} predictions via _file_name/stem → image_id")
                    if _warned > 10:
                        print(f"   ⚠️ Suppressed {_warned - 10} additional M2F unknown-file warnings")
                    if _resolved:
                        pred_json = _resolved_path
                    else:
                        print(f"   ⚠️ M2F prediction file had rows but none matched GT image names: {pred_json}")
            except Exception as _e_res:
                print(f"   ⚠️ _file_name→image_id resolution failed: {_e_res}")

            # Fix image_id mismatch if needed
            fixed_pred = RUN_DIR / "_bench_predictions_fixed.json"
            n_fixed = _be.fix_yolo_pred_image_ids(str(pred_json), str(gt_json), str(fixed_pred))
            if n_fixed > 0:
                print(f"   🔧 Fixed {n_fixed} prediction image_ids to match GT")
                pred_json = fixed_pred
            # Use masks_to_box_preds to derive boxes from masks; also run a
            # segm COCOeval on the original predictions so seg_P/R/F1 and the
            # S/M/L + AR_100 buckets get populated (the HF Trainer harvest
            # only emits a single seg_mAP50_95/seg_R triplet at best).
            bbox_json = RUN_DIR / "_bench_bbox_from_masks.json"
            n = _be.masks_to_box_preds(str(pred_json), str(bbox_json))
            if n > 0:
                extra = _be.coco_eval_full(str(gt_json),
                                           str(bbox_json), str(pred_json))
                # Box columns come entirely from COCOeval (M2F has no native
                # box head). For seg, fill only the cells the trainer left
                # at zero so the published mAP from HF stays canonical.
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
            print(f"   ⚠️ No predictions.json found for M2F box derivation")
    except Exception as e:
        print(f"⚠️ Box metrics derivation failed: {e}")

    # ---- SAM2-prompted segmentation (sam2_* family) ----
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
        # Use the derived bbox-only predictions (masks_to_box_preds output), not
        # the raw predictions_json which may lack clean bbox entries.
        _sam2_bbox_src = Path(bbox_json) if bbox_json else Path(pred_json) if pred_json else None
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
            print("⚠️ --seg_via_sam2 set but --sam2_ckpt/--sam2_config missing or images dir / pred_json not resolvable; skipping.")

    # ---- v2: params from the saved HF Mask2Former snapshot --------------
    # Carti-97 writes a full HF dump under best_model/ (or epoch_*/). Sum the
    # tensor numels in the safetensors / pytorch_model.bin shard(s) — no need
    # to reinstantiate the model.
    try:
        # Also try to harvest fps_eval from the patched evaluator's JSON.
        fps_path = RUN_DIR / "_bench_fps.json"
        if fps_path.is_file():
            with open(fps_path) as fh:
                fps_data = json.load(fh)
                metrics["fps_eval"] = float(fps_data.get("fps_eval", 0.0))
                print(f"   ⏱  fps_eval = {metrics['fps_eval']:.2f}")

        n_params = 0
        snap = RUN_DIR / "best_model"
        if not snap.is_dir():
            # Fall back to most recent epoch_* dir
            ep_dirs = sorted(Path(RUN_DIR).glob("epoch_*"),
                             key=lambda p: int(p.name.rsplit("_", 1)[-1]) if p.name.rsplit("_", 1)[-1].isdigit() else -1)
            snap = ep_dirs[-1] if ep_dirs else None
        if snap and snap.is_dir():
            # safetensors path is fastest; otherwise torch.load the .bin
            try:
                from safetensors import safe_open as _so  # noqa
                for st in snap.glob("*.safetensors"):
                    with _so(str(st), framework="pt") as fh:
                        for k in fh.keys():
                            t = fh.get_tensor(k); n_params += int(t.numel())
            except Exception:
                pass
            if n_params == 0:
                import torch as _t
                for bin_ in snap.glob("pytorch_model*.bin"):
                    sd = _t.load(str(bin_), map_location="cpu", weights_only=False)
                    for v in sd.values():
                        if hasattr(v, "numel"): n_params += int(v.numel())
            if n_params > 0:
                metrics["params_M"] = n_params / 1e6
    except Exception as _e:
        print(f"⚠️ params probe failed: {_e}")
    nz = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if v)
    print(f"📊 {nz or 'no metrics captured'}")

    # Report the actual micro-batch and realised effective batch (micro * grad_accum)
    # so the bench CSV reflects the real per-model batch, not the orchestrator arg.
    metrics["batch_sz"] = int(bs)
    metrics["batch_effective"] = int(bs * GRAD_ACCUM)

    try:
        # RUN_DIR is already under ~/bench_any/, so no separate mirror is needed.
        with open(RUN_DIR / 'results.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(metrics.keys()))
            w.writeheader(); w.writerow(metrics)
        print(f"✅ Results: {RUN_DIR}")
    except Exception as e:
        print(f"⚠️ Could not save results: {e}")

    print("🎉 Complete!")

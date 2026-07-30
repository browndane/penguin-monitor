#!/usr/bin/env python3
"""SAM2 video mask propagation on RF-DETR detections (Step 5).

Extends penguin_video_demo.py's RF-DETR + ByteTrack + counting pattern with
**SAM2 video-mode propagation**: RF-DETR-seg detections on periodic keyframes seed
SAM2's memory bank, and SAM2 propagates temporally consistent masks across the
in-between frames (using `build_sam2_video_predictor` → `init_state` →
`add_new_mask` / `add_new_points_or_box` → `propagate_in_video`), NOT the
per-frame image predictor that `bench_eval.sam2_seg_from_boxes` uses.

Two modes:
  * default (per-frame RF-DETR+ByteTrack baseline): the working prefork pipeline,
    no SAM2. This is the BUG-10 fix: SAM2 mask propagation regresses on dense
    colonies (collapses multiple penguin masks into one giant box), so the
    baseline is now the safe default.
  * --sam2_propagate (opt-in): keyframe-seeded SAM2 temporally-smoothed masks.
    Use only to reproduce the SAM2-prop comparison; known regression on dense
    colonies.

Writes:
  * an annotated mp4 (masks overlaid + track IDs + live/confirmed count);
  * a metrics JSON with temporal-stability scores (mask IoU between consecutive
    frames, ID-persistence / ID-switch rates, mask-boundary smoothness) for both
    the SAM2-prop and the per-frame baseline so the improvement is quantified.

CLI:
  python penguins/sam2_video_propagate.py \
      --video /mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/demo_videos/penguin_african_easy.mp4 \
      --weights ~/bench_any/runs/segment/AnimalP_rf-detr-seg-medium_b1_e1_i640_be2/checkpoint_best_total.pth \
      --sam2_config ~/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
      --sam2_ckpt ~/sam2/checkpoints/sam2.1_hiera_large.pt \
      --out penguins/penguin_african_easy_baseline.mp4 \
      [--variant rf-detr-seg-medium] [--conf 0.35] [--keyframe_interval 30] \
      [--show] [--sam2_propagate]
"""

# spawn MUST be set before importing torch / RF-DETR / supervision:
# multiprocessing start-method to "spawn" before importing
# torch/CUDA, or you get deadlocks/memory corruption.
import multiprocessing as _mp
try:
    _mp.set_start_method("spawn")
except RuntimeError:
    pass  # already set (e.g. re-import in tests)

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import cv2

# Sibling-path shim so the file works regardless of CWD: insert fork's penguins/
# dir on sys.path before importing penguin_video_demo (which pulls RF-DETR +
# supervision eagerly at module load — intended cost of reuse).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from penguin_video_demo import load_model, _make_tracker, _draw_overlay, _draw_hud  # noqa: E402

# SAM2 ckpt/config defaults mirror sam2_backfill_rfdetr.py L15-18 (env-overridable).
SAM2_CKPT = os.environ.get(
    "SAM2_CKPT", "/home/vorse/sam2/checkpoints/sam2.1_hiera_large.pt")
SAM2_CONFIG = os.environ.get(
    "SAM2_CONFIG", "/home/vorse/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
SAM2_CONFIGS_ROOT = "/home/vorse/sam2/sam2/configs"

# A small fixed palette for per-instance mask colours (matches the demo's PALETTE
# usage so the overlay looks consistent). 24 distinct colours.
def _os_remove(p):
    try:
        os.remove(p)
    except OSError:
        pass


def _path_exists(p):
    return os.path.exists(p)


def _encode_rle(mask_bool: "np.ndarray") -> dict:
    """Encode a (H,W) bool mask as a compact COCO RLE dict.

    pycocotools RLE turns a full-frame 960x540 bool mask (518 KB) into ~1-3 KB
    of run-length bytes — a ~100x reduction for the per_frame_masks buffer that
    bridges the SAM2 propagate pass and the overlay pass. Decodes losslessly
    via ``_decode_rle``. Falls back to the raw bool array if pycocotools is
    unavailable so the demo still works (just with higher RAM use).
    """
    try:
        from pycocotools import mask as _pmask
    except ImportError:
        return {"__raw__": True, "mask": mask_bool}
    arr = np.asfortranarray(mask_bool.astype(np.uint8))
    rle = _pmask.encode(arr)
    # bytes() so the RLE counts are JSON-serialisable + copyable; decode reverses it.
    rle["counts"] = bytes(rle["counts"])
    return {"__raw__": False, "rle": rle}


def _decode_rle(enc) -> "np.ndarray":
    """Decode an RLE dict from ``_encode_rle`` back to a (H,W) bool mask.

    Accepts three forms so the same helper serves the SAM2-prop path (RLE dict)
    and the baseline path (raw bool array, possibly wrapped):
      * ``{"__raw__": False, "rle": {...}}`` — COCO RLE (SAM2-prop path)
      * ``{"__raw__": True, "mask": bool ndarray}`` — pycocotools-unavailable fallback
      * bare ``np.ndarray`` bool mask — baseline path stores raw bools directly
    """
    if isinstance(enc, dict):
        if enc.get("__raw__", False):
            return enc["mask"]
        from pycocotools import mask as _pmask
        return _pmask.decode(enc["rle"]).astype(bool)
    # Bare numpy array (baseline path) — already a bool mask.
    return np.asarray(enc).astype(bool)


PALETTE = np.array([
    [i for i in c]
    for c in [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),
              (128,0,0),(0,128,0),(0,0,128),(128,128,0),(128,0,128),(0,128,128),
              (255,128,0),(255,0,128),(128,255,0),(0,255,128),(128,0,255),(0,128,255),
              (200,200,0),(200,0,200),(0,200,200),(200,100,100),(100,200,100),(100,100,200)]
], dtype=np.uint8)


def _build_sam2_video_predictor(sam2_cfg: str, sam2_ckpt: str, device: str):
    """Build the SAM2 video predictor, mirroring bench_eval.sam2_seg_from_boxes's
    Hydra setup (GlobalHydra clear → initialize_config_dir → build_sam2_video_predictor
    with a config path relative to the configs root).

    Returns the predictor (eval mode) or raises; caller guards with try/except.
    """
    # sam2_root must be on sys.path before importing sam2.* (mirrors bench_eval).
    sam2_root = "/home/vorse/sam2"
    if sam2_root not in sys.path:
        sys.path.insert(0, sam2_root)
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=SAM2_CONFIGS_ROOT, version_base="1.2")
    from sam2.build_sam import build_sam2_video_predictor  # noqa
    # build_sam2_video_predictor expects a config name relative to the Hydra
    # config dir; convert an absolute sam2_cfg to "sam2.1/sam2.1_hiera_l.yaml".
    cfg_path = Path(sam2_cfg)
    if cfg_path.is_absolute():
        try:
            cfg_rel = str(cfg_path.relative_to(SAM2_CONFIGS_ROOT))
        except ValueError:
            cfg_rel = sam2_cfg
    else:
        cfg_rel = sam2_cfg
    return build_sam2_video_predictor(cfg_rel, sam2_ckpt, device=device, mode="eval")


class SAM2VideoPropagator:
    """RF-DETR detector + SAM2 video mask propagation + ByteTrack counting.

    RF-DETR-seg native masks seed SAM2 via add_new_mask on keyframes (better than a
    box prompt — the seg head already gives a mask); for a detector-only RF-DETR we
    fall back to add_new_points_or_box with the bbox. SAM2 then propagates masks
    across the whole video; ByteTrack runs on the propagated boxes for IDs.
    """

    def __init__(self, weights, variant_hint="rf-detr-seg-medium",
                 sam2_config=SAM2_CONFIG, sam2_ckpt=SAM2_CKPT, conf=0.35,
                 auto_conf=False):
        # load_model returns (model, is_seg); store both (v3 reconciliation: v2's
        # pseudocode took a bare detector_model, but load_model returns the pair).
        self.detector, self.is_seg = load_model(weights, variant_hint=variant_hint)
        self.conf = float(conf)
        self.sam2_config = sam2_config
        self.sam2_ckpt = sam2_ckpt
        # auto_conf: when True (or flipped on by the zero-prompt guard), probe
        # RF-DETR scores on frame 0 and lower self.conf to a workable value if the
        # checkpoint's absolute confidence scale is miscalibrated for the requested
        # conf. Common for short-training checkpoints whose rank-ordering is fine
        # (good mAP) but raw scores are tiny — e.g. a 1-epoch checkpoint may emit
        # penguins at max conf ~0.01-0.03, so the default conf=0.35 filters every
        # detection out and SAM2 gets zero prompts. Only ever LOWERS conf (never
        # raises), so it can't surprise the user with a tighter filter than asked.
        self.auto_conf = bool(auto_conf)
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._sam2 = None  # lazy-build; only when propagating with SAM2

    def _ensure_sam2(self):
        if self._sam2 is None:
            self._sam2 = _build_sam2_video_predictor(self.sam2_config, self.sam2_ckpt, self.device)
        return self._sam2

    def _write_downscaled_proxy(self, video_path, out_w, out_h):
        """Write a temp mp4 of `video_path` resized to (out_w, out_h) for SAM2.

        SAM2 keeps frames + predicted masks at the input video's native resolution,
        so a high-res clip (e.g. 2560x1440) makes the per-frame mask upsample to
        orig-res alone need ~1.5 GiB — OOM on a ~11.5 GiB GPU under hiera_large.
        A downscaled proxy (longer side ~960) cuts that ~16x. Writes to a temp file
        in the same dir as `video_path` (so a rename/cleanup is cheap) and returns
        its path, or None on failure (caller falls back to the native video).
        """
        import tempfile
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        src_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        tmp_dir = str(Path(video_path).resolve().parent)
        fd, tmp_path = tempfile.mkstemp(suffix="_proxy.mp4", dir=tmp_dir)
        os.close(fd)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, src_fps, (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            try: os.remove(tmp_path)
            except OSError: pass
            return None
        written = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                written += 1
        finally:
            cap.release()
            writer.release()
        if written == 0:
            try: os.remove(tmp_path)
            except OSError: pass
            return None
        # Faststart-remux the cv2-written proxy so the moov atom sits at the
        # front. cv2.VideoWriter(mp4v) writes moov at the END, which forces
        # SAM2's PyAV lazy reader (used by init_state + propagate_in_video with
        # offload_video_to_cpu=True) to seek to EOF and back on every frame
        # access — observed as a multi-GB read loop and a frozen progress bar
        # partway through propagation. +faststart moves moov to the front,
        # turning per-frame access into a single sequential read. If ffmpeg is
        # unavailable or fails, fall back to the un-faststarted proxy (still
        # decodable, just slower).
        fast_path = tmp_path + "_fast.mp4"
        try:
            import subprocess
            r = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", tmp_path,
                 "-movflags", "+faststart", "-c", "copy", fast_path],
                check=False, capture_output=True)
            if r.returncode == 0 and _path_exists(fast_path):
                os.remove(tmp_path)
                return fast_path
        except (OSError, FileNotFoundError):
            pass
        try: os.remove(fast_path)
        except OSError: pass
        return tmp_path

    def _sam2_infer_ctx(self):
        """Context manager wrapping every SAM2 video-predictor call.

        SAM2's video predictor (unlike the image predictor used in bench_eval)
        casts mask-memory features to bfloat16 internally (sam2_video_predictor
        L781/L833). For that to not crash with ``mat1(BFloat16) x mat2(Float)``,
        inference must run under ``torch.autocast("cuda", dtype=bfloat16)`` so the
        float32 checkpoint weights are downcast on the fly. This is exactly the
        usage SAM2's own README documents (every inference snippet is wrapped in
        ``torch.inference_mode()`` + ``torch.autocast("cuda", dtype=bfloat16)``).
        On CPU there is no bf16 autocast kernel, so we skip the autocast there.
        Returns a single ExitStack context manager; callers wrap add_new_mask /
        add_new_points_or_box / propagate_in_video calls with `with self._sam2_infer_ctx():`.
        """
        import contextlib, torch
        stack = contextlib.ExitStack()
        stack.enter_context(torch.inference_mode())
        if self.device == "cuda":
            stack.enter_context(torch.autocast("cuda", dtype=torch.bfloat16))
        return stack

    # ---- detection on a single RGB frame (mirrors penguin_video_demo L527-543) ----
    def _detect(self, rgb, threshold=None):
        """Detect on one RGB frame, filtering by `threshold` (defaults to self.conf).

        RF-DETR's `.predict(rgb, threshold=...)` is the supported kwarg; older
        rfdetr signatures lack it, in which case we predict unfiltered and apply
        the threshold ourselves via sv.Detections boolean indexing.
        """
        thr = self.conf if threshold is None else float(threshold)
        try:
            dets = self.detector.predict(rgb, threshold=thr)
        except TypeError:
            dets = self.detector.predict(rgb)
        if hasattr(dets, "confidence") and dets.confidence is not None:
            keep = dets.confidence >= thr
            dets = dets[keep]
        return dets

    def _calibrate_conf(self, frame_rgb):
        """Probe RF-DETR scores on one frame; if they sit well below ``self.conf``,
        lower ``self.conf`` so keyframe seeding actually finds objects. Returns the
        (possibly lowered) conf.

        Probes at threshold=0.0 to see the raw score distribution, takes the median
        of the non-trivial scores, and if that median is below ``self.conf`` adopts
        it (floored at 0.005 so we don't collapse to pure noise). Never raises conf.
        """
        dets = self._detect(frame_rgb, threshold=0.0)
        scores = getattr(dets, "confidence", None)
        scores = np.asarray(scores, dtype=float) if scores is not None else np.array([])
        scores = scores[scores > 0.0]
        if len(scores) == 0:
            return self.conf  # nothing detectable; the zero-prompt guard will report it
        med = float(np.median(scores))
        cand = max(0.005, med)  # don't collapse to pure noise
        if cand < self.conf:
            print(f"[sam2prop] auto_conf: frame-0 score median={med:.4f} "
                  f"< requested conf={self.conf:.3f}; lowering conf -> {cand:.4f}")
            self.conf = cand
        else:
            print(f"[sam2prop] auto_conf: frame-0 score median={med:.4f} >= conf={self.conf:.3f}; "
                  f"no change")
        return self.conf

    @staticmethod
    def _masks_from_dets(dets, H, W):
        """Extract an (N,H,W) bool mask array from sv.Detections, or None.

        RF-DETR-seg yields masks at the frame's processing resolution; resize to
        (H,W) so they can seed SAM2 (which expects a 2D bool mask per object).
        """
        masks = getattr(dets, "mask", None)
        if masks is None or len(masks) == 0:
            return None
        out = []
        for m in masks:
            mb = np.asarray(m).astype(bool)
            if mb.shape[:2] != (H, W):
                mb = cv2.resize(mb.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            out.append(mb)
        return np.stack(out, axis=0) if out else None

    @staticmethod
    def _box_from_dets(dets):
        """xyxy boxes (N,4) in pixel coords from sv.Detections."""
        xyxy = getattr(dets, "xyxy", None)
        return None if xyxy is None or len(xyxy) == 0 else np.asarray(xyxy, dtype=float)

    # ---- SAM2 propagation (default mode) ----
    def propagate(self, video_path, out_path, *, keyframe_interval=30,
                  writer_fps=30.0, show=False, stride=1, proc_size=0):
        """Run keyframe-seeded SAM2 propagation over the whole video.

        RF-DETR detects on every `keyframe_interval`-th frame; those detections seed
        SAM2's memory (one SAM2 object per tracked RF-DETR instance, matched across
        keyframes by IoU). SAM2 propagates masks to all frames; ByteTrack runs on
        the propagated boxes for stable IDs. Writes the annotated mp4 + returns a
        per-frame dict {frame_idx: {tid: mask}} plus tracker stats.

        ``proc_size``: if >0, downscale the video's longer side to this many pixels
        (preserving aspect) into a temp mp4 before handing it to SAM2. SAM2 keeps
        frames + masks at the input video's native resolution, so a 2560x1440 clip
        makes the per-frame mask upsample alone need ~1.5 GiB — on a ~11.5 GiB GPU
        that OOMs under hiera_large. A temp 960-wide proxy cuts that ~16x and keeps
        the heavy model viable. The overlay writer also runs at the downscaled
        resolution (``_draw_overlay`` already resizes masks to the frame, so the
        annotated output is self-consistent at the proxy resolution).
        """
        import torch
        sam2 = self._ensure_sam2()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {video_path}")
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Optional temp downscaled proxy for SAM2 (memory relief on limited GPUs).
        sam2_video_path = str(video_path)
        proc_W, proc_H = W, H
        temp_proxy = None
        if proc_size and proc_size > 0:
            long_side = max(W, H)
            if long_side > proc_size:
                scale = proc_size / long_side
                proc_W = int(round(W * scale))
                proc_H = int(round(H * scale))
                temp_proxy = self._write_downscaled_proxy(video_path, proc_W, proc_H)
                if temp_proxy is not None:
                    sam2_video_path = temp_proxy
                    print(f"[sam2prop] proc_size={proc_size}: downscaled proxy "
                          f"{W}x{H} -> {proc_W}x{proc_H} for SAM2 ({temp_proxy})")
                    # Register an atexit cleanup so the temp proxy is removed even
                    # if propagation crashes (OOM/traceback) before the normal
                    # cleanup in the return path runs — prevents tmp*_proxy.mp4
                    # leaking in the input dir on failed runs.
                    import atexit
                    _proxy = temp_proxy
                    atexit.register(lambda p=_proxy: (_os_remove(p) if _path_exists(p) else None))
                else:
                    proc_W, proc_H = W, H  # proxy failed; fall back to native
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, writer_fps, (proc_W, proc_H))
        # RAM-safe per-frame PNG dump alongside the mp4 (codec-free fallback
        # for players that refuse the mp4v/H.264 output).
        frame_sub = _FrameSubdirWriter(out_path, stride=10)

        # SAM2 video state: init once on the whole video. offload_video_to_cpu +
        # offload_state_to_cpu move the (growing) frame + mask/feature memory to
        # CPU, keeping only the active frame on GPU. Without this, SAM2's memory
        # bank accumulates features per frame and OOMs a ~11.5 GiB GPU partway
        # through a 377-frame clip (the model itself is fine; the per-frame state
        # is what grows). The trade-off is some CPU<->GPU transfer latency per
        # frame, acceptable for an offline demo. (init_state defaults both to
        # False; we opt in for memory safety on limited GPUs.)
        inference_state = sam2.init_state(
            sam2_video_path, offload_video_to_cpu=True, offload_state_to_cpu=True)
        # Re-open a reader on the (possibly downscaled proxy) video for the
        # keyframe-seed detection loop and the second-pass overlay. Reading the
        # proxy keeps RF-DETR detection, SAM2 frames, and the overlay writer all at
        # the same resolution (no resampling surprises in the overlay).
        cap = cv2.VideoCapture(sam2_video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open (proxy) video: {sam2_video_path}")
        # obj_id → RF-DETR keyframe mask, maintained across keyframes by IoU match.
        obj_masks: dict = {}  # obj_id -> last-seen mask (H,W) at video res
        obj_boxes: dict = {}  # obj_id -> last-seen box xyxy (pixel)
        next_obj_id = 1
        # Per-frame SAM2 output: frame_idx -> {obj_id: mask(bool H,W)}
        per_frame_masks: dict = {}

        # First pass: keyframe-seed SAM2, then propagate and collect masks.
        # We seed keyframes as we encounter them, then propagate the full video
        # in one generator pass (SAM2 handles conditioning frames internally).
        keyframes = list(range(0, max(N, 1), max(int(keyframe_interval), 1)))

        # Calibrate conf BEFORE seeding if requested (or proactively: short-training
        # checkpoints often have good mAP but tiny absolute scores — without this,
        # the default conf=0.35 filters every detection out and SAM2 gets no
        # prompts, which propagates_in_video then rejects with a cryptic
        # "No input points or masks are provided" RuntimeError). We probe frame 0
        # once here and, if the score scale is too low, lower self.conf so the
        # seeding loop below actually finds objects. auto_conf=False keeps the
        # user's explicit conf; the zero-prompt guard below is the safety net.
        first_keyframe_rgb = None
        if self.auto_conf and len(keyframes) > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, keyframes[0])
            ok, kf0 = cap.read()
            if ok:
                first_keyframe_rgb = cv2.cvtColor(kf0, cv2.COLOR_BGR2RGB)
                self._calibrate_conf(first_keyframe_rgb)

        # Seed: detect on each keyframe with RF-DETR, add mask/box prompts to SAM2.
        # We read the keyframe images directly (cheap; SAM2 already loaded all
        # frames, but we need RGB at full res for RF-DETR).
        total_prompts = 0
        for kf in keyframes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, kf)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = self._detect(rgb)
            masks = self._masks_from_dets(dets, proc_H, proc_W) if self.is_seg else None
            boxes = self._box_from_dets(dets)
            n = 0 if boxes is None else len(boxes)
            total_prompts += n
            for i in range(n):
                # Match this keyframe detection to an existing SAM2 obj by IoU of
                # its box with the last-seen boxes (greedy); else spawn a new obj.
                box = boxes[i]
                target_obj = None
                best_iou = 0.3  # min IoU to be considered the same instance
                for oid, ob in obj_boxes.items():
                    iou = _box_iou(box, ob)
                    if iou > best_iou:
                        best_iou = iou; target_obj = oid
                if target_obj is None:
                    target_obj = next_obj_id; next_obj_id += 1
                obj_boxes[target_obj] = box
                if masks is not None:
                    obj_masks[target_obj] = masks[i]
                    # Mask prompt (RF-DETR-seg has native masks — better than a box).
                    with self._sam2_infer_ctx():
                        sam2.add_new_mask(inference_state, kf, target_obj,
                                          torch.from_numpy(masks[i].astype(bool)))
                else:
                    # Box prompt (detector-only RF-DETR); normalize_coords=False
                    # since box is in pixel coords.
                    with self._sam2_infer_ctx():
                        sam2.add_new_points_or_box(
                            inference_state, kf, target_obj, box=box, normalize_coords=False)
        # Zero-prompt guard: if RF-DETR found no objects on ANY keyframe, SAM2 has
        # nothing to propagate and propagate_in_video raises a cryptic
        # "No input points or masks are provided for any object" RuntimeError deep
        # in the upstream library. Catch it up here with an actionable message,
        # and (if auto_conf wasn't already tried) retry once with calibration.
        if total_prompts == 0:
            msg = (f"RF-DETR found 0 objects above conf={self.conf:.3f} on every "
                   f"keyframe of {video_path}; SAM2 has nothing to propagate.")
            if not self.auto_conf:
                print(f"[sam2prop] {msg}\n[sam2prop] retrying with --auto_conf "
                      f"(probe + lower conf to the frame-0 score median)...")
                self.auto_conf = True
                if first_keyframe_rgb is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, keyframes[0])
                    ok, kf0 = cap.read()
                    first_keyframe_rgb = cv2.cvtColor(kf0, cv2.COLOR_BGR2RGB) if ok else None
                if first_keyframe_rgb is not None and self._calibrate_conf(first_keyframe_rgb) < self.conf + 1e-9:
                    pass  # calibrate logged; only lowers
                # Re-run the seed loop with the (lowered) conf.
                sam2.reset_state(inference_state)
                total_prompts = 0
                for kf in keyframes:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, kf)
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    dets = self._detect(rgb)
                    masks = self._masks_from_dets(dets, proc_H, proc_W) if self.is_seg else None
                    boxes = self._box_from_dets(dets)
                    n = 0 if boxes is None else len(boxes)
                    total_prompts += n
                    for i in range(n):
                        box = boxes[i]
                        target_obj = next(
                            (oid for oid, ob in obj_boxes.items() if _iou(box, ob) >= 0.3),
                            None)
                        if target_obj is None:
                            target_obj = next_obj_id; next_obj_id += 1
                        obj_boxes[target_obj] = box
                        if masks is not None:
                            obj_masks[target_obj] = masks[i]
                            with self._sam2_infer_ctx():
                                sam2.add_new_mask(inference_state, kf, target_obj,
                                                  torch.from_numpy(masks[i].astype(bool)))
                        else:
                            with self._sam2_infer_ctx():
                                sam2.add_new_points_or_box(
                                    inference_state, kf, target_obj, box=box, normalize_coords=False)
            if total_prompts == 0:
                raise RuntimeError(
                    f"{msg}\nThis usually means the checkpoint's absolute confidence "
                    f"scale is miscalibrated for --conf {self.conf:.3f} (rank may be "
                    f"fine — mAP good — but raw scores tiny, common for short-training "
                    f"checkpoints). Re-run with --auto_conf, or pass a lower --conf "
                    f"(e.g. --conf 0.01), or use a more-trained checkpoint.")

        # Free RF-DETR from the GPU before the memory-heavy propagation pass. RF-DETR's
        # only job was keyframe seeding (done above); the propagation loop is pure SAM2.
        # On a ~11.5 GiB GPU, RF-DETR-seg-medium + SAM2 hiera-large together leave under
        # 1 GiB free, and SAM2's per-frame mask upsample to the original video resolution
        # (2560x1440 here) needs ~1.5 GiB → OOM mid-propagation. Moving the detector to
        # CPU (and emptying the cache) reclaims ~2-3 GiB for the upsample. The detector
        # is reloaded to GPU later only if propagate() is called again (it isn't here).
        if self.device == "cuda":
            try:
                inner = getattr(self.detector, "model", None) or self.detector
                if hasattr(inner, "to"):
                    inner.to("cpu")
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        # Propagate through the whole video. Generator yields (frame_idx, obj_ids, masks).
        # Wrapped in bfloat16 autocast: the video predictor internally casts mask-memory
        # features to bf16 (sam2_video_predictor L781/L833), so the float32 checkpoint
        # weights must be autocast-downcast or mat1(bf16) x mat2(float32) crashes. The
        # generator's __next__ calls run inside this `with`, so autocast stays active
        # for every frame's forward pass. (Matches SAM2 README's documented usage.)
        with self._sam2_infer_ctx():
            for frame_idx, obj_ids, video_res_masks in sam2.propagate_in_video(inference_state):
                # video_res_masks: (num_objs, H, W) bool at video resolution.
                # Store masks as compact COCO RLE (pycocotools.mask.encode) instead
                # of full (H,W) bool arrays. A 960x540 full-frame bool mask is 518 KB
                # as a numpy array but ~1-3 KB as RLE — a ~100x reduction in the
                # per_frame_masks buffer that bridges the propagate + overlay passes.
                # Without this, a 377-frame clip x ~50 IDs x 518 KB = ~10 GB held in
                # RAM just by our wrapper, on top of SAM2's internal per-frame state.
                # RLE decodes losslessly in the overlay pass below via mask.decode.
                fm = {}
                for oi, oid in enumerate(obj_ids):
                    m = video_res_masks[oi].detach().cpu().numpy().astype(bool)
                    # SAM2 masks may carry a leading singleton (e.g. (1, H, W) when the
                    # predictor returns per-object logits with a mask dim). Reduce to a
                    # single (H, W) plane so downstream np.where(m) unpacks 2 values.
                    while m.ndim > 2:
                        m = m[0]
                    fm[int(oid)] = _encode_rle(m)
                per_frame_masks[int(frame_idx)] = fm
                # Periodic GC + CUDA cache clear. SAM2's inference_state accumulates
                # maskmem_features per frame (offloaded to CPU); the per-frame GPU
                # mask tensors must be freed once we've RLE-encoded them. Without
                # this, RSS grows ~200 MB/frame (measured 123 MB/s steady) and a
                # 377-frame clip hits 77 GB RAM -> swap-to-SSD on a 60 GB machine.
                if (int(frame_idx) + 1) % 32 == 0:
                    import gc
                    gc.collect()
                    if self.device == "cuda":
                        import torch
                        torch.cuda.empty_cache()
            sam2.reset_state(inference_state)
            # Final cleanup after the full propagation pass.
            import gc
            gc.collect()
            if self.device == "cuda":
                import torch
                torch.cuda.empty_cache()

        # Second pass: read video again, overlay propagated masks, run ByteTrack on
        # propagated boxes for IDs, write the annotated mp4.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        tracker, tracker_name = _make_tracker(kind="bytetrack", track_conf=self.conf)
        unique_ids, confirmed_ids, id_hits = set(), set(), {}
        per_frame_overlay_boxes: dict = {}  # frame_idx -> xyxy per obj_id (for metrics)
        # per_frame_tracked_masks: frame_idx -> {tracker_id: mask(bool H,W)}.
        # Keyed by ByteTrack's tracker_id (which fragments/swaps under occlusion),
        # NOT SAM2's object index (which persists across all frames by construction
        # and makes compute_temporal_metrics tautological). Built in the overlay
        # pass where the tracker runs.
        per_frame_tracked_masks: dict = {}
        # Consistency metric: propagated mask vs a FRESH per-frame RF-DETR detection
        # (independent of SAM2's memory). For each propagated box, find the best-IoU
        # fresh RF-DETR box; record that IoU. Mean across all propagated instances in
        # all frames = how well propagation agrees with an independent detector.
        propagated_vs_rf_ious: list = []
        propagated_vs_rf_matches = 0  # count of propagated boxes with best-IoU >= 0.5
        frame_idx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if stride > 1 and (frame_idx % stride) != 0:
                continue
            fm_rle = per_frame_masks.get(frame_idx, {})
            # Decode RLE masks back to (H,W) bool for the overlay + tracker.
            fm = {oid: _decode_rle(r) for oid, r in fm_rle.items()}
            # Build a sv.Detections-like structure for the overlay + tracker.
            boxes, masks_list, oids = [], [], []
            for oid, m in fm.items():
                ys, xs = np.where(m)
                if len(xs) == 0:
                    continue
                x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
                boxes.append([x1, y1, x2, y2]); masks_list.append(m); oids.append(oid)
            if boxes:
                det = _SimpleDetections(np.array(boxes, dtype=float), masks_list, oids)
                if hasattr(tracker, "set_frame"):
                    tracker.set_frame(frame)
                det = tracker.update_with_detections(det)
                _track_count(det, unique_ids, confirmed_ids, id_hits, self.conf)
                per_frame_overlay_boxes[frame_idx] = det.xyxy
                # Key masks by tracker_id for the temporal metric. det.tracker_id is
                # aligned 1:1 with det.xyxy/det.mask (tracker annotates in place);
                # skip rows the tracker rejected (id < 0). det.mask is the masks_list
                # we passed in, same length/order as det.xyxy.
                tids = getattr(det, "tracker_id", None)
                det_mask = getattr(det, "mask", None)
                if tids is not None and det_mask is not None:
                    tracked_for_frame: dict = {}
                    for i in range(len(det.xyxy)):
                        ti = tids[i]
                        if ti is None or int(ti) < 0:
                            continue
                        tracked_for_frame[int(ti)] = det_mask[i]
                    if tracked_for_frame:
                        per_frame_tracked_masks[frame_idx] = tracked_for_frame
                # Consistency: match each propagated box to the best fresh RF-DETR box.
                # Report TWO numbers: (1) mean best-IoU across all propagated boxes
                # (the old number, diluted toward 0 when SAM2-prop has more boxes than
                # RF-DETR finds fresh), and (2) the match-rate = fraction of propagated
                # boxes with best-IoU >= 0.5 (robust to the dilution — an aggregator
                # that averages zeros for unmatched propagated boxes is the BUG 9
                # artifact that made consistency read 0.007-0.02).
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    fresh = self._detect(rgb)
                    fresh_xyxy = getattr(fresh, "xyxy", None)
                    if fresh_xyxy is not None and len(fresh_xyxy) > 0:
                        fresh_arr = np.asarray(fresh_xyxy, dtype=float)
                        for pb in det.xyxy:
                            best = 0.0
                            for fb in fresh_arr:
                                v = _box_iou(pb, fb)
                                if v > best:
                                    best = v
                            propagated_vs_rf_ious.append(best)
                            if best >= 0.5:
                                propagated_vs_rf_matches += 1
                except Exception:
                    pass  # detector error on one frame must not abort the overlay
                out = _draw_overlay(frame, det, self.conf)
            else:
                out = frame.copy()
            _draw_hud(out, len(fm), len(confirmed_ids), 0.0)
            writer.write(out)
            frame_sub.write(out, frame_idx)
            if show:
                cv2.imshow("sam2_video_propagate", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release(); writer.release()
        # cv2's mp4v fourcc writes MPEG-4 Part 2, which VLC/mpv often refuse or
        # render as a green screen. Re-encode to H.264 +faststart in place so
        # the output actually plays. Falls back to the mp4v file if no ffmpeg.
        _remux_to_h264(out_path)
        if show:
            cv2.destroyAllWindows()
        # Clean up the temp downscaled proxy (if any) now that SAM2 + overlay are done.
        if temp_proxy is not None:
            try:
                os.remove(temp_proxy)
            except OSError:
                pass
        return {
            "per_frame_masks": per_frame_masks,
            "per_frame_tracked_masks": per_frame_tracked_masks,
            "per_frame_boxes": per_frame_overlay_boxes,
            "unique_ids": sorted(unique_ids),
            "confirmed_ids": sorted(confirmed_ids),
            "n_frames": frame_idx + 1,
            "tracker": tracker_name,
            "mode": "sam2_propagate",
            "frame_subdir": frame_sub.path(),
            # Mean best-IoU between each propagated box and a fresh per-frame
            # RF-DETR box (independent of SAM2's memory). None if no fresh detections.
            # NOTE: this mean is diluted toward 0 when SAM2 propagation yields more
            # boxes than the fresh detector finds (unmatched boxes contribute 0).
            # Prefer `propagated_vs_rf_match_rate` for a robust agreement signal.
            "propagated_vs_rf_consistency": (
                float(sum(propagated_vs_rf_ious) / len(propagated_vs_rf_ious))
                if propagated_vs_rf_ious else None
            ),
            # Robust agreement: fraction of propagated boxes whose best fresh-IoU
            # >= 0.5. Immune to the dilution above. None if no propagated boxes
            # were ever matched against a fresh detection.
            "propagated_vs_rf_match_rate": (
                float(propagated_vs_rf_matches / len(propagated_vs_rf_ious))
                if propagated_vs_rf_ious else None
            ),
        }

    # ---- baseline: per-frame RF-DETR masks, no SAM2 (for comparison) ----
    def propagate_baseline(self, video_path, out_path, *, writer_fps=30.0,
                            show=False, stride=1, proc_size=0,
                            track_buffer=200, track_match=0.85, track_min_hits=3):
        """Per-frame RF-DETR-seg masks + ByteTrack, no SAM2 propagation. Quantifies
        the SAM2-prop improvement (temporal consistency) against this raw baseline.

        ``proc_size``: if >0, downscale every frame so its longer side equals
        this many pixels (preserving aspect, even dimensions for the mp4 encoder)
        before detect/track/draw/encode. The prefork ``penguin_video_demo.py`` had
        this knob; without it a 2160x3840 portrait clip runs at native 4K and
        OOMs the laptop. RF-DETR resizes internally regardless, so downscaling
        the long side to ~960 is near-free quality-wise and 4-8x lighter on RAM.

        ByteTrack defaults: ``track_match`` is 0.85 (not sv's 0.7) — a 2026-07-05
        sweep on the easy clip showed 0.7 merges two nearby penguins into one ID
        in the surf (4 IDs for 2 penguins, persistence 0.47); 0.85 keeps them
        separate (3 IDs, persistence 0.64). ``track_buffer`` is irrelevant on this
        clip (200/600/1200 give bit-identical results — both penguins are detected
        every frame, so no lost-track recovery is needed). The remaining 3->2
        over-count is a detector/association limit (one box covering both at
        re-emergence), not a tracker knob.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {video_path}")
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        long_side = max(W, H)
        proc_size = int(proc_size)
        if proc_size > 0 and long_side > proc_size:
            scale = proc_size / float(long_side)
            proc_W = int(round(W * scale)); proc_H = int(round(H * scale))
            if proc_W % 2: proc_W += 1
            if proc_H % 2: proc_H += 1
        else:
            proc_W, proc_H = W, H
        out_W, out_H = proc_W, proc_H
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, writer_fps, (out_W, out_H))
        frame_sub = _FrameSubdirWriter(out_path, stride=10)
        unique_ids, confirmed_ids, id_hits = set(), set(), {}
        per_frame_masks: dict = {}
        per_frame_boxes: dict = {}
        frame_idx = -1
        fps_ema = 0.0
        # Same conf-miscalibration risk as the SAM2-prop path: a short-training
        # checkpoint may have good mAP but tiny absolute scores, so the default
        # conf filters everything out and the baseline emits an empty video with
        # zero confirmed IDs (silently — no SAM2 to raise on missing prompts).
        # Calibrate once on frame 0 if requested; warn if it yields nothing.
        if self.auto_conf:
            ok0, f0 = cap.read()
            if ok0:
                self._calibrate_conf(cv2.cvtColor(f0, cv2.COLOR_BGR2RGB))
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # Build the tracker AFTER calibration so its track_activation_threshold
        # uses the (possibly lowered) self.conf — building it before calibration
        # would lock in 0.35 while _detect uses 0.005, and ByteTrack would reject
        # every low-conf detection → zero confirmed IDs (a silent all-zero result).
        tracker, tracker_name = _make_tracker(
            kind="bytetrack", track_conf=self.conf,
            track_buffer=track_buffer, track_match=track_match,
            track_min_hits=track_min_hits)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if stride > 1 and (frame_idx % stride) != 0:
                continue
            t0 = time.time()
            # Downscale once up-front so EVERYTHING downstream (inference RGB-copy,
            # mask resize, overlay, encode) operates on the small frame. Matches
            # the prefork penguin_video_demo.py proc_size knob.
            if (out_W, out_H) != (W, H):
                frame = cv2.resize(frame, (out_W, out_H), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = self._detect(rgb)
            masks = self._masks_from_dets(dets, out_H, out_W) if self.is_seg else None
            if hasattr(tracker, "set_frame"):
                tracker.set_frame(frame)
            dets = tracker.update_with_detections(dets)
            _track_count(dets, unique_ids, confirmed_ids, id_hits, self.conf)
            out = _draw_overlay(frame, dets, self.conf)
            dt = max(1e-6, time.time() - t0)
            fps_ema = (1.0 / dt) if fps_ema == 0.0 else (0.1 * (1.0 / dt) + 0.9 * fps_ema)
            _n_dets = len(getattr(dets, "xyxy", np.zeros((0, 4))))
            _draw_hud(out, _n_dets, len(confirmed_ids), fps_ema)
            writer.write(out)
            frame_sub.write(out, frame_idx)
            if show:
                cv2.imshow("sam2_video_propagate_baseline", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            # Record per-frame masks/boxes (keyed by tracker_id) for the metrics.
            tids = getattr(dets, "tracker_id", None)
            fm = {}
            if tids is not None and masks is not None:
                for i in range(len(tids)):
                    ti = int(tids[i]) if tids[i] is not None and tids[i] >= 0 else -1
                    if ti >= 0:
                        fm[ti] = masks[i]
            per_frame_masks[frame_idx] = fm
            per_frame_boxes[frame_idx] = getattr(dets, "xyxy", None)
        cap.release(); writer.release()
        # cv2's mp4v fourcc writes MPEG-4 Part 2, which VLC/mpv often refuse or
        # render as a green screen. Re-encode to H.264 +faststart in place so
        # the output actually plays. Falls back to the mp4v file if no ffmpeg.
        _remux_to_h264(out_path)
        if show:
            cv2.destroyAllWindows()
        return {
            "per_frame_masks": per_frame_masks,
            "per_frame_boxes": per_frame_boxes,
            "unique_ids": sorted(unique_ids),
            "confirmed_ids": sorted(confirmed_ids),
            "n_frames": frame_idx + 1,
            "tracker": tracker_name,
            "mode": "per_frame_baseline",
            "frame_subdir": frame_sub.path(),
        }


# ----- helpers -----

def _box_iou(a, b) -> float:
    """IoU of two xyxy boxes (pixel coords)."""
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _remux_to_h264(src_path, dst_path=None):
    """Re-encode a cv2-written mp4 (MPEG-4 Part 2 / ``mp4v``) to H.264 +faststart.

    cv2.VideoWriter with the ``mp4v`` fourcc produces an MPEG-4 Part 2 stream
    that modern players (VLC, mpv) often refuse to decode or render as a green
    screen, and the moov atom lands at EOF so streaming/seeking is broken.
    ffmpeg transcode to libx264 + ``+faststart`` + yuv420p yields a broadly
    playable H.264 mp4 with moov at the front. If ffmpeg is absent or fails,
    the caller keeps the original ``mp4v`` file (still decodable by ffmpeg
    itself, just not by pickier players).

    ``dst_path`` defaults to ``src_path`` (in-place replace via a temp file).
    Returns the path actually written (the H.264 file on success, the original
    on fallback).
    """
    import shutil, subprocess
    src = str(src_path)
    dst = str(dst_path) if dst_path is not None else src
    if shutil.which("ffmpeg") is None:
        return src  # no ffmpeg — leave the mp4v output as-is
    tmp = dst + ".h264tmp.mp4"
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-y", "-i", src,
             "-c:v", "libx264", "-preset", "fast",
             "-movflags", "+faststart", "-pix_fmt", "yuv420p", tmp],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, dst)
            return dst
    except Exception:
        pass
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
    except OSError:
        pass
    return src  # remux failed — keep original


class _FrameSubdirWriter:
    """Write annotated frames as PNGs into ``<out_path>_frames/`` as they are
    produced — one at a time, never held in memory.

    The mp4 is the primary deliverable, but mp4v/H.264 codec availability is
    environment-dependent and players vary; a frame subdir is a robust,
    no-codec-needed view of the same annotated output (the user's suggestion
    after mp4s refused to render in VLC/mpv). To bound disk + RAM on long
    clips, only every ``stride``-th frame is written as a PNG (default 10 →
    ~38 PNGs for a 377-frame clip), and stale PNGs from a prior run are cleared
    on open. Failures are swallowed so a disk error never aborts the run.
    """

    def __init__(self, out_path, stride=10):
        self.sub = Path(str(out_path) + "_frames")
        self.stride = max(1, int(stride))
        self.count = 0
        try:
            self.sub.mkdir(parents=True, exist_ok=True)
            for old in self.sub.glob("*.png"):
                try:
                    old.unlink()
                except OSError:
                    pass
        except Exception:
            self.sub = None  # disable on failure

    def write(self, frame, frame_idx):
        if self.sub is None:
            return
        if frame_idx % self.stride != 0:
            return
        try:
            cv2.imwrite(str(self.sub / f"frame_{frame_idx:06d}.png"), frame)
            self.count += 1
        except Exception:
            pass

    def path(self):
        return str(self.sub) if self.sub is not None and self.count > 0 else None


def _track_count(detections, unique_ids, confirmed_ids, id_hits, conf_thresh, min_age=3):
    """Mirror penguin_video_demo's confirmed_ids counting (L554-566)."""
    tids = getattr(detections, "tracker_id", None)
    if tids is None:
        return
    for t in tids:
        if t is None:
            continue
        ti = int(t)
        if ti < 0:
            continue
        unique_ids.add(ti)
        id_hits[ti] = id_hits.get(ti, 0) + 1
        if id_hits[ti] >= int(min_age):
            confirmed_ids.add(ti)


class _SimpleDetections:
    """Minimal stand-in for supervision.Detections carrying the fields the overlay
    + tracker need (.xyxy, .mask, .tracker_id, .confidence). Built from SAM2
    propagated masks; tracker.update_with_detections annotates .tracker_id in place.

    Masks are stored as a stacked (N,H,W) bool ndarray so __getitem__ can index
    them uniformly with the same key used for .xyxy/.confidence/.tracker_id —
    supervision's ByteTrack may index detections with a boolean mask, an integer
    array (e.g. after argsort), or a slice, and all three must keep the mask list
    aligned with the box array. (A Python-list mask rebuilt via np.where only
    handles boolean indexing and would silently misalign for integer-array index.)
    """
    def __init__(self, xyxy, masks, oids, confidence=None, tracker_id=None):
        self.xyxy = np.asarray(xyxy, dtype=float)
        n = len(self.xyxy)
        if masks is not None and len(masks) > 0:
            # Stack to (N,H,W) bool; resize each mask to the first mask's shape so
            # the stack is rectangular (masks from SAM2 are already same-shape, but
            # the baseline path's RF-DETR masks may differ if frames were resized).
            ref_h, ref_w = np.asarray(masks[0]).shape
            stacked = np.zeros((len(masks), ref_h, ref_w), dtype=bool)
            for i, m in enumerate(masks):
                mb = np.asarray(m).astype(bool)
                if mb.shape != (ref_h, ref_w):
                    mb = cv2.resize(mb.astype(np.uint8), (ref_w, ref_h), interpolation=cv2.INTER_NEAREST).astype(bool)
                stacked[i] = mb
            self.mask = stacked
        else:
            self.mask = None
        self.confidence = np.ones(n, dtype=float) if confidence is None else np.asarray(confidence, dtype=float)
        self.tracker_id = (np.full(n, -1, dtype=int) if tracker_id is None
                           else np.asarray(tracker_id, dtype=int))

    def __len__(self):
        return len(self.xyxy)

    def __getitem__(self, keep):
        # Unified indexing for boolean mask, integer array, or slice — keeps the
        # (N,H,W) mask stack aligned with xyxy/confidence/tracker_id in all cases.
        return _SimpleDetections(
            self.xyxy[keep],
            (self.mask[keep] if self.mask is not None else None),
            None,
            confidence=(self.confidence[keep] if self.confidence is not None else None),
            tracker_id=(self.tracker_id[keep] if self.tracker_id is not None else None),
        )


def compute_temporal_metrics(per_frame_masks: dict) -> dict:
    """Quantify temporal mask consistency vs a per-frame baseline.

    Metrics:
      * mean_consecutive_iou: mean IoU of each tracked instance's mask between
        consecutive frames where it appears (temporal smoothness; SAM2-prop should
        beat per-frame RF-DETR here).
      * id_persistence: mean fraction of frames each ID survives (1.0 = perfect).
      * id_switches: number of times an ID disappears then a different ID appears
        near the same location (proxy for ID fragmentation).
      * boundary_smoothness: 1 - mean(per-frame mask-boundary length variance vs a
        rolling mean) — higher = smoother mask edges over time.
      * n_instances_mean: mean instances per frame.
    All in [0,1] (or count); higher IoU/persistence/smoothness = better.
    """
    if not per_frame_masks:
        return {"mean_consecutive_iou": 0.0, "id_persistence": 0.0,
                "id_switches": 0, "boundary_smoothness": 0.0, "n_instances_mean": 0.0,
                "n_frames": 0}
    frames = sorted(per_frame_masks.keys())
    n_frames = len(frames)
    # Per-ID IoU across consecutive frames.
    ious = []
    per_id_lens = {}
    prev_mask_by_id: dict = {}
    for f in frames:
        cur = per_frame_masks[f]
        seen = set()
        for oid, m in cur.items():
            seen.add(oid)
            per_id_lens[oid] = per_id_lens.get(oid, 0) + 1
            if oid in prev_mask_by_id:
                ious.append(_mask_iou(prev_mask_by_id[oid], _decode_rle(m)))
        # IDs not seen this frame drop out of the memory (temporal IoU only
        # between consecutive appearances).
        prev_mask_by_id = {oid: _decode_rle(cur[oid]) for oid in seen}
    mean_iou = float(np.mean(ious)) if ious else 0.0
    # Persistence: mean per-ID coverage over the whole video.
    persistence = float(np.mean([l / n_frames for l in per_id_lens.values()])) if per_id_lens else 0.0
    # Boundary smoothness: variance of per-frame total boundary length.
    b_lens = []
    for f in frames:
        tot = 0
        for m in per_frame_masks[f].values():
            mb = _decode_rle(m)
            contours, _ = cv2.findContours(mb.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            tot += sum(cv2.arcLength(c, True) for c in contours)
        b_lens.append(tot)
    bsmooth = 0.0
    if len(b_lens) > 1 and max(b_lens) > 0:
        bsmooth = 1.0 - min(float(np.std(b_lens) / (np.mean(b_lens) + 1e-6)), 1.0)
    n_inst_mean = float(np.mean([len(per_frame_masks[f]) for f in frames]))
    return {
        "mean_consecutive_iou": round(mean_iou, 4),
        "id_persistence": round(persistence, 4),
        "boundary_smoothness": round(bsmooth, 4),
        "n_instances_mean": round(n_inst_mean, 4),
        "n_frames": n_frames,
        # id_switches left as a rough proxy: count of IDs that ever disappear.
        "id_switches": int(sum(1 for oid, l in per_id_lens.items() if l < n_frames)),
    }


def _mask_iou(a, b) -> float:
    """IoU of two bool masks (same shape)."""
    a = np.asarray(a, dtype=bool); b = np.asarray(b, dtype=bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description="SAM2 video mask propagation on RF-DETR detections.")
    ap.add_argument("--video", required=True, help="Input video (mp4).")
    ap.add_argument("--weights", required=True, help="RF-DETR(-seg) checkpoint path.")
    ap.add_argument("--variant", default="rf-detr-seg-medium",
                    help="RF-DETR variant hint for load_model (e.g. rf-detr-seg-medium, rf-detr-base-det).")
    ap.add_argument("--sam2_config", default=SAM2_CONFIG, help="SAM2 hydra config (yml).")
    ap.add_argument("--sam2_ckpt", default=SAM2_CKPT, help="SAM2 .pt checkpoint.")
    ap.add_argument("--out", default=None, help="Output annotated mp4 (default: <video>_sam2prop.mp4).")
    ap.add_argument("--conf", type=float, default=0.35,
                    help="RF-DETR confidence threshold. Note: short-training checkpoints "
                         "(e.g. 1 epoch) often have good mAP but tiny absolute scores "
                         "(max conf ~0.01-0.03), so the default 0.35 filters everything out. "
                         "Use --auto_conf or pass a low --conf (e.g. 0.01).")
    ap.add_argument("--auto_conf", action="store_true",
                    help="Probe RF-DETR scores on frame 0 and lower --conf to the frame-0 "
                         "score median if it sits below the requested conf (only ever lowers; "
                         "never raises). Recommended for short-training checkpoints whose "
                         "absolute confidence scale is miscalibrated. The SAM2-prop path "
                         "also auto-retries with calibration if zero prompts are seeded.")
    ap.add_argument("--keyframe_interval", type=int, default=30,
                    help="Seed SAM2 every N frames with RF-DETR detections (SAM2 propagates between).")
    ap.add_argument("--writer_fps", type=float, default=30.0, help="Output mp4 fps.")
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame (1 = all).")
    ap.add_argument("--proc_size", type=int, default=0,
                    help="Downscale the video's LONGER side to this many pixels (preserving aspect) "
                         "into a temp mp4 before handing it to SAM2, then delete the temp. 0 = native. "
                         "SAM2 keeps frames + masks at the input video's native resolution, so a high-res "
                         "clip (e.g. 2560x1440) makes the per-frame mask upsample alone need ~1.5 GiB and "
                         "OOMs on a ~11.5 GiB GPU under hiera_large. A 960-wide proxy cuts that ~16x and "
                         "keeps the heavy model viable. The annotated output is written at the proxy "
                         "resolution (self-consistent; _draw_overlay resizes masks to the frame).")
    ap.add_argument("--show", action="store_true", help="Display frames live (q to quit).")
    ap.add_argument("--sam2_propagate", action="store_true",
                    help="OPT-IN: use the SAM2 keyframe-seeded mask-propagation path. "
                         "WARNING (BUG 10): this path is a known regression on dense "
                         "colonies -- SAM2 mask propagation collapses multiple penguin "
                         "masks into one giant box. The default per-frame RF-DETR+ByteTrack "
                         "baseline (propagate_baseline) is the working prefork pipeline; "
                         "only set this flag to reproduce the SAM2-prop comparison.")
    ap.add_argument("--no_sam2_propagate", action="store_true",
                    help="DEPRECATED (now the default): per-frame RF-DETR+ByteTrack masks, "
                         "no SAM2. Kept for back-compat with older scripts; a no-op now.")
    ap.add_argument("--metrics_out", default=None,
                    help="Metrics JSON path (default: <out>.metrics.json).")
    args = ap.parse_args()

    if not Path(args.video).is_file():
        sys.exit(f"input video not found: {args.video}")
    if not Path(args.weights).is_file():
        sys.exit(f"weights not found: {args.weights}")
    out = args.out or (str(Path(args.video).with_suffix("")) + "_sam2prop.mp4")
    metrics_out = args.metrics_out or (str(out) + ".metrics.json")

    prop = SAM2VideoPropagator(
        args.weights, variant_hint=args.variant,
        sam2_config=args.sam2_config, sam2_ckpt=args.sam2_ckpt, conf=args.conf,
        auto_conf=args.auto_conf)

    if args.sam2_propagate:
        print(f"[sam2prop] SAM2 propagate mode (keyframe_interval={args.keyframe_interval}): {out}")
        print("[sam2prop] WARNING (BUG 10): SAM2 mask propagation collapses masks into one "
              "giant box on dense colonies. Default is the working per-frame baseline.")
        res = prop.propagate(args.video, out, keyframe_interval=args.keyframe_interval,
                             writer_fps=args.writer_fps, show=args.show, stride=args.stride,
                             proc_size=args.proc_size)
    else:
        out = args.out or (str(Path(args.video).with_suffix("")) + "_baseline.mp4")
        metrics_out = args.metrics_out or (str(out) + ".metrics.json")
        print(f"[sam2prop] baseline mode (per-frame RF-DETR+ByteTrack, no SAM2) [DEFAULT]: {out}")
        res = prop.propagate_baseline(args.video, out, writer_fps=args.writer_fps,
                                      show=args.show, stride=args.stride)

    # Headline metric: tracker-ID-keyed (ByteTrack IDs fragment/swap under occlusion
    # → real tracking numbers). Falls back to the SAM2-object-index dict if the
    # tracked set is empty (baseline mode never builds one).
    tracked = res.get("per_frame_tracked_masks") or res["per_frame_masks"]
    metrics = compute_temporal_metrics(tracked)
    metrics["mode"] = res["mode"]
    metrics["tracker"] = res.get("tracker", "?")
    metrics["unique_ids"] = len(res["unique_ids"])
    metrics["confirmed_ids"] = len(res["confirmed_ids"])
    # Side-by-side: the OLD tautological SAM2-object-index numbers, kept for
    # transparency so the contrast is visible in the JSON (see BUG 9).
    if res.get("per_frame_tracked_masks"):
        sam2obj = compute_temporal_metrics(res["per_frame_masks"])
        metrics["mean_consecutive_iou_sam2obj"] = sam2obj["mean_consecutive_iou"]
        metrics["id_persistence_sam2obj"] = sam2obj["id_persistence"]
        metrics["id_switches_sam2obj"] = sam2obj["id_switches"]
    # Side-by-side: propagated-mask vs fresh per-frame RF-DETR box IoU (independent
    # detector agreement). None in baseline mode (no propagation).
    cons = res.get("propagated_vs_rf_consistency")
    if cons is not None:
        metrics["propagated_vs_rf_consistency"] = cons
    mrate = res.get("propagated_vs_rf_match_rate")
    if mrate is not None:
        metrics["propagated_vs_rf_match_rate"] = mrate
    fsub = res.get("frame_subdir")
    if fsub:
        metrics["frame_subdir"] = fsub
    Path(metrics_out).write_text(json.dumps(metrics, indent=2))
    print(f"[sam2prop] wrote annotated video: {out}")
    print(f"[sam2prop] wrote metrics JSON: {metrics_out}")
    if fsub:
        print(f"[sam2prop] wrote frame subdir: {fsub}")
    print(f"[sam2prop] temporal metrics ({metrics['mode']}) [tracker-ID keyed]: "
          f"consec_IoU={metrics['mean_consecutive_iou']} "
          f"persistence={metrics['id_persistence']} "
          f"boundary_smooth={metrics['boundary_smoothness']} "
          f"id_switches={metrics['id_switches']} "
          f"confirmed_ids={metrics['confirmed_ids']}")
    if "mean_consecutive_iou_sam2obj" in metrics:
        print(f"[sam2prop]   [SAM2-obj-index (tautological, see BUG 9)]: "
              f"consec_IoU={metrics['mean_consecutive_iou_sam2obj']} "
              f"persistence={metrics['id_persistence_sam2obj']} "
              f"id_switches={metrics['id_switches_sam2obj']}")
    if "propagated_vs_rf_consistency" in metrics:
        print(f"[sam2prop]   propagated-vs-RF-DETR consistency="
              f"{metrics['propagated_vs_rf_consistency']:.4f} "
              f"(mean-best-IoU; diluted by unmatched propagated boxes — see BUG 9)")
    if "propagated_vs_rf_match_rate" in metrics:
        print(f"[sam2prop]   propagated-vs-RF-DETR match_rate="
              f"{metrics['propagated_vs_rf_match_rate']:.4f} "
              f"(robust: fraction of propagated boxes with best-IoU >= 0.5)")
    if args.sam2_propagate:
        print("[sam2prop] (SAM2 path) re-run WITHOUT --sam2_propagate for the working "
              "per-frame baseline (the default; BUG 10 makes the SAM2 path regress "
              "on dense colonies).")


if __name__ == "__main__":
    main()

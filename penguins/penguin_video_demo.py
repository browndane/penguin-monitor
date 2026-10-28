# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Real-time-ish penguin video demo: RF-DETR-seg inference + ByteTrack + counting.

Loads a fine-tuned RF-DETR-seg checkpoint, runs per-frame instance segmentation
on a video, tracks instances across frames (ByteTrack via `supervision`), and
overlays masks + IDs + a live/total counter. Writes an annotated mp4 and
optionally shows a live window.

Defaults are tuned for the penguin demo on this machine, so plain
    python penguin_video_demo.py
just works.

USAGE
    python penguin_video_demo.py \
        --weights /home/user/bench_any/runs/segment/penguin-seg-combined_rf-detr-seg-large/checkpoint_best_total.pth \
        --video   /mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/demo_videos/penguin_underwater.mp4 \
        --out     ~/penguins/penguin_underwater_annotated.mp4 \
        --conf 0.35 --show
"""
import argparse
import os
import sys
import time
from pathlib import Path

# ---- Silence Qt font-directory spam from cv2's bundled Qt plugin ----
# cv2's wheel ships a Qt plugin but no fonts dir, so Qt prints a warning per
# imshow call ("QFontDatabase: Cannot find font directory .../cv2/qt/fonts").
# 1) Mute the warning category outright.
# 2) Best-effort: symlink the system fonts dir into the cv2 qt/fonts location so
#    Qt actually has fonts to use (purely cosmetic for the on-screen text).
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts.warning=false;qt.qpa.fonts=false")

# ---- Silence noisy-but-harmless warnings ----
# - torch.jit TracerWarnings from `model.optimize_for_inference()` tracing RF-DETR's
#   DINOv2 backbone (`Converting a tensor to a Python boolean ...`). These are expected
#   when tracing a dynamic-shape model that we'll only ever call at one shape.
# - transformers' `use_return_dict is deprecated` / `loss_type=None` chatter.
# - boxmot's `pkg_resources is deprecated` UserWarning.
import warnings as _warnings
try:
    from torch.jit import TracerWarning as _TracerWarning  # type: ignore
    _warnings.filterwarnings("ignore", category=_TracerWarning)
except Exception:
    pass
_warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*")
_warnings.filterwarnings("ignore", message=r".*use_return_dict.*")
_warnings.filterwarnings("ignore", message=r".*loss_type=None.*")
# transformers prints some of these via its own logger (not warnings.warn), mute it too.
try:
    import logging as _logging
    _logging.getLogger("transformers").setLevel(_logging.ERROR)
except Exception:
    pass
# rf-detr emits its checkpoint-init warnings once at load time; we want to KEEP those
# (they're informative), so we do NOT silence the `rf-detr` logger.
def _try_link_qt_fonts():
    try:
        import cv2 as _cv2_probe  # noqa
        qt_dir = Path(_cv2_probe.__file__).resolve().parent / "qt" / "fonts"
        if qt_dir.parent.is_dir() and not qt_dir.exists():
            for sys_fonts in ("/usr/share/fonts", "/usr/local/share/fonts"):
                if Path(sys_fonts).is_dir():
                    try:
                        os.symlink(sys_fonts, qt_dir)
                    except Exception:
                        pass
                    break
    except Exception:
        pass
_try_link_qt_fonts()

import cv2
import numpy as np

DEFAULT_WEIGHTS = "/home/user/bench_any/runs/segment/african-penguin_rf-detr-seg-xlarge_b1_e150_i640_be64_fair_grid/checkpoint_best_total.pth"
DEFAULT_VIDEO = "/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/demo_videos/penguin_underwater.mp4"

# COCO-80 class names (index = class_id as emitted by RF-DETR's default COCO-pretrained
# checkpoints, which use 0-indexed 80-class labels via supervision's Detections).
# This is only used when --pretrained is set (no fine-tuned weights) so we can filter
# detections by name (e.g. --classes bird).
COCO80_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush",
]


def _resolve_class_filter(spec: str):
    """Parse --classes 'bird,cat' or '14,15' into a set of int class_ids.
    Returns None if spec is empty/falsy (= no filtering)."""
    if not spec:
        return None
    ids = set()
    name_to_id = {n: i for i, n in enumerate(COCO80_NAMES)}
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok.isdigit():
            ids.add(int(tok))
        elif tok in name_to_id:
            ids.add(name_to_id[tok])
        else:
            print(f"[demo] ⚠️ unknown class token '{tok}' (not in COCO-80); ignoring")
    return ids or None


def _pick_rfdetr_class(name_hint: str):
    """Same routing logic as rfdetr_any_tune.pick_rfdetr_class, kept local so this
    demo script has zero project-internal deps and stays self-contained."""
    import rfdetr as _rf
    n = name_hint.lower()
    is_seg = "seg" in n or not ("-det" in n or n.endswith("det") or "detect" in n)
    for size, attr_det, attr_seg in [
        ("xlarge", "RFDETRXLarge", "RFDETRSegXLarge"),
        ("nano",   "RFDETRNano",   "RFDETRSegNano"),
        ("small",  "RFDETRSmall",  "RFDETRSegSmall"),
        ("medium", "RFDETRMedium", "RFDETRSegMedium"),
        ("large",  "RFDETRLarge",  "RFDETRSegLarge"),
    ]:
        if size in n:
            return getattr(_rf, attr_seg if is_seg else attr_det), is_seg
    return (_rf.RFDETRSegMedium if is_seg else _rf.RFDETRBase), is_seg


def load_model(weights_path: str, variant_hint: str = "rf-detr-seg-xlarge",
               pretrained: bool = False, optimize: bool = True):
    """Instantiate RF-DETR. If `pretrained` is True, use the variant's default
    COCO weights (auto-downloaded by rfdetr). Otherwise load `weights_path`.
    If `optimize` is True (default), call `model.optimize_for_inference()` to
    silence the "Model is not optimized for inference" warning and get a
    measurable speedup (fused ops, AMP/half on CUDA where supported)."""
    ModelCls, is_seg = _pick_rfdetr_class(variant_hint)
    if pretrained:
        print(f"[demo] using {ModelCls.__name__} (seg={is_seg}) with DEFAULT COCO weights")
        model = ModelCls()
    else:
        print(f"[demo] using {ModelCls.__name__} (seg={is_seg}) with weights={weights_path}")
        model = ModelCls(pretrain_weights=weights_path)
    # Move to GPU if available + eval mode. The rfdetr wrapper handles this
    # internally in .predict(), but doing it once up-front trims first-call latency.
    try:
        import torch
        if torch.cuda.is_available():
            inner = getattr(model, "model", None)
            if inner is not None and hasattr(inner, "to"):
                inner.to("cuda")
    except Exception:
        pass
    # Fuse / AMP / inference-graph optimizations. RF-DETR exposes this as a
    # one-shot method; older versions may not have it, so we guard.
    if optimize:
        opt = getattr(model, "optimize_for_inference", None)
        if callable(opt):
            try:
                opt()
                print("[demo] model.optimize_for_inference() applied")
            except Exception as e:
                print(f"[demo] ⚠️ optimize_for_inference() failed ({e}); continuing un-optimized")
        else:
            print("[demo] ⚠️ this rfdetr version has no optimize_for_inference(); continuing")
    return model, is_seg


class _BoxmotAdapter:
    """Adapt a boxmot tracker (which expects np.ndarray [N,6]=xyxy,conf,cls + raw BGR
    frame) to the supervision-style `update_with_detections(sv.Detections)` API
    used elsewhere in this script. Call `.set_frame(bgr)` before each update."""
    def __init__(self, tracker):
        self._t = tracker
        self._last_frame = None

    def set_frame(self, bgr):
        self._last_frame = bgr

    def update_with_detections(self, detections):
        n = len(detections.xyxy) if hasattr(detections, "xyxy") else 0
        if n == 0 or self._last_frame is None:
            detections.tracker_id = np.zeros((n,), dtype=int) - 1
            return detections
        confs = getattr(detections, "confidence", None)
        clss = getattr(detections, "class_id", None)
        if confs is None: confs = np.ones((n,), dtype=float)
        if clss is None:  clss = np.zeros((n,), dtype=int)
        dets = np.concatenate([
            detections.xyxy.astype(float),
            np.asarray(confs, dtype=float).reshape(-1, 1),
            np.asarray(clss, dtype=float).reshape(-1, 1),
        ], axis=1)
        tracks = self._t.update(dets, self._last_frame)
        ids = np.zeros((n,), dtype=int) - 1
        if tracks is not None and len(tracks) > 0:
            tr = np.asarray(tracks)
            tboxes = tr[:, :4]; tids = tr[:, 4].astype(int)
            for j, b in enumerate(detections.xyxy):
                bx1, by1, bx2, by2 = b
                best_iou, best_id = 0.3, -1
                for k in range(len(tr)):
                    tx1, ty1, tx2, ty2 = tboxes[k]
                    iw = max(0.0, min(bx2, tx2) - max(bx1, tx1))
                    ih = max(0.0, min(by2, ty2) - max(by1, ty1))
                    inter = iw * ih
                    ua = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                    ub = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
                    iou = inter / (ua + ub - inter + 1e-6)
                    if iou > best_iou:
                        best_iou, best_id = iou, int(tids[k])
                ids[j] = best_id
        detections.tracker_id = ids
        return detections


def _make_botsort(track_buffer, track_match, track_conf, frame_rate,
                  reid_weights="osnet_x0_25_msmt17.pt"):
    """BoTSORT via boxmot with OSNet appearance ReID. Returns (adapter, name) or
    (None, error_msg) so the caller can gracefully degrade to ByteTrack."""
    try:
        from pathlib import Path as _P
        import torch as _torch
        try:
            from boxmot import BotSort as _BotSort
        except ImportError:
            from boxmot import BoTSORT as _BotSort  # older boxmot alias
    except Exception as e:
        return None, f"boxmot unavailable ({e}); install with: pip install boxmot"
    # boxmot's select_device wants "0" / "0,1" / "cpu" — NOT the literal "cuda".
    has_cuda = bool(_torch.cuda.is_available()) and _torch.cuda.device_count() > 0
    device = "0" if has_cuda else "cpu"
    try:
        tracker = _BotSort(
            reid_weights=_P(reid_weights),
            device=device,
            half=has_cuda,
            track_high_thresh=float(track_conf),
            track_low_thresh=0.1,
            new_track_thresh=max(0.5, float(track_conf) + 0.1),
            track_buffer=int(track_buffer),
            match_thresh=float(track_match),
            frame_rate=float(frame_rate),
        )
    except TypeError:
        tracker = _BotSort(reid_weights=_P(reid_weights),
                           device=device, half=has_cuda)
    return _BoxmotAdapter(tracker), (f"boxmot.BoTSORT(reid={reid_weights}, "
                                     f"buffer={track_buffer}, match={track_match}, dev={device})")


def _make_tracker(kind: str = "bytetrack",
                  track_conf: float = 0.35,
                  track_buffer: int = 200,
                  track_match: float = 0.85,
                  track_min_hits: int = 3,
                  frame_rate: float = 30.0,
                  reid_weights: str = "osnet_x0_25_msmt17.pt"):
    """Build a tracker. `kind` in {'bytetrack', 'botsort'}.
    BoTSORT uses appearance ReID (OSNet via boxmot) so IDs survive long dives /
    off-screen gaps even when the predicted Kalman box no longer overlaps.
    Falls back: botsort -> bytetrack -> IoU if upstream libs are missing.
    """
    if kind == "botsort":
        bot, name = _make_botsort(track_buffer, track_match, track_conf, frame_rate, reid_weights)
        if bot is not None:
            return bot, name
        print(f"[demo] ⚠️ {name}; falling back to ByteTrack")
    try:
        import supervision as sv
        # Try the full modern kwarg set first; fall back gracefully for older sv versions.
        kw = dict(
            track_activation_threshold=float(track_conf),
            lost_track_buffer=int(track_buffer),
            minimum_matching_threshold=float(track_match),
            minimum_consecutive_frames=int(track_min_hits),
            frame_rate=float(frame_rate),
        )
        try:
            t = sv.ByteTrack(**kw)
        except TypeError:
            # Older supervision used different kwarg names: track_thresh / match_thresh.
            legacy = dict(
                track_thresh=float(track_conf),
                track_buffer=int(track_buffer),
                match_thresh=float(track_match),
                frame_rate=float(frame_rate),
            )
            try:
                t = sv.ByteTrack(**legacy)
            except TypeError:
                t = sv.ByteTrack()  # last-ditch: defaults only
                print("[demo] ⚠️ ByteTrack ignored tuning kwargs; using sv defaults")
        return t, (f"supervision.ByteTrack(buffer={track_buffer}, match={track_match}, "
                   f"conf={track_conf}, min_hits={track_min_hits}, fps={frame_rate:.1f})")
    except Exception as e:
        print(f"[demo] supervision.ByteTrack unavailable ({e}); using IoU fallback")
        return _IoUTracker(max_age=int(track_buffer)), "iou-fallback"


class _IoUTracker:
    """Minimal IoU-based tracker used only if `supervision` is missing.
    Mirrors the subset of the sv.ByteTrack API we actually use:
        update_with_detections(detections) -> detections (with .tracker_id set)
    """
    def __init__(self, iou_thresh=0.3, max_age=20):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self._next_id = 1
        self._tracks = {}  # id -> (xyxy, age)

    @staticmethod
    def _iou(a, b):
        xa1, ya1, xa2, ya2 = a; xb1, yb1, xb2, yb2 = b
        ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
        ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
        ub = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
        union = ua + ub - inter + 1e-6
        return inter / union

    def update_with_detections(self, detections):
        boxes = detections.xyxy if len(detections) else np.zeros((0, 4))
        assigned = [-1] * len(boxes)
        used_track_ids = set()
        for i, box in enumerate(boxes):
            best_id, best_iou = -1, self.iou_thresh
            for tid, (tbox, _) in self._tracks.items():
                if tid in used_track_ids:
                    continue
                v = self._iou(box, tbox)
                if v > best_iou:
                    best_id, best_iou = tid, v
            if best_id >= 0:
                assigned[i] = best_id
                used_track_ids.add(best_id)
                self._tracks[best_id] = (box, 0)
        for i in range(len(boxes)):
            if assigned[i] < 0:
                tid = self._next_id; self._next_id += 1
                assigned[i] = tid
                self._tracks[tid] = (boxes[i], 0)
        # Age + cull
        for tid in list(self._tracks):
            if tid not in used_track_ids and tid not in assigned:
                box, age = self._tracks[tid]
                age += 1
                if age > self.max_age:
                    del self._tracks[tid]
                else:
                    self._tracks[tid] = (box, age)
        detections.tracker_id = np.array(assigned, dtype=int)
        return detections


def _palette(n=64):
    rng = np.random.default_rng(0)
    return (rng.integers(40, 255, size=(n, 3))).astype(np.uint8)


PALETTE = _palette()


def _draw_overlay(frame, detections, conf_thresh):
    """Draw masks, boxes, IDs. Works on a sv.Detections-like object with
    .xyxy, .confidence, .tracker_id, optional .mask."""
    out = frame.copy()
    n = len(detections.xyxy) if hasattr(detections, "xyxy") else 0
    if n == 0:
        return out
    masks = getattr(detections, "mask", None)
    confs = getattr(detections, "confidence", None)
    tids = getattr(detections, "tracker_id", None)
    for i in range(n):
        c = float(confs[i]) if confs is not None else 1.0
        if c < conf_thresh:
            continue
        tid = int(tids[i]) if (tids is not None and tids[i] is not None and tids[i] >= 0) else -1
        color = PALETTE[(tid if tid >= 0 else i) % len(PALETTE)].tolist()
        if masks is not None:
            m = masks[i].astype(bool)
            if m.shape[:2] != out.shape[:2]:
                m = cv2.resize(m.astype(np.uint8), (out.shape[1], out.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            # Reject pathological masks: a real penguin covers a small fraction of
            # the frame. If a mask fills >40% of the frame it is a SAM2 propagation
            # artefact (mask "ballooning" off a keyframe seed), and painting it
            # would tint the whole frame solid pink/cyan — the bug where frames
            # 4+ render as a uniform colour with only the HUD visible. Skip the
            # tint for these but keep the bbox + ID label so the track is still
            # visible (the bbox is tight to the seed, even when the mask leaks).
            fill_frac = float(m.mean())
            if fill_frac < 0.40:
                overlay = out.copy()
                overlay[m] = color
                out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
        x1, y1, x2, y2 = detections.xyxy[i].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"ID {tid}  {c:.2f}" if tid >= 0 else f"{c:.2f}"
        # Transparent label background (was a solid-fill rectangle -> it
        # occluded adjacent penguins in the dense hard clip). Composite the
        # label patch at 45% over the frame so birds behind/around the label
        # stay visible. Box outline + ID text remain fully opaque.
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lx1, ly1, lx2, ly2 = x1, y1 - th - 6, x1 + tw + 6, y1
        lx1 = max(0, lx1); ly1 = max(0, ly1)
        lx2 = min(out.shape[1], lx2); ly2 = min(out.shape[0], ly2)
        if lx2 > lx1 and ly2 > ly1:
            patch = out[ly1:ly2, lx1:lx2]
            fill = np.full_like(patch, color, dtype=patch.dtype)
            out[ly1:ly2, lx1:lx2] = cv2.addWeighted(fill, 0.45, patch, 0.55, 0)
        cv2.putText(out, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _draw_hud(frame, live_n, total_unique, fps):
    h, w = frame.shape[:2]
    pad = 10
    lines = [
        f"live:  {live_n}",
        f"total: {total_unique}",
        f"fps:   {fps:5.1f}",
    ]
    box_w = 180
    box_h = 20 * len(lines) + 2 * pad
    cv2.rectangle(frame, (pad, pad), (pad + box_w, pad + box_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (pad, pad), (pad + box_w, pad + box_h), (255, 255, 255), 1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (pad + 8, pad + 20 * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def run(args):
    weights = str(Path(args.weights).expanduser())
    video = str(Path(args.video).expanduser())
    out_path = Path(args.out).expanduser() if args.out else \
        Path(video).with_name(Path(video).stem + "_annotated.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.pretrained and not Path(weights).is_file():
        sys.exit(f"❌ weights not found: {weights}  (use --pretrained to skip and use default COCO weights)")
    if not Path(video).is_file():
        sys.exit(f"❌ video not found: {video}")

    model, is_seg = load_model(weights, variant_hint=args.variant,
                               pretrained=args.pretrained,
                               optimize=not args.no_optimize)
    if not is_seg:
        print("[demo] ⚠️ chosen variant is detect-only; masks will be unavailable.")
    class_filter = _resolve_class_filter(args.classes)
    if class_filter is not None:
        names = [COCO80_NAMES[i] if 0 <= i < len(COCO80_NAMES) else str(i) for i in sorted(class_filter)]
        print(f"[demo] class filter: ids={sorted(class_filter)} names={names}")

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"❌ cannot open video: {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Build the tracker now that we know src_fps + stride, so its lost-buffer math
    # is correct in wall-clock terms (frames ↔ seconds).
    _stride = max(1, int(args.stride))
    _track_fps = (src_fps / _stride) if src_fps > 0 else 30.0
    tracker, tracker_name = _make_tracker(
        kind=args.tracker,
        track_conf=args.track_conf,
        track_buffer=args.track_buffer,
        track_match=args.track_match,
        track_min_hits=args.track_min_hits,
        frame_rate=_track_fps,
        reid_weights=args.reid_weights,
    )
    print(f"[demo] tracker: {tracker_name}")

    # Speed knob: downscale every frame so its LONGER side equals `proc_size`,
    # preserving aspect ratio. Works correctly for landscape AND portrait video
    # (phone-shot 2160x3840 etc.). RF-DETR resizes internally to its own pretrained
    # resolution regardless, so on 4K sources almost all per-frame cost is actually
    # BGR->RGB conversion, mask resize back to native, mp4 encode, and (with --show)
    # imshow. Dropping the long side to ~960 typically gives 4-8x end-to-end speedup
    # at negligible quality cost for visualization.
    proc_size = int(args.proc_size if args.proc_size > 0 else args.proc_width)
    long_side = max(W, H)
    if proc_size > 0 and long_side > proc_size:
        scale = proc_size / float(long_side)
        proc_W = int(round(W * scale))
        proc_H = int(round(H * scale))
        # mp4 encoders require even dimensions
        if proc_W % 2: proc_W += 1
        if proc_H % 2: proc_H += 1
        print(f"[demo] downscaling for inference (long-side={proc_size}): {W}x{H} -> {proc_W}x{proc_H}")
    else:
        proc_W, proc_H = W, H

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Write at the effective fps after stride so saved mp4 plays at real-world speed.
    _stride_for_writer = max(1, int(args.stride))
    writer_fps = src_fps / _stride_for_writer
    writer = cv2.VideoWriter(str(out_path), fourcc, writer_fps, (proc_W, proc_H))
    print(f"[demo] input: {video}  {W}x{H}  src_fps={src_fps:.2f}  frames={total_frames}")
    print(f"[demo] output: {out_path}  ({proc_W}x{proc_H})")

    unique_ids = set()           # every tracker_id we've ever seen (noisy)
    confirmed_ids = set()        # tracker_ids that survived >= args.count_min_age frames (real count)
    id_hits = {}                 # tracker_id -> consecutive/total frame hits
    fps_ema = 0.0
    alpha = 0.1
    frame_idx = 0
    stride = max(1, int(args.stride))
    # Effective fps after stride: 1 displayed frame represents `stride` source frames.
    effective_fps = src_fps / stride if stride > 0 else src_fps
    target_dt = (1.0 / effective_fps) if args.playback in ("realtime", "source") else 0.0
    t_start = time.time()
    _window_name = "penguin demo (q to quit)"
    _window_ready = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            # Skip frames per --stride (still need cap.read() to advance the decoder).
            if stride > 1 and (frame_idx % stride) != 0:
                continue
            t0 = time.time()

            # Downscale once up-front so EVERYTHING downstream (inference RGB-copy,
            # mask resize, overlay, encode, imshow) operates on the small frame.
            if (proc_W, proc_H) != (W, H):
                frame = cv2.resize(frame, (proc_W, proc_H), interpolation=cv2.INTER_AREA)

            # RF-DETR expects RGB ndarray / PIL; opencv gives BGR.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                detections = model.predict(rgb, threshold=args.conf)
            except TypeError:
                # Older rfdetr signatures may not accept `threshold=`.
                detections = model.predict(rgb)

            # Filter by confidence (defensive: predict() may not honor threshold).
            if hasattr(detections, "confidence") and detections.confidence is not None:
                keep = detections.confidence >= args.conf
                detections = detections[keep]

            # Optional class-id filter (useful with --pretrained for things like
            # --classes bird on a generic COCO model).
            if class_filter is not None and hasattr(detections, "class_id") \
                    and detections.class_id is not None and len(detections) > 0:
                keep = np.isin(detections.class_id, list(class_filter))
                detections = detections[keep]

            # BoTSORT (boxmot) needs the raw BGR frame for appearance ReID.
            if hasattr(tracker, "set_frame"):
                tracker.set_frame(frame)
            detections = tracker.update_with_detections(detections)

            # Live + cumulative counts. We separate noisy `unique_ids` (every id
            # the tracker ever emitted) from `confirmed_ids` (ids that survived at
            # least `--count_min_age` frames). The HUD shows confirmed — that's
            # what the user actually wants when there are "only 2 penguins".
            tids = getattr(detections, "tracker_id", None)
            if tids is not None:
                live_n = int(np.sum(tids >= 0)) if len(tids) else 0
                for t in tids:
                    if t is None: continue
                    ti = int(t)
                    if ti < 0: continue
                    unique_ids.add(ti)
                    id_hits[ti] = id_hits.get(ti, 0) + 1
                    if id_hits[ti] >= int(args.count_min_age):
                        confirmed_ids.add(ti)
            else:
                live_n = len(detections.xyxy)

            out = _draw_overlay(frame, detections, args.conf)
            dt = max(1e-6, time.time() - t0)
            inst_fps = 1.0 / dt
            fps_ema = inst_fps if fps_ema == 0.0 else (alpha * inst_fps + (1 - alpha) * fps_ema)
            _draw_hud(out, live_n, len(confirmed_ids), fps_ema)

            writer.write(out)
            if args.show:
                if not _window_ready:
                    # Resizable window so the user can maximize it freely. We do NOT
                    # auto-center anymore (pointless on multi-monitor setups). Pass
                    # --window_pos X,Y to pin a specific position; otherwise OpenCV's
                    # default placement is used.
                    cv2.namedWindow(_window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(_window_name, proc_W, proc_H)
                    if args.window_pos:
                        try:
                            sx, sy = args.window_pos.lower().split(",")
                            cv2.moveWindow(_window_name, int(sx), int(sy))
                        except Exception:
                            print(f"[demo] ⚠️ bad --window_pos '{args.window_pos}', expected 'X,Y'")
                    _window_ready = True
                cv2.imshow(_window_name, out)
                # Pace the live window. In 'infer' mode we just yield ~1ms so the
                # GUI can repaint (max FPS = inference FPS). In realtime/source
                # modes we sleep enough to hit `target_dt` per displayed frame.
                if target_dt > 0.0:
                    leftover = target_dt - (time.time() - t0)
                    wait_ms = max(1, int(leftover * 1000))
                else:
                    wait_ms = 1
                if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                    print("[demo] interrupted by user")
                    break
            if args.max_frames and frame_idx >= args.max_frames:
                break
            if frame_idx % 50 == 0:
                elapsed = time.time() - t_start
                print(f"[demo] {frame_idx}/{total_frames or '?'}  "
                      f"live={live_n}  confirmed={len(confirmed_ids)}  "
                      f"raw_ids={len(unique_ids)}  "
                      f"fps={fps_ema:.1f}  elapsed={elapsed:.1f}s")
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\n[demo] done. processed {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / max(elapsed, 1e-6):.1f} FPS avg)")
    print(f"[demo] total CONFIRMED penguins seen (>= {args.count_min_age} hits): {len(confirmed_ids)}")
    print(f"[demo] total raw tracker ids (incl. flicker): {len(unique_ids)}")
    print(f"[demo] annotated video: {out_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="RF-DETR-seg penguin video demo with tracking + counting")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help="Path to fine-tuned RF-DETR-seg checkpoint (.pth).")
    ap.add_argument("--video", default=DEFAULT_VIDEO,
                    help="Input video file.")
    ap.add_argument("--out", default="",
                    help="Output annotated mp4 (default: <video>_annotated.mp4 next to input).")
    ap.add_argument("--variant", default="rf-detr-seg-xlarge",
                    help="Variant hint for selecting the RFDETR* class (e.g. rf-detr-seg-xlarge, rf-detr-seg-large, rf-detr-large).")
    ap.add_argument("--pretrained", action="store_true",
                    help="Ignore --weights and load the variant's default COCO-pretrained weights "
                         "(use with --classes to demo generic detection, e.g. 'bird').")
    ap.add_argument("--classes", default="",
                    help="Comma-separated COCO-80 class names or ids to keep (e.g. 'bird' or '14'). "
                         "Empty = no filter. Only meaningful with --pretrained.")
    ap.add_argument("--proc_size", type=int, default=960,
                    help="Cap the LONGER side of each frame to this many pixels before inference/"
                         "overlay/encode, preserving aspect ratio. Works for landscape AND portrait "
                         "video (phone vertical 2160x3840 etc.). Set 0 to keep native resolution.")
    ap.add_argument("--proc_width", type=int, default=0,
                    help="DEPRECATED alias for --proc_size (kept for back-compat). Despite the name, "
                         "it now also caps the longer side and preserves aspect. Prefer --proc_size.")
    ap.add_argument("--conf", type=float, default=0.35,
                    help="Confidence threshold.")
    ap.add_argument("--imgsz", type=int, default=0,
                    help="Informational only — RF-DETR uses its built-in pretrained resolution. Accepted for CLI parity with bench_any.py.")
    ap.add_argument("--show", action="store_true",
                    help="Show a live cv2 window (off by default for headless/SSH).")
    ap.add_argument("--window_pos", default="",
                    help="With --show: pin the window to 'X,Y' in screen pixels. "
                         "Default empty = OpenCV's default placement (then maximize manually); "
                         "no auto-centering (pointless on multi-monitor setups).")
    ap.add_argument("--tracker", choices=["bytetrack", "botsort"], default="bytetrack",
                    help="Tracker backend. 'bytetrack' (default, supervision) is motion+IoU only. "
                         "'botsort' uses boxmot.BoTSORT with OSNet appearance ReID so IDs survive "
                         "long dives / off-screen gaps even when the predicted box no longer "
                         "overlaps. Requires `pip install boxmot`.")
    ap.add_argument("--reid_weights", default="osnet_x0_25_msmt17.pt",
                    help="ReID weights for BoTSORT (only used with --tracker botsort). boxmot "
                         "auto-downloads this on first use.")
    ap.add_argument("--no_optimize", action="store_true",
                    help="Skip model.optimize_for_inference(). Use only if it errors on your "
                         "rfdetr version. By default the demo always optimizes (silences the "
                         "'Model is not optimized for inference' warning and runs faster).")
    ap.add_argument("--max_frames", type=int, default=0,
                    help="Stop after N frames (0 = whole video). Useful for smoke tests.")
    ap.add_argument("--track_buffer", type=int, default=200,
                    help="ByteTrack lost_track_buffer: frames a missing track is kept alive before "
                         "being deleted. Bigger = same ID survives longer underwater/off-screen "
                         "(default 200 ≈ 6-7s at 30fps effective).")
    ap.add_argument("--track_match", type=float, default=0.85,
                    help="ByteTrack minimum_matching_threshold (lower = easier re-association). "
                         "Default 0.85 — tuned 2026-07-05 on the penguin easy clip to keep two "
                         "nearby surf penguins as separate IDs (sv's 0.7 merges them). See "
                         "memory bytetrack-tuning-penguin-demo.")
    ap.add_argument("--track_conf", type=float, default=0.35,
                    help="ByteTrack track_activation_threshold (higher = fewer flicker tracks). Default 0.35.")
    ap.add_argument("--track_min_hits", type=int, default=3,
                    help="ByteTrack minimum_consecutive_frames: a new track must persist this many "
                         "frames before being confirmed (kills flicker inflating unique counts). Default 3.")
    ap.add_argument("--count_min_age", type=int, default=15,
                    help="A tracker_id is only added to the 'confirmed' unique count after it has "
                         "been seen this many frames total. Filters out brief ID-switch flicker so "
                         "the on-screen 'total' actually reflects the number of distinct penguins. "
                         "Raise (e.g. 30) for noisier scenes, lower to ~5 if you want every blip.")
    ap.add_argument("--stride", type=int, default=1,
                    help="Process every Nth frame (1=all, 2=every other, etc.). Speeds up "
                         "processing and makes --show playback closer to real-time on high-FPS sources.")
    ap.add_argument("--playback", choices=["infer", "realtime", "source"], default="infer",
                    help="With --show: 'infer' = display as fast as inference allows (default); "
                         "'realtime' = pace to wall-clock so 1s of video takes 1s; "
                         "'source' = pace to source FPS (same as realtime for normal videos).")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())

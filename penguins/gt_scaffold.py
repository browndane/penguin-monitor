#!/usr/bin/env python3
"""Ground-truth annotation scaffolder for the penguin field-video clips.

WHY THIS EXISTS
---------------
The 4th paper (difficulty-stratified multi-animal tracking on dense penguin
video) needs per-frame ground truth: object count, bounding boxes, and
persistent identities, on at least one clip (the easy 2-penguin clip is the
cleanest case, ~50-100 frames). The demo (`penguin_video_demo.py`) currently
emits only self-referential metrics (id_persistence, id_switches) computed
*from the tracker's own output* — those cannot evaluate the tracker, they
are the tracker. This tool scaffolds the manual GT that *can* evaluate it.

WHAT IT PRODUCES
----------------
1. Extracted frames (PNG) at a configurable stride, named frame_NNNNNN.png.
2. A COCO-style predictions/seed JSON: one row per (frame, detected instance)
   with bbox, score, and a *suggested* track id taken from ByteTrack. This is
   a STARTING POINT, not ground truth — see the circularity warning below.
3. A human-editable CSV annotation template, one row per (frame, instance),
   with the seed boxes/ids pre-filled in an `gt_count`, `gt_box`, `gt_id`
   column block the annotator overwrites by hand.
4. A short PROTOCOL.md describing exactly how to annotate (count rule, id
   rule, occlusion rule) so the GT is reproducible across annotators.

THE CIRCULARITY WARNING (read before trusting any "GT" from this tool)
----------------------------------------------------------------------
The detector + tracker prefill is CIRCULAR if used as identity ground truth:
MOTA/IDF1 measure how well a tracker reproduces the *true* identity
assignment, so seeding GT identities from a tracker (even a different one)
imports that tracker's association errors as "truth". The boxes are mostly
trustworthy (a strong detector on easy frames is ~correct); the *count* is
trustworthy on easy frames (2 birds rarely hide); the *identities* are NOT
trustworthy and must be hand-verified frame-by-frame. The CSV is structured
so the human fixes identities; the seed ids are just to speed up the easy
frames where the tracker obviously has it right.

COUNT-ACCURACY (the one metric that does NOT need identity GT)
-------------------------------------------------------------
Per-frame object *count* is ground-truthable without identity: a human can
count "there are 2 visible penguins" without naming them. So this tool's
output immediately supports a count-accuracy / MAE-on-count experiment,
which is the realistic first deliverable before full MOTA/IDF1. The identity
columns exist for the eventual MOTA/IDF1 run but are flagged optional.

USAGE
-----
  ~/cvML/bin/python penguins/gt_scaffold.py \
      --video /mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/demo_videos/penguin_african_easy.mp4 \
      --out penguins/research/gt_easy \
      --stride 5 --variant rf-detr-seg-xlarge \
      --track_match 0.85 --conf 0.15 --proc_size 960

This mirrors the demo's strongest config (step-1 re-run). The extracted
frames + seed JSON + CSV template land in --out. Then a human opens the
frames in any viewer and fills the gt_* columns in the CSV.

This is NOT a smoke test and does not touch ~/bench_any/runs/ or Fish/YOLO/.
It reads one video, writes one output dir under penguins/research/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Reuse the demo's exact model-loading + tracker + drawing plumbing so the
# scaffold is byte-identical to what the demo runs (no second implementation
# that drifts). We import from the sibling module.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import penguin_video_demo as _demo  # noqa: E402


def _probe_video(video_path: str):
    """Return (W, H, src_fps, total_frames) via cv2.VideoCapture."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"[gt_scaffold] cannot open video: {video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return W, H, src_fps, total


def _extract_frames(cap, out_dir: Path, stride: int, proc_size: int,
                   src_W: int, src_H: int, max_frames: int | None):
    """Decode frames at `stride`, downscale to proc_size, write PNGs.
    Returns dict {frame_idx_1based -> (proc_W, proc_H)} and the list of kept
    frame indices. Frame index here is the SOURCE frame index (1-based) so the
    CSV/JSON refer to real video positions, not compacted indices."""
    import cv2
    kept = []
    dims = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if stride > 1 and (frame_idx % stride) != 0:
            continue
        if max_frames is not None and len(kept) >= max_frames:
            break
        if proc_size and (proc_size < src_W or proc_size < src_H):
            scale = proc_size / max(src_W, src_H)
            new_w = max(1, int(round(src_W * scale)))
            new_h = max(1, int(round(src_H * scale)))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            new_w, new_h = src_W, src_H
        name = f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(out_dir / name), frame)
        kept.append(frame_idx)
        dims[frame_idx] = (new_w, new_h)
    return kept, dims


def _detect_and_track(model, tracker, cap, kept_frames, proc_size,
                      src_W, src_H, conf: float, class_filter):
    """Detection + tracking over EVERY source frame (stride 1 for the tracker),
    but only RECORD output for the kept (annotation-stride) frames. ByteTrack
    needs consecutive frames to maintain association — feeding it the
    annotation stride (e.g. every 5th frame) breaks its Kalman/IoU matching
    and spuriously loses identities mid-dive. We decouple: tracker always
    runs at stride 1; we sample its output at the annotation stride. Returns
    {frame_idx -> [instance dicts]} for kept frames only."""
    import cv2
    import numpy as np
    frame_idx = 0
    per_frame = {}
    kept_set = set(kept_frames)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        # Process EVERY frame (stride 1) so the tracker sees consecutive
        # frames. Only record results for the kept (annotation-stride) frames.
        if proc_size and (proc_size < src_W or proc_size < src_H):
            scale = proc_size / max(src_W, src_H)
            frame = cv2.resize(frame,
                                (max(1, int(round(src_W * scale))),
                                 max(1, int(round(src_H * scale)))),
                                interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            detections = model.predict(rgb, threshold=conf)
        except TypeError:
            detections = model.predict(rgb)
        if hasattr(detections, "confidence") and detections.confidence is not None:
            keep = detections.confidence >= conf
            detections = detections[keep]
        if class_filter is not None and hasattr(detections, "class_id") \
                and detections.class_id is not None and len(detections) > 0:
            keep = np.isin(detections.class_id, list(class_filter))
            detections = detections[keep]
        if hasattr(tracker, "set_frame"):
            tracker.set_frame(frame)
        detections = tracker.update_with_detections(detections)
        if frame_idx not in kept_set:
            continue  # tracker state updated; output not recorded
        tids = getattr(detections, "tracker_id", None)
        rows = []
        n = len(detections.xyxy) if hasattr(detections, "xyxy") else 0
        for i in range(n):
            x1, y1, x2, y2 = [float(v) for v in detections.xyxy[i]]
            score = float(detections.confidence[i]) if hasattr(detections, "confidence") and detections.confidence is not None else 1.0
            tid = int(tids[i]) if tids is not None and i < len(tids) else -1
            rows.append({
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "score": round(score, 3),
                "seed_track_id": tid,
            })
        per_frame[frame_idx] = rows
    return per_frame


def _overlay_boxes_on_frames(frames_dir: Path, per_frame: dict):
    """Draw the seed boxes + seed track ids onto the extracted PNGs so the
    annotator can SEE what the seed got right/wrong while filling GT. Writes
    annotated copies alongside the originals as frame_NNNNNN_ann.png (does not
    overwrite the clean frames). Uses the demo's transparent-label style."""
    import cv2
    import numpy as np
    # Stable colour per seed track id.
    palette = [(0, 200, 0), (220, 40, 40), (40, 120, 255), (220, 200, 0),
               (200, 0, 220), (0, 200, 200), (255, 140, 0)]
    for fi, insts in per_frame.items():
        src = frames_dir / f"frame_{fi:06d}.png"
        if not src.is_file():
            continue
        img = cv2.imread(str(src))
        if img is None:
            continue
        for inst in insts:
            x1, y1, x2, y2 = [int(v) for v in inst["bbox"]]
            tid = inst["seed_track_id"]
            col = palette[tid % len(palette)] if tid >= 0 else (128, 128, 128)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            label = f"id{tid}({inst['score']:.2f})" if tid >= 0 else f"({inst['score']:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # Transparent label background (matches demo's _draw_overlay fix).
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, max(0, y1 - th - 4)),
                          (x1 + tw + 4, max(0, y1 - 2)), col, -1)
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
            cv2.putText(img, label, (x1 + 2, max(th + 1, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(frames_dir / f"frame_{fi:06d}_ann.png"), img)


def _write_seed_json(out_dir: Path, per_frame: dict, video: str, dims: dict):
    seed = {
        "video": str(video),
        "note": ("Detector+ByteTrack SEED only — NOT ground truth. Identities "
                 "are circular (imported from a tracker); boxes/count need human "
                 "verification; see PROTOCOL.md."),
        "frames": [
            {"frame_idx": fi, "img_w": dims[fi][0], "img_h": dims[fi][1],
             "instances": per_frame.get(fi, [])}
            for fi in sorted(per_frame)
        ],
    }
    with open(out_dir / "seed_predictions.json", "w") as f:
        json.dump(seed, f, indent=2)


def _write_csv_template(out_dir: Path, per_frame: dict, dims: dict, video: str):
    """One row per (frame, instance). gt_* columns are blank for the human to
    fill; seed_* columns carry the detector/tracker prefill."""
    fields = [
        "frame_idx", "instance_idx",
        "seed_bbox_x1", "seed_bbox_y1", "seed_bbox_x2", "seed_bbox_y2",
        "seed_score", "seed_track_id",
        "gt_count",          # human: total visible penguins in THIS frame
        "gt_bbox_x1", "gt_bbox_y1", "gt_bbox_x2", "gt_bbox_y2",  # human: correct box
        "gt_id",             # human: persistent identity (1,2,... or "" if occluded/unsure)
        "gt_visible",        # human: 1 fully visible / 0 occluded / -1 unsure
        "gt_notes",          # human: free text (e.g. "diving", "merged")
    ]
    path = out_dir / "gt_annotations.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for fi in sorted(per_frame):
            insts = per_frame[fi]
            # If detector found nothing, still emit one blank row so the human
            # can record gt_count=0 / gt_visible etc. for that frame.
            if not insts:
                w.writerow({k: "" for k in fields} | {
                    "frame_idx": fi, "instance_idx": 0,
                    "gt_count": "", "gt_visible": "",
                })
                continue
            for j, inst in enumerate(insts):
                b = inst["bbox"]
                w.writerow({
                    "frame_idx": fi, "instance_idx": j,
                    "seed_bbox_x1": b[0], "seed_bbox_y1": b[1],
                    "seed_bbox_x2": b[2], "seed_bbox_y2": b[3],
                    "seed_score": inst["score"], "seed_track_id": inst["seed_track_id"],
                    "gt_count": "", "gt_bbox_x1": "", "gt_bbox_y1": "",
                    "gt_bbox_x2": "", "gt_bbox_y2": "",
                    "gt_id": "", "gt_visible": "", "gt_notes": "",
                })
    return path


def _write_protocol(out_dir: Path, video: str, stride: int, n_frames: int):
    txt = f"""# Ground-truth annotation protocol — {Path(video).name}

**Clip:** {video}
**Frames extracted:** {n_frames} (every {stride}th source frame)
**Tool:** `penguins/gt_scaffold.py`

## What is and is NOT ground truth here

- The `seed_*` columns are detector+ByteTrack output. They are a *scaffold to
  speed you up*, not truth.
- **Count is ground-truthable** without identity: write the true number of
  visible penguins in `gt_count` for each frame (0 if none). This alone
  supports a count-accuracy / MAE-on-count experiment — the realistic first
  result before full MOTA/IDF1.
- **Boxes** should be corrected in `gt_bbox_*` only where the seed box is
  clearly wrong (merged two birds, missed one, bad crop). Otherwise copy the
  seed values across.
- **Identities (`gt_id`)** are the circular part: do NOT trust `seed_track_id`.
  Assign identities yourself: 1, 2, ... for each distinct penguin you can
  follow across frames. Leave blank / -1 when a bird is fully occluded or you
  cannot tell who it is. The identity GT is what MOTA/IDF1 measure, and it
  must come from a human watching the clip, not from any tracker.

## Counting rule (write into gt_count)

- Count a penguin if any part of its body is visible enough that a detector
  *should* have found it. Do NOT count a bird that is fully behind another or
  fully underwater with nothing showing.
- Two penguins touching / overlapping = count 2 (even if only one box fits).
- A bird mid-dive (only splash/wake visible, no body) = count 0 for that frame.

## Identity rule (write into gt_id)

- Use the same small integer (1, 2) for the same physical bird across all
  frames it appears in.
- If a bird leaves and you are unsure the re-emerging bird is the same one,
  give it a NEW id and note "re-emerge, unsure" in gt_notes. Under-counting
  identity (merging two birds) and over-counting (splitting one) are both
  errors the tracker makes; honest unsure-split is better than a guess.

## Occlusion flag (gt_visible)

- 1 = fully visible, well-separated
- 0 = partially occluded but identity still clear
- -1 = heavily occluded / identity uncertain

## When you are done

Save the filled CSV as `gt_annotations.csv` in this same dir. The analysis
CLIs (`ap_difficulty`, the count-accuracy script) read this file directly.
"""
    with open(out_dir / "PROTOCOL.md", "w") as f:
        f.write(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="input clip (mp4)")
    ap.add_argument("--out", required=True, help="output dir for frames+CSV+JSON")
    ap.add_argument("--weights", default=_demo.DEFAULT_WEIGHTS,
                    help="RF-DETR weights (default: xlarge 150ep strong ckpt)")
    ap.add_argument("--variant", default="rf-detr-seg-xlarge")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--track_match", type=float, default=0.85,
                    help="ByteTrack match threshold (demo default 0.85)")
    ap.add_argument("--proc_size", type=int, default=960,
                    help="downscale long edge (960 for 4K, matches demo)")
    ap.add_argument("--stride", type=int, default=5,
                    help="ANNOTATION stride: extract/record every Nth frame "
                         "(5 -> ~75 frames on a 377f clip). The tracker ALWAYS "
                         "runs at stride 1 internally (ByteTrack needs "
                         "consecutive frames); this only sets which frames "
                         "become annotation templates.")
    ap.add_argument("--max_frames", type=int, default=None,
                    help="cap number of extracted frames (None = whole clip)")
    ap.add_argument("--tracker", default="bytetrack",
                    help="bytetrack (default, deprecated) or botsort")
    args = ap.parse_args()

    video = args.video
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    src_W, src_H, src_fps, total = _probe_video(video)
    print(f"[gt_scaffold] {video}  {src_W}x{src_H}  src_fps={src_fps:.2f}  total={total}")
    print(f"[gt_scaffold] stride={args.stride} -> ~{total // max(1, args.stride)} frames to extract")

    import cv2
    # Pass 1: extract frames.
    cap = cv2.VideoCapture(str(video))
    kept, dims = _extract_frames(cap, frames_dir, args.stride, args.proc_size,
                                  src_W, src_H, args.max_frames)
    cap.release()
    print(f"[gt_scaffold] extracted {len(kept)} frames -> {frames_dir}")

    # Load the same strong model + tracker the demo uses (consistency with step 1).
    model, is_seg = _demo.load_model(args.weights, variant_hint=args.variant)
    # _make_tracker returns (tracker, name); we only need the tracker.
    tracker, _tname = _demo._make_tracker(args.tracker,
                                          track_match=args.track_match,
                                          frame_rate=src_fps)
    print(f"[gt_scaffold] tracker: {_tname}")
    class_filter = None  # fine-tuned penguin model emits its own class; no COCO filter

    # Pass 2: detect+track over exactly the kept frames.
    cap = cv2.VideoCapture(str(video))
    per_frame = _detect_and_track(model, tracker, cap, kept, args.proc_size,
                                   src_W, src_H, args.conf, class_filter)
    cap.release()

    n_inst = sum(len(v) for v in per_frame.values())
    print(f"[gt_scaffold] seeded {n_inst} instances across {len(per_frame)} frames")

    _overlay_boxes_on_frames(frames_dir, per_frame)
    _write_seed_json(out_dir, per_frame, video, dims)
    csv_path = _write_csv_template(out_dir, per_frame, dims, video)
    _write_protocol(out_dir, video, args.stride, len(kept))

    print(f"[gt_scaffold] wrote:")
    print(f"  frames:           {frames_dir} ({len(kept)} PNGs)")
    print(f"  seed predictions: {out_dir / 'seed_predictions.json'}")
    print(f"  GT CSV template:  {csv_path}  <-- fill the gt_* columns by hand")
    print(f"  protocol:         {out_dir / 'PROTOCOL.md'}")
    print()
    print("REMINDER: seed_track_id is CIRCULAR (from a tracker), not identity GT.")
    print("Count GT is safe; identity GT must be human-verified frame-by-frame.")


if __name__ == "__main__":
    main()

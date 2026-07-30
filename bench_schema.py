"""Shared metric schema for bench_any and its tuner wrappers.

Keep this file dependency-free so every child script can import it before any
large ML stack is available.
"""

METRIC_KEYS = [
    "box_P", "box_R", "box_F1", "box_mAP50", "box_mAP75", "box_mAP50_95",
    "seg_P", "seg_R", "seg_F1", "seg_mAP50", "seg_mAP75", "seg_mAP50_95",
]

SIZE_METRIC_KEYS = [
    "box_mAP_S", "box_mAP_M", "box_mAP_L",
    "seg_mAP_S", "seg_mAP_M", "seg_mAP_L",
    "box_AR_100", "seg_AR_100",
]

# SAM2-prompted segmentation family, mirroring the native ``seg_*`` columns.
# Populated when a model's masks come from prompting SAM2 with predicted boxes
# (pure detectors, plus YOLO-seg which can report both native ``seg_*`` and
# ``sam2_*``). Keeps cross-model segmentation comparison fair by separating
# native mask-head quality from foundation-prompted mask quality.
SAM2_METRIC_KEYS = [
    "sam2_P", "sam2_R", "sam2_F1",
    "sam2_mAP50", "sam2_mAP75", "sam2_mAP50_95",
    "sam2_mAP_S", "sam2_mAP_M", "sam2_mAP_L", "sam2_AR_100",
]

EXTRA_METRIC_KEYS = [
    "mIoU", "mAcc", "fwIoU", "pixel_acc", "boundary_IoU",
    "fg_mIoU", "fg_mAcc", "fg_fwIoU", "fg_pixel_acc", "fg_boundary_IoU",
    "fg_seg_P", "fg_seg_R", "fg_seg_F1",
]

PERF_KEYS = ["params_M", "fps_eval"]
SUPPORT_KEYS = ["n_images", "n_instances"]

# AP-Difficulty stratified columns (Step 2). Per-tier mAP50_95 for box + seg
# families, plus a single scalar Easy−Hard delta per family (``diff_delta_*``).
# These are *analytical* (derived post-hoc from existing predictions), not
# scored model metrics, so bench_any subtracts them from SCORE_KEYS (see
# bench_any.py). delta is a scalar float — never the dict stratified_coco_eval
# returns (v3 fix #4: dict-in-CSV-cell would corrupt the numeric column).
DIFFICULTY_TIERS = ["Easy", "Medium", "Hard"]
DIFFICULTY_METRIC_KEYS = (
    [f"{pfx}_mAP50_95_{t}" for t in DIFFICULTY_TIERS for pfx in ("box", "seg")]
    + [f"diff_delta_{pfx}" for pfx in ("box", "seg")]  # Easy - Hard mAP50_95
)

ALL_METRIC_KEYS = (
    METRIC_KEYS + SIZE_METRIC_KEYS + SAM2_METRIC_KEYS
    + EXTRA_METRIC_KEYS + PERF_KEYS + SUPPORT_KEYS
    + DIFFICULTY_METRIC_KEYS
)

RUN_META_KEYS = ["status", "return_code", "metric_status", "manifest_path"]

# Per-model batch bookkeeping a wrapper may report back in its results.csv so
# the aggregated CSV can show the micro-batch each wrapper actually ran
# (``batch_sz``) and the realised effective batch (``batch_effective`` =
# micro_bs * grad_accum). These are informational, not scored metrics, so they
# are kept out of ALL_METRIC_KEYS / SCORE_KEYS.
RUN_BATCH_KEYS = ["batch_sz", "batch_effective"]


def zero_metrics():
    return {k: 0.0 for k in ALL_METRIC_KEYS}

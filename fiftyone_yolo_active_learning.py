# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
import fiftyone as fo, os, shutil, sys, yaml, numpy as np, argparse, glob, tempfile, torch, gc, cv2, subprocess, csv, datetime, functools, re, hashlib, select, threading, termios, fcntl, contextlib, io
from bson import ObjectId
from ultralytics.data.converter import merge_multi_segment
from collections import Counter, defaultdict
from fiftyone import ViewField as F
from fiftyone.core.odm.dataset import ActiveFields
import fiftyone.zoo as foz, fiftyone.brain as fob
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
default_conf=0.4
_ALLOW_DELETE_LIVE = False
_INPUT_DIRS = []   # realpaths of every --local_*_dirs, read by _refuse_overwrite at the rmtree itself

def _del_dataset(name, why):
    """A persistent dataset needs --allow_delete_live, so a stray delete cannot drop a live one."""
    if not _ALLOW_DELETE_LIVE and fo.dataset_exists(name) and fo.load_dataset(name).persistent:
        sys.exit(f"REFUSED: '{name}' is a persistent dataset and this run tried to delete it ({why}). Nothing was touched. Pass --allow_delete_live if you really mean it.")
    fo.delete_dataset(name)

def _overlaps(a, b):
    """Equal, or one inside the other: an export rmtrees output_dir while reading off disk."""
    return a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep)

def parse_args():
    parser = argparse.ArgumentParser(description="Load, merge, improve, split, and export FiftyOne datasets to YOLO format.", allow_abbrev=False)   # the disk line test reads raw argv: an abbreviated flag lost his clicks
    parser.add_argument("--local_yolo_dirs", nargs='+', help="Paths to local YOLO dataset directories")
    parser.add_argument("--local_coco_dirs", nargs='+', help="Paths to local flat COCO dataset directories (each contains _annotations.coco.json + images)")
    parser.add_argument("--dataset_name", nargs='+', help="Datasets to load from zoo")
    parser.add_argument("--output_dir", help="Export directory (YOLO format)")
    parser.add_argument("--output_coco_dir", help="Also export to COCO format at this directory (train/valid/test/_annotations.coco.json + images)")
    parser.add_argument("--label_types", nargs='+', default=['detections', 'segmentations'], choices=['detections', 'segmentations', 'obb', 'keypoints'], help="Label types to load")
    parser.add_argument("--remap_classes", nargs='+', help="Remap classes (format: old:new)")
    parser.add_argument("--resplit", nargs='*', type=float, default=None, metavar='RATIO', help="Re-split dataset into train/val/test. Pass 3 floats (e.g. --resplit 0.7 0.2 0.1) or no args to use default 0.7/0.2/0.1. Omit flag to keep source splits.")
    parser.add_argument("--max_samples", type=int, help="Max samples per split, zoo datasets only: a local folder always loads whole")
    parser.add_argument("--splits_to_run", nargs='+', choices=['train', 'val', 'test', 'validation'], help="Splits to process. A local folder still loads whole; this narrows the pass, and the --to_valid screen")
    parser.add_argument("--classes", nargs='+', help="Classes to use")
    parser.add_argument("--all_coco_classes", action="store_true", help="Use all 80 COCO classes")
    parser.add_argument("--find_missed_labels", action="store_true", help="Find missed labels")
    parser.add_argument("--full_inference", action="store_true", help="Run inference on full image (default: skip labeled regions)")
    parser.add_argument("--eval_model", default="/home/user/yolo_general/models/yolov8x6-oiv7.pt", help="Evaluation model")
    parser.add_argument("--reevaluate", action="store_true", help="Force re-evaluation")
    parser.add_argument("--eval_iou_thresh", type=float, default=0.25, help="IoU threshold for evaluation metrics, for SAM3 disagreement matching and for consensus vote matching (mask IoU)")
    parser.add_argument("--nms_iou", type=float, default=0.45, help="YOLO NMS IoU threshold (higher = more predictions, lower = fewer)")
    parser.add_argument("--pred_dedup_iou", type=float, default=0.3, help="Self-dedup IoU for predictions across scales/variants/superres (does NOT affect merge into ground_truth for polylines)")
    parser.add_argument("--auto_approve_threshold", nargs='+', help="Auto-approve confidence (single value or class:threshold pairs, e.g., 0.2 or dog:0.3 person:0.5)")
    parser.add_argument("--refine_boxes", action="store_true", help="Refine labels with SAM")
    parser.add_argument("--sam_model", default="segment-anything-2.1-hiera-large-image-torch", help="SAM model")
    parser.add_argument("--sam_confidence_threshold", type=float, default=0.5, help="SAM refine quality gate: area of the SAM mask's own extent over the prompt box area, in [0,1]; a refined mask below this keeps the original label. 0 = off.")
    parser.add_argument("--sam_convert_to_boxes", action=argparse.BooleanOptionalAction, default=True, help="Convert polylines to boxes")
    parser.add_argument("--sam3_refine", action="store_true", help="SAM3 box-prompt refinement of existing GT (one bbox per label). Samples without GT are skipped unless --sam3_prompt is also set.")
    parser.add_argument("--sam3_prompt", nargs="+", default=None, help="SAM3 text prompt(s), one per class (class name derived by stripping a/an/the). If set without --sam3_refine, runs text-mode on every image ignoring GT. If combined with --sam3_refine, used only for samples without GT.")
    parser.add_argument("--sam3_script", default=None, help="Path to sam3_coco_annotator.py (default: <this_dir>/point_crowdsource_annotate/sam3_coco_annotator.py)")
    parser.add_argument("--sam3_overwrite", action="store_true", help="Replace existing GT with SAM3 outputs. COCO's labels, the cars already there and any label a SAM3 car sits on stay, and SAM3's cars are never written. Default: each SAM3 label redraws the label it fits best at --seg_iou and keeps that label's class. A label two want goes to the same-class one if it covers --adopt_match_iou, else to the tighter fit, and the other tries its next fit. One that lands on a COCO label is dropped, and one that finds no free label is added unless it is a car. Two SAM3 labels on one object at mask IoU --adopt_match_iou, from two prompts or one prompt twice, keep one by --sam3_adopt's rule: the lower confidence goes, and a redraw of your label outlives a new one.")
    parser.add_argument("--sam3_eval", action="store_true", help="Evaluate SAM3 text predictions against ground_truth (eval_key sam3_eval) instead of merging them into it. Use to flag label errors.")
    parser.add_argument("--sam3_disagree_out", default=None, help="With --sam3_eval: write the ranked per-instance disagreement CSV here (default: ~/bench_any/runs/sam3_disagree/<dataset>/disagreement_<date>.csv).")
    parser.add_argument("--sam3_score", type=float, default=None, help="SAM3 confidence threshold for --sam3_refine/--sam3_prompt (default 0.25, or the lowest --auto_approve_threshold if given).")
    parser.add_argument("--sam3_add_score", type=float, default=None, help="With --sam3_adopt: separate, stricter confidence a SAM3 mask must clear to ADD an object no annotation covers. --sam3_score stays the floor for disputing the class of a box you already drew. Default: no extra gate.")
    parser.add_argument("--classchange_iou", type=float, default=0.1, metavar="IOU", help="The class errors adopt filed as a new object beside your label because they overlapped under --adopt_match_iou (0.5). ON BY DEFAULT at 0.1: the classchange pass shows them alongside the ones adopt booked, same queue, same verb, a tick or a click keeps YOUR class. 0 turns them off. Read time only, off the masks already stored: no re-adopt, no vote, nothing reviewed is touched.")
    parser.add_argument("--adopt_match_iou", type=float, default=0.5, help="With --sam3_adopt: mask IoU above which a SAM3 mask and an annotation are the SAME object. At or above it, agreeing on class is adopted silently and disagreeing is a class error; below it a same-class mask queues as lowiou (the mask moved) and a different-class mask is a new object beside yours, never a class error.")
    parser.add_argument("--sam3_border_factor", type=float, default=0.4, help="Drop a border-touching SAM3 mask smaller than this fraction of the image's median object area. 0 disables the rule.")  # in `sig`, so it only bites a re-sieve
    parser.add_argument("--sam3_min_area", type=int, default=144, help="SAM3 sieve: minimum mask area in pixels, the authors' 12x12 ruling (W118)")
    parser.add_argument("--min_label_area", type=int, default=0, help="With --sam3_adopt: DELETE every ground_truth polyline whose bounding box is under this many pixels, his labels and SAM3's alike, recorded as `floored` in the adopt CSV. Author 2026-09-14: 256 (16x16) on the house buckets. With --sam3_add_only it floors SAM3's additions only and every annotation already there survives (Author 2026-09-28, 'those objects should have never been there for review'). Default 0 is off, because at 256 sama-coco-birdpass loses 10028 of its 28726 labels and 8027 of its 17734 birds.")
    parser.add_argument("--sam3_nms", type=float, default=0.8, help="SAM3 sieve: class-agnostic non-maximum suppression IoU")
    parser.add_argument("--sam3_mask_threshold", type=float, default=0.3, help="SAM3 mask binarisation, a PROBABILITY in 0-1, not a logit: transformers 5.13.0 does `pred_masks.sigmoid()` and then `masks > mask_threshold`. Lower keeps more pixels and gives whole objects, higher shaves the edge and wriggles around them. HF defaults to 0.5, this to 0.3.")
    parser.add_argument("--sam3_hierarchy", nargs="+", default=None, help="With --sam3_eval or --sam3_adopt: child:parent pairs (e.g. bike:vehicle dog:animal). A parent prompt covering a child annotation is 'agrees', not 'renamed', and parent/child never count as rivals.")
    parser.add_argument("--bioclip_classify", action="store_true", help="Relabel existing ground_truth labels (species-level, per-image majority vote) with BioCLIP 2. Labels marked as COCO ground truth are skipped and counted, never renamed.")
    parser.add_argument("--bioclip_taxa", nargs='+', default=None, help="Override candidate scientific names. Default: 18 extant penguin species across 6 genera.")
    parser.add_argument("--bioclip_genus", action="store_true", help="Write genus instead of species as the class label.")
    parser.add_argument("--bioclip_classes", nargs='+', default=['penguin'], help="Only reclassify labels whose (normalized) name is in this list (default: penguin).")
    parser.add_argument("--bioclip_restrict", nargs='+', default=None, help="Restrict BioCLIP candidates to these genera or species (e.g. spheniscus or 'spheniscus demersus'). Also filenames are auto-scanned: common-name/species hits (e.g. 'Emperor_penguin') skip BioCLIP; genus hits (e.g. 'Aptenodytes11.jpg') narrow candidates for that image.")
    parser.add_argument("--consensus", nargs=2, metavar=("YOLO_A", "YOLO_B"), help="Consensus corrector: two YOLO seg checkpoints voted against SAM3 text prompts (--sam3_prompt required). Auto-renames and mask-replaces where 2 of 3 agree, proposes adds/deletes, per-instance review in the app. YOLO votes at conf 0.4 (default_conf), vocabulary = ground_truth labels.")
    parser.add_argument("--consensus_auto", type=float, default=0.7, help="min YOLO confidence of the agreeing YOLO votes for an AUTO rename")
    parser.add_argument("--consensus_vote_conf", type=float, default=0.0, help="drop YOLO polylines below this confidence before voting, so a high-recall low-precision checkpoint cannot invent proposals. SAM3 is gated by --sam3_score instead.")
    parser.add_argument("--consensus_sam3_add", type=float, default=None, metavar="CONF", help="SAM3-as-default: a SAM3 find above CONF that covers no annotation goes straight into ground_truth as verdict 'added', so the review is a rejection pass. Never deletes an annotation SAM3 failed to cover.")
    parser.add_argument("--consensus_revert", action="store_true", help="restore ground_truth from ground_truth_orig and exit. The rename consumes the snapshot, so this runs once and cannot be undone, and it also deletes sam3_predictions and every review field. Refuses on an already-reviewed bucket without --allow_revert_reviewed.")
    parser.add_argument("--allow_revert_reviewed", action="store_true", help="permit --consensus_revert on a bucket carrying adopt_reviewed tags, throwing that review away. Off by default since 2026-09-14.")
    parser.add_argument("--allow_delete_live", action="store_true", help="permit this run to delete a live house-* dataset. Off by default since 2026-09-13, when every house dataset was deleted out of the shared FiftyOne database.")
    parser.add_argument("--dedup_contained", action="store_true", help="With --sam3_adopt: drop same-class ground_truth polylines that are one object drawn twice (mask intersection over the smaller mask >= 0.7, mask IoU < 0.5). Prints a census, writes a CSV of every drop, repairs the dataset and exits without opening the app.")
    parser.add_argument("--sam3_adopt", action="store_true", help="SAM3-as-default: SAM3 output becomes ground_truth, and the review is a rejection pass over triggered images only.")
    parser.add_argument("--sam3_adopt_fill", default=None, metavar="CKPT", help="YOLO checkpoint used ONLY on images where SAM3 returned nothing.")
    parser.add_argument("--yolo_judge", default=None, metavar="CKPT", help="The reverse of the SAM3 pass: a YOLO seg checkpoint proposes into yolo_judge_preds and the SAME disagreement scorer ranks ground_truth against it. For sets SAM3 labelled on its own. Never merges and never rewrites ground_truth; only a click on a ground_truth label deletes it. Use a -seg checkpoint or every mask IoU is scored against a rectangle.")
    parser.add_argument("--sam3_add_only", action="store_true", help="With --sam3_adopt: SAM3 may only ADD objects where there was no annotation. Every annotation already there survives verbatim, nothing is replaced, renamed or deleted, and the only review pass is `more`. For sets whose labels are right and merely incomplete, like sama-coco.")
    parser.add_argument("--review_cap", type=int, default=None, metavar="N", help="show at most this many images in the review queue, worst first. On --to_valid_confusion, --to_valid_small, --border_to_valid and --overlap_to_valid, at most this many frames a launch; without it every waiting frame.")
    parser.add_argument("--review_files", default=None, metavar="FILE", help="With --sam3_adopt: serve only the queue images named in FILE, one image file name per line. Names that are no frame here, or not in this queue, are counted and skipped. With a --to_valid screen (--overlap_to_valid, --border_to_valid, --to_valid_small, --to_valid_confusion too): serve only the frames named, by file name or stem, from train/ and valid/.")
    parser.add_argument("--review_reason", choices=sorted(set(ADOPT_RANK) | {"unsupported"}), default=None, help="With --sam3_adopt: review ONE fault at a time. Queues only images carrying that reason and greys every label you and SAM3 already agree on, so the fault of this pass is what stands out. Each reason drains its own queue.")
    parser.add_argument("--review_pair", default=None, metavar="WAS:NOW", help="With --review_reason classchange: only the disputes that went one way, e.g. person:dog. Keeps your eye in one mode so you are hunting outliers instead of re-reading every crop.")
    parser.add_argument("--review_origin", nargs="+", default=None, metavar="CLASS", help="With --review_reason classchange: every dispute that STARTED as one of these classes, whichever way it went. What a dispute was before SAM3 touched it predicts whether SAM3 is wrong far better than its confidence does. Takes several, so --review_origin car dog is one launch for everything worth your eyes.")
    parser.add_argument("--unsup_sam3_conf", type=float, default=None, metavar="CONF", help="With --review_reason unsupported: a SAM3 mask scoring under this no longer counts as backing one of your labels, so weakly-backed labels join the queue. Read time only, off the masks already stored: it never re-runs SAM3 and never moves the adopt signature, so no review edit is touched. Measured on the seven reviewed buckets, 12512 images: 1904 objects at no floor, 1938 at 0.85.")
    parser.add_argument("--review_more_conf", type=float, default=0.0, metavar="CONF", help="With --review_reason more: hide every SAM3 ADDITION scoring under this. Default 0: every addition adopt exported is shown, whatever the build's score floor. A SAM3 mask that redrew one of your own labels is not an addition and is shown at every confidence. Display and queue only: nothing is deleted, and the ones held back stay in ground_truth exactly as they are.")
    parser.add_argument("--tick_wipes", nargs="+", default=None, metavar="CLASS", help="With --review_reason more: a tick on an image DELETES every label of these classes on it, SAM3's additions and the original annotations alike, instead of only taking SAM3's additions off. For an image where the class is simply not there, like SAM3 and the original both calling a doll a bird. The originals go from ground_truth_orig too, so --consensus_revert cannot bring them back. Every class must be one of --sam3_prompt, so the flag can only ever aim at what the pass is already about.")
    parser.add_argument("--review_reset", action="store_true", help="Put this pass back in the queue before opening it. Closing the app marks everything on screen reviewed, so one accidental close drains a pass; this undoes that. Scoped to the same --review_reason/--review_pair you pass, or the whole dataset if you pass neither. Touches only the reviewed tag: annotations, deletions and reverts are not affected.")
    parser.add_argument("--similar_model", default="dinov2-vitb14-reg-torch", help="Model for finding similar images. clip-vit-base32-torch")
    parser.add_argument('--find_similar', type=float, nargs='?', const=1.0, help='Show only duplicates with uniqueness < threshold (default 1.0 all). Select to remove or use with --subset to split.')
    parser.add_argument('--exact_dup', action='store_true', help='Find exact duplicate images by hash')
    parser.add_argument('--to_valid', action='store_true', help='Open the bucket camera by camera, ground_truth alone. TICK a frame to move it from train/ to valid/ on disk, image and label, retagged validation, after the y/N prompt on close. Click a polygon to delete it, in FiftyOne and in the label file. Author 2026-09-22: partial detections, a dog cut by the border or a person half behind a wall, go to validation this way, by his hand, no grouping picks them.')
    parser.add_argument('--to_valid_confusion', action='store_true', help='The train/ frames where a person box covers more than half of a dog or animal box, the --overlap_to_valid way: TICK a frame to keep it in train/, every frame left unticked moves to valid/ on close and y, each tagged confusion_judged and never shown again. Every waiting frame, or --review_cap N of them. Author 2026-09-25: a person label with a dog inside it teaches the model that dog is a person. The last pass on sama-merge-5. 2026-10-04: "Flip tiny + person/dog".')
    parser.add_argument('--to_valid_small', action='store_true', help='The train/ frames where over a quarter of the labels are under 8x8 px at 320, the size the model trains at, the --overlap_to_valid way: TICK a frame to keep it in train/, every frame left unticked moves to valid/ on close and y, each tagged small_judged and never shown again. Every waiting frame, or --review_cap N of them. Author 2026-09-25: the pass before --to_valid_confusion on sama-merge-5. 2026-10-04: "Flip tiny + person/dog".')
    parser.add_argument('--border_to_valid', action='store_true', help='Every frame of the bucket, train/ and valid/, where a labelled object touches the frame edge and its box is under 0.4 of the frame\'s median object area (the --sam3_border_factor rule on the labels), the --overlap_to_valid way: TICK a frame to keep it in train/. On close and y a ticked frame in valid/ moves to train/ and an unticked one in train/ moves to valid/, each tagged border_judged and never shown again. Every waiting frame, or --review_cap N of them. Author 2026-10-04: "Small edge object", moved to valid/ by rule first, this screen brings back what he ticks.')
    parser.add_argument('--overlap_to_valid', nargs='?', type=float, const=0.5, default=None, metavar='T', help='Open every frame of the bucket, train/ and valid/, where a car box covers over T of a smaller box that is no car, or a dog box and a person box cover each other over T, the bigger over the smaller. T defaults to 0.5. Camera by camera, ground_truth alone, every waiting frame or --review_cap N of them. TICK a frame to keep it in train/. On close and y, a ticked frame in valid/ moves to train/ and an unticked frame in train/ moves to valid/, image and label. A frame that lands where you put it is tagged overlap_judged and never shown again, so the next run serves the rest. A lost tick (grid Escape, shift-select) sends its frame to valid/ and hides it: check the TICKED count in the prompt. Click a polygon to delete it, in FiftyOne and in the label file. Author 2026-10-01: a box inside a car box, or a dog box and a person box on each other, teaches false positives.')
    parser.add_argument('--find_redundant', nargs='?', const='0.7:10', default=None, metavar='IOU:HAM', help='Tag frames of one camera in train/ that carry the same objects in the same place with the same look, so a burst of a sleeping dog becomes one frame. Camera from the stem (cam2-<epoch>-id), never across cameras; stems with no camera stamp (CatDog, sama-coco) are one camera with no clock, ordered by name. Two frames match when their class multisets agree, every same-class box pair overlaps at IoU >= IOU, and every matched object crop is within HAM of 64 bits by dHash. Frames with no label match on whole-frame dHash <= 4 or the same 10-minute clock window. Frames already in valid/ are never shown. The earliest frame of a group is tagged kept, the rest duplicate, and both are shown group by group. A TICKED frame moves to valid/, as on --to_valid: close the app and the terminal asks y/N, naming how many ticked frames lead their group. Unticked frames stay in train/. A clicked polygon is deleted, in FiftyOne and in the label file on disk, permanent. The pass exits after the move, it exports nothing. Default 0.7:10, measured 2026-09-20: 757 of 13642 house frames, 611 of them on 15Dec. The DINOv2 finder at 0.2 tagged 642 of 1005 21Nov frames and 221 of those pairs carried different labels.')
    parser.add_argument('--find_duplicates', type=float, default=None, metavar='THRESH', help='Tag near-duplicate images via fiftyone.brain.compute_near_duplicates with --similar_model; THRESH is an embedding DISTANCE (brain default 0.2), not a similarity. Review/delete/export like --exact_dup.')
    parser.add_argument('--subset', action='store_true', help='Split dataset based on similarity selection into two outputs')
    parser.add_argument("--force_reload", action="store_true", help="Force dataset reload")
    parser.add_argument("--no_launch", action="store_true", help="Don't launch app")
    parser.add_argument('--max_bbox_width', type=float, default=1.0, help='Max bbox width')
    parser.add_argument('--min_bbox_width', type=float, default=0.01, help='Min bbox width')
    parser.add_argument('--max_aspect_ratio', type=float, default=0.0, help='Max aspect ratio w/h, 0 = off (W10)')
    parser.add_argument('--min_aspect_ratio', type=float, default=0.0, help='Min aspect ratio')
    parser.add_argument('--clean_4point_seg_outliers', action=argparse.BooleanOptionalAction, default=True, help='Also clean outliers from 4-point polylines/segmentations (default: detections only)')
    parser.add_argument('--clean_complex_seg_outliers', action='store_true', help='Also clean outliers from complex polylines/segmentations with >4 points (default: detections only)')
    parser.add_argument('--seg_iou', default=0.25, type=float, help='Save predictions overlapping existing complex polylines less than this IoU, lower is strict')
    parser.add_argument('--scales', nargs='+', type=int, default=list(range(960, 1920, 64)), help='Inference scales for multi-scale detection')
    parser.add_argument('--sample', type=int, help='Process only specific sample number (1-based index)')
    parser.add_argument('--vertical_scaling', type=float, default=1.0, help='Vertical perspective scaling factor (1.0=no scaling, >1.0=smaller at top/larger at bottom)')
    parser.add_argument('--region', nargs=4, type=float, metavar=('TOP', 'BOTTOM', 'LEFT', 'RIGHT'), help='Filter predictions to region: top bottom left right (0.0-1.0)')
    parser.add_argument('--superres', action='store_true', help='Run inference both with and without super-resolution, combining results (multiplies scale runs by 2)')
    parser.add_argument('--imgproc', nargs='+', choices=['contrast', 'contrast2', 'denoise', 'denoise2', 'denoise3', 'retinex', 'supir', 'swinir'], help='Image processing: contrast, contrast2 (CLAHE), denoise, denoise2, denoise3 (edge-aware), retinex, supir, swinir')
    parser.add_argument('--intensity', type=float, default=1.0, help='Global intensity multiplier for all image processing and superres (>=0.1)')
    parser.add_argument('--clip_outliers', action='store_true', help='Use CLIP outlier detection instead of manual thresholds')
    parser.add_argument('--outlier_classes', nargs='+', help='Classes to check for outliers (keeps all other classes intact)')
    parser.add_argument('--keep_tn', action='store_true', help='With --find_missed_labels: show unlabeled images, select false negatives, export TN/FN/autolabeled separately')
    parser.add_argument('--filter_shadows', action='store_true', help='Filter out shadow detections based on low saturation/brightness')
    parser.add_argument('--clean_overlap_on', help='Remove overlaps with priority class (e.g., dog)')
    parser.add_argument('--resize', type=int, help='Resize images based on shortest side (e.g., 640)')
    parser.add_argument('--force_export', action='store_true', help='Allow an export to delete a non-empty output directory (default: refuse)')
    parser.add_argument('--own_dir', action='store_true', help='restrict review queue and export to samples under the single --local_yolo_dirs dir (a merged dataset holds other dirs\' samples, and a sample tick on one destroys another bucket\'s labels)')
    parser.epilog = "Removed 2026-09-07: --annotate_labelstudio/--load_annotations/--delete_annotation_run (Label Studio round trip). Use --consensus and per-instance review instead."
    args = parser.parse_args()
    # the widened half has no prev_label to scope by; on by default, so dropped, not an error
    if args.review_origin or args.review_pair: args.classchange_iou = 0
    # one if/elif chain runs the screens, so a second is dropped silently. Read before --to_valid absorbs the other three
    _scr = [f for f, on in (("--exact_dup", args.exact_dup), ("--to_valid", args.to_valid), ("--to_valid_confusion", args.to_valid_confusion), ("--to_valid_small", args.to_valid_small), ("--border_to_valid", args.border_to_valid), ("--overlap_to_valid", args.overlap_to_valid is not None), ("--find_redundant", args.find_redundant is not None), ("--find_duplicates", args.find_duplicates is not None), ("--find_similar", args.find_similar is not None)) if on]
    if len(_scr) > 1: parser.error(f"{' and '.join(_scr)} are {len(_scr)} screens: run one, then the other (W216)")
    if _scr and args.no_launch: parser.error(f"{_scr[0]} opens the app whatever --no_launch says: drop --no_launch (W234)")
    if _scr and args.bioclip_classify: parser.error(f"--bioclip_classify skips every screen, so {_scr[0]} would never open: run one, then the other (W239)")
    if args.subset and _scr and _scr[0] not in ("--exact_dup", "--find_duplicates", "--find_similar"): parser.error(f"{_scr[0]} moves ticked frames and exits before any --subset split: drop --subset (W239)")
    if args.overlap_to_valid is not None and not 0 < args.overlap_to_valid < 1: parser.error(f"--overlap_to_valid wants an overlap between 0 and 1, e.g. 0.5; got {args.overlap_to_valid}")
    args.to_valid = args.to_valid or args.to_valid_confusion or args.to_valid_small or args.border_to_valid
    # same test as is_outlier_workflow
    outlier_branch = (args.clip_outliers or args.max_bbox_width != 1.0 or args.min_bbox_width != 0.01
                      or args.max_aspect_ratio != 0.0 or args.min_aspect_ratio != 0.0) and not args.find_missed_labels and not args.refine_boxes
    if args.consensus and (args.exact_dup or args.find_similar is not None or args.find_duplicates is not None or args.subset or args.find_missed_labels or args.classes or outlier_branch or args.consensus_revert or args.sam3_refine or args.bioclip_classify or args.remap_classes):
        parser.error("--consensus cannot be combined with --exact_dup/--find_similar/--find_duplicates/--subset/--find_missed_labels/--classes/--clip_outliers (or the bbox and aspect-ratio flags)/--consensus_revert/--sam3_refine/--bioclip_classify/--remap_classes (its vocabulary is the ground_truth labels; the class filter and the class remap edit ground_truth before the snapshot; the outlier cleaning loop and the per-instance review loop both act on the same app selection; box-prompted SAM3 labels every prediction with the first class; BioCLIP would rewrite the labels the verdicts describe)")
    if args.consensus and not args.sam3_prompt: parser.error("--consensus requires --sam3_prompt")
    if args.sam3_adopt and (args.sam3_eval or args.consensus or args.sam3_overwrite or args.sam3_refine or args.exact_dup or args.find_similar is not None or args.find_duplicates is not None or args.subset or args.find_missed_labels or args.classes or outlier_branch or args.bioclip_classify or args.consensus_revert or args.refine_boxes or args.clean_overlap_on or args.remap_classes):
        parser.error("--sam3_adopt cannot be combined with --sam3_eval/--consensus/--sam3_overwrite/--sam3_refine/--exact_dup/--find_similar/--find_duplicates/--subset/--find_missed_labels/--classes/--clip_outliers (or the bbox and aspect-ratio flags)/--bioclip_classify/--consensus_revert/--refine_boxes/--clean_overlap_on/--remap_classes (--clean_overlap_on deletes every ground_truth label that overlaps a priority-class label above --pred_dedup_iou, keeping the priority class itself, with no queue row and no CSV line, and it runs before the adopt pass takes its snapshot; --refine_boxes rewrites live ground_truth geometry before the adopt pass reads it; consensus rewrites ground_truth from its own verdicts before the adopt pass reads it; --sam3_overwrite and --sam3_refine make SAM3 write ground_truth itself, and box-prompted SAM3 labels every prediction with the first class; the duplicate and similarity reviews run their own app pass and exit before the adopt pass, so adopt would silently never run; the class filter and the class remap edit ground_truth before the snapshot; the outlier cleaning loop and the per-instance review loop both act on the same app selection; BioCLIP would rewrite the labels the adopt verdicts describe; --consensus_revert only restores and exits)")
    # SAM3 flags are checked in sam3_adopt_workflow: reopening a review never runs SAM3
    if args.yolo_judge and (args.sam3_prompt or args.sam3_adopt or args.sam3_eval or args.sam3_overwrite or args.sam3_refine or args.refine_boxes or args.min_label_area or args.clean_overlap_on or outlier_branch or args.find_missed_labels or args.bioclip_classify):
        parser.error("--yolo_judge cannot be combined with --sam3_prompt/--sam3_adopt/--consensus/--sam3_eval/--sam3_overwrite/--sam3_refine/--refine_boxes/--min_label_area/--clean_overlap_on/--clip_outliers (or the bbox and aspect-ratio flags)/--find_missed_labels/--bioclip_classify. --yolo_judge IS the eval pass with YOLO as the proposer, so --sam3_eval would score twice; --min_label_area deletes ground_truth labels under its floor; each of the rest writes ground_truth itself, and the outlier loop and the review loop act on the same app selection")
    if args.sam3_add_only:
        if not args.sam3_adopt: parser.error("--sam3_add_only only applies to --sam3_adopt")
        if args.tick_wipes: parser.error("--sam3_add_only and --tick_wipes contradict each other: add-only promises no annotation of yours leaves the data, and a wipe deletes them from ground_truth_orig as well")
        if args.reevaluate: parser.error("--sam3_add_only with --reevaluate votes against ground_truth as it stands now, so SAM3's own earlier additions become the annotations it must not touch. Rebuild with --force_reload instead")
    if args.sam3_score is None:   # the floor SAM3 runs at, picked once so every reader sees it
        _cd, _dt = parse_class_thresholds(args.auto_approve_threshold); args.sam3_score = _dt if _dt is not None else min(_cd.values()) if _cd else 0.25
    if args.sam3_add_score is not None:
        if not args.sam3_adopt: parser.error("--sam3_add_score only applies to --sam3_adopt (it gates which SAM3 masks may ADD an object; every other SAM3 mode has one job and one threshold)")
        if args.sam3_add_score <= args.sam3_score:
            parser.error(f"--sam3_add_score {args.sam3_add_score} is at or below --sam3_score {args.sam3_score}, so it can never suppress anything: SAM3 never emitted a mask below {args.sam3_score}. Raise it, or drop the flag.")
    # here, before any SAM3 subprocess runs
    if args.sam3_hierarchy:
        _h = {}
        for x in args.sam3_hierarchy:
            if ":" not in x: parser.error(f"--sam3_hierarchy takes child:parent pairs; got '{x}'")
            k, v = x.split(":", 1); _h[_norm(k)] = _norm(v)
        if (nested := sorted(set(_h.values()) & set(_h))):
            parser.error(f"--sam3_hierarchy is one level deep; {nested} appear as both child and parent")
    if args.resplit and len(args.resplit) != 3:
        parser.error(f"--resplit takes exactly 3 ratios or none; got {len(args.resplit)}: {args.resplit}")
    if args.resplit and abs(sum(args.resplit) - 1.0) >= 1e-6:
        parser.error(f"--resplit ratios must sum to 1.0; got {args.resplit} (sum={sum(args.resplit)})")
    if not args.sam3_adopt:
        for _f in ("review_reset", "review_origin", "review_pair", "review_cap", "dedup_contained"):
            if _f == "review_cap" and (args.to_valid_confusion or args.to_valid_small or args.border_to_valid or args.overlap_to_valid is not None): continue   # Author 2026-10-10: the author cap on the judged screens, never a 300 of mine
            if getattr(args, _f): parser.error(f"--{_f} only affects the --sam3_adopt review queue" + (" and the four judged --to_valid screens" if _f == "review_cap" else "") + " and does nothing on any other pass. Add --sam3_adopt.")
        if args.review_files and not (args.to_valid or args.overlap_to_valid is not None): parser.error("--review_files narrows the --sam3_adopt queue or a --to_valid screen and does nothing on any other pass")
    if args.review_files is not None:
        try: args.review_files = {l.strip() for l in open(args.review_files, encoding="utf-8-sig") if l.strip()}
        except (OSError, UnicodeDecodeError) as e: parser.error(f"--review_files: {e}")
        # an empty set is falsy and would serve the whole queue
        if not args.review_files: parser.error("--review_files names no image")
    # never alone: a closed data-loss path
    if args.review_reset and not (args.review_reason or args.review_pair):
        parser.error("--review_reset with no --review_reason clears the reviewed tag for EVERY pass on the dataset "
                     "at once, so every pass you have finished comes back on screen and nothing records which "
                     "ones they were. Name the pass you want back: --review_reset --review_reason unsupported.")
    if args.dedup_contained and args.output_dir:
        parser.error("--dedup_contained cannot be combined with --output_dir: --output_dir names a SECOND cloned dataset, not an export path, so the repair would land on the clone and every later run would reload it. Repair first, export after.")
    if args.own_dir and len(args.local_yolo_dirs or []) != 1:
        parser.error("--own_dir restricts the review and the export to one input dir; pass exactly one --local_yolo_dirs")
    if args.review_origin and args.review_pair:
        parser.error("--review_origin and --review_pair both scope the same queue; --review_pair is one direction, --review_origin is every direction out of that class. Pass one.")
    if args.tick_wipes and (args.review_reason != "more" or not args.sam3_adopt):
        parser.error("--tick_wipes needs --sam3_adopt and --review_reason more: it changes what a tick means, and only the `more` tick shows SAM3 additions over your annotations")
    if args.tick_wipes and (_bad := [c for c in args.tick_wipes if _norm(c) not in {_cls_of(x) for x in (args.sam3_prompt or [])}]):
        parser.error(f"--tick_wipes {' '.join(_bad)}: every class must also be in --sam3_prompt ({' '.join(args.sam3_prompt or []) or 'none on this line'}), or a tick deletes annotations this pass never looked at")
    if (args.review_origin or args.review_pair) and args.review_reason not in SHEET_REASONS:
        parser.error(f"--review_origin/--review_pair need --review_reason from {sorted(SHEET_REASONS)}; every other reason is sample-level and names no label to scope by")
    # --consensus carries the rename and classchange passes
    if args.review_reason and not (args.sam3_adopt or args.consensus):
        parser.error("--review_reason needs --sam3_adopt, or --consensus for the rename and classchange passes: the queue, the review sheet, --review_reset and every click and tick handler sit behind one of those two flags, so with neither the reason writes no label at all and only rewrites the saved app layers and leaves the drawing unfenced")
    if args.consensus and args.review_reason not in (None, "rename", "classchange"):
        parser.error(f"--consensus carries only the rename and classchange passes; --review_reason {args.review_reason} would open a queue nothing on this pass fills (W5)")
    if (args.to_valid or args.overlap_to_valid is not None or args.find_redundant is not None) and (args.sam3_adopt or args.consensus or args.output_dir):
        parser.error("--to_valid/--overlap_to_valid/--find_redundant run their own screen and exit: --sam3_adopt/--consensus would never run, and under --output_dir the move lands on the clone while the files leave the input bucket (W2)")
    if args.find_redundant is not None:
        try: _iou, _ham = map(float, args.find_redundant.split(":"))
        except ValueError: parser.error(f"--find_redundant wants IOU:HAM, e.g. 0.7:10; got '{args.find_redundant}' (W81)")
    if args.review_pair and not all(args.review_pair.partition(":")[::2]):
        parser.error(f"--review_pair wants WAS:NOW, e.g. person:dog; got '{args.review_pair}' (W84)")
    if args.remap_classes and (_bad := [x for x in args.remap_classes if x.count(":") != 1 or not all(x.split(":"))]):
        parser.error(f"--remap_classes wants old:new pairs; got {' '.join(_bad)} (W84)")
    if args.review_reset and args.no_launch:
        parser.error("--review_reset runs only when the app opens, so under --no_launch it resets nothing; drop --no_launch (W83)")
    if args.review_files and (args.no_launch or args.dedup_contained):
        parser.error("--review_files narrows only the queue the app opens, so under --no_launch it narrows nothing and --dedup_contained still repairs the whole set; drop --no_launch, or run --dedup_contained on its own")
    if args.review_cap is not None and args.review_cap < 1:
        parser.error(f"--review_cap wants 1 or more; {args.review_cap} would show the whole queue (W85)")
    if args.consensus_revert and args.output_dir:
        parser.error("--consensus_revert with --output_dir reverts the clone, never the dataset you reviewed; drop --output_dir (W85)")
    # argparse accepts a child flag alone and it does nothing
    for _c, _ps in (("keep_tn", ("find_missed_labels",)), ("sam3_adopt_fill", ("sam3_adopt",)), ("consensus_sam3_add", ("consensus",)),
                    ("allow_revert_reviewed", ("consensus_revert",)), ("sam3_hierarchy", ("sam3_adopt", "consensus", "sam3_eval", "yolo_judge"))):
        if getattr(args, _c) not in (None, False) and not any(getattr(args, x) for x in _ps):
            parser.error(f"--{_c} does nothing without --{' or --'.join(_ps)}")
    # the export rmtrees output_dir and every derived path; _refuse_overwrite re-checks at the rmtree
    _dirs = (args.local_yolo_dirs or []) + (args.local_coco_dirs or [])
    _ins = [os.path.realpath(d) for d in _dirs]
    if args.output_dir:
        for _o in (args.output_dir, f"{args.output_dir}1", f"{args.output_dir}2", f"{args.output_dir.rstrip('/')}-clean",
                   f"{args.output_dir.rstrip('/')}-fp", *(f"{args.output_dir.rstrip('/')}-{s}" for s in ("train", "valid", "test"))):
            _out = os.path.realpath(_o)
            for _d, _in in zip(_dirs, _ins):
                if _overlaps(_out, _in):
                    parser.error(f"--output_dir would write {_o}, which overlaps the input dir {_d}; the export deletes that path first and the images it is still reading live under it")
    global _INPUT_DIRS
    _INPUT_DIRS = _ins
    return args

_norm = lambda s: s.lower().replace(' ', '')

def _sast(t):
    """A FiftyOne UTC stamp printed in SAST. Compare in UTC, print here."""
    if not t: return "unknown"
    if t.tzinfo is None: t = t.replace(tzinfo=datetime.timezone.utc)
    return t.astimezone(datetime.timezone(datetime.timedelta(hours=2))).strftime("%Y-%m-%d %H:%M SAST")

def _norm_classes(class_names, source):
    """Normalising collapses 'Big Cat' and 'big cat' into one class and export's .index() then merges them."""
    out, seen = [], {}
    for c in class_names:
        out.append(n := _norm(c))
        if seen.setdefault(n, c) != c: raise SystemExit(f"class name collision in {source}: '{seen[n]}' and '{c}' both normalise to '{n}'. Rename one in the class list.")
    return out

def _yaml_names(names):
    """A data.yaml names list; the dict form {0: 'bird'} yields int keys, so it is read in key order."""
    return [names[k] for k in sorted(names)] if isinstance(names, dict) else names

# a SAM3 prompt's class: "a penguin" -> penguin
_cls_of = lambda p: _norm(next((p[len(a):].strip() for a in ("a ", "an ", "the ") if p.lower().startswith(a)), p) or "object")

def parse_class_thresholds(values):
    """Parse auto_approve_threshold into (class_dict, default_threshold)."""
    thresholds, default = {}, None
    for v in values or ():
        cls, _, thresh = v.rpartition(':')
        if ':' in cls: sys.exit(f"--auto_approve_threshold: '{v}' has more than one colon; use class:number")
        try: t = float(thresh)
        except ValueError: sys.exit(f"--auto_approve_threshold: '{v}' is not a number or class:number pair")
        if cls: thresholds[_norm(cls)] = t
        elif default is not None: sys.exit(f"--auto_approve_threshold: two bare thresholds ({default} and {t}); one default only, the rest as class:number")
        else: default = t
    return thresholds, default

def _resolve_split_path(dataset_dir, split_value):
    """Resolve a data.yaml split against the dataset dir. Roboflow ships 'train: ../train/images', which
    escapes the root and loads empty, so leading '../' and './' are stripped and retried."""
    if os.path.isabs(split_value): return split_value
    direct = os.path.join(dataset_dir, split_value)
    if os.path.isdir(direct): return direct
    stripped = split_value
    while stripped.startswith(('../', './')):
        stripped = stripped.split('/', 1)[1]
    candidate = os.path.join(dataset_dir, stripped)
    if stripped != split_value and os.path.isdir(candidate):
        print(f"  Split '{split_value}' escapes the dataset root; using '{stripped}' instead.")
        return candidate
    return direct

SPLIT_MAPPING = defaultdict(lambda: 'validation', {'train': 'train', 'val': 'validation', 'valid': 'validation', 'validation': 'validation', 'test': 'test'})

IMG_EXTS = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']   # the importer's and SAM3's, one list

def import_yolo_dataset(dataset_dir, label_field="ground_truth", max_samples=None):
    yaml_path = next((os.path.join(dataset_dir, f) for f in ["data.yaml", "dataset.yaml"] if os.path.exists(os.path.join(dataset_dir, f))), None)
    if not yaml_path: raise FileNotFoundError(f"no data.yaml or dataset.yaml in {dataset_dir}")
    config = yaml.safe_load(open(yaml_path, 'r'))
    # the merge builder's per-stem hashes of COCO's own lines, stamped coco=True
    _op = os.path.join(dataset_dir, "COCO_ORIGIN.tsv")
    coco_lines = {st: Counter(h.split()) for st, _, h in (l.rstrip("\n").partition("\t") for l in open(_op))} if os.path.isfile(_op) else {}
    n_badcls = n_coco = 0; by_label = defaultdict(set)
    class_names = _yaml_names(config['names'])
    dataset = fo.Dataset()
    for split_key in (k for k in config if k in SPLIT_MAPPING):
        split_path = _resolve_split_path(dataset_dir, config[split_key])
        imgs = sum((glob.glob(os.path.join(split_path, f'*.{ext}')) + glob.glob(os.path.join(split_path, f'*.{ext.upper()}')) for ext in IMG_EXTS), [])[:max_samples or None]
        samples = []
        for img_path in imgs:
            # the LAST '/images/' only: a dataset root may itself contain one
            label_path = '/labels/'.join(img_path.rsplit('/images/', 1)).rsplit('.', 1)[0] + '.txt'; by_label[label_path].add(img_path)
            sample = fo.Sample(filepath=img_path)
            sample.tags.append(SPLIT_MAPPING[split_key])
            polylines, boxes = [], []
            own = coco_lines.get(os.path.splitext(os.path.basename(img_path))[0], Counter())
            if os.path.exists(label_path):
                for line in open(label_path, 'r'):
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    if not 0 <= (class_idx := int(parts[0])) < len(class_names): n_badcls += 1; continue
                    coords, coco = [float(x) for x in parts[1:]], own[_h := hashlib.md5(line.strip().encode()).hexdigest()[:10]] > 0
                    own[_h] -= coco   # a multiset: his rename onto COCO's points and class takes no second stamp
                    n_coco += coco
                    if len(coords) == 4:
                        x_c, y_c, w, h = coords
                        boxes.append(fo.Polyline(label=_norm(class_names[class_idx]), points=[bbox_to_polyline_points([x_c - w/2, y_c - h/2, w, h])], filled=True, closed=True, **({"coco": True} if coco else {})))
                    elif len(coords) >= 6 and len(coords) % 2 == 0:
                        points = [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]
                        polylines.append(fo.Polyline(label=_norm(class_names[class_idx]), points=[points], filled=True, closed=True, **({"coco": True} if coco else {})))
            # boxes as Polylines too, after the rings: one field type must hold across the whole dataset
            if polylines := polylines + boxes: sample[label_field] = fo.Polylines(polylines=polylines)
            samples.append(sample)
        dataset.add_samples(samples)
    # the export would drop the line again, and two images of one stem read one label file
    if n_badcls: sys.exit(f"{yaml_path}: {n_badcls} label lines carry a class number outside its {len(class_names)} names. Nothing loaded: an export would drop them. Fix the lines or the yaml.")
    if _two := [v for v in by_label.values() if len(v) > 1]: sys.exit(f"{dataset_dir}: {len(_two)} label files are read by two images, e.g. {sorted(_two[0])}. Nothing loaded: rename one image of each pair.")
    if coco_lines: print(f"  COCO_ORIGIN.tsv: {n_coco} labels stamped coco")
    return dataset, _norm_classes(class_names, 'the YOLO class list')

COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def _discover_coco_sources(dataset_dir):
    """[(json_path, img_root, split_tag)]: one for a flat dir, else one per subfolder holding _annotations.coco.json."""
    flat = os.path.join(dataset_dir, "_annotations.coco.json")
    if os.path.exists(flat): return [(flat, dataset_dir, 'train')]
    return [(jp, sub, SPLIT_MAPPING.get(name.lower(), 'train')) for name in sorted(os.listdir(dataset_dir))
            if os.path.isdir(sub := os.path.join(dataset_dir, name)) and os.path.exists(jp := os.path.join(sub, "_annotations.coco.json"))]

def _coco_cats(data):
    """[(id, normalised name)] in id order, skipping Roboflow's dummy background at id 0."""
    return [(c['id'], _norm(c['name'])) for c in sorted(data.get('categories', []), key=lambda c: c['id']) if c['id'] != 0]

SIMPLIFY_EPS = 0.005   # the authors' ruling. Fraction of a ring's own perimeter, Douglas-Peucker.
SPECK_MIN_AREA = 256   # the authors' ruling. Contour area px. A ring must fail BOTH this and SPECK_MIN_FRAC to be dropped.
SPECK_MIN_FRAC = 0.01  # ...of the label's own largest ring, so a real small part of a small object survives.

def _drop_specks(rings, W, H):
    """A 2-30 px contour speck makes the export bridge draw a line across the frame and inflates the box.
    Both tests, so a real small part of a small object survives."""
    if not W or len(rings) <= 1: return rings
    a = [abs(cv2.contourArea((np.array(r, np.float64) * (W, H)).astype(np.float32))) for r in rings]
    return [r for r, q in zip(rings, a) if not (q < SPECK_MIN_AREA and q < SPECK_MIN_FRAC * max(a))]

def import_coco_dataset(dataset_dir, max_samples=None, stamp_coco=False):
    """stamp_coco marks every label as COCO ground truth: True only where the json IS human ground truth
    (--local_coco_dirs), never for SAM3's output, and never for this script's own COCO export."""
    import json
    sources = _discover_coco_sources(dataset_dir)
    if not sources:
        raise FileNotFoundError(f"No _annotations.coco.json found in '{dataset_dir}' or its immediate subfolders.")
    id_to_idx, class_names = {}, []
    dataset, samples = fo.Dataset(), []
    for json_path, img_root, split_tag in sources:
        data = json.load(open(json_path, 'r'))
        # export_coco_dataset writes this key: post-review, it holds SAM3's adoptions
        _stamp = stamp_coco and not data.get("info", {}).get("fyal_export")
        # COCO's own parts never go, specks included, as in the export; SAM3's re-import and our own export keep the speck rule
        _specks = (lambda rings, W, H: rings) if _stamp else _drop_specks
        if stamp_coco and not _stamp:
            print(f"  NOT stamping as COCO ground truth: '{json_path}' was written by this script's own "
                  f"COCO exporter, so its labels are post-review, not human annotations.")
        for cid, name in _coco_cats(data):   # merged across splits
            if name not in class_names: class_names.append(name)
            id_to_idx[(json_path, cid)] = class_names.index(name)
        images_by_id = {img['id']: img for img in data.get('images', [])}
        anns_by_image = defaultdict(list)
        for ann in data.get('annotations', []):
            if ann.get('category_id', 0) == 0: continue
            anns_by_image[ann['image_id']].append(ann)
        image_ids = list(images_by_id)[:max_samples or None]   # max_samples is per split
        n_gone = n_gone_lbl = 0
        for img_id in image_ids:
            img = images_by_id[img_id]
            img_path = os.path.join(img_root, img['file_name'])
            if not os.path.exists(img_path): n_gone += 1; n_gone_lbl += len(anns_by_image.get(img_id, [])); continue
            W, H = float(img['width']), float(img['height'])
            sample = fo.Sample(filepath=img_path)
            sample.tags.append(split_tag)
            detections, polylines = [], []
            for ann in anns_by_image.get(img_id, []):
                cls_idx = id_to_idx.get((json_path, ann['category_id']))
                if cls_idx is None: continue
                label = class_names[cls_idx]
                seg = ann.get('segmentation')
                rings = []
                # one annotation is ONE object however many rings or blobs it has
                if isinstance(seg, list) and seg:
                    rings = _specks([[[poly[i]/W, poly[i+1]/H] for i in range(0, len(poly), 2)] for poly in seg
                                     if isinstance(poly, list) and len(poly) >= 6 and len(poly) % 2 == 0], W, H)
                elif isinstance(seg, dict) and 'counts' in seg and 'size' in seg:
                    from pycocotools import mask as maskUtils
                    rle = maskUtils.frPyObjects(seg, seg['size'][0], seg['size'][1]) if isinstance(seg['counts'], list) else seg
                    contours, _ = cv2.findContours(maskUtils.decode(rle), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
                    if contours:   # TC89_KCOS never smooths, so the eps here takes the wriggle off
                        # a COCO outline the eps flattens stays as traced, as the export keeps it as drawn
                        contours = [_ac if len(_ac := cv2.approxPolyDP(c, SIMPLIFY_EPS * cv2.arcLength(c, True), True)) >= 3 or not _stamp else c for c in contours]
                        rings = _specks([[[float(pt[0][0])/W, float(pt[0][1])/H] for pt in c] for c in contours if len(c) >= 3], W, H)
                if rings:
                    polylines.append(fo.Polyline(label=label, points=rings, filled=True, closed=True, confidence=ann.get("score")))
                elif ann.get('bbox'):
                    x, y, w, h = ann['bbox']
                    detections.append(fo.Detection(label=label, bounding_box=[x/W, y/H, w/W, h/H], confidence=ann.get("score")))
            polylines.extend([fo.Polyline(label=det.label, points=[bbox_to_polyline_points(det.bounding_box)], filled=True, closed=True, confidence=det.confidence) for det in detections])
            # the COCO mark originates here, the only place that knows the origin
            if _stamp:
                for _p in polylines: _p.coco = True
            if polylines: sample["ground_truth"] = fo.Polylines(polylines=polylines)
            samples.append(sample)
        if n_gone: print(f"  Skipped {n_gone} of {len(image_ids)} image(s) in '{json_path}' whose file is gone, with {n_gone_lbl} COCO label(s)")   # W77
    dataset.add_samples(samples)
    return dataset, class_names

def get_class_configuration(args):
    if args.local_yolo_dirs and not any([args.dataset_name, args.classes, args.all_coco_classes]):
        for f in ["data.yaml", "dataset.yaml"]:
            if os.path.exists(yaml_path := os.path.join(args.local_yolo_dirs[0], f)):
                config = yaml.safe_load(open(yaml_path, 'r'))
                if 'names' in config:
                    lower_names = _norm_classes(_yaml_names(config['names']), yaml_path)
                    return lower_names, lower_names, True
    if args.local_coco_dirs and not any([args.dataset_name, args.classes, args.all_coco_classes, args.local_yolo_dirs]):
        import json
        names = []
        for coco_dir in args.local_coco_dirs:
            for json_path, _, _ in _discover_coco_sources(coco_dir):
                for _, name in _coco_cats(json.load(open(json_path, 'r'))):
                    if name not in names: names.append(name)
        if names:
            return names, names, True
    # every stored label goes through _norm, so the class list must too ('traffic light' -> 'trafficlight')
    if args.all_coco_classes or args.classes:
        lower_classes = [_norm(c) for c in (COCO_CLASSES if args.all_coco_classes else args.classes)]
        return lower_classes, lower_classes, True
    default_classes = ["bird", "cow", "bus", "car", "cat", "dog", "horse", "motorcycle", "person", "truck"]
    return default_classes, [_norm(c) for c in COCO_CLASSES if _norm(c) in default_classes], False

_box = lambda l: l.to_detection().bounding_box if isinstance(l, fo.Polyline) else l.bounding_box

def compute_iou(box1, box2):
    x1, y1, w1, h1, x2, y2, w2, h2 = *box1, *box2
    inter_area = max(0, min(x1+w1, x2+w2) - max(x1, x2)) * max(0, min(y1+h1, y2+h2) - max(y1, y2))
    return inter_area / (w1*h1 + w2*h2 - inter_area) if inter_area > 0 else 0.0

def _dhash(grey):
    g = cv2.resize(grey, (9, 8), interpolation=cv2.INTER_AREA)
    return np.packbits(g[:, 1:] > g[:, :-1])

def _taken(dataset, ids, to):
    """The frames whose name <to>/ already holds, image or label, a file or a link to nothing. A move leaves them."""
    out = set()
    for i, img in zip(*dataset.select(list(ids)).values(["id", "filepath"])):
        b, _, _, n = img.rsplit(os.sep, 3)
        if any(os.path.lexists(dst) for dst in (os.path.join(b, to, "images", n), os.path.join(b, to, "labels", os.path.splitext(n)[0] + ".txt"))): out.add(i)   # counted in the prompt, held by the move (MV1)
    return out

def _move_to_valid(dataset, ids, args, to="valid", what="ticked"):
    """Move samples from <bucket>/train/ to <bucket>/<to>/ on disk, image and label, and retag them validation. Nothing deleted.
    to="train" takes them back from valid/, retagged train (--overlap_to_valid).
    to="archived/dedup_<date>" takes a frame from either split: the duplicate screens' tick, which then drops the sample."""
    if not _bare_bucket_argv(sys.argv[1:]):   # a clone or an a+b set would move files his bucket's own dataset still points at
        print("  Nothing moved: the line is not one --local_yolo_dirs bucket and its screen flags."); return 0
    n, nolbl, m = 0, 0, len(ids)
    frm = {"valid": "train", "train": "valid"}.get(to)   # the split a screen moves out of
    old, new = ("validation", "train") if to == "train" else ("train", "validation")
    if frm and (stay := _unsaved(dataset, ids, args)):   # a file difference is checked here, not remembered, so a relaunch sees it too; a new off-yaml label only in its own pass. An archived duplicate leaves the set, its file with it
        off, nm = _off_yaml_new(dataset, stay, args), lambda xs: ', '.join(sorted(os.path.basename(f) for f in dataset.select(list(xs)).values('filepath')))
        if rest := stay - set(off): print(f"  {len(rest)} {what} frames not moved, their label file does not match FiftyOne: {nm(rest)}. A full export of this bucket levels the two, ask me for it")
        if off: print(f"  {len(off)} {what} frames not moved, a label there has a class not in data.yaml ({', '.join(sorted({repr(p.label) for v in off.values() for p in v}))}): {nm(off)}. Delete it or give it a data.yaml class in the Annotate menu, then {'tick again' if what == 'ticked' else 'run the screen again'}")
        ids = [i for i in ids if i not in stay]
    for s in dataset.select(ids).iter_samples(autosave=True):
        img = s.filepath; split = os.path.dirname(os.path.dirname(img)); bucket = os.path.dirname(split)
        if (frm and os.path.basename(split) != frm) or not os.path.isfile(img): print(f"  not a file under {frm or 'train'}/, left where it is: {img}"); continue
        lbl = os.path.join(split, "labels", os.path.splitext(os.path.basename(img))[0] + ".txt")
        mv = [(src, sub, os.path.join(bucket, to, sub, os.path.basename(src))) for src, sub in ((img, "images"), (lbl, "labels"))]
        if any(os.path.lexists(dst) for *_, dst in mv): print(f"  {to}/ already holds a frame of this name, left where it is: {img}"); continue   # a move would overwrite it (MV1)
        for src, sub, dst in mv:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.isfile(src): shutil.move(src, dst)
            elif sub == "labels": nolbl += 1
        s.filepath = os.path.join(bucket, to, "images", os.path.basename(img))
        if s.has_field("original_filepath") and s["original_filepath"] == img: s["original_filepath"] = s.filepath   # the exporter reads it first
        s.tags = [t for t in s.tags if t not in (old, new)] + [new]; n += 1
    dataset.save(); print(f"  moved {n} of {m} frames to {to}/" + (f", tagged {new}" if frm else "") + (f", {nolbl} had no label file" if nolbl else ""))
    return n

# sheet boxes that argue a NAME, not the object: a click keeps the object on every screen (lowiou's box deletes)
CANCEL_KINDS = {"classchange", "loose", "rename"}

def _sheet_hits(sample, clicked):
    """{live ground_truth id: box kind} under his clicked review_sheet boxes. A box over nothing live is not in it."""
    gt = sample["ground_truth"]; live = {p.id for p in gt.polylines} if gt else set()
    boxes = sample["review_sheet"].detections if sample.has_field("review_sheet") and sample["review_sheet"] else []
    return {q.gt_id: getattr(q, "kind", None) for q in boxes if q.id in clicked and getattr(q, "gt_id", None) in live}

def _unspared_sheet_samples(dataset, picked):
    """Samples where a clicked review_sheet box deletes a live label; any other box click keeps its frame off the disk plan."""
    by = defaultdict(set)
    for l in picked:
        if l["field"] == "review_sheet": by[l["sample_id"]].add(l["label_id"])
    if not by: return set()
    return {s.id for s in dataset.select(list(by)) if any(k not in CANCEL_KINDS for k in _sheet_hits(s, by[s.id]).values())}

def _orig_under(p, obys, oboxs, W, H, args):
    """Index of his label that adopt's mask p drew over, read the way adopt claimed it, or None. alt_name_prev is
    the class before a rename click renamed it, or 'animal' stops matching his 'dog' and a click deletes, not restores."""
    ag = _agree(_hier(args), _norm(p.label), _norm(getattr(p, "alt_name_prev", None) or p.label))
    b, iou, _ = _best_overlap(p, p.to_detection().bounding_box, obys, oboxs, W, H, boundary=False, prefer=ag, prefer_floor=args.adopt_match_iou)
    return b if b is not None and iou >= args.adopt_match_iou and ag(_norm(obys[b].label)) else None

def _delete_picked(dataset, picked, ticked=(), args=None):
    """No pass is read-only. A clicked ground_truth label is deleted; a ticked image is wiped except a COCO label of a
    class the build never prompted, which only his click removes. A clicked review_sheet box deletes the label under it
    (gt_id) and goes; a CANCEL_KINDS box goes but keeps its label (rename/loose: name answered, classchange: nothing changes)."""
    by, sheet, other = defaultdict(set), defaultdict(set), Counter()
    for l in picked:
        if l["field"] == "ground_truth": by[l["sample_id"]].add(l["label_id"])
        elif l["field"] == "review_sheet": sheet[l["sample_id"]].add(l["label_id"])
        else: other[l["field"]] += 1
    keepcls, n_del, n_img, n_wipe, n_kept, n_box, n_live, n_ans, n_cc, n_res = {_cls_of(p) for p in dataset.info.get("sam3_prompts") or ()}, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for sample in dataset.select(list(set(by) | set(sheet) | set(ticked))).iter_samples(autosave=True):
        gt = sample["ground_truth"]
        if sample.id in sheet:
            boxes = list(sample["review_sheet"].detections) if sample.has_field("review_sheet") and sample["review_sheet"] else []
            res = _sheet_hits(sample, sheet[sample.id])
            # a naming box over a label he also clicked counts with the deletes: his click on the label wins
            hit = {g for g, k in res.items() if k not in CANCEL_KINDS or g in by[sample.id]}
            ans = {g for g, k in res.items() if k in ("rename", "loose")} - hit
            cc = {g for g, k in res.items() if k == "classchange"} - hit
            # a 'replaced' box hands his label back, as the more screen does for the same click
            rep = {g for g, k in res.items() if k == "replaced"} - by[sample.id] if args else set()
            origs = list(sample["ground_truth_orig"].polylines) if rep and sample.has_field("ground_truth_orig") and sample["ground_truth_orig"] else []
            obys, oboxs = _poly_boxes(origs); W, H = _wh(sample)
            for p in (gt.polylines if origs and W else []):
                if p.id not in rep or getattr(p, "adopt", None) != "adopted": continue
                b = _orig_under(p, obys, oboxs, W, H, args)
                if b is None or any(o.points == obys[b].points for o in gt.polylines if o is not p): continue   # p itself at his exact points is no copy
                q = obys[b]; p.points = [list(r) for r in q.points]; p.label, p.adopt = q.label, "restored"; hit.discard(p.id); n_res += 1
                if getattr(q, "coco", False): p.coco = True
            for p in (gt.polylines if ans else []):
                if p.id in ans: p.alt_name_done = True; p.clear_field("alt_name")
            by[sample.id] |= hit; n_box += len(sheet[sample.id]); n_live += len(hit); n_ans += len(ans); n_cc += len(cc)
            if boxes: sample["review_sheet"] = fo.Detections(detections=[q for q in boxes if q.id not in sheet[sample.id]])
        if not gt: continue
        clicked, tick = by.get(sample.id, ()), sample.id in ticked
        n_hit = sum(p.id in clicked for p in gt.polylines); n_del += n_hit; n_img += n_hit > 0
        keep = [p for p in gt.polylines if p.id not in clicked and (not tick or getattr(p, "coco", False) and _norm(p.label) not in keepcls)]   # prompt and label both normalised
        if tick: n_wipe += len(gt.polylines) - n_hit - len(keep); n_kept += len(keep)
        sample["ground_truth"] = fo.Polylines(polylines=keep)
    if ticked: print(f"Wiped {n_wipe} labels on {len(ticked)} ticked images; {n_kept} COCO labels of a class this build never prompted kept.")
    if n_box: print(f"{n_box} review_sheet boxes clicked: {n_live} sat on a live ground_truth label, deleted with it; {n_box - n_live - n_ans - n_cc - n_res} sat on nothing live, box removed."
                    + (f" {n_res} put your own label back under SAM3's replacement." if n_res else "")
                    + (f" {n_ans} asked about a name: name kept, not asked again." if n_ans else "")
                    + (f" {n_cc} asked about a class change: nothing changed, answer it on the classchange screen." if n_cc else ""))
    if (n_del or n_wipe) and dataset.has_sample_field("ground_truth_orig"):   # the revert guard reads this: these screens tag nothing
        dataset.info["screen_clicks"] = dataset.info.get("screen_clicks", 0) + n_del + n_wipe; dataset.save()
    print(f"Deleted {n_del} ground_truth labels across {n_img} images."
          + (" Ignored " + ", ".join(f"{n} on {f}" for f, n in sorted(other.items())) + ": only a ground_truth label is yours to delete here." if other else ""))

def _build_lock(name):
    """One disk build per dataset name, held for the build only, never across his review screen. The --output_dir
    clone is not covered. Named by hash: a long a+b+c name is no file name."""
    f = open(os.path.join(tempfile.gettempdir(), f"fyal_build_{hashlib.md5(name.encode()).hexdigest()}.lock"), "w")
    try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"Another launch is already building '{name}'. Waiting for it instead of building it twice.", flush=True)
        fcntl.flock(f, fcntl.LOCK_EX)
    return f

# the flags a click-and-delete screen opens with, its --review_files list too. Anything else on the line and the disk stays as it was.
_WRITE_BACK_SCREENS = {"--find_redundant", "--find_duplicates", "--find_similar", "--exact_dup", "--similar_model", "--to_valid", "--to_valid_confusion", "--to_valid_small", "--overlap_to_valid", "--border_to_valid", "--review_files", "--review_cap"}
_EDITED = {}   # {sample id: its ground_truth polylines as the last pass opened} for each frame he changed in the app's editor, set by _wait_or_enter
TARGET = ["animal", "bike", "dog", "person", "vehicle"]   # sama-merge-5's classes in order; its builder asserts these two match its own
# At merge time only, by NAME, never by index.
FOLD = {"cat": "animal", "bird": "animal", "sheep": "animal", "cow": "animal",
        "horse": "animal", "zebra": "animal", "donkey": "animal",
        "car": "vehicle", "truck": "vehicle", "bus": "vehicle",
        "bicycle": "bike", "motorcycle": "bike"}
_RIDE = {"bike", *(k for k, v in FOLD.items() if v == "bike")}

def _rider(a, b):
    """A person and a bike on one spot is a rider and his bike, two objects: never one name twice, never a class change. Author 2026-10-03:
    "Bike -> Person is extremely problematic and should probably just be left out in the pass i completed and future ones"."""
    return {_norm(a or ""), _norm(b or "")} in ({"person", r} for r in _RIDE)

def _bare_bucket_argv(argv):
    """True when the line is one --local_yolo_dirs <bucket> plus screen flags and their values, nothing else."""
    rest, i = [], 0
    while i < len(argv):
        a = argv[i]; i += 1
        if a.split("=", 1)[0] in _WRITE_BACK_SCREENS:   # --find_redundant=0.7:10 is the same line to argparse
            # swallows every value after the flag; argparse (0 positionals) refuses a stray token first
            while i < len(argv) and not argv[i].startswith("-"): i += 1
        else: rest.extend(a.split("=", 1))
    return len(rest) == 2 and rest[0] == "--local_yolo_dirs"

def _export_rings(poly, W, H, st):
    """One polyline's parts as the exporter writes them, or None: every ring at the SAM3 build's eps, COCO's too. A part the
    eps flattens to a line goes and is counted, but a flattened COCO part stays as drawn and no COCO speck goes.
    st counts vin, vout, flat, degenerate, specks."""
    rings = [r for r in (poly.points or []) if len(r) >= 3]
    coco = getattr(poly, "coco", False)
    if W and rings:
        st["vin"] += sum(map(len, rings))
        simp = [[(float(x) / W, float(y) / H) for x, y in cv2.approxPolyDP(_px, SIMPLIFY_EPS * cv2.arcLength(_px, True), True)[:, 0]]
                for r in rings for _px in [(np.array(r, np.float64) * (W, H)).astype(np.float32).reshape(-1, 1, 2)]]
        rings = [s if len(s) >= 3 else r for s, r in zip(simp, rings) if coco or len(s) >= 3]
        st["flat"] += len(simp) - len(rings); st["vout"] += sum(map(len, rings))
    if not rings: st["degenerate"] += 1; return None
    if not coco:
        keep = _drop_specks(rings, W, H); st["specks"] += len(rings) - len(keep); rings = keep
    return rings

def _export_pts(poly, W, H, st):
    """_export_rings as one YOLO line, or None."""
    if (rings := _export_rings(poly, W, H, st)) is None: return None
    if len(rings) == 1: return np.array(rings[0])
    st["merged"] += 1   # one YOLO line is one polygon, so bridge blobs the way ultralytics' converter does
    return np.concatenate(merge_multi_segment([np.array(r) for r in rings]), axis=0)

def _yolo_line(cls, pts):
    return f"{cls} {' '.join(map(str, np.asarray(pts).ravel()))}\n"

def _line_key(cls, pts):
    """A label line as (class id, vertex count, vertex set at 4 dp): a rotated ring or a reformatted float is the same label."""
    p = [(round(float(x), 4), round(float(y), 4)) for x, y in pts]
    return (str(cls), len(p), frozenset(p))

def _vk(q):
    """A polyline's vertex set at 5 dp: one shape, whatever vertex it starts on."""
    return frozenset(tuple(round(v, 5) for v in pt) for seg in q.points or () for pt in seg)

def _gt_name(gt, q):
    """The name the export holds for q's shape: q's own if a twin carries it, else its twin's. The screens draw this."""
    t = [g.label for g in gt if _vk(g) == _vk(q)]
    return q.label if not t or q.label in t else t[0]

def _file_keys(path):
    """[(line, key)] of a label file, a box line keyed as its four corners. [] when there is no file, None when it
    cannot be read or parsed: unreadable is not empty."""
    if not os.path.exists(path): return []   # a directory at the path is unreadable, not empty
    try:
        out = []
        for l in open(path):
            t = l.split()
            if not t: continue
            v = list(map(float, t[1:]))
            if len(v) == 4: cx, cy, w, h = v; v = [cx - w / 2, cy - h / 2, cx + w / 2, cy - h / 2, cx + w / 2, cy + h / 2, cx - w / 2, cy + h / 2]
            out.append((l.rstrip("\n") + "\n", _line_key(t[0], zip(v[::2], v[1::2]))))
        return out
    except Exception: return None

def _poly_keys(p, W, H, classes):
    """The keys a disk line of this label can carry: its points as imported, and as the exporter writes them."""
    if p.label not in classes: return set()
    c, pts = classes.index(p.label), _export_pts(p, W, H, Counter())
    return ({_line_key(c, p.points[0])} if len(p.points or []) == 1 else set()) | ({_line_key(c, pts)} if pts is not None else set())

def _write_back(dataset, picked, ticked, args):
    """A click deletes on disk too, only on an argv allowlist line; a frame whose file does not pair one to one with ground_truth
    is left alone. A kept label writes back its own line verbatim, a restored one as the exporter writes it. A frame changed in the
    editor pairs by what it held as the pass opened and is written as FiftyOne holds it now. No disk-side failure may cost a click."""
    ids = list({l["sample_id"] for l in picked if l["field"] == "ground_truth"} | _unspared_sheet_samples(dataset, picked) | set(ticked) | set(_EDITED))   # a sheet box click deletes the label under it; a naming box, or one over nothing live, leaves the file alone
    try: plan = _disk_plan(dataset, ids, args, _EDITED) if ids else None   # read before the delete
    except Exception as e: print(f"  disk plan failed, disk unchanged: {type(e).__name__}: {e}"); plan = None
    _delete_picked(dataset, picked, ticked, args)
    try: ok = _fold_or_add(dataset, args); classes = get_class_configuration(args)[1]   # after the delete: a label he clicked away, or a tick wiped, adds no class
    except Exception as e: print(f"  fold or add failed, no label file written: {type(e).__name__}: {e}"); ok = False
    if not (plan and ok): return
    n, bad, miss, thin = 0, [], Counter(), []
    for s in dataset.select(list(plan)).select_fields(["filepath", "metadata", "ground_truth"]):
        try:   # one frame per try: a bad file never stops the frames after it
            dst, lines, owner = plan[s.id]; W, H = _wh(s); out, st = [], Counter()
            if u := sorted({repr(p.label) for p in _polys(s, "ground_truth") if p.id in owner and p.label not in classes}):   # its line would go
                print(f"  given a class not in data.yaml ({', '.join(u)}), its line would go, left alone: {dst}"); continue
            for p in _polys(s, "ground_truth"):
                i, pts, lab = owner.get(p.id, (None, None, None))
                if p.label not in classes: miss[p.label] += 1
                elif i is not None and p.points == pts: out.append((i, lines[i] if p.label == lab else re.sub(r"^\s*\S+", str(classes.index(p.label)), lines[i], count=1)))   # a rename, in the editor or by a box: the class token alone, the points byte for byte
                elif (q := _export_pts(p, W, H, st)) is not None: out.append((len(lines) if i is None else i, _yolo_line(classes.index(p.label), q)))   # Author 2026-09-26, "follow the editor": a moved label keeps its line's place, a new one goes last
            if st["degenerate"]: thin.append(os.path.basename(dst))   # the exporter drops it too
            new = [l for _, l in sorted(out, key=lambda t: t[0])]
            if new != lines:
                try:   # a temp file beside it, then one rename: a failed write leaves the old file whole
                    with contextlib.suppress(FileNotFoundError): os.unlink(dst + ".tmp")   # a stale temp, a link above all, is never written through
                    with open(dst + ".tmp", "w") as f: f.writelines(new)   # closed here, so a failed flush is raised and named
                    os.replace(dst + ".tmp", dst)
                finally:
                    with contextlib.suppress(OSError): os.unlink(dst + ".tmp")   # a failed remove never hides the write's own error
                n += 1
        except Exception as e: bad.append(f"{os.path.basename(s.filepath)} ({type(e).__name__}: {e})")
    print(f"  {n} of {len(plan)} label files rewritten on disk" + (f". NOT on disk, class not in data.yaml: {dict(miss)}" if miss else "")
          + (f". NOT on disk, too thin for a line, redraw or delete it: {thin}" if thin else ""))
    if bad: print(f"  disk write FAILED on {len(bad)} frames. FiftyOne holds your edits there, the label file does not: {'; '.join(bad)}")

def _fold_or_add(dataset, args):
    """Author 2026-09-27, "fold else add, obviously keep the rename if I do it": a label he draws or renames in the pass with a class
    not in data.yaml keeps it. A sama-merge-5 set folds it as the merge does, cat to animal. A house set adds the class last in
    data.yaml, no old number moves. sama-merge-5 stays five classes (he picked the add for house sets only): a name there with no
    fold rule, or a blank one anywhere, is named and stays as typed with no line: a renamed one leaves its frame's file whole (the write loop).
    False when data.yaml could not take a class: no label file of the pass is written and every touched frame holds."""
    if not (off := _off_yaml_new(dataset, _EDITED, args)): return True
    classes, d = get_class_configuration(args)[1], args.local_yolo_dirs[0]
    fold = FOLD if classes[:len(TARGET)] == TARGET else {}   # a prefix, so a class added to merge-5 once does not stop the fold
    to = {p.label: fold.get(_norm(p.label or ""), _norm(p.label or "")) for ps in off.values() for p in ps}
    if new := [] if fold else list(dict.fromkeys(c for c in to.values() if c and c not in classes)):
        try: yp = _yaml_add(d, new)
        except Exception as e: print(f"  data.yaml could not take {new}, no label file written, your edits stay in FiftyOne: {type(e).__name__}: {e}"); return False
        print(f"  data.yaml gains {', '.join(f'{c} as class {len(classes) + k}' for k, c in enumerate(new))}, no old class number moves: {yp}")
    if done := sorted({(_norm(a or ""), b) for a, b in to.items() if _norm(a or "") in fold}): print(f"  folded as the merge does: {', '.join(f'{a} to {b}' for a, b in done)}")
    if bad := sorted({c for c in to.values() if c not in classes and c not in new}):
        print(f"  no class for {', '.join(map(repr, bad))}" + (f", not one of sama-merge-5's {len(TARGET)} and no fold rule" if fold else "")
              + ". It stays in FiftyOne as you typed it with no line on disk: a renamed one leaves its frame's file as it was. Rename it to a data.yaml class or delete it.")
    ids = {p.id for ps in off.values() for p in ps}
    for s in dataset.select(list(off)).iter_samples(autosave=True):   # after the yaml: a failed write leaves FiftyOne as he left it
        for p in _polys(s, "ground_truth"):
            if p.id in ids and p.label and (to[p.label] in classes or to[p.label] in new): p.label = to[p.label]
    return True

def _yaml_add(d, new):
    """new class names last in d's data.yaml, list or dict form, nc kept in step, no old number moved, the day's first original
    kept in archived/. A temp file, then one rename: on any failure the old yaml stands whole and the error is raised."""
    yp = os.path.realpath(os.path.join(d, next((f for f in ("data.yaml", "dataset.yaml") if os.path.exists(os.path.join(d, f))), "data.yaml")))
    try:
        cfg = yaml.safe_load(open(yp))
        if isinstance(cfg["names"], dict): cfg["names"].update({max(cfg["names"]) + 1 + k: c for k, c in enumerate(new)})
        else: cfg["names"] += new
        if "nc" in cfg: cfg["nc"] = len(cfg["names"])
        with contextlib.suppress(FileNotFoundError): os.unlink(yp + ".tmp")   # a stale temp, a link above all, is never written through
        with open(yp + ".tmp", "w") as f: yaml.safe_dump(cfg, f, sort_keys=False)
        os.makedirs(bk := os.path.join(os.path.dirname(yp), "archived"), exist_ok=True)
        if not os.path.exists(bak := os.path.join(bk, f"{os.path.basename(yp)}.bak_addclass_{datetime.date.today()}")): shutil.copy2(yp, bak)   # the day's first add keeps the original
        os.replace(yp + ".tmp", yp)
    except Exception:
        with contextlib.suppress(OSError): os.unlink(yp + ".tmp")
        raise
    return yp

def _disk_plan(dataset, ids, args, before=None):
    """Before the delete: {sample id: (label file, its lines, {label id: (line index, points, class)})} for every touched
    frame whose labels and lines pair up one to one, or None with the reason printed. A frame in before pairs by the labels
    it held as the pass opened, so an editor delete, move or rename made in the pass is written, not held."""
    if not _bare_bucket_argv(sys.argv[1:]):
        print(f"  {len(ids)} touched frames: disk unchanged, only a line of --local_yolo_dirs <one bucket>, alone or with a screen flag ({', '.join(sorted(_WRITE_BACK_SCREENS))}) writes label files back"); return None
    root, classes = os.path.realpath(args.local_yolo_dirs[0]), get_class_configuration(args)[1]
    plan, stale, n_out = {}, 0, 0
    for s in dataset.select(ids).select_fields(["filepath", "metadata", "ground_truth"]):
        fp = s.filepath
        # the image's dir, not the image: the 189 set's images are symlinks out of the bucket, its label files are not
        if not os.path.realpath(os.path.dirname(fp)).startswith(root + os.sep): n_out += 1; continue
        dst = os.path.join(os.path.dirname(os.path.dirname(fp)), "labels", os.path.splitext(os.path.basename(fp))[0] + ".txt")
        lines, W, H = _file_keys(dst), *_wh(s)
        free, owner = dict(enumerate(lines or [])), {}
        ps = [p for p in (before[s.id] if s.id in (before or {}) else _polys(s, "ground_truth")) if p.label in classes]   # a class not in data.yaml has no line to pair, the export drops it too
        for p in ps:
            ks = _poly_keys(p, W, H, classes)
            i = next((i for i, (_, k) in free.items() if k in ks), None)
            if i is not None: owner[p.id] = (i, p.points, p.label); del free[i]
        if lines is None or free or len(owner) != len(ps):   # a disk line no label claims is never dropped, edited or not
            stale += 1; print(f"  {'could not read' if lines is None else 'differed from FiftyOne before this pass, your clicks and editor changes there stay in FiftyOne'}, left alone: {dst}"); continue
        plan[s.id] = (dst, [l for l, _ in lines], owner)
    if n_out: print(f"  {n_out} touched frames are not under {root}: disk unchanged there")
    if stale: print(f"  {stale} of {len(ids)} touched frames not written: a full export of this bucket levels the two, ask me for it")
    return plan or None

def _unsaved(dataset, ids, args):
    """The frames whose label file does not match FiftyOne, a failed save or an older difference: a move leaves them.
    Quiet, the save step already named each file. A label drawn or renamed this pass whose class data.yaml could not take has no
    line to go to, so its frame stays too."""
    if not _bare_bucket_argv(sys.argv[1:]): return set()   # no bucket on disk to compare, and the move refuses the line
    with contextlib.redirect_stdout(io.StringIO()): out = set(ids) - set(_disk_plan(dataset, ids, args) or ())
    return out | set(_off_yaml_new(dataset, ids, args))

def _stay_note(dataset, moves, args):
    """The frames a move takes, and the y/N prompt's count of those it leaves with the reason. moves: {target split: ids}."""
    ids = {i for v in moves.values() for i in v}; tk = set().union(*(_taken(dataset, v, to) for to, v in moves.items()))
    diff = _unsaved(dataset, ids, args); go = ids - diff - tk; off = len(_off_yaml_new(dataset, diff, args))
    why = [w for w, k in (("their label file does not match FiftyOne", len(diff) - off), ("a label there has a class not in data.yaml", off), ("the other split already holds that name", len(tk - diff))) if k]
    return go, (f"{len(ids) - len(go)} of them stay, {' or '.join(why)}. " if why else "")

def _off_yaml_new(dataset, ids, args):
    """{sample id: labels} for the frames among ids he edited this pass, each label drawn or renamed in it with a class not in data.yaml.
    A relaunch forgets the pass: a new drawing then holds nothing, a renamed one still holds as a file difference."""
    classes = get_class_configuration(args)[1]
    return {s.id: u for s in dataset.select([i for i in ids if i in _EDITED]).select_fields("ground_truth")
            if (u := [p for p in _polys(s, "ground_truth") if p.label not in classes and {q.id: q.label for q in _EDITED[s.id]}.get(p.id) != p.label])}

def _gt_only(view, *keep):
    """Screens that are not a review pass draw ground_truth alone, plus the stale review_sheet his click still answers
    and any layer in keep. Every other label field is cut, the saved layer list reset."""
    d, show = view._dataset, ["ground_truth", "review_sheet", *keep]
    d.app_config.active_fields = ActiveFields(exclude=False, paths=[f for f in show if d.has_sample_field(f)]); d.save()
    shadow = [f for f in d.get_field_schema(embedded_doc_type=fo.Label) if f not in show]
    return view.exclude_fields(shadow) if shadow else view

def _class_menu(view, args):
    """Author 2026-09-28: a class change in the Annotate menu offers every class. With no stored schema the app's setup scans the frames it
    shows and lists only their classes, one on a person-only queue. So each polyline field it serves gets a stored dropdown of data.yaml's
    classes, merge-5's, their folds and every class already on a polyline, switched on: the app offers no menu for a field that is off.
    The outline layer goes off: its names are the pass's notes ("added: person"), so its panel opened invalid (browser 2026-09-28).
    A stored schema for a field the served set lacks goes: the copy of an adopt screen fails on it."""
    d = view._dataset
    try:
        pl, old = list(view.get_field_schema(embedded_doc_type=fo.Polylines)), {f: v for f, v in d.label_schemas.items() if d.has_sample_field(f)}
        names = sorted({*(get_class_configuration(args)[1] or ()), *TARGET, *FOLD, *(c for f in pl for c in [*d.distinct(f"{f}.polylines.label"), *old.get(f, {}).get("classes", [])] if c)})
        gen = d.generate_label_schemas(fields=pl, scan_samples=True) if pl else {}
        d.set_label_schemas({**old, **{f: {**gen[f], **old.get(f, {}), "classes": names, "component": "dropdown"} for f in pl}}); d.activate_label_schemas(pl)
        if off := [f for f in view.get_field_schema(embedded_doc_type=fo.Detections) if f in d.active_label_schemas]: d.deactivate_label_schemas(off)
    except Exception as e: print(f"  the Annotate menu lists only the classes on screen: {type(e).__name__}: {e}")
    return view

# every screen opens camera by camera, cam1 then cam2 then cam10, numeric; the pass's own order
# holds inside a camera. Stems with no cam<N>- sort last. Raw Mongo: a ViewExpression has no capturing group.
_CAM_NUM = {"$ifNull": [{"$toInt": {"$arrayElemAt": [{"$getField": {"field": "captures", "input":
           {"$regexFind": {"input": "$filepath", "regex": r"/cam(\d+)-[^/]*$"}}}}, 0]}}, 10**9]}

def _cam_first(view, *keys):
    return view.sort_by([(_CAM_NUM, 1), *keys, ("filepath", 1)])

def _confused(dataset):
    """Frames where a person box covers over half a dog or animal box: that person label holds a dog too."""
    has, ids = lambda *c: F("ground_truth.polylines").filter(F("label").is_in(c)).length() > 0, []
    for s in dataset.match(has("person") & has("dog", "animal")).select_fields("ground_truth"):
        b = [(p.label, p.to_detection().bounding_box) for p in _polys(s, "ground_truth")]
        if any(max(0, min(d[0] + d[2], q[0] + q[2]) - max(d[0], q[0])) * max(0, min(d[1] + d[3], q[1] + q[3]) - max(d[1], q[1])) > 0.5 * d[2] * d[3]
               for c, d in b if c in ("dog", "animal") for k, q in b if k == "person"): ids.append(s.id)
    print(f"{len(ids)} of {len(dataset)} frames in train/: a person box covers over half a dog or animal box")
    return ids

def _overlap(dataset, t):
    """Frames where a car box covers over t of a smaller box that is no car, or a dog box over a person box or the other way round."""
    car, ids = {k for k, v in FOLD.items() if v == "vehicle"} | {"vehicle"}, []
    for i, ps in zip(*dataset.values(["id", "ground_truth.polylines"])):
        b = [(p.label, p.to_detection().bounding_box) for p in ps or []]   # an empty polyline is a zero box, never counted
        if any(0 < d[2] * d[3] < q[2] * q[3] and max(0, min(d[0] + d[2], q[0] + q[2]) - max(d[0], q[0])) * max(0, min(d[1] + d[3], q[1] + q[3]) - max(d[1], q[1])) > t * d[2] * d[3]
               for c, d in b for k, q in b if {c, k} == {"dog", "person"} or k in car and c not in car): ids.append(i)
    print(f"{len(ids)} frames wait to be judged: a car box covers over {t} of a smaller box, or a dog box and a person box do")
    return ids

def _touches(b, eps=0.005): return b[0] <= eps or b[1] <= eps or b[0] + b[2] >= 1 - eps or b[1] + b[3] >= 1 - eps

def _border(dataset, f=0.4):
    """Frames where a labelled object touches the frame edge and its box is under f of the frame's median object area, edge objects
    in the median: the --sam3_border_factor rule, on the labels. Author 2026-10-04 "Small edge object"."""
    ids = []
    for i, ps in zip(*dataset.values(["id", "ground_truth.polylines"])):
        b = [q for p in ps or [] if (q := p.to_detection().bounding_box)[2] * q[3] > 0]   # an empty polyline is no label
        thr = f * float(np.median([q[2] * q[3] for q in b])) if b else 0
        if any(_touches(q) and q[2] * q[3] < thr for q in b): ids.append(i)
    print(f"{len(ids)} frames wait to be judged: an object touches the frame edge and is under {f} of the frame's median object area")
    return ids

def _small(dataset):
    """Frames where over a quarter of the labels are under 8x8 px at 320, the size the model trains at."""
    dataset.compute_metadata(); ids = []
    for i, W, H, pts in zip(*dataset.values(["id", "metadata.width", "metadata.height", "ground_truth.polylines.points"])):   # values(), not samples: 53 s, not 269 s, on 110772 frames
        W, H = (W, H) if W else (0, 0); ps = [a for p in pts or [] if (a := _px_area(p, W, H)) > 0]   # an empty ring is no label, small or not
        if W and ps and sum(a * (320 / max(W, H)) ** 2 < 64 for a in ps) > len(ps) / 4: ids.append(i)
    print(f"{len(ids)} of {len(dataset)} frames in train/: over a quarter of the labels are under 8x8 px at the 320 training size")
    return ids

def _redundant_groups(dataset, iou, ham, empty_ham=4, empty_win=600):
    """Groups of same-camera frames carrying the same objects in the same place with the same look.
    Returns lists of sample ids, earliest first, biggest group first. Stems without cam<N>-<epoch>- (CatDog, sama-coco)
    are one camera of their own with no clock: compared on labels and look alone, ordered by name."""
    recs = []
    for s in dataset.select_fields(["filepath", "ground_truth"]).iter_samples():
        m = re.match(r"(cam\d+)-(\d+(?:\.\d+)?)-", os.path.basename(s.filepath))
        cam, epoch = (m.group(1), float(m.group(2))) if m else (None, None)
        labels = s.ground_truth.polylines if isinstance(s.ground_truth, fo.Polylines) else (s.ground_truth.detections if s.ground_truth else [])
        boxes = [(l.label, _box(l)) for l in labels]
        grey = cv2.imread(s.filepath, cv2.IMREAD_GRAYSCALE)
        if grey is None: print(f"  skipped unreadable {s.filepath}"); continue
        H, W = grey.shape
        crops = [_dhash(grey[int(y * H):max(int(y * H) + 1, int((y + h) * H)), int(x * W):max(int(x * W) + 1, int((x + w) * W))]) for _, (x, y, w, h) in boxes]
        recs.append((cam, epoch, s.id, boxes, _dhash(grey), crops, os.path.basename(s.filepath)))
    recs.sort(key=lambda r: (r[0] or "", r[1] or 0.0, r[6]))
    parent = {r[2]: r[2] for r in recs}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def same(a, b):
        if not a[3]: return a[1] is not None and a[1] // empty_win == b[1] // empty_win or np.count_nonzero(np.unpackbits(a[4] ^ b[4])) <= empty_ham
        used = set()
        for i, (la, ba) in enumerate(a[3]):
            score, j = max(((compute_iou(ba, bb), j) for j, (lb, bb) in enumerate(b[3]) if lb == la and j not in used), default=(0.0, -1))
            if score < iou or np.count_nonzero(np.unpackbits(a[5][i] ^ b[5][j])) > ham: return False
            used.add(j)
        return True
    by_key = defaultdict(list)
    for r in recs: by_key[(r[0], tuple(sorted(l for l, _ in r[3])))].append(r)
    # ponytail: O(n^2) per camera and class set, 2852 frames of one camera took under a minute; bucket by epoch if a pull gets 10x bigger
    for rs in by_key.values():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                if same(rs[i], rs[j]): parent[find(rs[i][2])] = find(rs[j][2])
    groups = defaultdict(list)
    for r in recs: groups[find(r[2])].append(r[2])
    nocam = sum(r[0] is None for r in recs)
    if nocam: print(f"  {nocam} of {len(dataset)} stems carry no cam<N>-<epoch>- stamp: compared as one camera with no clock window")
    return sorted((g for g in groups.values() if len(g) > 1), key=len, reverse=True)

def deduplicate_detections(detections, iou_threshold=0.5):
    if not detections: return detections
    keep = []
    for det in sorted(detections, key=lambda d: getattr(d, 'confidence', 1.0) or 1.0, reverse=True):
        if not any(det.label == k.label and compute_iou(det.bounding_box, k.bounding_box) > iou_threshold for k in keep):
            keep.append(det)
    return keep

def filter_overlapping_classes(detections, priority_class, iou_threshold):
    priority_dets = [d for d in detections if _norm(d.label) == _norm(priority_class)]
    return detections if not priority_dets else [d for d in detections if _norm(d.label) == _norm(priority_class) or not any(compute_iou(d.bounding_box, p.bounding_box) > iou_threshold for p in priority_dets)]

def bbox_to_polyline_points(bbox): return [[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]], [bbox[0], bbox[1] + bbox[3]]]

def print_dataset_stats(dataset, title="Dataset stats", label_field="ground_truth"):
    print(f"\n{title}:")
    for split in ["train", "validation", "test"]:
        if len(view := dataset.match_tags(split)):
            if dataset.has_sample_field(label_field):
                labels = view.count(f"{label_field}.detections") + view.count(f"{label_field}.polylines")
                class_counts = view.count_values(f"{label_field}.detections.label") if view.count(f"{label_field}.detections") > 0 else view.count_values(f"{label_field}.polylines.label")
                class_str = ", ".join(f"{k}:{v}" for k, v in sorted(class_counts.items()))
            else:
                labels, class_str = 0, ""
            print(f"  {split}: {len(view)} samples, {labels} labels {f'[{class_str}]' if class_str else ''}")

def get_labels_from_sample(sample, label_field):
    if not (sample.has_field(label_field) and sample[label_field]): return []
    field_data = sample[label_field]
    return getattr(field_data, 'detections', None) or getattr(field_data, 'polylines', None) or []

def should_check_label_for_outliers(label, clean_4point_seg_outliers, clean_complex_seg_outliers):
    if isinstance(label, fo.Detection): return not (hasattr(label, 'mask') and label.mask) or clean_complex_seg_outliers
    if isinstance(label, fo.Polyline):
        num_points = len(label.points[0]) if label.points else 0
        return (num_points == 4 and clean_4point_seg_outliers) or (num_points > 4 and clean_complex_seg_outliers)
    return False

def _tmp_for(image_path, what):
    """Hash the full source path: two --local_yolo_dirs holding the same basename would share one temp file."""
    stem, ext = os.path.splitext(os.path.basename(image_path))
    return os.path.join(tempfile.gettempdir(), f'{stem}_{hashlib.md5(os.path.realpath(image_path).encode()).hexdigest()[:8]}_{what}{ext or ".jpg"}')

def _region_px(img, region):
    h, w = img.shape[:2]; top, bottom, left, right = region
    return int(top * h), int((1.0 - bottom) * h), int(left * w), int((1.0 - right) * w)

def enhance_region(image_path, region, intensity=1.0):
    try:
        from realesrgan import RealESRGANer; from basicsr.archs.srvgg_arch import SRVGGNetCompact
        tile_size, tile_pad, pre_pad = 256, 16, 16; print(f"Superres: intensity={intensity}, tile_size={tile_size}, pre_pad={pre_pad}")
        if not hasattr(enhance_region, 'upsampler'): enhance_region.upsampler = RealESRGANer(scale=4, model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth', dni_weight=1, model=SRVGGNetCompact(num_in_ch=3,num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'), tile=tile_size, tile_pad=tile_pad, pre_pad=pre_pad, device='cuda')
        img = cv2.imread(image_path, cv2.IMREAD_COLOR); y1, y2, x1, x2 = _region_px(img, region); enhanced_crop, _ = enhance_region.upsampler.enhance(img[y1:y2, x1:x2], outscale=4); img[y1:y2, x1:x2] = cv2.resize(enhanced_crop, (x2-x1, y2-y1)); temp_path = _tmp_for(image_path, f'enhanced_4x_{intensity:.2f}'); cv2.imwrite(temp_path, img); return temp_path
    except Exception as e: print(f"Superres failed: {e}"); return image_path

def process_image(image_path, methods, region=None, intensity=1.0):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if region:
            y1, y2, x1, x2 = _region_px(img, region)
            full_img = img.copy(); img = img[y1:y2, x1:x2]
        if 'contrast' in methods: img = cv2.convertScaleAbs(img, alpha=1.0 + 0.05*intensity, beta=int(10*intensity))
        if 'contrast2' in methods:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = cv2.createCLAHE(clipLimit=0.25, tileGridSize=(16,16)).apply(lab[:,:,0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        if 'retinex' in methods:
            img_f = img.astype(np.float32); log_img = np.log1p(img_f); sigma = max(1, int(80*intensity))
            log_blur = np.log1p(cv2.GaussianBlur(img_f, (0,0), sigma))
            img = cv2.normalize(np.exp(0.3*intensity * (log_img - log_blur)) * img_f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if 'denoise' in methods:
            d, sigma = min(15, int(9 * intensity)), max(30, min(100, int(75 * intensity)))
            print(f"denoise: d={d}, σ={sigma}")
            img = cv2.bilateralFilter(img, d, sigma, sigma)
        if 'denoise2' in methods:
            h_param = min(10, int(5*intensity)); print(f"denoise2: h={h_param}")
            img = cv2.fastNlMeansDenoisingColored(img, None, h_param, h_param, 7, 21)
        if 'denoise3' in methods:
            k_size = max(1, int(2 * intensity)); print(f"denoise3: kernel={k_size}")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            img = cv2.morphologyEx(cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
        if 'swinir' in methods:
            try:
                from basicsr.utils import img2tensor, tensor2img
                import requests, time
                h, w = img.shape[:2]; print(f"SwinIR: {w}x{h}px, intensity={intensity}")
                if not hasattr(process_image, 'swinir_model'):
                    noise_level = 'noise15' if intensity < 0.9 else ('noise25' if intensity < 1.0 else 'noise50')
                    model_path = os.path.join(tempfile.gettempdir(), f'swinir_colordn_{noise_level}.pth')
                    if not os.path.exists(model_path):
                        print(f"Downloading SwinIR {noise_level}...")
                        response = requests.get(f'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_{noise_level}.pth'); response.raise_for_status()
                        with open(model_path + '.tmp', 'wb') as f: f.write(response.content)   # a cut or failed download never sits at the path the exists check trusts
                        os.replace(model_path + '.tmp', model_path)
                    from basicsr.models import build_model
                    opt = {'model_type': 'SwinIRModel', 'num_gpu': 1, 'manual_seed': 42, 'is_train': False, 'dist': False, 'network_g': {'type': 'SwinIR', 'upscale': 4, 'in_chans': 3, 'img_size': 128, 'window_size': 8, 'img_range': 1.0, 'depths': [6]*6, 'embed_dim': 180, 'num_heads': [6]*6, 'mlp_ratio': 2, 'upsampler': '', 'resi_connection': '1conv'}, 'path': {'pretrain_network_g': model_path}}
                    process_image.swinir_model = build_model(opt)
                start_time = time.time()
                img_tensor = img2tensor(img / 255., bgr2rgb=False, float32=True).unsqueeze(0).cuda()
                process_image.swinir_model.feed_data({'lq': img_tensor})
                with torch.no_grad(): process_image.swinir_model.test()
                img = tensor2img(process_image.swinir_model.get_current_visuals()['result'], rgb2bgr=False, min_max=(0, 1))
                print(f"SwinIR: done in {time.time() - start_time:.1f}s")
            except Exception as e: print(f"SwinIR failed: {e}")
        if 'supir' in methods:
            try:
                from PIL import Image; import time
                h, w = img.shape[:2]
                print(f"SUPIR: {w}x{h}px, intensity={intensity}{', may take >360s...' if max(w, h) > 512 else ''}")
                temp_input = f"{tempfile.gettempdir()}/supir_{os.getpid()}.png"
                Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(temp_input)
                control_scale = max(0.8, min(1.5, 0.75 * intensity)); start_time = time.time()
                result = subprocess.run([sys.executable, "run_supir_cli.py", "--img_path", temp_input, "--save_dir", tempfile.gettempdir(), "--SUPIR_sign", "Q", "--upscale", "2", "--edm_steps", str(max(50, int(100 * intensity))), "--cfg_scale_start", str(max(1.0, 2.0 * intensity)), "--cfg_scale_end", str(max(4.0, 6.0 * intensity)), "--control_scale_start", str(control_scale), "--control_scale_end", str(control_scale)], cwd="/mnt/LinuxBackup/SUPIR-Demo", capture_output=True, timeout=3600)
                if result.returncode == 0 and os.path.exists(output_path := f"{tempfile.gettempdir()}/{os.path.splitext(os.path.basename(temp_input))[0]}_SUPIR.png"):
                    out = cv2.cvtColor(np.array(Image.open(output_path)), cv2.COLOR_RGB2BGR)
                    img = cv2.resize(out, (img.shape[1], img.shape[0])) if region else out
                    os.unlink(output_path); print(f"SUPIR: done in {time.time() - start_time:.1f}s")
                else: print(f"SUPIR failed: {result.stderr if result.returncode != 0 else 'output not found'}")
                os.unlink(temp_input)
            except Exception as e: print(f"SUPIR failed: {e}")
        if region: full_img[y1:y2, x1:x2] = img; img = full_img
        temp_path = _tmp_for(image_path, f'{"_".join(methods)}_{intensity:.2f}'); cv2.imwrite(temp_path, img); return temp_path
    except Exception as e: print(f"Image processing failed: {e}"); import traceback; traceback.print_exc(); return image_path

def add_region_visualization(sample, region):
    if not region or hasattr(sample, '_region_added'): return
    top, bottom, left, right = region
    sample['region_viz'] = fo.Polylines(polylines=[fo.Polyline(label="region_boundary",
        points=[[[left, top], [1.0-right, top], [1.0-right, 1.0-bottom], [left, 1.0-bottom]]], filled=False, closed=True)])
    sample._region_added = True

def is_bbox_in_region(bbox, region): return True if not region else region[2] <= bbox[0]+bbox[2]/2 <= 1.0-region[3] and region[0] <= bbox[1]+bbox[3]/2 <= 1.0-region[1]

def is_outlier_label(label, max_bbox_width, min_bbox_width, max_aspect_ratio, min_aspect_ratio, vertical_scaling=1.0):
    bbox = _box(label)
    w, h = bbox[2], bbox[3]
    if vertical_scaling != 1.0:
        scale_factor = (1.0/vertical_scaling) + (bbox[1] + bbox[3]/2) * (1.0 - 1.0/vertical_scaling)
        min_bbox_width, max_bbox_width = min_bbox_width * scale_factor, max_bbox_width * scale_factor
    return not (min_bbox_width <= w <= max_bbox_width) or (h > 0 and ((max_aspect_ratio and w / h > max_aspect_ratio) or (min_aspect_ratio and w / h < min_aspect_ratio)))

def _det_to_polyline(d, morph=True):
    """One detection -> one Polyline. Mask contour when there is a mask, box otherwise.
    Morphology closes pinholes and drops speckle; skipped on masks under 32 px, where a 3x3 kernel eats the object."""
    if hasattr(d, 'mask') and d.mask is not None:
        mask_uint8 = (d.mask * 255).astype(np.uint8)
        if morph and min(d.mask.shape) >= 32:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            m = cv2.morphologyEx(cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
            if np.count_nonzero(m) > 0: mask_uint8 = m
        if np.count_nonzero(mask_uint8) > 0:
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            if contours:
                largest_area = cv2.contourArea(max(contours, key=cv2.contourArea))
                # drops any blob under a quarter of the largest; bites model output only
                keep = [c for c in contours if len(c) >= 3 and cv2.contourArea(c) > largest_area * 0.25] or [max(contours, key=cv2.contourArea)]
                keep = [cv2.approxPolyDP(c, SIMPLIFY_EPS * cv2.arcLength(c, True), True) for c in keep]
                bbox = d.bounding_box
                h, w = d.mask.shape
                rings = [[(float(bbox[0] + (pt[0][0]/w)*bbox[2]), float(bbox[1] + (pt[0][1]/h)*bbox[3])) for pt in c] for c in keep if len(c) >= 3]
                if rings: return fo.Polyline(label=d.label, points=rings, filled=True, closed=True)
        return d.to_polyline(tolerance=1e-10)
    return fo.Polyline(label=d.label, points=[bbox_to_polyline_points(d.bounding_box)], filled=True, closed=True)

def merge_predictions_to_ground_truth(approved_dets, sample, label_field, pred_dedup_iou):
    """Merge approved detections into sample's ground_truth field. Always writes Polylines (segmentation-focused script)."""
    # a zoo import arrives as Detections, and fo.Polylines(polylines=[Detection, ...]) raises
    existing = [l if isinstance(l, fo.Polyline) else _det_to_polyline(l) for l in get_labels_from_sample(sample, label_field)]
    existing_boxes = [(e.label, e.to_detection().bounding_box) for e in existing]
    # a true positive the model re-found is already annotated: unfiltered, the export holds it twice
    new_polylines = [p for d in approved_dets for p in [_det_to_polyline(d)]
                     if not any(lbl == p.label and compute_iou(box, p.to_detection().bounding_box) > pred_dedup_iou for lbl, box in existing_boxes)]
    sample[label_field] = fo.Polylines(polylines=existing + new_polylines)
    return len(new_polylines)   # what was added, after the dedupe (W14)

def is_shadow_detection(image_path, bbox, confidence):
    """Shadow = low saturation + darker than surroundings."""
    try:
        if confidence > 0.8 or (img := cv2.imread(image_path)) is None: return False
        h, w = img.shape[:2]
        x1, y1, x2, y2 = max(0, int(bbox[0] * w)), max(0, int(bbox[1] * h)), min(w, int((bbox[0] + bbox[2]) * w)), min(h, int((bbox[1] + bbox[3]) * h))
        if x2 <= x1 or y2 <= y1: return False
        border = int(max(x2-x1, y2-y1) * 0.1)
        ex1, ey1, ex2, ey2 = max(0, x1-border), max(0, y1-border), min(w, x2+border), min(h, y2+border)
        mean_sat = np.mean(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)[:, :, 1])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness_diff = np.mean(gray[ey1:ey2, ex1:ex2]) - np.mean(gray[y1:y2, x1:x2])
        is_shadow = mean_sat < 50 and brightness_diff > 15
        if is_shadow:
            print(f"  ✓ SHADOW REJECTED: sat={mean_sat:.1f}, diff={brightness_diff:.1f}, conf={confidence:.2f}")
        elif confidence <= 0.8 and mean_sat < 50:
            print(f"  ? Near-shadow (kept): sat={mean_sat:.1f}, diff={brightness_diff:.1f}, conf={confidence:.2f}")
        return is_shadow
    except Exception: return False

def filter_outlier_detections(sample_collection, max_bbox_width, min_bbox_width, max_aspect_ratio, min_aspect_ratio, label_field="ground_truth", clean_4point_seg_outliers=False, clean_complex_seg_outliers=False, vertical_scaling=1.0, region=None, clip_model=None, outlier_classes=None):
    outlier_classes_lower = [_norm(c) for c in outlier_classes] if outlier_classes else None
    # last run's tags go first, or a label this run's thresholds clear is still served and deleted on a tick
    sample_collection.untag_labels("outlier", label_fields=label_field); sample_collection.untag_samples("outlier_candidate")
    view = sample_collection.match(~F("tags").contains("manually_reviewed"))
    clip_outliers = None
    if clip_model:
        from sklearn.neighbors import LocalOutlierFactor
        print("Running CLIP outlier detection...")
        model = foz.load_zoo_model(clip_model)
        view.compute_patch_embeddings(model, patches_field=label_field, embeddings_field="clip_embeddings")
        all_embeddings = []
        for sample in view:
            for i, det in enumerate(get_labels_from_sample(sample, label_field)):
                if not hasattr(det, 'clip_embeddings'): continue
                if outlier_classes_lower and _norm(det.label) not in outlier_classes_lower: continue
                all_embeddings.append((sample.id, i, det, det.clip_embeddings))
        if len(all_embeddings) >= 5:
            outliers = LocalOutlierFactor(n_neighbors=5, contamination="auto").fit_predict([e[3] for e in all_embeddings])
            clip_outliers = {(sid, idx) for (sid, idx, det, _), is_outlier in zip(all_embeddings, outliers) if is_outlier == -1}
            print(f"CLIP found {len(clip_outliers)} outliers from {len(all_embeddings)} embeddings")
        else: print(f"CLIP skipped: {len(all_embeddings)} embeddings, LOF needs 5. Nothing will be tagged.")
        try: sample_collection._dataset.delete_sample_field(f"{label_field}.polylines.clip_embeddings")
        except Exception: pass

    print(f"Finding outlier detections in '{label_field}'{f' in region {region}' if region else ''}...")
    if len(view) < len(sample_collection): print(f"Skipping {len(sample_collection) - len(view)} manually reviewed samples")
    if outlier_classes_lower: print(f"Only checking classes: {outlier_classes_lower}")
    total_outlier_detections, outlier_samples = 0, set()
    with fo.ProgressBar() as pb:
        for sample in pb(view.iter_samples(autosave=True)):
            add_region_visualization(sample, region)
            is_candidate = False
            for i, label in enumerate(get_labels_from_sample(sample, label_field)):
                if outlier_classes_lower and _norm(label.label) not in outlier_classes_lower: continue
                if getattr(label, "coco", False): continue   # a COCO label is never an outlier: the tick would delete it
                if not (is_bbox_in_region(_box(label), region) and should_check_label_for_outliers(label, clean_4point_seg_outliers, clean_complex_seg_outliers)): continue
                if clip_model:   # never the size rules under CLIP, even when it found nothing
                    should_process = (sample.id, i) in (clip_outliers or ())
                else:
                    should_process = is_outlier_label(label, max_bbox_width, min_bbox_width, max_aspect_ratio, min_aspect_ratio, vertical_scaling)
                if should_process:
                    is_candidate = True
                    if "outlier" not in label.tags:
                        label.tags.append("outlier")
                        total_outlier_detections += 1
                        if total_outlier_detections <= 5:
                            print(f"Outlier {total_outlier_detections}: {label.label} (bbox_width={_box(label)[2]:.3f})")
            if is_candidate:
                if "outlier_candidate" not in sample.tags: sample.tags.append("outlier_candidate")
                outlier_samples.add(sample.id)
    if total_outlier_detections:
        print(f"\nFound {total_outlier_detections} outlier labels in {len(outlier_samples)} samples. Tagging for review..")
        print("\n--- HOW TO CLEAN OUTLIERS ---\n1. SELECT samples to clean in the app\n2. CLOSE the app browser tab when finished")
    else: print("No outlier detections were found.")

def load_and_merge_datasets(dataset_names, local_dirs, target_classes, label_types, max_samples, splits_to_run, user_classes_arg=None, find_missed_labels=False, local_coco_dirs=None):
    merged_dataset = fo.Dataset()
    configs = defaultdict(lambda: {"source_map": {k: k for k in target_classes}, "class_mapping": {k: k for k in target_classes}, "extra_sources": {}, "excluded_classes": set()})
    configs["open-images-v7"] = {"source_map": {"person": "Person", "car": "Car", "motorcycle": "Motorcycle", "bus": "Bus", "truck": "Truck", "bird": "Bird", "cat": "Cat", "dog": "Dog", "horse": "Mule", "cow": "Bull"},
                                "class_mapping": {"Person": "person", "Car": "car", "Motorcycle": "motorcycle", "Bus": "bus", "Truck": "truck", "Van": "truck", "Bird": "bird", "Cat": "cat", "Dog": "dog", "Mule": "horse", "Bull": "cow"},
                                "extra_sources": {"truck": ["Van"]},
                                "excluded_classes": {"person"}}
    # FIXME: cannot exclude person as it appears in most other object's images...
    configs["voc-2012"] = {"source_map": {"bird": "bird", "cat": "cat", "cow": "cow", "dog": "dog", "horse": "horse", "bus": "bus", "car": "car", "motorcycle": "motorbike"},
                          "class_mapping": {"bird": "bird", "cat": "cat", "cow": "cow", "dog": "dog", "horse": "horse", "bus": "bus", "car": "car", "motorbike": "motorcycle"},
                          "extra_sources": {},
                          "excluded_classes": {"person"}}
    splits_to_load = sorted({SPLIT_MAPPING[s] for s in splits_to_run}) if splits_to_run else ['train', 'validation', 'test']
    if splits_to_run and dataset_names: print(f"Loading only splits: {splits_to_load}")
    all_local_classes, local_dataset_classes = set(), {}

    if dataset_names:
        for name in dataset_names:
            config = configs[name]
            excluded_classes = config.get("excluded_classes", set())
            filtered_target_classes = [c for c in target_classes if c not in excluded_classes]
            source_classes = [config["source_map"][c] for c in filtered_target_classes if c in config["source_map"]]
            source_classes.extend(sum([extras for target, extras in config.get("extra_sources", {}).items() if target in filtered_target_classes], []))
            source_classes, class_mapping = list(set(source_classes)), config["class_mapping"]
            zoo_persistent_name = f"zoo-{name}-{'_'.join(sorted(splits_to_load))}-{len(source_classes)}c-{max_samples or 'all'}"
            print(f"\nLoading ZOO dataset '{name}' (or using cached '{zoo_persistent_name}')...")
            temp_dataset = foz.load_zoo_dataset(name, splits=splits_to_load, label_types=label_types, classes=source_classes, max_samples=max_samples, only_matching=True, persistent=True, dataset_name=zoo_persistent_name)
            label_fields = [field for field, info in temp_dataset.get_field_schema().items() if hasattr(info, 'document_type') and info.document_type in [fo.Detections, fo.Polylines, fo.Segmentation]]
            processed_samples = []
            for sample in temp_dataset:
                new_sample = fo.Sample(filepath=sample.filepath)
                if hasattr(sample, 'tags'): new_sample.tags = sample.tags
                all_detections = []
                for field in label_fields:
                    if not (sample.has_field(field) and sample[field]): continue
                    field_data = sample[field]
                    if isinstance(field_data, fo.Detections):
                        all_detections.extend([det for det in field_data.detections if det.label in class_mapping and class_mapping[det.label] not in excluded_classes and (setattr(det, 'label', class_mapping[det.label]) or True)])
                    elif isinstance(field_data, fo.Polylines):
                        all_detections.extend([fo.Detection(label=class_mapping[poly.label], bounding_box=poly.to_detection().bounding_box) for poly in field_data.polylines if poly.label in class_mapping and class_mapping[poly.label] not in excluded_classes])
                if all_detections:
                    # Always Polylines, as both local importers emit, or a zoo+local mix crashes add_samples
                    new_sample["ground_truth"] = fo.Polylines(polylines=[fo.Polyline(label=det.label, points=[bbox_to_polyline_points(det.bounding_box)], filled=True, closed=True, confidence=det.confidence)
                                                                        for det in deduplicate_detections(all_detections)])
                    processed_samples.append(new_sample)
            merged_dataset.add_samples(processed_samples)
            print(f"Merged {len(processed_samples)} of {len(temp_dataset)} samples, the rest held no label of these classes. Total: {len(merged_dataset)}")

    def _merge_local(dirs, loader, kind):
        if not dirs: return
        seen, dups = set(), 0
        for idx, d in enumerate(dirs):
            print(f"\nLoading LOCAL {kind} dataset {idx+1}/{len(dirs)} from '{d}'...")
            try:
                local_dataset, dataset_classes = loader(d)   # whole: cached under the folder's name, so a split or cap would stand in for the folder
                local_dataset_classes[d] = dataset_classes
                all_local_classes.update(dataset_classes)
                print(f"  Classes in this dataset: {dataset_classes}")
            except Exception as e:
                # not bare exit(): rc 0 reads a failed bucket as success in a for loop
                print(f"FATAL ERROR: Failed to load {kind} dataset from '{d}'. Details: {e}")
                sys.exit(1)

            # find_missed_labels keeps all ground_truth
            target_classes_normed = [_norm(c) for c in target_classes or []]
            if not find_missed_labels and target_classes and {_norm(c) for c in dataset_classes} - set(target_classes_normed):   # names, not counts (W15)
                _n_bg = sum(1 for x in local_dataset if not (x.has_field("ground_truth") and x["ground_truth"]))
                _off = [l for x in local_dataset for l in get_labels_from_sample(x, "ground_truth") if _norm(l.label) not in target_classes_normed]
                _n_coco = sum(getattr(l, "coco", False) for l in _off)
                if _n_bg or _n_coco:   # Author 2026-09-24: refuse and name the drops
                    sys.exit(f"'{d}': the class filter {target_classes} would drop {_n_coco} COCO labels of other classes and {_n_bg} frames with no labels. "
                             f"Nothing loaded. Rerun with --classes naming every class to keep.")
                filtered_samples = []
                for sample in local_dataset.iter_samples():
                    if filtered_labels := [l for l in get_labels_from_sample(sample, "ground_truth") if _norm(l.label) in target_classes_normed]:
                        new_sample = sample.copy()
                        new_sample["ground_truth"] = (fo.Detections(detections=filtered_labels) if isinstance(sample["ground_truth"], fo.Detections)
                                                  else fo.Polylines(polylines=filtered_labels))
                        filtered_samples.append(new_sample)
                _n_off = len(local_dataset) - len(filtered_samples)
                _del_dataset(local_dataset.name, "replaced by the class-filtered copy")
                local_dataset = fo.Dataset()
                local_dataset.add_samples(filtered_samples)
                print(f"  Filtered to {len(filtered_samples)} samples with classes: {target_classes}")
                if _off: print(f"  DROPPED {len(_off)} labels of other classes; {_n_off} of {len(local_dataset) + _n_off} frames held nothing else and went with them.")

            converted, skipped_count = [], 0
            for sample in local_dataset:
                fp = os.path.realpath(sample.filepath)
                if fp in seen:
                    dups += 1; skipped_count += 1; continue
                seen.add(fp)
                new_sample = fo.Sample(filepath=sample.filepath, tags=sample.tags)
                if sample.has_field("ground_truth") and isinstance(sample["ground_truth"], fo.Detections):
                    new_sample["ground_truth"] = fo.Polylines(polylines=[fo.Polyline(label=det.label, points=[bbox_to_polyline_points(det.bounding_box)], filled=True, closed=True) for det in sample["ground_truth"].detections])
                elif sample.has_field("ground_truth"):
                    new_sample["ground_truth"] = sample["ground_truth"]
                converted.append(new_sample)
            if skipped_count: print(f"  ⚠️  Skipped {skipped_count} duplicate image(s) already seen in a previous {kind} dataset.")
            merged_dataset.add_samples(converted)
            print_dataset_stats(local_dataset, f"Local {kind} dataset {idx+1} loaded")
            print(f"Merged {len(converted)} samples (skipped {skipped_count} duplicates). Total: {len(merged_dataset)}")
            _del_dataset(local_dataset.name, "merged into the combined dataset")

        if dups: print(f"\n⚠️  Removed {dups} duplicate image(s) across {kind} datasets.")

    _merge_local(local_dirs, import_yolo_dataset, "YOLO")
    if local_dirs and len(local_dirs) > 1:
        yolo_class_lists = [local_dataset_classes[d] for d in local_dirs if d in local_dataset_classes]
        if len(set(tuple(c) for c in yolo_class_lists)) > 1:
            print("⚠️  WARNING: Local datasets have different class configurations!")
            for d in local_dirs:
                if d in local_dataset_classes: print(f"  {os.path.basename(d)}: {local_dataset_classes[d]}")

    # a local COCO json is human ground truth, so its labels are born stamped; SAM3's re-import is not
    _merge_local(local_coco_dirs, functools.partial(import_coco_dataset, stamp_coco=True), "COCO")

    return merged_dataset

def _yolo_dets(model, path, classes, conf, nms_iou, imgsz, retina_masks):
    """One YOLO forward pass -> FiftyOne detections with _norm'd labels. Instances when the model has masks."""
    import fiftyone.utils.ultralytics as fou
    results = model(path, classes=classes, conf=conf, iou=nms_iou, imgsz=imgsz, max_det=10000, retina_masks=retina_masks, verbose=False)
    if not results: return []
    r = results[0]
    if r.masks is not None: dets = fou.to_instances(r).detections
    elif r.boxes is not None: dets = fou.to_detections(r).detections
    else: return []
    for det in dets: det.label = _norm(det.label)   # store space-free, like every other label
    return dets

def active_learning_workflow(dataset, model_name, target_classes, classes_were_specified, reevaluate, yolo_conf_thresh, eval_iou_thresh, nms_iou, pred_dedup_iou, splits_to_run=None, label_field="ground_truth", pred_field="predictions", eval_key="eval", max_bbox_width=1.0, min_bbox_width=0.01, max_aspect_ratio=0.0, min_aspect_ratio=0.0, seg_iou=None, scales=None, full_inference=False, sample_num=None, vertical_scaling=1.0, region=None, superres=None, imgproc=None, intensity=1.0, clip_model=None, find_missed_labels=False, filter_shadows=False, clean_overlap_on=None, class_thresholds=None):
    print("\n--- Active Learning ---")
    # tags are written through SPLIT_MAPPING: a raw "val" matches nothing and the run silently no-ops
    view = dataset.match_tags([SPLIT_MAPPING[s] for s in splits_to_run]) if splits_to_run else dataset.view()
    if sample_num: view = view.skip(sample_num-1).limit(1); print(f"Processing only sample {sample_num}")
    use_t = bool(class_thresholds and (class_thresholds[0] or class_thresholds[1] is not None))
    if use_t:
        class_dict, default_t = class_thresholds
        if default_t is None: default_t = 0.3   # `is None`, not falsy: an explicit 0.0 keeps everything
        get_t = lambda lbl: class_dict.get(_norm(lbl), default_t)

    # manual review mode: only samples holding a target class
    if find_missed_labels and not use_t and target_classes:
        target_classes_normed = [_norm(c) for c in target_classes]
        original_count = len(view)
        view = view.match((F(f"{label_field}.polylines").filter(F("label").is_in(target_classes_normed)).length() > 0) | (F(f"{label_field}.detections").filter(F("label").is_in(target_classes_normed)).length() > 0))
        print(f"Manual review mode: filtered to {len(view)}/{original_count} samples containing classes {target_classes}")
    if not len(view): print("No samples to process."); return view   # evaluate would run on a field never written
    # per sample, or one inferred sample (--sample, one split) skips the rest of the view
    todo = view if reevaluate else view.select(view.exists(pred_field, False).values("id"))   # frozen: the loop writes the field it is keyed on
    if not len(todo):
        print(f"Predictions already exist in '{pred_field}' field. Use --reevaluate to force re-inference.")
    else:
        from ultralytics import YOLO
        model = YOLO(model_name)
        model_classes = list(model.names.values()) if hasattr(model, 'names') else []
        target_classes_for_model = target_classes if classes_were_specified else model_classes
        # _norm both sides: model.names keep the space ('traffic light'), every stored label does not.
        target_normed = [_norm(tc) for tc in target_classes_for_model]
        target_indices = [i for i, model_class in enumerate(model_classes) if _norm(model_class) in target_normed]
        retina_masks = 'seg' in model_name.lower()
        total_predictions = 0
        for sample in todo.iter_samples(progress=True, autosave=True):
            if not sample.metadata: sample.compute_metadata()
            if not sample.has_field("original_filepath") or not sample["original_filepath"]: sample["original_filepath"] = sample.filepath
            if imgproc: sample.filepath = process_image(sample["original_filepath"], imgproc, region, intensity); sample.save()

            # enhance ONCE, off the unenhanced path, so 'original' stays the unenhanced image
            unenhanced = sample.filepath
            superres_path = enhance_region(unenhanced, region, intensity) if (region and superres) else None
            if superres_path: sample.filepath = superres_path; sample.save()

            orig_scale = max(sample.metadata.width, sample.metadata.height)
            # a scale within 20% of the original is a duplicate; an emptied list falls back to the original
            inference_scales = [orig_scale] if scales is None else (sorted({s for s in scales if abs(s - orig_scale) / orig_scale >= 0.2}) or [orig_scale])
            image_variants = [('original', unenhanced)]
            if superres_path: image_variants.append(('superres', superres_path))
            print(f"Running YOLO inference with classes {target_classes_for_model}, scale={inference_scales}, variants={[v[0] for v in image_variants]} ({len(image_variants) * len(inference_scales)} combinations) and conf={yolo_conf_thresh}...")

            all_detections = []
            for variant_name, image_path in image_variants:
                for scale in inference_scales:
                    all_detections.extend(det for det in _yolo_dets(model, image_path, target_indices, yolo_conf_thresh, nms_iou, scale, retina_masks) if det.label in target_normed)

            raw_prediction_count = len(all_detections)
            detections = []
            if all_detections:
                all_detections.sort(key=lambda x: x.confidence, reverse=True)
                add_region_visualization(sample, region)
                for det in all_detections:
                    # size gates read the true extent, not the region-clipped sliver
                    true_box = det.bounding_box
                    if region:
                        top, bottom, left, right = region
                        x1, y1, x2, y2 = np.clip([det.bounding_box[0], det.bounding_box[1], det.bounding_box[0]+det.bounding_box[2], det.bounding_box[1]+det.bounding_box[3]], [left, top, left, top], [1.0-right, 1.0-bottom, 1.0-right, 1.0-bottom])
                        if x2 <= x1 or y2 <= y1: continue
                        # the mask is box-relative: crop it by the same fractions or it stretches over the clipped box
                        if getattr(det, "mask", None) is not None and (bw := true_box[2]) > 0 and (bh := true_box[3]) > 0:
                            mh, mw = det.mask.shape[:2]
                            r0, r1 = int(round((y1-true_box[1])/bh*mh)), int(round((y2-true_box[1])/bh*mh))
                            c0, c1 = int(round((x1-true_box[0])/bw*mw)), int(round((x2-true_box[0])/bw*mw))
                            if r1 > r0 and c1 > c0: det.mask = det.mask[r0:r1, c0:c1]
                        det.bounding_box = [x1, y1, x2-x1, y2-y1]
                    if (is_bbox_in_region(det.bounding_box, region) and
                        not any(det.label == existing.label and compute_iou(det.bounding_box, existing.bounding_box) > pred_dedup_iou for existing in detections) and
                        (not (max_bbox_width != 1.0 or min_bbox_width != 0.01 or max_aspect_ratio != 0.0 or min_aspect_ratio != 0.0 or clip_model) or not is_outlier_label(fo.Detection(label=det.label, bounding_box=list(true_box)), max_bbox_width, min_bbox_width, max_aspect_ratio, min_aspect_ratio, vertical_scaling)) and
                        (not filter_shadows or not is_shadow_detection(sample.filepath, det.bounding_box, det.confidence))):
                        detections.append(det)

            if clean_overlap_on:
                detections = filter_overlapping_classes(detections, clean_overlap_on, max(0.5, pred_dedup_iou*2))
            if use_t: detections = [d for d in detections if d.confidence > get_t(d.label)]
            sample[pred_field] = fo.Detections(detections=detections)
            if detections:
                print(f"  Sample {sample.id[:8]}: YOLO generated {raw_prediction_count} raw → deduplicated to {len(detections)} predictions")
                print(f"  Bbox stats: {sum(d.bounding_box[1] for d in detections) / len(detections):.3f} {sum(d.bounding_box[1] + d.bounding_box[3] for d in detections) / len(detections):.3f} {sum(d.bounding_box[0] for d in detections) / len(detections):.3f} {sum(d.bounding_box[0] + d.bounding_box[2] for d in detections) / len(detections):.3f}")
            total_predictions += len(detections)
            if imgproc or (region and superres):
                # point the sample back at the real image; the processed /tmp copy stays addressable here
                sample["processed_filepath"] = sample.filepath
                sample.filepath = sample["original_filepath"]
                sample.save()

        print(f"Generated {total_predictions} predictions across {len(todo)} samples")
        del model
        torch.cuda.empty_cache()
        gc.collect()

    has_ground_truth = view.has_sample_field(label_field) and (view.count(f"{label_field}.detections") > 0 or view.count(f"{label_field}.polylines") > 0)
    if has_ground_truth and (reevaluate or len(todo) or eval_key not in view.list_evaluations()):
        print("Running evaluation...")
        is_poly = dataset.count(f"{label_field}.polylines") > 0
        temp_gt_field = f"{label_field}_temp_detections"
        if is_poly:
            for sample in view.iter_samples(progress=True):
                if sample.has_field(label_field) and sample[label_field]:
                    sample[temp_gt_field] = sample[label_field].to_detections()
                    sample.save()
        view.evaluate_detections(pred_field, gt_field=temp_gt_field if is_poly else label_field, eval_method="coco", eval_key=eval_key, iou_threshs=[val/100.0 for val in range(0, 105, 5)], iou=eval_iou_thresh)
        if is_poly:
            # the marks land on the temp detections: copy them onto the polylines before the temp field goes
            eval_attrs = [eval_key, f"{eval_key}_iou", f"{eval_key}_id"]
            for sample in view.iter_samples(progress=True, autosave=True):
                if not (sample.has_field(temp_gt_field) and sample[temp_gt_field] and sample.has_field(label_field) and sample[label_field]): continue
                for poly, det in zip(sample[label_field].polylines, sample[temp_gt_field].detections):
                    for attr in eval_attrs:
                        val = getattr(det, attr, None)
                        if val is not None: poly[attr] = val
            dataset.delete_sample_field(temp_gt_field)
            dataset.add_dynamic_sample_fields()   # declare the copied eval_* so the app can filter on them
    elif has_ground_truth: print("Evaluation results already exist. Use --reevaluate to force re-evaluation.")

    if use_t:
        thresh_desc = f"per-class: {class_dict}, default: {default_t}" if class_dict else f"{default_t}"
        print(f"Auto-approving with thresholds: {thresh_desc}")
        approved_count = n_car = 0
        for sample in view.iter_samples(autosave=True, progress=True):
            if not sample.has_field(pred_field) or not sample[pred_field]: continue
            is_polyline_format = sample.has_field(label_field) and isinstance(sample[label_field], fo.Polylines)
            over = [det for det in sample[pred_field].detections if det.confidence > get_t(det.label)]
            n_car += sum(_norm(det.label) == "car" for det in over)   # car is never auto-accepted: his lawnmower reads as car
            approved_dets = [det for det in over if _norm(det.label) != "car" and (not seg_iou or not is_polyline_format or not any(poly.label == det.label and len(poly.points[0]) > 4 and compute_iou(det.bounding_box, poly.to_detection().bounding_box) >= seg_iou for poly in sample[label_field].polylines))]
            if approved_dets: approved_count += merge_predictions_to_ground_truth(approved_dets, sample, label_field, pred_dedup_iou)
        print(f"Auto-approved {approved_count} detections" if approved_count > 0 else "No detections met the per-class confidence thresholds")
        if n_car: print(f"  {n_car} car detections over the threshold were NOT auto-approved: car is never auto-accepted, a car:<number> threshold included")

    return view

def sam3_refine_workflow(dataset, args, class_thresholds=None, gt_field="ground_truth", pred_field="sam3_predictions"):
    """Run SAM3 into `pred_field`: --sam3_refine prompts with GT boxes, --sam3_prompt with text; both: text only on images without GT.
    Merge into `gt_field`: --sam3_overwrite replaces GT, else each SAM3 poly replaces the geometry of the GT it fits best at IoU >= seg_iou, label kept
    (a GT two polys want goes to the same-class one if it covers half, else to the tighter fit); the rest is added.
    Two polys on one object keep one, by adopt's cross-prompt rule (_dupes)."""
    import json as _json
    print("\n--- SAM3 Refine ---")
    prompts = [p.strip() for p in (args.sam3_prompt or [])]
    cd, dt = class_thresholds or ({}, None)
    score = args.sam3_score if args.sam3_score is not None else dt if dt is not None else (min(cd.values()) if cd else 0.25)
    # the annotator lives in the read-only pre-fork dir: executed, never written
    _here = os.path.dirname(os.path.abspath(__file__))
    _candidates = [args.sam3_script] if args.sam3_script else [
        os.path.join(_here, 'point_crowdsource_annotate', 'sam3_coco_annotator.py'),
        os.path.join(os.path.dirname(_here), 'Fish', 'YOLO', 'point_crowdsource_annotate', 'sam3_coco_annotator.py')]
    script = next((c for c in _candidates if os.path.isfile(c)), _candidates[0])
    # everything that changes what SAM3 emits, so a change re-runs SAM3 instead of voting on stale masks;
    # SPECK_* and SIMPLIFY_EPS shape the re-imported masks
    sig = [prompts, score, args.sam3_min_area, args.sam3_nms, args.sam3_border_factor, args.sam3_mask_threshold,
           SIMPLIFY_EPS, SPECK_MIN_AREA, SPECK_MIN_FRAC] + (["refine"] if args.sam3_refine else [])   # box-prompted masks are never text masks
    # re-running the same --sam3_eval opens the review; keys on the prompts alone, so adopt and consensus skip it
    if args.sam3_eval and not args.sam3_adopt and not args.consensus and not args.reevaluate and dataset.has_sample_field("sam3_worst") and not dataset.info.get("sam3_worst_by") and dataset.info.get("sam3_prompts", prompts) == prompts:
        print(f"SAM3 disagreement already on '{dataset.name}' for these prompts (sam3_worst present). Opening it as is; pass --reevaluate to run SAM3 again.")
        return
    # --reevaluate under --consensus re-votes, it does not re-detect: reuse the stored predictions when nothing changed.
    # another annotator is other masks; compared only once stamped, so no stored sig goes stale over it
    if (args.consensus or args.sam3_adopt) and dataset.info.get("sam3_sig") == sig and dataset.info.get("sam3_script", os.path.realpath(script)) == os.path.realpath(script) and dataset.exists(pred_field).count() == len(dataset):
        print(f"SAM3 predictions on '{dataset.name}' match this prompt set, score and sieve; reusing them. Change any of those, or delete '{pred_field}', to re-run SAM3.")
        return
    if not args.sam3_adopt and dataset.has_sample_field("review_worst"):
        sys.exit(f"'{dataset.name}' is adopted: this run would replace the SAM3 masks his review resolves against and its prompt fence. Use --sam3_adopt --reevaluate, or --output_dir for a copy. Nothing touched.")
    if not os.path.isfile(script):
        if args.sam3_adopt: sys.exit("SAM3 script not found; --sam3_adopt cannot run without SAM3 (it adopts SAM3's masks as ground_truth, so a skip would report success having changed nothing). Tried:\n  " + "\n  ".join(_candidates))
        if args.consensus: sys.exit("SAM3 script not found; --consensus is a 2-of-3 vote and would silently run as 2-of-2, letting the two YOLO models rewrite labels unopposed. Tried:\n  " + "\n  ".join(_candidates))
        print(f"SAM3 script not found at {script}; skipping."); return
    use_refine, use_text = bool(args.sam3_refine), bool(args.sam3_prompt)
    cls_names = [_cls_of(p) for p in prompts]
    cls_name = cls_names[0] if cls_names else "object"
    _bbox = lambda l: (l.to_detection().bounding_box if isinstance(l, fo.Polyline) and l.points and l.points[0] else l.bounding_box if isinstance(l, fo.Detection) and l.bounding_box else None)
    dataset.compute_metadata()
    with tempfile.TemporaryDirectory() as tmp:
        text_dirs = [os.path.join(tmp, f"text_{i}") for i in range(len(prompts))]
        # one box dir per GT class, so each mask carries the class of the box that prompted it, not the first prompt's
        bn2fp, gt_dirs, coco, aid = {}, {}, defaultdict(lambda: ([], [])), 1
        for i, sample in enumerate(dataset.iter_samples(progress=True), 1):
            bn = f"{i:08d}_{os.path.basename(sample.filepath)}"
            W, H = _wh(sample)
            by_cls = defaultdict(list)
            for l in get_labels_from_sample(sample, gt_field):
                if W and H and use_refine and (b := _bbox(l)): by_cls[l.label or cls_name].append((b[0]*W, b[1]*H, b[2]*W, b[3]*H))
            tgts = [gt_dirs.setdefault(c, os.path.join(tmp, f"with_gt_{len(gt_dirs)}")) for c in by_cls] if by_cls else (text_dirs if use_text else [])
            if not tgts: continue
            for tgt in tgts: os.makedirs(tgt, exist_ok=True); os.symlink(os.path.realpath(sample.filepath), os.path.join(tgt, bn))
            bn2fp[bn] = sample.filepath
            for c, boxes in by_cls.items():
                images, anns = coco[c]
                images.append({"id": (iid := len(images)+1), "file_name": bn, "width": int(W), "height": int(H)})
                for bx, by, bw, bh in boxes:
                    anns.append({"id": aid, "image_id": iid, "category_id": 1, "bbox": [float(bx), float(by), float(bw), float(bh)], "area": float(bw*bh), "iscrowd": 0}); aid += 1
        if not bn2fp: print("No images to process."); return
        for c, (images, anns) in coco.items():
            with open(os.path.join(gt_dirs[c], "_annotations.coco.json"), "w") as f:
                _json.dump({"images": images, "annotations": anns, "categories": [{"id": 1, "name": c}]}, f)
        mode = "boxes" if use_refine and not use_text else ("text" if use_text and not use_refine else "hybrid")
        print(f"Running SAM3 on {len(bn2fp)} images (mode={mode}, score={score}, prompts={prompts}, classes={cls_names or [cls_name]})...")
        for d, pr, cn in [(d, None, c) for c, d in gt_dirs.items()] + list(zip(text_dirs, prompts, cls_names)):
            if not os.path.isdir(d): continue
            print(f"  SAM3 {os.path.basename(d)}: class={cn!r} prompt={pr!r}")
            subprocess.check_call([sys.executable, script, d, "--score", str(score), "--class-name", cn, "--min-area", str(args.sam3_min_area), "--nms", str(args.sam3_nms), "--mask-threshold", str(args.sam3_mask_threshold),
                                   "--ext", ",".join(f".{e}" for x in IMG_EXTS for e in (x, x.upper()))] + (["--prompt", pr] if pr else []))
        refined, _ = import_coco_dataset(tmp)
        fp2preds = defaultdict(list)
        for s in refined.iter_samples():
            if (bn := os.path.basename(s.filepath)) in bn2fp and s.has_field(gt_field) and s[gt_field]:
                fp2preds[bn2fp[bn]].extend(s[gt_field].polylines)
        seg_iou = args.seg_iou
        def _clone(p, label=None, conf=False):
            out = fo.Polyline(label=label or p.label, points=p.points, closed=getattr(p, 'closed', True), filled=getattr(p, 'filled', True))
            if conf and getattr(p, 'confidence', None) is not None: out.confidence = p.confidence
            # a fresh Polyline carries no dynamic attribute: carry the coco stamp
            if getattr(p, 'coco', False): out.coco = True
            return out
        n_samples = n_raw = n_added = n_replaced = n_kept = n_border = n_car = n_gone = n_block = n_dup = 0
        hier = _hier(args)
        dataset.info.pop("sam3_sig", None); dataset.save()   # an interrupt below leaves no sig claiming half-new masks
        for sample in dataset.iter_samples(autosave=True, progress=True):
            if sample.filepath not in fp2preds:
                # clear, don't skip, or the previous prompt set's polylines still vote
                sample[pred_field] = fo.Polylines(polylines=[])
                continue
            n_samples += 1
            # a point-less polyline has no box to match on but is still his annotation: keep it
            all_gt = [l if isinstance(l, fo.Polyline) else l.to_polyline() for l in get_labels_from_sample(sample, gt_field)]
            existing_polys = [l for l in all_gt if _bbox(l)]
            unmatchable = [l for l in all_gt if not _bbox(l)]
            ex_boxes = [_bbox(l) for l in existing_polys]
            sam_polys = list(fp2preds[sample.filepath])
            sam_boxes = [p.to_detection().bounding_box for p in sam_polys]
            n_raw += len(sam_polys)

            # drop border-touching SAM3 polys under --sam3_border_factor of the median object area (0 = off);
            # border objects are IN the median, or an all-border image collapses to an over-aggressive threshold
            areas = [b[2]*b[3] for b in ex_boxes + sam_boxes] if args.sam3_border_factor > 0 else []
            if areas:
                thr = args.sam3_border_factor * float(np.median(areas))
                keep_idx = [i for i, b in enumerate(sam_boxes) if not (_touches(b) and b[2]*b[3] < thr)]
                n_border += len(sam_polys) - len(keep_idx)
                sam_polys = [sam_polys[i] for i in keep_idx]; sam_boxes = [sam_boxes[i] for i in keep_idx]

            sample[pred_field] = fo.Polylines(polylines=[_clone(p, conf=True) for p in sam_polys])
            if args.sam3_eval: continue   # GT untouched: SAM3 is scored against it instead
            if args.sam3_overwrite:   # SAM3's cars are never written, so the cars already there and any label of his a SAM3 car sat on, at seg_iou, stay
                cars = [pb for p, pb in zip(sam_polys, sam_boxes) if _norm(p.label) == "car"]; n_car += len(cars)
                stay = [l for l in all_gt if getattr(l, "coco", False) or _norm(l.label or "") == "car" or (_bbox(l) and any(compute_iou(_bbox(l), b) >= seg_iou for b in cars))]
                # COCO's labels, and a kept label of his of SAM3's class, win at seg_iou; his other kept labels only over the same object, --adopt_match_iou
                keep = [p for p, pb in zip(sam_polys, sam_boxes) if _norm(p.label) != "car" and all(compute_iou(pb, _bbox(l)) < (seg_iou if getattr(l, "coco", False) or _norm(l.label or "") == _norm(p.label) else args.adopt_match_iou) for l in stay if _bbox(l))]
                # adopt's cross-prompt rule: two of SAM3's at mask IoU >= adopt_match_iou are one object, the lower confidence goes
                lost = {k for k, *_ in _dupes(keep, [None] * len(keep), set(range(len(keep))), hier, *_wh(sample), args.adopt_match_iou)}
                keep = [p for k, p in enumerate(keep) if k not in lost]
                sample[gt_field] = fo.Polylines(polylines=stay + [_clone(p) for p in keep])
                n_added += len(keep); n_kept += len(stay); n_gone += len(all_gt) - len(stay); n_block += len(sam_polys) - len(cars) - len(keep) - len(lost); n_dup += len(lost)
                continue

            # best fit first (Author 2026-09-24): each SAM3 poly redraws the GT it fits best at IoU >= seg_iou, and tries its next fit if that GT is
            # taken. A GT two polys want goes to the same-class one if it covers half (--adopt_match_iou), else to the tighter fit
            ious = [[compute_iou(pb, eb) for eb in ex_boxes] for pb in sam_boxes]
            same = lambda i, j: ious[i][j] >= args.adopt_match_iou and _norm(sam_polys[i].label) == _norm(existing_polys[j].label or "")
            out_polys = list(existing_polys); used, done, rep = set(), set(), {}
            for *_, i, j in sorted((iou < max(row), not same(i, j), -iou, i, j) for i, row in enumerate(ious) for j, iou in enumerate(row) if iou >= seg_iou and not _rider(sam_polys[i].label, existing_polys[j].label)):
                if i in done or j in used: continue
                done.add(i)
                # COCO ground truth is never overwritten by a script
                if getattr(out_polys[j], "coco", False): n_block += 1; continue   # COCO wins, no SAM3 duplicate beside it
                # conf=True: a refined GT keeps its SAM3 score so a review can sort on it
                out_polys[j] = _clone(sam_polys[i], label=out_polys[j].label, conf=True); used.add(j); rep[i] = j
            # car is never auto-accepted: his lawnmower reads as car
            new = [i for i, p in enumerate(sam_polys) if i not in done and _norm(p.label) != "car"]
            n_car += len(sam_polys) - len(done) - len(new)
            # adopt's cross-prompt rule: two of SAM3's at mask IoU >= adopt_match_iou are one object. A redraw exports his label, so it answers to
            # his label and SAM3's class (his dog redrawn as a cat: a second cat and a new dog both go). A redraw never loses to a new
            # mask, so the redraws alone settle which lose; each loser's label comes back as drawn under both names and drops a new mask on it; then one
            # walk over the rest, so a mask that goes never drops another first
            his = lambda i: existing_polys[rep[i]] if i in rep else None
            dup = lambda ids, poly, alt=his: [(ids[l], ids[k]) for l, k, _ in _dupes([poly(i) for i in ids], [rep.get(i) for i in ids], {x for x, i in enumerate(ids) if i not in rep}, hier, *_wh(sample), args.adopt_match_iou, [alt(i) for i in ids])]
            back = {l for l, _ in dup(sorted(rep), lambda i: sam_polys[i])}
            lost = back | {l for r in back for l, k in dup([r, *new], lambda i: his(i) if i == r else sam_polys[i], lambda i: sam_polys[r] if i == r else None) if k == r}
            lost |= {l for l, _ in dup([i for i in sorted([*rep, *new]) if i not in lost], lambda i: sam_polys[i])}
            for i in lost & rep.keys(): out_polys[rep[i]] = existing_polys[rep[i]]; used.discard(rep[i])   # his label back as drawn
            out_polys += [_clone(sam_polys[i]) for i in new if i not in lost]
            n_replaced += len(rep.keys() - lost); n_added += len(set(new) - lost); n_dup += len(lost)
            n_kept += len(all_gt) - len(used)
            sample[gt_field] = fo.Polylines(polylines=out_polys + unmatchable)
        _del_dataset(refined.name, "temporary SAM3 refinement dataset")
        # only now: an interrupt before this point would leave the sig claiming masks that were never stored
        dataset.info["sam3_prompts"], dataset.info["sam3_sig"], dataset.info["sam3_script"] = prompts, sig, os.path.realpath(script); dataset.info.pop("sam3_worst_by", None); dataset.save()
        print(f"SAM3: {n_raw} raw preds across {n_samples}/{len(dataset)} samples (seg_iou={seg_iou}) -> "
              f"stored in '{pred_field}'; " + (f"'{gt_field}' untouched (--sam3_eval), " if args.sam3_eval else
              f"merged into '{gt_field}': replaced {n_replaced} GT (refinement), added {n_added} new, kept {n_kept} GT as drawn, "
              + (f"removed {n_gone} GT (--sam3_overwrite), " if args.sam3_overwrite else "") + (f"did not write {n_block} over a label that stays, " if n_block else "")
               + (f"dropped {n_dup} SAM3 polys that a second prompt, or the same prompt twice, put on one object (adopt's rule), " if n_dup else ""))
              + f"dropped {n_border} small border.")
        if n_car: print(f"  {n_car} SAM3 car finds " + ("were NOT written: the cars already there, and any label of yours a SAM3 car sat on, stay" if args.sam3_overwrite else "that redrew no label of yours were NOT added") + ": car is never auto-accepted")
        if args.sam3_eval:
            sam3_disagreement_eval(dataset, args, prompts, cls_names, n_samples,
                                   sieve=(args.sam3_min_area, args.sam3_nms,
                                          f"{args.sam3_border_factor}x median area on border-touching" if args.sam3_border_factor > 0 else "off"),
                                   gt_field=gt_field, pred_field=pred_field)

# Weights of the per-instance disagreement score, each term in [0,1], summing to 1. Class dominates on
# purpose: a tight mask on his bird called a rock is the most suspicious thing on the list.
DISAGREE_W_CLASS, DISAGREE_W_SEG, DISAGREE_W_BOUNDARY, DISAGREE_W_CONF = 0.55, 0.20, 0.15, 0.10
DISAGREE_COMPETE_IOU = 0.5   # two SAM3 polys this close are competing for the same object

SHEET_REASONS = {"classchange", "lowiou"}   # the only two that name a per-polyline adopt value; 'more' is sample-level
COVERED_IOU = 0.4   # another label this close is already on the object, so the dispute is a duplicate
# Share of classchange disputes REJECTED on review per origin class (173 ruled on house-24Sept2024-final).
# Ranks SAM3's likeliest faults first; an unmeasured class sits between car and person.
ORIGIN_REJECT_RATE, ORIGIN_REJECT_UNKNOWN = {"person": 0.06, "car": 0.41, "dog": 0.50}, 0.25
_ORIGINS = lambda a: sorted({_norm(c) for c in (a.review_origin or [])})  # sorted, or `car dog` and `dog car` write two done tags for one queue

def _reviewed_origins(tags, reason):
    """Which origin classes were already PUT ON SCREEN for this sample, read out of its done tags; None means every one.
    An accepted dispute stays adopt=='classchange' forever, so the tag is the only record he looked."""
    out = set()
    for t in tags:
        if t == "adopt_reviewed": return None            # a no-reason pass showed every fault on the image
        if not t.startswith(pre := f"adopt_reviewed_{reason}"): continue
        if not (rest := t[len(pre):].strip("_")): return None
        # the loose-IoU scope is a tag segment, not an origin class
        if rest.startswith("iou"): rest = rest.split("_", 1)[1] if "_" in rest else ""
        if not rest: return None
        out |= set(rest[5:].split("_")) if rest.startswith("from_") else {rest.split("_")[0]}  # from_car_dog, or the pair tag car_person
    return out

def _poly_boxes(polys):
    """A polyline list and its bounding boxes, in the (gts, gboxes) shape _best_overlap wants."""
    return polys, [p.to_detection().bounding_box if p.points and p.points[0] else (0.0, 0.0, 0.0, 0.0) for p in polys]

def _poly_pixel_mask(poly, W, H, x0, y0, w, h):
    """Rasterise one polyline into a (h, w) uint8 map, cropped to the pixel box (x0, y0, w, h)."""
    canvas = np.zeros((max(1, h), max(1, w)), np.uint8)
    for ring in (poly.points or []):
        if len(ring) < 3: continue
        cv2.fillPoly(canvas, [np.array([[p[0]*W - x0, p[1]*H - y0] for p in ring], np.int32)], 1)
    return canvas

def _pair_ious(pred_poly, gt_poly, W, H, boundary=True):
    """Mask IoU and Boundary-IoU (bench_eval.boundary_iou_mean) for one pair, on a map cropped to the two.
    The band ratio is rescaled to 2 % of the IMAGE diagonal (Cheng et al. 2021), not the crop's."""
    bs = [l.to_detection().bounding_box for l in (pred_poly, gt_poly)]
    x0 = int(min(b[0] for b in bs) * W); y0 = int(min(b[1] for b in bs) * H)
    x1 = int(np.ceil(max(b[0]+b[2] for b in bs) * W)); y1 = int(np.ceil(max(b[1]+b[3] for b in bs) * H))
    w, h = x1 - x0, y1 - y0
    if w < 2 or h < 2: return 0.0, 0.0
    pm = _poly_pixel_mask(pred_poly, W, H, x0, y0, w, h)
    gm = _poly_pixel_mask(gt_poly, W, H, x0, y0, w, h)
    union = int(np.logical_or(pm, gm).sum())
    m_iou = float(np.logical_and(pm, gm).sum()) / union if union else 0.0
    if not boundary: return m_iou, 0.0
    try: from bench_eval import boundary_iou_mean
    except Exception: return m_iou, m_iou
    ratio = 0.02 * float(np.hypot(W, H)) / float(np.hypot(w, h))
    return m_iou, boundary_iou_mean(pm, gm, num_classes=2, dilation_ratio=ratio)

def _best_overlap(poly, box, gts, gboxes, W, H, boundary=True, prefer=None, prefer_floor=0.0):
    """Best mask-IoU annotation for one prediction; box IoU only prefilters the top 3. `prefer(label)` is true
    when a normalised label AGREES, kin included: an agreeing one at or above `prefer_floor` wins, agreement is never a fault."""
    cand = []
    for k in sorted((k for k in range(len(gts)) if compute_iou(box, gboxes[k]) > 0.05),
                    key=lambda k: -compute_iou(box, gboxes[k]))[:3]:
        m, b = _pair_ious(poly, gts[k], W, H, boundary)
        if m > 0: cand.append((k, m, b))   # a zero-IoU candidate was never a match: keep returning None
    if not cand: return None, 0.0, 0.0
    if prefer:
        agree = [c for c in cand if prefer(_norm(gts[c[0]].label)) and c[1] >= prefer_floor]
        if agree: return max(agree, key=lambda c: c[1])
    return max(cand, key=lambda c: c[1])

def _best_of(p, box, gts, gboxes, idx, W, H):
    """_best_overlap over the annotations at `idx` only, no boundary; the index returned is into `idx`."""
    return _best_overlap(p, box, [gts[i] for i in idx], [gboxes[i] for i in idx], W, H, boundary=False)

UNSUP_CLASH_IOU, UNSUP_COVER_IOU = 0.5, 0.25   # SAM3 is on the object and calls it something else / SAM3 is not there at all
_UNSUP_LIVE = {None, "kept_orig", "restored"}   # his own points; `reverted` is his class on SAM3's geometry

# No green: the footage is grass. A class wears its merge class's colour and the pool repeats none, so only dog is pink.
CLASS_COLORS = [{"value": v, "color": c} for c, vs in (("#00FFFF", "person"), ("#2E6BFF", "car vehicle truck bus"), ("#FF69B4", "dog"),
                ("#FF0000", "animal cat bird sheep cow horse zebra donkey"), ("#0E6E8C", "bicycle bike"), ("#008080", "motorcycle")) for v in vs.split()]

# --review_reason only: grey out the agreed, outline the fault. Black scores highest on every neighbour.
ADOPT_FIELDS = [
    {"path": "ground_truth", "colorByAttribute": "adopt", "valueColors": [
        {"value": "adopted", "color": "#C8C8C8"}, {"value": "kept_orig", "color": "#C8C8C8"},
        {"value": "reverted", "color": "#C8C8C8"}, {"value": "restored", "color": "#C8C8C8"},
        {"value": "classchange", "color": "#FF00FF"}, {"value": "lowiou", "color": "#FFFF00"},
        {"value": "dup", "color": "#FF8000"}, {"value": "yolo_fill", "color": "#00FFFF"}]},
    {"path": "review_sheet", "colorByAttribute": "kind", "fieldColor": "#FF00FF", "valueColors": [
        {"value": "added", "color": "#FF00FF"}, {"value": "replaced", "color": "#FFFF00"},
        {"value": "mismatch", "color": "#000000"}, {"value": "nocover", "color": "#FF8000"},
        {"value": "rename", "color": "#00FF00"}, {"value": "loose", "color": "#B026FF"},
        {"value": "classchange", "color": "#FF00FF"}, {"value": "lowiou", "color": "#FFFF00"}]}]

def _color_scheme(extra_fields=()):
    """The one scheme every launch uses. A path in `extra_fields` is not class-coloured."""
    taken = {f["path"] for f in extra_fields}
    return fo.ColorScheme(color_by="value", opacity=0.35,
        color_pool=["#FFFF00", "#B026FF", "#FF8000"],
        fields=[{"path": p, "valueColors": CLASS_COLORS}
                for p in ("ground_truth", "ground_truth_orig", "sam3_predictions", "yolo_judge_preds") if p not in taken]
               + list(extra_fields))

UNSUP_MIN_AREA = 0   # hides a smaller label from unsupported, his and SAM3's. Author 2026-09-16: "hiding stuff is moronic"

def _yolo_into(view, model, field, conf, nms_iou):
    """Run `model` over `view`, write its detections to `field` as polylines, return the count.
    A sample with no metadata width is skipped, not blanked, which would read as "found nothing"."""
    n = 0
    for sample in view.iter_samples(autosave=True, progress=True):
        if not (sample.metadata and sample.metadata.width): continue
        W, H, polys = sample.metadata.width, sample.metadata.height, []
        for d in _yolo_dets(model, sample.filepath, None, conf, nms_iou, max(W, H), True):
            p = _det_to_polyline(d); p.confidence = d.confidence; polys.append(p)
        sample[field] = fo.Polylines(polylines=polys); n += len(polys)
    return n

def _sheet_box(p, label, **kw):
    """One review_sheet outline over a ground_truth polyline."""
    return fo.Detection(label=label, bounding_box=p.to_detection().bounding_box,
                        confidence=getattr(p, "confidence", None), gt_id=p.id, **kw)

def _scan(dataset):
    """The three queue counters' walk. The field list is a contract: read a fourth field and it is empty."""
    return dataset.select_fields(["metadata", "ground_truth", "sam3_predictions"]).iter_samples(progress=True)

def _polys(sample, field, live=False):
    """The polylines on `field`, tolerating a missing or None field. live=True drops the ones with no usable points."""
    ps = list(sample[field].polylines) if sample.has_field(field) and sample[field] else []
    return [p for p in ps if p.points and p.points[0]] if live else ps

def _wh(sample):
    """(width, height) from metadata, (0, 0) when it was never computed."""
    return (sample.metadata.width, sample.metadata.height) if (sample.metadata and sample.metadata.width) else (0, 0)

def _px_area(points, W, H):
    """Bounding-box area of a polyline's points in pixels. 0 when it has no usable points."""
    pts = [pt for r in points for pt in r] if points else []
    if len(pts) < 3: return 0.0
    xs = [x * W for x, _ in pts]; ys = [y * H for _, y in pts]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))

def _review_scope(args):
    """The classes a pass may act on: the ones --sam3_prompt named, or None for all. A tick answers the pass,
    nothing else; a rerun without the flag reads it off the build (sam3_adopt_workflow fills args from the stamp)."""
    return {_cls_of(x) for x in (args.sam3_prompt or [])} or None   # SAM3 labels 'a dog' as dog (W87)

def _unsupported(sample, gt_field="ground_truth", pred_field="sam3_predictions", min_conf=None, classes=None, hier={}):
    """His own live polylines SAM3 does not back, as (poly, kind, sam3 label, iou): mismatch = SAM3 on it at >= UNSUP_CLASH_IOU
    under another class, nocover = nothing of SAM3's reaches UNSUP_COVER_IOU. `classes` (_review_scope) fences the pass."""
    polys, preds, (W, H) = _polys(sample, gt_field), _polys(sample, pred_field, live=True), _wh(sample)
    if not polys or not W: return []
    # min_conf drops a weak SAM3 mask from the BACKING side only, so his label queues
    gts, gboxes = _poly_boxes([q for q in preds if _px_area(q.points, W, H) >= UNSUP_MIN_AREA and (min_conf is None
                               or (getattr(q, "confidence", None) is not None and float(q.confidence) >= min_conf))])
    out, gk, mk = [], {_vk(q) for q in polys}, lambda q: _poly_pixel_mask(q, W, H, 0, 0, W, H).astype(bool)
    for p in polys:
        if getattr(p, "adopt", None) not in _UNSUP_LIVE or not (p.points and p.points[0]): continue
        if getattr(p, "coco", False): continue   # a COCO label is never on trial: only his click removes one
        if classes and _norm(p.label) not in classes: continue
        if _px_area(p.points, W, H) < UNSUP_MIN_AREA: continue
        b, iou, _ = _best_overlap(p, p.to_detection().bounding_box, gts, gboxes, W, H, boundary=False)
        # SAM3 puts two prompts on one object: first ask whether it said his class anywhere on it
        _same = [i for i, g in enumerate(gts) if _norm(g.label) == _norm(p.label)]
        if _same and _best_of(p, p.to_detection().bounding_box, gts, gboxes, _same, W, H)[1] >= UNSUP_COVER_IOU: continue
        # W322: a same-class or kin mask over half or more of it with no export copy is the addition adopt dropped as this object
        _w = [g for g in gts if _vk(g) not in gk and _agree(hier, _norm(g.label))(_norm(p.label))]
        if _w and (mp := mk(p)).any() and any((mk(g) & mp).sum() / mp.sum() >= 0.5 for g in _w): continue
        if b is not None and iou >= UNSUP_CLASH_IOU and _norm(gts[b].label) != _norm(p.label): out.append((p, "mismatch", gts[b].label, iou))
        elif iou < UNSUP_COVER_IOU: out.append((p, "nocover", "", iou))
    return out

def _more_added(sample, match_iou, min_conf, gt_field="ground_truth", orig_field="ground_truth_orig", classes=None, hier={}):
    """What a 'more' pass draws, as (poly, kind, iou): added = no label of his under it, a tick deletes it; replaced = his
    same-class label under it at match_iou, a tick hands it back. min_conf holds back weak ADDITIONS only.
    `classes` (_review_scope) fences the pass. The cover is chosen as adopt chose it, agreeing or kin label first."""
    polys, (W, H) = _polys(sample, gt_field), _wh(sample)
    if not polys or not W: return []
    gts, gboxes = _poly_boxes(_polys(sample, orig_field))
    out = []
    for p in polys:
        if getattr(p, "adopt", None) not in ("adopted", "yolo_fill") or not (p.points and p.points[0]): continue
        if classes and _norm(p.label) not in classes: continue
        ag = _agree(hier, _norm(p.label))
        b, iou, _ = _best_overlap(p, p.to_detection().bounding_box, gts, [(0.0,) * 4 if _rider(p.label, g.label) else x for g, x in zip(gts, gboxes)], W, H, boundary=False, prefer=ag, prefer_floor=match_iou)
        # a same-class cover at the match threshold is a REPLACEMENT of his own polyline
        if b is not None and iou >= match_iou:
            if ag(_norm(gts[b].label)): out.append((p, "replaced", iou))
            continue
        c = getattr(p, "confidence", None)
        if c is not None and float(c) < min_conf: continue   # an unscored one is shown, never hidden
        out.append((p, "added", iou))
    return out

def _wrong_class(sample, iou_floor, gt_field="ground_truth", pred_field="sam3_predictions"):
    """{polyline id: (SAM3's class, its confidence)} where a SAM3 mask sits on one of his labels at iou_floor or more
    and calls it something else: the class errors adopt filed as new objects under --adopt_match_iou, read off the stored masks."""
    # a COCO human label is never on trial, the same rule as _unsupported
    polys = [p for p in _polys(sample, gt_field, live=True)
             if not getattr(p, "alt_name_done", None) and not getattr(p, "coco", False)]
    W, H = _wh(sample)
    preds = _polys(sample, pred_field, live=True)
    if not polys or not preds or not W: return {}
    qs, qboxes = _poly_boxes(preds)
    out = {}
    for p in polys:
        box = p.to_detection().bounding_box
        _diff = [i for i, q in enumerate(qs) if _norm(q.label) != _norm(p.label) and not _rider(q.label, p.label)]
        if not _diff: continue
        b, iou, _ = _best_of(p, box, qs, qboxes, _diff, W, H)
        if b is None or iou < iou_floor: continue
        # a second name for an object SAM3 already agreed on is `rename`'s question, not a class error;
        # the agreeing mask must be ON the object (UNSUP_CLASH_IOU), not merely touching it
        _same = [i for i, q in enumerate(qs) if _norm(q.label) == _norm(p.label)]
        if _same and _best_of(p, box, qs, qboxes, _same, W, H)[1] >= UNSUP_CLASH_IOU: continue
        out[p.id] = (qs[_diff[b]].label, round(_conf_of(qs[_diff[b]]), 4))
    return out

def _promote_map(sample, args, reason):
    """{polyline id: the coarser name SAM3 gave it} for `rename`, the one pass that APPLIES a name. Tick and click both read it."""
    if reason != "rename": return {}
    return {k: v[0] for k, v in _alt_names(sample, _hier(args), args.adopt_match_iou).items()}

def _loose_class(sample, args):
    """_wrong_class for the classchange pass, empty unless --classchange_iou is given. Queue, sheet and tick all read this one call."""
    if args.review_reason != "classchange" or not args.classchange_iou: return {}
    return _wrong_class(sample, args.classchange_iou)

def _alt_names(sample, hier, match_iou, gt_field="ground_truth", pred_field="sam3_predictions"):
    """{polyline id: (coarser label, its confidence)} for objects SAM3 also called by a parent name (--sam3_hierarchy dog:animal).
    Stored `alt_name` when the build wrote one, else recomputed off the stored proposals: re-adopting would destroy his clicking."""
    polys = [p for p in _polys(sample, gt_field, live=True) if not getattr(p, "coco", False) and not getattr(p, "alt_name_done", None)]
    out = {p.id: (p.alt_name, getattr(p, "alt_name_conf", None) or 0.0) for p in polys
           if getattr(p, "alt_name", None) and (not hier or hier.get(_norm(p.label)) == _norm(p.alt_name))}   # a classchange revert leaves a stale one
    W, H = _wh(sample)
    todo = [p for p in polys if p.id not in out and hier.get(_norm(p.label))]
    if not todo or not W or not hier: return out
    preds = _polys(sample, pred_field, live=True)
    if not preds: return out
    qs, qboxes = _poly_boxes(preds)
    for p in todo:
        par = hier[_norm(p.label)]
        cand = [i for i, q in enumerate(qs) if _norm(q.label) == par]
        if not cand: continue
        b, iou, _ = _best_of(p, p.to_detection().bounding_box, qs, qboxes, cand, W, H)
        if b is not None and iou >= match_iou:
            out[p.id] = (qs[cand[b]].label, round(_conf_of(qs[cand[b]]), 4))
    return out

def _hier(args):
    # permissive: nesting is rejected in parse_args, before any GPU work
    return {_norm(k): _norm(v) for k, v in (x.split(":", 1) for x in (args.sam3_hierarchy or []))}
def _kin(hier, a, b): return hier.get(a) == b or hier.get(b) == a
def _agree(hier, *names): return lambda G: any(G == n or _kin(hier, G, n) for n in names)   # adopt's claim test, kin included
def _conf_of(p): return float(getattr(p, "confidence", None) or 0.0)

def _dupes(polys, claims, new, hier, W, H, iou, alt=None):
    """Cross-prompt suppression, adopt's and the plain SAM3 merge's: two masks at mask IoU >= iou are one object named
    twice, if same label, kin, or two new masks. claims[k] is the label mask k redraws, or None. alt[k], when given, is a label
    whose name mask k answers to as well. Yields (loser, keeper, kind)."""
    nm = lambda k: {_norm(polys[k].label or ""), *([_norm(alt[k].label or "")] if alt[k] else [])}
    parents, sup = set(hier.values()), set()
    # without a hierarchy the loser is always the later of a pair in this order (a claim before a new mask, then the higher score),
    # so a mask is dropped only by one that stays. ponytail: a parent rule can drop the earlier one, so a hierarchy keeps index order
    o = list(range(len(polys))) if hier else sorted(range(len(polys)), key=lambda k: (claims[k] is None, -_conf_of(polys[k])))
    for i, j in ((o[x], o[y]) for x in range(len(o)) for y in range(x + 1, len(o))):
        if i in sup or j in sup: continue
        a, b = _norm(polys[i].label or ""), _norm(polys[j].label or "")   # adopt's labels are normed already, the merge's are not
        if alt and nm(i) & nm(j): b = a   # the merge's redraw exports his label, so it answers to both names
        if _rider(a, b): continue
        # a different-class pair needs both masks new, unless kin: a parent mask on his child is agreement
        if a != b and not (i in new and j in new) and not _kin(hier, a, b): continue
        if _pair_ious(polys[i], polys[j], W, H, False)[0] < iou: continue
        # the mask on his label outlives a new one, or his label comes back as dropped beside it
        _mine = lambda l: i + j - l if claims[l] is not None and claims[i + j - l] is None else l
        lower = i if _conf_of(polys[i]) < _conf_of(polys[j]) else j
        # same label is one object named twice as well
        if a == b: lose, kind = _mine(lower), "same"
        elif _kin(hier, a, b): lose, kind = _mine(i if hier.get(a) is None else j), "kin"
        # non-kin rivals: the name his hierarchy calls a parent loses, else the lower confidence
        else: lose, kind = (i if a in parents else j) if (a in parents) != (b in parents) else lower, "rival"
        sup.add(lose); yield lose, i + j - lose, kind

def _contained_dups(polys, W, H, thr=0.7, iou_max=0.5):
    """Ids of the polylines to drop when one object was drawn twice: same label, mask intersection over the
    SMALLER mask >= thr, and mask IoU < iou_max, the band _adopt_vote's same-class suppression does not own.
    The smaller mask goes, his original included, but a COCO label is always the keeper over a SAM3 mask:
    COCO first, then largest, skipping what is already dropped, so a chain collapses to its largest node."""
    live, boxes = _poly_boxes([p for p in polys if p.points and p.points[0]])
    px = lambda b: (int(b[0]*W), int(b[1]*H), max(1, int(np.ceil(b[2]*W))), max(1, int(np.ceil(b[3]*H))))
    areas = [int(_poly_pixel_mask(p, W, H, *px(b)).sum()) for p, b in zip(live, boxes)]
    order = sorted(range(len(live)), key=lambda i: (not getattr(live[i], "coco", False), -areas[i]))
    drop = {}
    for k, i in enumerate(order):
        if live[i].id in drop or not areas[i]: continue
        for j in order[k+1:]:
            if live[j].id in drop or not areas[j] or _norm(live[i].label) != _norm(live[j].label): continue
            if areas[j] > areas[i] and getattr(live[j], "adopt", None) in (None, "kept_orig"): continue   # a COCO keeper takes a SAM3 container, never one of his
            if (m := _pair_ious(live[i], live[j], W, H, False)[0]) >= iou_max: continue
            bi, bj = boxes[i], boxes[j]
            x0, y0 = int(min(bi[0], bj[0])*W), int(min(bi[1], bj[1])*H)
            x1 = int(np.ceil(max(bi[0]+bi[2], bj[0]+bj[2])*W)); y1 = int(np.ceil(max(bi[1]+bi[3], bj[1]+bj[3])*H))
            mi = _poly_pixel_mask(live[i], W, H, x0, y0, x1-x0, y1-y0); mj = _poly_pixel_mask(live[j], W, H, x0, y0, x1-x0, y1-y0)
            r = int(np.logical_and(mi, mj).sum()) / max(1, min(int(mi.sum()), int(mj.sum())))
            if r >= thr: drop[live[j].id] = (live[i].id, round(r, 4), round(m, 4), areas[j])
    return drop
def _viewsig(session):
    # the VALUE, not the object: fiftyone hands Python a new view on every state update
    v = getattr(session, "view", None)
    return None if v is None else str([(st.get("_cls"), st.get("kwargs")) for st in v._serialize()])

def _app_copy(dataset, view):
    """W371, browser 2026-09-27: the app's editor edits by list index, counted in the list it shows and applied to the stored list.
    On a filtered field it renamed a polygon he never picked, and its next save removed a hidden one. So the app gets a copy of
    what it shows, same sample and label ids, and _app_diff brings his edits back. A killed pass leaves its copy to the next launch."""
    for n in fo.list_datasets(f"{dataset.name}__app_*"):
        if (pid := n.rsplit("_", 1)[-1]).isdigit() and (pid == str(os.getpid()) or not os.path.exists(f"/proc/{pid}")): fo.delete_dataset(n)   # this pid's: a dead pass's, the pid reused
    cp = view.clone(f"{dataset.name}__app_{os.getpid()}", persistent=True)   # a clone keeps his editor schemas
    # a shape shows the export's name: a renamed export label (cam10's dogs) kept SAM3's name on screen
    gt = {s.id: _polys(s, "ground_truth") for s in dataset.select(cp.values("id")).select_fields("ground_truth")}
    for s in cp.iter_samples(autosave=True, progress=False):
        for q in _polys(s, "sam3_predictions") + _polys(s, "ground_truth_orig"): q.label = _gt_name(gt[s.id], q)
    return cp

def _app_diff(dataset, copy, shown, iou):
    """W371: his editor work in the copy, against the live set, before the pass writes it. Returns the editor deletes as click picks,
    the renames and reshapes, each with its export labels: the ones holding its old class and shape, or for his own box renamed where
    the export holds SAM3's shape, the one over it at `iou`. And the drawings (a box drawn on the outline layer is a label too). An
    outline box renamed or moved is no label and is only counted. His editor schemas go back to the set. Then the copy goes. A failed
    read costs no click, a failed schema copy no edit: the copy is kept and named."""
    labs = lambda s, f: {p.id: p for p in (getattr(s[f], "polylines", None) or getattr(s[f], "detections", None) or [])} if s.has_field(f) and s[f] else {}
    name = f"{dataset.name}__app_{os.getpid()}"   # not copy.name: a copy gone from the database raises on it
    try:
        F2 = [f for f in ("sam3_predictions", "ground_truth_orig", "review_sheet") if copy.has_sample_field(f)]
        vis = {v.id: {f: set(labs(v, f)) for f in F2} for v in shown.select_fields(F2).iter_samples()}
        live = {s.id: s for s in dataset.select(list(vis)).select_fields([*F2, "ground_truth", "metadata"]).iter_samples()}
        dels, edits, drawn, box = [], [], [], 0
        for c in copy.select_fields(F2).iter_samples():
            s = live[c.id]; gt, (W, H) = _polys(s, "ground_truth"), _wh(s)
            for f in F2:
                cur, was = labs(c, f), labs(s, f)
                dels += [{"sample_id": c.id, "field": f, "label_id": i} for i in vis[c.id][f] - set(cur)]
                for i, p in cur.items():
                    if i not in was: drawn.append((c.id, p)); continue
                    q = was[i]
                    if f == "review_sheet": box += (p.label, p.bounding_box) != (q.label, q.bounding_box); continue
                    if (p.label, p.points) == (ql := _gt_name(gt, q), q.points): continue   # the copy showed the export's name
                    # its export twin holds its shape; the class first, since a rename (W327's dogs) leaves the export under another name
                    _t = [g for g in gt if _vk(g) == _vk(q)]
                    tw, name_only = [g.id for g in _t if g.label == q.label] or [g.id for g in _t], False
                    if not tw and f == "ground_truth_orig" and p.label != q.label and W:
                        b, m, _ = _best_overlap(q, q.to_detection().bounding_box, *_poly_boxes(gt), W, H, boundary=False)
                        if b is not None and m >= iou: tw, name_only = [gt[b].id], True   # the name goes over, SAM3's shape stays
                    edits.append((c.id, f, i, p, tw, name_only, ql))
        if box: print(f"{box} outline boxes given a new class or moved in the Annotate menu: an outline box is no label, nothing changed")
    except Exception as e:
        print(f"  could not read your editor edits, your clicks still count. They stay in {name} until the next launch on this set: {type(e).__name__}: {e}")
        return [], [], []
    try:   # the schema apart: a failure here keeps his edits
        copy.reload()   # the app saves his schemas from its own process: this process's doc of the copy predates them
        if (sc := copy._doc.label_schemas) and (sc, copy._doc.active_label_schemas) != (dataset._doc.label_schemas, dataset._doc.active_label_schemas):
            dataset._doc.label_schemas = {**(dataset._doc.label_schemas or {}), **sc}
            dataset._doc.active_label_schemas = list(dict.fromkeys([*(dataset._doc.active_label_schemas or []), *(copy._doc.active_label_schemas or [])]))
            dataset.save()
    except Exception as e:
        print(f"  could not copy your Annotate menu schema back, your edits and clicks still count. It stays in {name} until the next launch on this set: {type(e).__name__}: {e}")
        return dels, edits, drawn
    copy.delete()
    return dels, edits, drawn

def _app_apply(dataset, edits, drawn, picked):
    """W371, after the clicks and ticks: a delete wins, else his edit lands last. A click on the edited polygon wins. An edit goes onto
    its SAM3 or orig polygon and onto its export labels that the pass left. A drawing goes into the export and onto his layer, where the
    next screens draw it and a click deletes it. A clicked outline box beats an edit of SAM3's polygon over its label: a click always
    takes SAM3's contribution off the object (W375). An edit of his own label still lands last."""
    clicked, n, by, lost = {l["label_id"] for l in picked}, Counter(), defaultdict(list), []
    for e in edits + drawn: by[e[0]].append(e)
    for s in dataset.select(list(by)).iter_samples(autosave=True, progress=False):
        gt, boxed = _polys(s, "ground_truth"), set(_sheet_hits(s, clicked))
        for e in by[s.id]:
            p = e[-1] if len(e) == 2 else e[3]
            p = p.to_polyline() if isinstance(p, fo.Detection) else p; pts = [[list(pt) for pt in r] for r in p.points]   # a box drawn on the outline layer
            if not (p.label or "").strip(): n["named blank, not carried"] += 1; continue
            if len(e) == 2:
                gt.append(fo.Polyline(label=p.label, points=pts, closed=True, filled=True)); n["drawn, into the export and your layer"] += 1
                s["ground_truth_orig"] = fo.Polylines(polylines=_polys(s, "ground_truth_orig") + [fo.Polyline(label=p.label, points=pts, closed=True, filled=True)]); continue
            q = next((q for q in _polys(s, e[1]) if q.id == e[2]), None)
            if e[2] in clicked or q is None or (e[1] == "sam3_predictions" and boxed & set(e[4])): n["edited and clicked, the click wins"] += 1; continue
            # a click or tick this pass can put his box back in the export under SAM3's id: found by his box's vertex set, before the edit moves it
            v = _vk(q); tw, ql = [g for g in gt if g.id in e[4]] if e[4] else [g for g in gt if _vk(g) == v], e[6]
            q.label, q.points = p.label if p.label != ql else q.label, pts
            for g in tw: g.label, g.points = p.label if p.label != ql else g.label, g.points if e[5] and _vk(g) != v else pts   # a reshape alone keeps the export's name
            k = "into the export" if tw else "its export label went in this pass" if e[4] else "no export label under it"
            n[f"class changed or reshaped, {k}"] += 1
            if not tw: lost.append(os.path.basename(s.filepath))
        s["ground_truth"] = fo.Polylines(polylines=gt)
    print(f"Annotate menu edits: {dict(n)}" + (f"\nThese edits changed SAM3's or your layer only, no export label took them: {sorted(set(lost))}" if lost else ""))

def _wait_or_enter(session, dataset=None):
    """_harvest, and on a write-back line the frames he changed in the app's editor (_EDITED), Author 2026-09-26 "follow the editor": ground_truth is copied
    Mongo-side before the app is waited on and compared after, then the copy goes. A failed copy or compare costs
    no click: the editor change is held, and said so."""
    global _EDITED; _EDITED, tmp = {}, None
    if dataset is not None and _bare_bucket_argv(sys.argv[1:]):
        try:
            sc = dataset._dataset._sample_collection; tmp = sc.database[f"zz_before_{sc.name}_{os.getpid()}"]
            for n in sc.database.list_collection_names(filter={"name": {"$regex": "^zz_before_"}}):   # a killed pass leaves its copy
                if not os.path.exists(f"/proc/{n.rsplit('_', 1)[-1]}"): sc.database.drop_collection(n)
            sc.aggregate([{"$project": {"ground_truth": 1}}, {"$out": tmp.name}])
        except Exception as e: tmp = None; print(f"  could not copy ground_truth, an editor change this pass stays in FiftyOne: {type(e).__name__}: {e}")
    try:
        out = _harvest(session)
        if tmp is not None:
            try:
                ids = [x["_id"] for x in sc.aggregate([{"$lookup": {"from": tmp.name, "localField": "_id", "foreignField": "_id", "as": "b"}},
                                                       {"$match": {"$expr": {"$ne": ["$ground_truth", {"$arrayElemAt": ["$b.ground_truth", 0]}]}}}, {"$project": {"_id": 1}}])]
                _EDITED = {str(x["_id"]): [fo.Polyline.from_dict(p) for p in (x.get("ground_truth") or {}).get("polylines") or []] for x in tmp.find({"_id": {"$in": ids}})}
            except Exception as e: print(f"  could not compare ground_truth, an editor change this pass stays in FiftyOne: {type(e).__name__}: {e}")
        return out
    finally:
        if tmp is not None:
            with contextlib.suppress(Exception): tmp.drop()

def _harvest(session):
    # session.wait() returns only when EVERY tab is closed, so Enter in the terminal ends the pass too.
    # Esc in the modal EMPTIES the selection, so it is harvested every half second and unioned.
    # Flush the tty, or a spare Enter skips the next bucket in a for loop.
    if sys.stdin.isatty(): termios.tcflush(sys.stdin, termios.TCIFLUSH)
    _t = threading.Thread(target=session.wait, daemon=True); _t.start()
    print("Close every FiftyOne tab, or press Enter in this terminal, when this pass is done.")
    _fds, picked, prev = [sys.stdin], {}, set()
    ticked, prev_t, kept, spared = set(), set(), set(), set()
    # A VIEW change (a class change in the app) clears both selections in one poll, so the TICK half is held
    # for three polls after the view VALUE changes.
    _vs, _wipe = _viewsig(session), 0
    while _t.is_alive():
        # READ ORDER IS THE GUARD: selections first, view last, and the _wipe update ABOVE `gone`.
        cur, n = list(session.selected_labels), len(picked)
        cleared = bool(prev and picked) and not cur   # the app's Annotate button and Escape both empty its highlight
        curt, nt = set(session.selected), len(ticked)
        _nvs = _viewsig(session)
        if _nvs != _vs: _vs, _wipe = _nvs, 3
        elif _wipe: _wipe -= 1
        # only a selected->unselected transition drops a pick, and only on an image where a pick SURVIVED
        # (`alive`), or a wipe plus a fresh click reads as an unclick. Spared picks are printed after the loop.
        # NOT gated on _wipe: a view change empties `cur`, so the window would only hide an unclick.
        if cur:
            live = {l["label_id"] for l in cur}
            spared -= live   # a re-click is his own act
            alive = {l["sample_id"] for l in cur if l["label_id"] in prev}
            _go = [k for k, v in picked.items() if k in prev and k not in live]
            spared |= {k for k in _go if picked[k]["sample_id"] not in alive}
            for k in [k for k in _go if picked[k]["sample_id"] in alive]: del picked[k]
        for l in cur: picked.setdefault(l["label_id"], l)
        prev = {l["label_id"] for l in cur}   # an Esc clear ends the window, so a reopened image starts fresh
        # TICKS are a plain toggle, but a grid Escape empties every tick at once, so only a single untick, or the
        # same selection minus some images, subtracts; a multi-image vanish is kept.
        # Blind spot: a clear with identical view stages still loses a lone tick.
        gone = prev_t - curt
        if _wipe and gone: kept |= gone
        elif (prev_t & curt) or (len(gone) <= 1 and not curt): ticked -= gone
        elif gone: kept |= gone
        ticked |= curt; kept -= curt; prev_t = curt   # a re-tick is his own act
        if cleared: print(f"\nThe highlight went empty (the app cleared it, or you unclicked the last mark). Your {len(picked)} marked polygons still count.")
        if len(picked) != n or len(ticked) != nt or cleared: print(f"\r{len(picked)} polygons marked, {len(ticked)} images ticked", end="", flush=True)
        if select.select(_fds, [], [], 0.5)[0]:
            if sys.stdin.readline(): break
            _fds = []   # EOF is not Enter: stop polling a closed stdin, keep waiting on the tabs
    # a tab close ends the loop at the top, so a "y" typed early would answer the next input() unread
    if sys.stdin.isatty(): termios.tcflush(sys.stdin, termios.TCIFLUSH)
    for l in session.selected_labels: picked.setdefault(l["label_id"], l)
    ticked |= set(session.selected)
    if picked or ticked: print()
    # the guard cannot tell an Escape wipe from the tick-count pill's clear, so say what it kept; printed
    # here, never in the loop, where the \r status line paints over it
    if kept & ticked:
        print(f"KEPT {len(kept & ticked)} ticks through a cleared selection. If that was not your doing, "
              f"rerun the pass and check them. Images: {sorted(kept & ticked)}")
    # same report for spared picks; neither line may say the marks are applied, or "press Enter and rerun"
    if (_sp := spared & set(picked)):
        print(f"KEPT {len(_sp)} polygon marks through a cleared selection. A mark that was really an unmark "
              f"deletes your own label. If that was not your doing, rerun the pass and check them. "
              f"Frames: {sorted({picked[k]['sample_id'] for k in _sp})}")
    return list(picked.values()), ticked

def _ckpt_key(p):   # best.pt retrained in place is another model; a hub name ultralytics downloads has no file yet
    return [os.path.realpath(p), os.path.getmtime(p) if os.path.isfile(p) else None]

def yolo_judge_workflow(dataset, args, gt_field="ground_truth", pred_field="yolo_judge_preds"):
    """YOLO as the second opinion on a SAM3-labelled set: the --sam3_eval review with YOLO proposing.
    `gt_field` is never written here. Only a click deletes, in the post-session branch."""
    from ultralytics import YOLO
    print("\n--- YOLO Judge ---")
    # stale adopt outlines cleared before the queue opens; `.exists()` so last_modified_at is not bumped
    if dataset.has_sample_field("review_sheet"):
        dataset.exists("review_sheet").set_field("review_sheet", None).save("review_sheet")
    dataset.compute_metadata()
    model = YOLO(args.yolo_judge)
    # every annotation is judged whatever its class; cls_names only weights the score. --sam3_hierarchy maps the rest
    live = {_norm(c) for c in dataset.distinct(f"{gt_field}.polylines.label") if c}
    names, hier = {_norm(v) for v in model.names.values()}, _hier(args)
    # a proposal in a parent class the annotations never use is renamed to the live child
    fold = {_norm(par): _norm(ch) for ch, par in (x.split(":", 1) for x in (args.sam3_hierarchy or []))
            if _norm(par) not in live and _norm(ch) in live}
    cls_names = sorted({c for c in live if c in names or hier.get(c) in names or c in fold.values()})
    if fold: print(f"  YOLO proposals renamed to your classes: {', '.join(f'{p}->{c}' for p, c in sorted(fold.items()))}")
    if (via := sorted(c for c in cls_names if c not in names and c not in fold.values())):
        print(f"  matched through --sam3_hierarchy: {', '.join(f'{c}->{hier[c]}' for c in via)}")
    if (unmapped := sorted(live - set(cls_names))):
        n_un = sum(dataset.count_values(f"{gt_field}.polylines.label").get(c, 0) for c in unmapped)
        print(f"  {os.path.basename(args.yolo_judge)} has no class for {unmapped} and no --sam3_hierarchy parent: "
              f"those {n_un} annotations will still be scored, so expect 'missed' and 'renamed' on them that are "
              f"NOT real faults. Map them with --sam3_hierarchy child:parent or ignore that class in the app.")
    # the forward pass is reused, the scoring never is; `fold` is baked into the stored labels, so it is in the sig
    sig = [*_ckpt_key(args.yolo_judge), default_conf, args.nms_iou, SIMPLIFY_EPS,
           sorted(f"{p}->{c}" for p, c in fold.items())]
    nonempty = F(f"{pred_field}.polylines").length() > 0
    if not args.reevaluate and dataset.info.get("yolo_judge_sig") == sig and dataset.exists(pred_field).count() == len(dataset):
        n_raw, n_samples = dataset.count(f"{pred_field}.polylines"), dataset.match(nonempty).count()
        print(f"YOLO judge: reusing {n_raw} stored proposals for this checkpoint (--reevaluate to run YOLO again).")
    else:
        dataset.info.pop("yolo_judge_sig", None); dataset.save()
        n_raw = _yolo_into(dataset, model, pred_field, default_conf, args.nms_iou)
        if fold:
            for sample in dataset.iter_samples(autosave=True, progress=True):
                ps = _polys(sample, pred_field)
                if not any(_norm(p.label) in fold for p in ps): continue
                for p in ps: p.label = fold.get(_norm(p.label), p.label)
                sample[pred_field] = fo.Polylines(polylines=ps); sample.save()
        n_samples = dataset.match(nonempty).count()
        dataset.info["yolo_judge_sig"] = sig; dataset.save()
        print(f"YOLO judge: {n_raw} proposals across {n_samples}/{len(dataset)} samples -> '{pred_field}'; "
              f"'{gt_field}' untouched. Judged classes: {cls_names}")
    del model; torch.cuda.empty_cache(); gc.collect()
    dataset.add_dynamic_sample_fields()
    args.sam3_eval = True   # opens the ranked disagreement review, keeps this off every merge path
    # marked before the scorer writes sam3_worst, so a later --sam3_eval never reopens it as SAM3's
    dataset.info["sam3_worst_by"] = f"<yolo_judge:{os.path.basename(args.yolo_judge)}>"; dataset.save()
    sam3_disagreement_eval(dataset, args, cls_names, cls_names, n_samples,
                           sieve=(0, args.nms_iou, "off"), gt_field=gt_field, pred_field=pred_field)
    print(f"YOLO's proposals are in '{pred_field}'. SAM3's masks are untouched in 'sam3_predictions'.")

def sam3_disagreement_eval(dataset, args, prompts, cls_names, n_samples, sieve=(64, 0.8, ""),
                           gt_field="ground_truth", pred_field="sam3_predictions"):
    """Rank every prediction and annotation by disagreement, each prediction against its OWN best-overlapping
    annotation (not one-to-one), so competing prompts on one object both keep their claim. `gt_field` geometry is never touched."""
    min_area, nms_iou, border_rule = sieve
    thr = args.eval_iou_thresh
    _who = "YOLO" if pred_field == "yolo_judge_preds" else "SAM3"
    print(f"\n--- {_who} Disagreement (mask IoU, best-overlap, iou>={thr}) ---")
    out = args.sam3_disagree_out or os.path.join(os.path.expanduser("~/bench_any/runs/sam3_disagree"),
                                                 dataset.name, f"disagreement_{datetime.datetime.now():%Y-%m-%d_%H%M%S}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows, n_agree, n_renamed, n_invented, n_missed = [], 0, 0, 0, 0
    cls_set = set(cls_names)   # an out-of-vocabulary invention scores 0
    hier = _hier(args)   # a superset prompt is not a dispute
    for sample in dataset.iter_samples(autosave=True, progress=True):
        preds = _polys(sample, pred_field)
        # to_polyline() copies, so the verdict lands on the originals the app edits
        gt_src = list(get_labels_from_sample(sample, gt_field) or [])
        gts = [l if isinstance(l, fo.Polyline) else l.to_polyline() for l in gt_src]   # an emptied image scores 0, not its old sam3_worst (W32)
        W, H = (sample.metadata.width, sample.metadata.height) if sample.metadata and sample.metadata.width else (1, 1)
        boxes = [p.to_detection().bounding_box for p in preds]
        gboxes = [g.to_detection().bounding_box for g in gts]
        covered, gt_claim = set(), {}   # gt index -> (score, what SAM3 called it, verdict)
        for i, p in enumerate(preds):
            conf = float(getattr(p, "confidence", None) or 0.0)
            # the best-scoring poly of a DIFFERENT class over the same object
            rival = max((float(getattr(q, "confidence", None) or 0.0) for j, q in enumerate(preds)
                         if j != i and q.label != p.label and not _kin(hier, p.label, q.label) and compute_iou(boxes[i], boxes[j]) >= DISAGREE_COMPETE_IOU),
                        default=None)
            margin = conf if rival is None else conf - rival
            best, seg_iou, b_iou = _best_overlap(p, boxes[i], gts, gboxes, W, H)
            matched = best is not None and seg_iou >= thr
            gt_label = gts[best].label if matched else ""
            if matched: covered.add(best)
            renamed = matched and gt_label != p.label and hier.get(gt_label) != p.label   # parent over child agrees
            verdict = "renamed" if renamed else ("agrees" if matched else "invented")
            # three disjoint stories, three branches, not one weighted sum
            if renamed:      # a different label on your object: credible with agreement and a won margin
                score = DISAGREE_W_CLASS * seg_iou * max(0.0, margin) + DISAGREE_W_CONF * conf
            elif matched:    # same label, so the only story is geometry, and only where SAM3 is sure
                score = (DISAGREE_W_SEG * (1.0 - seg_iou) + DISAGREE_W_BOUNDARY * (1.0 - b_iou)) * conf
            else:            # nothing of yours here: a miss only if the dataset annotates this label
                score = (DISAGREE_W_CLASS + DISAGREE_W_CONF) * conf if p.label in cls_set else 0.0
            p.sam3_gt_label, p.sam3_verdict = gt_label, verdict
            p.sam3_seg_iou, p.sam3_boundary_iou, p.sam3_class_margin, p.sam3_disagree = seg_iou, b_iou, margin, score
            n_agree += int(verdict == "agrees"); n_renamed += int(verdict == "renamed"); n_invented += int(verdict == "invented")
            # the claim that speaks for your annotation: a won renaming beats an agreement beats a lost renaming
            if matched:
                rank = (2 if (renamed and margin > 0) else 1 if not renamed else 0, score)
                if rank > gt_claim.get(best, ((-1, -1.0),))[0]:
                    gt_claim[best] = (rank, p.label, verdict, score)
            rows.append((sample.filepath, "sam3_pred", p.label, gt_label, conf, margin, seg_iou, b_iou, score, verdict))
        worst = max([float(getattr(p, "sam3_disagree", 0.0) or 0.0) for p in preds], default=0.0)
        for k, g in enumerate(gts):
            if k in covered:   # the verdict goes on your annotation too, so the app can sort what you edit
                _, lab, v, s = gt_claim.get(k, (None, "", "agrees", 0.0))
                gt_src[k].sam3_verdict, gt_src[k].sam3_label, gt_src[k].sam3_disagree = v, lab, s
                continue
            n_missed += 1
            miss = DISAGREE_W_SEG + DISAGREE_W_BOUNDARY
            gt_src[k].sam3_verdict, gt_src[k].sam3_label, gt_src[k].sam3_disagree = "missed", "", miss
            worst = max(worst, miss)
            rows.append((sample.filepath, "gt_missed", "", g.label, 0.0, 0.0, 0.0, 0.0, miss, "missed"))
        sample["sam3_worst"] = worst   # over predictions AND annotations, so a miss ranks its image too
    rows.sort(key=lambda r: r[8], reverse=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# prompts", " ".join(prompts), "classes", " ".join(cls_names), "match", "best_overlap_mask_iou",
                    "match_iou", thr, "sieve_min_area", min_area, "sieve_nms", nms_iou, "border_rule", border_rule])
        w.writerow(["image", "kind", "sam3_label", "gt_label", "confidence", "class_margin", "seg_iou", "boundary_iou", "disagree", "verdict"])
        w.writerows(rows)
    dataset.add_dynamic_sample_fields()   # declare sam3_* so the app can show, filter and sort on them
    print(f"{_who} disagreement: {len(rows)} rows over {n_samples} samples -> {out}\n"
          f"  {n_agree} agree, "
          f"{n_renamed} renamed ({_who} covers your instance and calls it something else), "
          f"{n_invented} invented (covers nothing you annotated), "
          f"{n_missed} missed (annotation {'no prompt covered' if _who == 'SAM3' else 'YOLO saw nothing at'}).\n"
          f"  sieve as run: min_area={min_area}, nms={nms_iou}, border={border_rule}. '{gt_field}' geometry untouched.")

CONSENSUS_SRC = {"a": "yolo_a", "b": "yolo_b", "s": "sam3_predictions"}
# review order, worst first; consensus_worst = rank + conf/10. `single` sits under the >= 2 queue cut on purpose
CONSENSUS_RANK = dict(unverifiable=0, confirmed=0, mask_replaced=1, renamed=1.5, single=1.8, add_proposed=2, added=2.2, rename_proposed=3, disputed=4, unsupported=5)

def _consensus_vote(sample, args, hier, vocab, emit, gt_field="ground_truth"):
    """Vote two YOLO checkpoints and SAM3 against each annotation. No model, no I/O; edits the annotation objects in place.
    Two of three agreeing rewrites the label or the mask in place; everything else is proposed.
    Returns the CSV rows; the caller iterates with autosave, so nothing is saved here."""
    if not (sample.metadata and sample.metadata.width): sample["consensus_worst"] = 0.0; return []
    thr, W, H = args.eval_iou_thresh, sample.metadata.width, sample.metadata.height
    gts = [l for l in (get_labels_from_sample(sample, gt_field) or []) if isinstance(l, fo.Polyline) and l.points and l.points[0]]
    gboxes = [g.to_detection().bounding_box for g in gts]
    _conf = lambda p: float(getattr(p, "confidence", None) or 0.0); same = lambda a, b: a == b or _kin(hier, a, b)
    # floor applies to a and b only: SAM3 is already gated by --sam3_score, a second floor double-thresholds
    polys = {t: [p for p in _polys(sample, f, live=True) if t == "s" or _conf(p) >= args.consensus_vote_conf] for t, f in CONSENSUS_SRC.items()}
    finest = lambda a, b: a if a in hier else b   # for a kin pair exactly one is a hier key (the child)
    _allsame = lambda Ls: all(same(a, b) for i, a in enumerate(Ls) for b in Ls[i+1:])
    votes, voted = [dict() for _ in gts], set()
    for t, ps in polys.items():
        for j, p in enumerate(ps):
            box = p.to_detection().bounding_box
            # ponytail: one blob over two touching objects votes twice; split votes if that shows up in review
            for k in sorted((k for k in range(len(gts)) if compute_iou(box, gboxes[k]) > 0.05), key=lambda k: -compute_iou(box, gboxes[k]))[:3]:
                m, _ = _pair_ious(p, gts[k], W, H, boundary=False)
                if m >= thr:
                    voted.add((t, j))   # matched anything at all, so never an addition candidate
                    if m > votes[k].get(t, (0, 0, None, 0.0))[3]: votes[k][t] = (p.label, _conf(p), p, m)
    rows, worst = [], []
    _pts = lambda P: [[tuple(map(float, q)) for q in r] for r in (P or [])]   # tuple-vs-list is not a difference
    for k, g in enumerate(gts):
        G, v, X = g.label, votes[k], ""
        sup = [t for t, (L, *_) in v.items() if same(L, G)]
        if not (G in emit or hier.get(G) in emit or any(hier.get(c) == G and c in emit for c in hier)):
            verdict = "unverifiable"
        elif not v:
            verdict = "unsupported"
        elif len(sup) >= 2:
            verdict = "confirmed"
            if "s" in sup and not getattr(g, "coco", False) and _pts(v["s"][2].points) != _pts(g.points):
                g.points, g.confidence, verdict = v["s"][2].points, v["s"][1], "mask_replaced"
        elif len(v) >= 2 and not sup and _allsame([L for L, *_ in v.values()]):
            X = functools.reduce(finest, [L for L, *_ in v.values()])
            if X not in vocab: verdict = "disputed"
            elif X != "car" and not getattr(g, "coco", False) and min(c for t, (L, c, *_) in v.items() if t != "s") >= args.consensus_auto:
                g.label = X
                if "s" in v: g.points, g.confidence = v["s"][2].points, v["s"][1]
                verdict = "renamed"
            else: verdict = "rename_proposed"
        else:   # one lone agreeing vote is thin evidence, not a contest; a contest needs a contradicting label
            verdict = "single" if len(v) == 1 and sup else "disputed"
        g.consensus, g.consensus_new_label = verdict, X
        g.consensus_votes = " ".join(f"{t}:{L}" for t, (L, *_) in sorted(v.items()))
        g.consensus_conf = min((c for _, c, *_ in v.values()), default=0.0)
        rows.append((sample.filepath, G, verdict, X, g.consensus_votes, g.consensus_conf, *(v[t][3] if t in v else 0.0 for t in "abs")))
        worst.append(CONSENSUS_RANK[verdict] + g.consensus_conf / 10)

    # Additions: source polys that spoke for no annotation and sit clear of every existing one.
    pool = sorted(((t, j) for t in polys for j in range(len(polys[t])) if (t, j) not in voted
                   and all(compute_iou(polys[t][j].to_detection().bounding_box, gb) <= args.pred_dedup_iou for gb in gboxes)),
                  key=lambda tj: -_conf(polys[tj[0]][tj[1]]))
    clusters, proposals = [], []
    for t, j in pool:   # ponytail: greedy and order-dependent, seeded by the most confident poly
        p = polys[t][j]
        c = next((c for c in clusters if t not in c and _pair_ious(p, next(iter(c.values())), W, H, boundary=False)[0] >= thr), None)
        if c is None: clusters.append({t: p})
        else: c[t] = p
    def _new(X, geom, conf, verdict, c):
        q = fo.Polyline(label=X, points=geom.points, closed=True, filled=True)
        q.confidence = q.consensus_conf = conf
        q.consensus, q.consensus_new_label = verdict, X
        q.consensus_votes = " ".join(f"{t}:{p.label}" for t, p in sorted(c.items()))
        rows.append((sample.filepath, "", verdict, X, q.consensus_votes, q.confidence, 0.0, 0.0, 0.0))
        worst.append(CONSENSUS_RANK[verdict] + q.confidence / 10)
        return q
    accepted = []
    for c in clusters:
        labs = [p.label for p in c.values()]
        # a confident SAM3 find goes straight in and the review rejects; SAM3 silence never deletes
        if args.consensus_sam3_add and "s" in c and _conf(c["s"]) >= args.consensus_sam3_add:
            X = functools.reduce(finest, labs) if _allsame(labs) else _norm(c["s"].label)
            if X not in vocab: continue
            if X != "car":   # car is never auto-accepted, his lawnmower reads as car
                accepted.append(_new(X, c["s"], _conf(c["s"]), "added", c))
                continue
        if len(c) < 2 or not _allsame(labs): continue
        X = functools.reduce(finest, labs)
        if X not in vocab: continue
        proposals.append(_new(X, c["s"] if "s" in c else max(c.values(), key=_conf), min(_conf(p) for p in c.values()), "add_proposed", c))
    # append to the raw field, not `gts`: `gts` dropped point-less polylines
    if accepted: sample[gt_field] = fo.Polylines(polylines=_polys(sample, gt_field) + accepted)
    sample["consensus_proposals"] = fo.Polylines(polylines=proposals)
    sample["consensus_worst"] = max(worst, default=0.0)
    return rows

def _snap_polys(labels):
    """Fresh-id copies of the polylines for a `*_orig` snapshot (clone_sample_field keeps ids, so a
    per-label delete would hit the snapshot too). The `coco` stamp is carried."""
    out = []
    for p in (labels or []):
        if not isinstance(p, fo.Polyline): continue
        q = fo.Polyline(label=p.label, points=[list(r) for r in p.points], closed=p.closed, filled=p.filled)
        if getattr(p, "coco", False): q.coco = True
        out.append(q)
    return out

def _snap_orig(dataset, gt_field):
    """Snapshot `gt_field` into `<gt_field>_orig`, keyed on the samples lacking it, not the field."""
    for sample in (dataset.exists(f"{gt_field}_orig", False) if dataset.has_sample_field(f"{gt_field}_orig") else dataset).iter_samples(autosave=True, progress=True):
        sample[f"{gt_field}_orig"] = fo.Polylines(polylines=_snap_polys(get_labels_from_sample(sample, gt_field)))

def _hier_one_hop(args):
    """_hier(args), refused when a parent is itself a child: the kin test would depend on pair order."""
    hier = _hier(args)
    if (nested := sorted(set(hier.values()) & set(hier))):
        sys.exit(f"--sam3_hierarchy is one level deep; {nested} appear as both child and parent")
    return hier

def consensus_workflow(dataset, args, vocab, class_thresholds, gt_field="ground_truth"):
    """Run both YOLO checkpoints and SAM3 over the dataset, then vote them against the annotations."""
    print("\n--- Consensus corrector ---")
    if not args.reevaluate and dataset.has_sample_field("consensus_worst"):
        print(f"Consensus already on '{dataset.name}' (consensus_worst present). Opening the review; pass --reevaluate to vote again.")
        return
    if dataset.count(f"{gt_field}.detections"): exit(f"--consensus needs polyline {gt_field}; this dataset holds Detections (zoo import). Both local importers emit polylines.")
    dataset.compute_metadata()
    _snap_orig(dataset, gt_field)
    from ultralytics import YOLO
    emit = set()
    for tag, ckpt in zip("ab", args.consensus):
        # load the ckpt either way: `emit` needs its class names or a class it cannot say scores as a decline
        model = YOLO(ckpt); emit |= {_norm(n) for n in model.names.values()}
        # the file, its age and every knob that shapes the polylines: another run's best.pt is not this one
        fld, ck = f"yolo_{tag}", [*_ckpt_key(ckpt), default_conf, args.nms_iou, SIMPLIFY_EPS]
        # --reevaluate re-votes, it does not re-detect: stored dets are reused, conf filters at vote time
        if dataset.info.get(f"consensus_{fld}") == ck and dataset.exists(fld).count() == len(dataset):
            print(f"  {fld} = {os.path.basename(ckpt)}: reusing stored polylines")
        else:
            dataset.info.pop(f"consensus_{fld}", None); dataset.save()
            n = _yolo_into(dataset, model, fld, default_conf, args.nms_iou)
            dataset.info[f"consensus_{fld}"] = ck; dataset.save()
            print(f"  {fld} = {os.path.basename(ckpt)}: {n} polylines")
        del model; torch.cuda.empty_cache(); gc.collect()
    args.sam3_eval = True
    sam3_refine_workflow(dataset, args, class_thresholds)
    emit |= {_cls_of(p) for p in args.sam3_prompt}   # what SAM3 CAN name, not what it happened to find; _cls_of, not _norm, or "a penguin" emits 'apenguin' and never matches
    hier = _hier_one_hop(args)

    rows = []
    for sample in dataset.iter_samples(autosave=True, progress=True):
        rows += _consensus_vote(sample, args, hier, vocab, emit, gt_field)
    out = os.path.join(os.path.expanduser("~/bench_any/runs/consensus"), dataset.name, f"consensus_{datetime.datetime.now():%Y-%m-%d_%H%M%S}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows.sort(key=lambda r: (CONSENSUS_RANK[r[2]], r[5]), reverse=True)
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows([["# models", *[os.path.realpath(c) for c in args.consensus], "prompts", " ".join(args.sam3_prompt),
                                  "match_iou", args.eval_iou_thresh, "auto", args.consensus_auto, "dedup_iou", args.pred_dedup_iou,
                                  "vocab", " ".join(sorted(vocab)), "hier", " ".join(f"{k}:{v}" for k, v in hier.items())],
                                 ["image", "gt_label", "verdict", "new_label", "votes", "conf", "iou_a", "iou_b", "iou_s"], *rows])
    dataset.add_dynamic_sample_fields()   # declare consensus_* so the app can show, filter and sort on them
    print(f"Consensus: {len(rows)} rows -> {out}\n  " + ", ".join(f"{c} {v}" for v, c in Counter(r[2] for r in rows).most_common()) +
          "\n  AUTO applied: renamed, mask_replaced. PROPOSED (review in app): rename_proposed, add_proposed"
          " (consensus_proposals field), unsupported (delete?). single = one agreeing vote only. disputed = sources disagree."
          " unverifiable = no source can emit this class.")

# queue rank, worst first; a wrong class tops it. `empty`, `fewer` and `dup` write a CSV row but never queue
ADOPT_RANK = dict(classchange=7, more=6, dropped=5, lowiou=4, rename=3)
# every review pass; `dropped` is OUT, it queues no images
GALLERY_REASONS = (set(ADOPT_RANK) | {"unsupported"}) - {"dropped"}

def _adopt_vote(sample, args, hier, vocab, gt_field="ground_truth", pred_field="sam3_predictions"):
    """Adopt this sample's SAM3 masks as the annotations and flag what a human must look at. No model, no I/O, no save
    (the caller autosaves). Returns (csv rows, out-of-vocab skips, instances adopted, empty scene, thin, kin, rival, same)."""
    add_thr = args.sam3_add_score or 0.0
    n_thin = n_kin = n_rival = n_same = 0
    if not (sample.metadata and sample.metadata.width):
        sample["review_reasons"], sample["review_worst"] = [], 0.0
        return [], 0, 0, 0, 0, 0, 0, 0
    W, H = sample.metadata.width, sample.metadata.height
    # the size floor deletes, off by default and house-only; points tested first; COCO is exempt: Author kept 288 tiny COCO sheep on this rule
    small = lambda a: args.min_label_area and a.points and a.points[0] and _px_area(a.points, W, H) < args.min_label_area and not getattr(a, "coco", False)
    # RAW list, never filtered: rebuilding gt from a subset deletes what the filter dropped
    gts = _polys(sample, gt_field)
    # vocab filter MUST precede the count, or out-of-vocab masks read as finds and empty the gt
    raw = _polys(sample, pred_field, live=True)
    sams = [p for p in raw if _norm(p.label) in vocab]
    n_gt, n_sam, skipped = len(gts), len(sams), len(raw) - len(sams)
    # an older --sam3_eval's attributes would render beside a verbatim label and read as a confidence
    for g in gts:
        for k in ("sam3_disagree", "sam3_verdict", "sam3_label"):
            if g.has_attribute(k): g.delete_attribute(k)
    # `quiet` goes to the CSV and never queues
    reasons, quiet, pend, adopted = set(), set(), [], []

    if n_sam == 0:
        # only where he had nothing: the fill model never overwrites a real annotation
        if not n_gt and args.sam3_adopt_fill and (fills := _polys(sample, "yolo_fill", live=True)):   # a stale fill of an older run is not this one's (W39)
            for p in fills:
                if _norm(p.label) not in _review_scope(args): continue   # the fill model knows 80 classes; only the prompted ones
                a = fo.Polyline(label=_norm(p.label), points=p.points, closed=True, filled=True)
                if getattr(p, "confidence", None) is not None: a.confidence = p.confidence
                a.adopt = "yolo_fill"
                adopted.append(a)
            sample[gt_field] = fo.Polylines(polylines=adopted)
        if n_gt: quiet.add("empty")
        if adopted: reasons.add("more")   # the more pass shows a fill as an addition and a tick deletes it
        # the floor holds where SAM3 found nothing too: 4 20Sept seeds under 256 px kept theirs. Add-only keeps his verbatim
        if not args.sam3_add_only and (fdrop := {g.id for g in gts if small(g)}):
            pend += [("floored", _norm(g.label), getattr(g, "adopt", None) or "", 0.0, _conf_of(g)) for g in gts if g.id in fdrop]
            sample[gt_field] = fo.Polylines(polylines=[g for g in gts if g.id not in fdrop])
    else:   # an unmatched annotation is carried through as kept_orig below
        # a point-less annotation gets a zero box: it never matches and the index stays aligned with gts
        gboxes = [g.to_detection().bounding_box if g.points and g.points[0] else (0.0, 0.0, 0.0, 0.0) for g in gts]
        new_idx, sup, claims, agreed, ious = set(), set(), [], set(), []
        for p in sams:
            X = _norm(p.label)
            a = fo.Polyline(label=X, points=p.points, closed=True, filled=True)
            if getattr(p, "confidence", None) is not None: a.confidence = p.confidence
            # an annotation that already AGREES with this mask wins the claim over one that disputes it
            # a bike mask never claims his person, nor a person mask his bike: a rider and his bike are two objects
            best, seg_iou, _ = _best_overlap(p, p.to_detection().bounding_box, gts, [(0.0,) * 4 if _rider(X, g.label) else b for g, b in zip(gts, gboxes)], W, H, boundary=False, prefer=_agree(hier, X), prefer_floor=args.adopt_match_iou)
            if best is not None and seg_iou < args.adopt_match_iou: best = None   # a new object beside his
            # add-only: a mask on something already annotated is dropped; only his own click removes a COCO label
            if best is not None and args.sam3_add_only: continue
            # a COCO label SAM3 agreed with goes through verbatim below, not rebuilt as 'dropped'
            if best is not None and getattr(gts[best], "coco", False): agreed.add(best); continue
            # an addition takes the strict gate; a class dispute over his box wants recall
            if best is None and (getattr(a, "confidence", None) or 0.0) < add_thr: n_thin += 1; continue
            G = _norm(gts[best].label) if best is not None else ""
            # no `lowiou` tag can be produced here: `best` is nulled under adopt_match_iou above
            swap = best is not None and G != X and not _kin(hier, G, X)
            tag = "classchange" if swap else "adopted"
            # a kin mask keeps his class and only proposes its own: a child of his asks nothing, a parent asks rename
            if best is not None and G != X and not swap:
                a.label = G
                if hier.get(G) == X: a.alt_name, a.alt_name_conf = X, round(_conf_of(a), 4); reasons.add("rename")
            if tag != "adopted": a.prev_label = gts[best].label
            a.adopt = tag
            if best is None: new_idx.add(len(adopted))
            adopted.append(a); claims.append(best); ious.append(seg_iou)
        # cross-prompt suppression after the add gate: two prompts on one object is one object named twice
        for lose, keep, kind in _dupes(adopted, claims, new_idx, hier, W, H, args.adopt_match_iou):
            sup.add(lose); n_same += kind == "same"; n_kin += kind == "kin"; n_rival += kind == "rival"
            # keep the child, the parent goes on it as alt_name for the rename pass
            if kind == "kin" and hier.get(adopted[keep].label) == adopted[lose].label:
                adopted[keep].alt_name, adopted[keep].alt_name_conf = adopted[lose].label, round(_conf_of(adopted[lose]), 4); reasons.add("rename")
        if sup: adopted, claims, ious = ([x for k, x in enumerate(l) if k not in sup] for l in (adopted, claims, ious))
        # booked after suppression, or a dispute whose mask lost still queues with a CSV row
        for a, c, v in zip(adopted, claims, ious):
            if a.adopt != "classchange": continue
            if max(_px_area(a.points, W, H), _px_area(gts[c].points, W, H)) >= UNSUP_MIN_AREA: reasons.add("classchange")
            pend.append(("classchange", a.label, a.prev_label, round(v, 4), getattr(a, "confidence", None) or 0.0))
        # surviving claims only: a suppressed mask takes its claim with it, or the annotation it claimed silently leaves ground_truth
        _cl = [c for c in claims if c is not None]
        if len(_cl) != len(set(_cl)): quiet.add("dup")   # two masks left on one of your objects: recorded, not queued
        if new_idx - sup: reasons.add("more")   # genuinely new objects after the add gate and suppression
        _added = {a.id for a in adopted}     # SAM3's own, fixed before the annotations are carried through
        _fresh = {a.id for a, c in zip(adopted, claims) if c is None}   # the additions: SAM3's that claimed no box of his
        # an uncovered annotation is kept and queued
        for i, g in enumerate(gts):
            if i in _cl: continue
            # add-only: the annotation is not up for review, so it goes back as it was, with no `dropped`
            if args.sam3_add_only or i in agreed: adopted.append(g); continue
            # a point-less annotation can never match: carry it through, never overwrite it away
            if not (g.points and g.points[0]): adopted.append(g); continue
            k = fo.Polyline(label=_norm(g.label), points=[list(r) for r in g.points], closed=True, filled=True)
            if getattr(g, "coco", False): k.coco = True   # a fresh Polyline carries no stamp
            k.adopt = "kept_orig"; adopted.append(k)
            reasons.add("dropped"); pend.append(("dropped", _norm(g.label), g.label, 0.0, 0.0))
        # add-only floors SAM3's own only, his annotations survive
        if (fdrop := {a.id for a in adopted if small(a) and (a.id in _added or not args.sam3_add_only)}):
            pend += [("floored", _norm(a.label), getattr(a, "adopt", None) or "", 0.0, _conf_of(a)) for a in adopted if a.id in fdrop]
            adopted = [a for a in adopted if a.id not in fdrop]
        # W322, Author 2026-09-27: "if it is same class on top of another it is likely the same object", with his car rule "They stay except
        # when there is already a car underneath it". A SAM3 addition covering half or more of a same-class or kin box of his, a box that
        # is still labelled after the size floor, is that box's object: the addition goes, his box stays
        _mk = lambda p: _poly_pixel_mask(p, W, H, 0, 0, W, H).astype(bool)
        _held = lambda g, a: any(k is not a and (_vk(k) == _vk(g) or _agree(hier, _norm(k.label))(_norm(g.label)) and _pair_ious(k, g, W, H, False)[0] >= args.adopt_match_iou) for k in adopted)
        if (sdrop := {a.id for a in adopted if a.id in _fresh and a.points and a.points[0] and any(
                _agree(hier, _norm(a.label))(_norm(g.label)) and g.points and g.points[0] and (mg := _mk(g)).any()
                and (_mk(a) & mg).sum() / mg.sum() >= 0.5 and _held(g, a) for g in gts)}):
            pend += [("same_object", _norm(a.label), "added", 0.0, _conf_of(a)) for a in adopted if a.id in sdrop]
            adopted = [a for a in adopted if a.id not in sdrop]
        # one object drawn twice with loose overlap: containment over the smaller mask catches it. Never a COCO label, only his click removes one,
        # and add-only drops only SAM3's own doubles, never the annotation one of them landed on
        _coco = {a.id for a in adopted if getattr(a, "coco", False)}
        cdrop = {i: v for i, v in _contained_dups(adopted, W, H).items() if i not in _coco and (i in _added or not args.sam3_add_only)}
        if cdrop:
            pend += [("contained", _norm(a.label), getattr(a, "adopt", None) or "", cdrop[a.id][1], _conf_of(a)) for a in adopted if a.id in cdrop]
            adopted = [a for a in adopted if a.id not in cdrop]
        if cdrop or fdrop:
            # drop a reason no surviving stamp backs, or the image queues with nothing on screen; `more` is rebuilt live
            _live = {getattr(a, "adopt", None) for a in adopted}
            reasons -= {r for r, st in dict(dropped="kept_orig", classchange="classchange").items() if st not in _live}
        sample[gt_field] = fo.Polylines(polylines=adopted)

    rows = [(sample.filepath, n_gt, n_sam, len(adopted), *t) for t in pend]
    rows += [(sample.filepath, n_gt, n_sam, len(adopted), r, "", "", "", "") for r in sorted((reasons & {"more"}) | quiet)]
    sample["review_reasons"] = sorted(reasons)
    sample["review_worst"] = float(max((ADOPT_RANK[r] for r in reasons), default=0.0))
    return rows, skipped, len(adopted), int(n_sam == 0 and not n_gt and not adopted), n_thin, n_kin, n_rival, n_same

def sam3_adopt_workflow(dataset, args, class_thresholds, gt_field="ground_truth", pred_field="sam3_predictions"):
    """SAM3-as-default: SAM3 output BECOMES the annotations and the human pass rejects over a triggered
    queue. SAM3 finding nothing removes no annotation but one under --min_label_area; an uncovered one is carried through as kept_orig."""
    print("\n--- SAM3 adopt ---")
    # add-only is a property of the BUILD, whatever the command line says
    if dataset.info.get("sam3_add_only") and not args.sam3_add_only:
        args.sam3_add_only = True
        print(f"  '{dataset.name}' was BUILT --sam3_add_only: add-only is back on for this run, command line or not")
    # the class fence is a build property too, on the review rerun only (not a first adopt, not --reevaluate)
    _rerun = not args.reevaluate and dataset.has_sample_field("review_worst")
    if _rerun and (_bp := dataset.info.get("sam3_prompts")):
        if not args.sam3_prompt:
            args.sam3_prompt = list(_bp)
            print(f"  '{dataset.name}' was BUILT --sam3_prompt {' '.join(_bp)}: the pass is fenced to those classes, command line or not")
        elif _extra := sorted({_cls_of(p) for p in args.sam3_prompt} - {_cls_of(p) for p in _bp}):
            sys.exit(f"--sam3_prompt {' '.join(_extra)}: '{dataset.name}' was built with --sam3_prompt {' '.join(_bp)}, so SAM3 never looked for {' '.join(_extra)} and every such label would queue as unsupported. Drop them or rebuild.")
    # replaced/added were decided at the build's IoU; a review at another IoU flips them and the tick acts on the flip
    if _rerun and (_mi := dataset.info.get("adopt_match_iou", 0.5)) != args.adopt_match_iou:
        print(f"  '{dataset.name}' was BUILT at --adopt_match_iou {_mi}: using that, not {args.adopt_match_iou}"); args.adopt_match_iou = _mi
    # argparse only sees these two beside the flag, so re-check them once the build's own record is in
    if args.sam3_add_only and args.reevaluate:
        sys.exit("--sam3_add_only with --reevaluate votes against ground_truth as it stands now, so SAM3's own earlier additions become the annotations it must not touch. Rebuild with --force_reload instead")
    # Author 2026-09-14, floor scope "House only" (at 144 it deleted 7457 of 28726 bird labels): COCO's labels are exempt by stamp (_adopt_vote), SAM3's small birds there are not
    if args.min_label_area and (dataset.name.startswith("sama-coco") or dataset.has_sample_field(gt_field) and len(dataset.filter_labels(gt_field, F("coco") == True, only_matches=True))):
        sys.exit(f"--min_label_area is for the house buckets only: '{dataset.name}' is a sama-coco set or holds COCO labels, and the floor would delete SAM3's small additions there (at 256 sama-coco-birdpass loses 10028 of its 28726 labels). Drop the flag")
    # a second adopt touches nothing, or it undoes every deletion he already made
    if not args.reevaluate and dataset.has_sample_field("review_worst"):
        print(f"SAM3 adopt already on '{dataset.name}' (review_worst present). Opening the review as it stands;"
              f" pass --reevaluate to adopt again, which REWRITES ground_truth from SAM3 and DISCARDS every review edit you have made.")
        return
    if not args.sam3_prompt:
        sys.exit("--sam3_adopt needs --sam3_prompt: " + (f"--reevaluate adopts '{dataset.name}' afresh" if args.reevaluate else f"'{dataset.name}' has no review_worst, so this would be a first adopt") + " and SAM3 has nothing to look for.")
    # stamped on the dataset: the later review command no longer carries the flags
    dataset.info["sam3_add_only"] = bool(args.sam3_add_only); dataset.info["adopt_match_iou"] = args.adopt_match_iou; dataset.save()
    # a raw pull has no annotations: SAM3 is the annotator and every prediction lands as an addition
    if not dataset.has_sample_field(gt_field):
        dataset.add_sample_field(gt_field, fo.EmbeddedDocumentField, embedded_doc_type=fo.Polylines)
    if dataset.count(f"{gt_field}.detections"): exit(f"--sam3_adopt needs polyline {gt_field}; this dataset holds Detections (zoo import). Both local importers emit polylines.")
    dataset.compute_metadata()
    _snap_orig(dataset, gt_field)
    # the vocabulary is what he annotated PLUS what SAM3 was asked to find
    vocab = {_norm(l) for l in dataset.distinct(f"{gt_field}.polylines.label")} | {_cls_of(x) for x in (args.sam3_prompt or [])}
    if not dataset.count(f"{gt_field}.polylines"): print(f"  no annotations on '{dataset.name}': SAM3 is the annotator, vocabulary = {' '.join(sorted(vocab))}")
    hier = _hier_one_hop(args)   # up front, before the GPU
    args.sam3_eval = True   # leaves ground_truth untouched; SAM3 lands in pred_field for the adopt pass below
    sam3_refine_workflow(dataset, args, class_thresholds, gt_field=gt_field, pred_field=pred_field)

    if args.sam3_adopt_fill:
        blank = dataset.match(~F(pred_field).exists() | (F(f"{pred_field}.polylines").length() == 0))
        if (n_blank := len(blank)):
            from ultralytics import YOLO
            model = YOLO(args.sam3_adopt_fill)
            n = _yolo_into(blank, model, "yolo_fill", default_conf, args.nms_iou)
            del model; torch.cuda.empty_cache(); gc.collect()
            print(f"  yolo_fill = {os.path.basename(args.sam3_adopt_fill)}: {n} polylines over {n_blank} SAM3-empty images")
        else: print("  --sam3_adopt_fill: SAM3 covered every image, nothing to fill")

    # a re-adopt rewrites what he passed, so every adopt_reviewed* tag goes back in the queue, once SAM3 has run
    if (_t := [t for t in dataset.distinct("tags") if t.startswith("adopt_reviewed")]) and (n_tagged := len(dataset.match_tags(_t))):
        dataset.untag_samples(_t)
        dataset.info["review_reset_cleared"] = sorted(set(dataset.info.get("review_reset_cleared") or []) | set(_t)); dataset.save()
        print(f"  cleared {', '.join(_t)} from {n_tagged} images; this adopt rewrites their annotations, so they go back in the queue")
    rows, skipped, n_inst, n_img, n_blank, n_thin, n_kin, n_rival, n_same = [], 0, 0, 0, 0, 0, 0, 0, 0
    for sample in dataset.iter_samples(autosave=True, progress=True):
        r, s, a, b, t, k, v, m = _adopt_vote(sample, args, hier, vocab, gt_field, pred_field)
        rows += r; skipped += s; n_inst += a; n_img += a > 0; n_blank += b; n_thin += t; n_kin += k; n_rival += v; n_same += m
    out = os.path.join(os.path.expanduser("~/bench_any/runs/sam3_adopt"), dataset.name, f"adopt_{datetime.datetime.now():%Y-%m-%d_%H%M%S}.csv")   # to the second: a same-day rerun must not overwrite the record of what it deleted
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows.sort(key=lambda r: ADOPT_RANK.get(r[4], 0.0), reverse=True)
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows([["# prompts", " ".join(args.sam3_prompt), "score", args.sam3_score, "add_score", args.sam3_add_score or "", "min_area", args.sam3_min_area, "min_label_area", args.min_label_area,
                                  "nms", args.sam3_nms, "border_factor", args.sam3_border_factor, "match_iou", args.adopt_match_iou,
                                  "review_cap", args.review_cap, "fill", args.sam3_adopt_fill or "", "vocab", " ".join(sorted(vocab))],
                                 ["image", "n_gt", "n_sam", "adopted", "reasons", "label", "prev_label", "best_iou", "confidence"], *rows])
    dataset.add_dynamic_sample_fields()   # declare adopt/prev_label/review_* so the app can show, filter and sort on them
    trig = Counter(r[4] for r in rows)
    print(f"SAM3 adopt: {len(rows)} rows -> {out}\n  " + ", ".join(f"{c} {v}" for v, c in trig.most_common()) +
          f"\n  {n_img} images written, {n_inst} labels in all (SAM3's, your kept ones and fills); recorded but NOT queued: {trig['empty']} left as they"
          f" were (SAM3 found nothing there), {trig['dup']} with two masks on one object; {n_blank} empty scenes not queued"
          f" (nothing annotated, nothing found, nothing written); {skipped} SAM3 masks skipped, class in neither the ground_truth labels nor --sam3_prompt."
          + (f"\n  cross-prompt suppression dropped {n_kin + n_rival} masks two prompts put on one object"
             f" ({n_kin} kin, coarser name dropped; {n_rival} rival classes, lower confidence dropped)." if n_kin or n_rival else "")
          + (f"\n  same-class suppression dropped {n_same} masks SAM3 put twice on one object (lower confidence dropped)." if n_same else "")
          # the only three paths that delete without queuing, so they say so out loud
          + (f"\n  DELETED, not queued: {trig['floored']} labels under --min_label_area {args.min_label_area} px." if trig["floored"] else "")
          + (f"\n  DELETED, not queued: {trig['contained']} where one object was drawn twice (smaller removed)." if trig["contained"] else "")
          + (f"\n  DELETED, not queued: {trig['same_object']} SAM3 additions over half of a box of yours of the same class or kin (your box stays)." if trig["same_object"] else "")
          + ("\n  Each is listed in the CSV above." if trig["floored"] or trig["contained"] or trig["same_object"] else "")
          + (f"\n  --sam3_add_score {args.sam3_add_score}: {n_thin} SAM3 masks covering no annotation were NOT added"
             f" (they still had to clear --sam3_score {args.sam3_score} to exist); class disputes over boxes you"
             f" already drew were judged at {args.sam3_score}, untouched by this gate." if args.sam3_add_score is not None else "")
          + "\n  ground_truth_orig is the pre-adopt state; --consensus_revert restores it.")

# filename common name -> scientific name, the default penguin taxa
_FN_SPECIES = {"emperor": "Aptenodytes forsteri", "king": "Aptenodytes patagonicus",
    "adelie": "Pygoscelis adeliae", "ad\u00e9lie": "Pygoscelis adeliae",
    "chinstrap": "Pygoscelis antarcticus", "gentoo": "Pygoscelis papua",
    "rockhopper": "Eudyptes chrysocome", "macaroni": "Eudyptes chrysolophus",
    "fiordland": "Eudyptes pachyrhynchus", "snares": "Eudyptes robustus",
    "royal": "Eudyptes schlegeli", "erect-crested": "Eudyptes sclateri", "erectcrested": "Eudyptes sclateri",
    "little": "Eudyptula minor", "fairy": "Eudyptula minor", "blue": "Eudyptula minor",
    "yellow-eyed": "Megadyptes antipodes", "yelloweyed": "Megadyptes antipodes",
    "african": "Spheniscus demersus", "humboldt": "Spheniscus humboldti",
    "magellanic": "Spheniscus magellanicus", "galapagos": "Spheniscus mendiculus", "gal\u00e1pagos": "Spheniscus mendiculus"}
_DEFAULT_PENGUIN_TAXA = sorted(set(_FN_SPECIES.values()) | {"Eudyptes moseleyi"})  # moseleyi (Northern rockhopper) has no distinct filename token

def bioclip_classify_workflow(dataset, args, gt_field="ground_truth"):
    """Reclassify existing labels in `gt_field` using BioCLIP 2 with per-image confidence-weighted majority vote.
    Filename species hit supersedes BioCLIP; filename genus hit narrows candidates; --bioclip_restrict narrows globally."""
    print("\n--- BioCLIP 2 Classify ---")
    try: import open_clip; from PIL import Image
    except ImportError: print("pip install open_clip_torch pillow"); return
    taxa, targets = args.bioclip_taxa or _DEFAULT_PENGUIN_TAXA, {_norm(c) for c in args.bioclip_classes}
    genera = {t.split()[0].lower() for t in taxa}
    restrict = {s.lower() for s in (args.bioclip_restrict or [])}
    g_mask = [(t.lower() in restrict or t.split()[0].lower() in restrict) for t in taxa] if restrict else [True]*len(taxa)
    if restrict and not any(g_mask): print(f"--bioclip_restrict {sorted(restrict)} matched no taxa; ignoring."); g_mask = [True]*len(taxa)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading BioCLIP 2 on {device} ({len(taxa)} taxa, {sum(g_mask)} active)...")
    model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
    model = model.to(device).eval()
    with torch.no_grad():
        tf = model.encode_text(open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")([f"a photo of {t}." for t in taxa]).to(device))
        text_feats = tf / tf.norm(dim=-1, keepdim=True)
    fmt = lambda sci: (sci.split()[0] if args.bioclip_genus else sci.replace(' ', '-')).lower()
    mask_t = torch.tensor(g_mask, device=device)
    has_word = lambda t, s: re.search(rf"(?<![^\W\d_]){re.escape(t)}(?![^\W\d_])", s)   # whole tokens bounded by non-letters: "king" never matches "hiking"
    n_changed = n_fn_species = n_fn_genus = n_coco = 0
    for sample in dataset.iter_samples(progress=True, autosave=True):
        tgt = [l for l in (get_labels_from_sample(sample, gt_field) or []) if _norm(getattr(l, "label", "") or "") in targets]
        if not tgt: continue
        b = os.path.basename(sample.filepath).lower()
        sp = next((s for tok, s in _FN_SPECIES.items() if has_word(tok, b) and s in taxa), None)
        if sp:  # filename species match: skip BioCLIP
            for l in tgt:   # a COCO label is never renamed by a script
                if getattr(l, "coco", False): n_coco += 1; continue
                l.label = fmt(sp); n_changed += 1
            n_fn_species += 1; continue
        gen = next((g for g in genera if has_word(g, b)), None)
        sample_mask = mask_t & torch.tensor([t.split()[0].lower() == gen for t in taxa], device=device) if gen else mask_t
        if not bool(sample_mask.any()): sample_mask = mask_t
        if gen: n_fn_genus += 1
        try: img = Image.open(sample.filepath).convert("RGB"); W, H = img.size
        except Exception: continue
        preds, votes = [], defaultdict(float)
        for l in tgt:
            box = (l.to_detection().bounding_box if isinstance(l, fo.Polyline) and l.points and l.points[0] else l.bounding_box if isinstance(l, fo.Detection) else None)
            if not box: continue
            x, y, w, h = box
            x1, y1, x2, y2 = max(0, int(x*W)), max(0, int(y*H)), min(W, int((x+w)*W)), min(H, int((y+h)*H))
            if x2 - x1 < 16 or y2 - y1 < 16: continue
            with torch.no_grad():
                feat = model.encode_image(preprocess(img.crop((x1, y1, x2, y2))).unsqueeze(0).to(device))
                logits = (100.0 * (feat / feat.norm(dim=-1, keepdim=True)) @ text_feats.T).masked_fill(~sample_mask.unsqueeze(0), float('-inf'))
                probs = logits.softmax(dim=-1)[0]
            idx = int(probs.argmax()); preds.append((l, idx)); votes[idx] += float(probs[idx])
        if not preds: continue
        lab = fmt(taxa[max(votes, key=votes.get)])
        for l, _ in preds:
            if getattr(l, "coco", False): n_coco += 1; continue
            l.label = lab; n_changed += 1
    del model; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"BioCLIP: relabeled {n_changed} (filename-species skips: {n_fn_species}, filename-genus narrowed: {n_fn_genus})."
          + (f" {n_coco} COCO labels left alone: only your own click renames one." if n_coco else ""))

def refine_boxes_workflow(dataset, sam_model_name, splits_to_run=None, gt_field="ground_truth", convert_to_boxes=True, clean_4point_seg_outliers=True, clean_complex_seg_outliers=False, sam_confidence_threshold=0.5):
    print("\n--- Refining Boxes ---")
    temp_field, sam_model = "sam_refined", foz.load_zoo_model(sam_model_name)
    view = dataset.match((F(f"{gt_field}.polylines").length() > 0) | (F(f"{gt_field}.detections").length() > 0))
    if splits_to_run: view = view.match_tags([SPLIT_MAPPING[s] for s in splits_to_run])
    if not view: print(f"No labels found in '{gt_field}' to refine."); return
    if clean_4point_seg_outliers or clean_complex_seg_outliers:
        filtered = [s.id for s in view if s.has_field(gt_field) and s[gt_field] and any(should_check_label_for_outliers(l, clean_4point_seg_outliers, clean_complex_seg_outliers) for l in get_labels_from_sample(s, gt_field))]
        if not filtered: print("No refineable labels found."); return
        view = view.select(filtered)
    prompt_field = gt_field
    if convert_to_boxes and dataset.count(f"{gt_field}.polylines") > 0:
        print("Converting polylines to detections for SAM...")
        prompt_field = "sam_prompts"
        for sample in view.iter_samples(autosave=True):
            if sample.has_field(gt_field) and sample[gt_field]: sample[prompt_field] = sample[gt_field].to_detections()
    print("Running SAM with batching...")
    for i in range(0, len(sid_list := list(view.values("id"))), 100):
        batch_ids, high_det = sid_list[i:i+100], [sid for sid in sid_list[i:i+100] if dataset[sid].has_field(prompt_field) and dataset[sid][prompt_field] and len(dataset[sid][prompt_field].detections) > 300]
        if normal := [sid for sid in batch_ids if sid not in high_det]:
            try: dataset.select(normal).apply_model(sam_model, label_field=temp_field, prompt_field=prompt_field)
            except Exception as e:
                print(f"  SAM failed on the {len(normal)}-sample batch, those labels keep their original geometry: {e}")
                [dataset[sid].__setitem__(temp_field, fo.Detections(detections=[])) or dataset[sid].save() for sid in normal]
            finally: torch.cuda.empty_cache() if torch.cuda.is_available() else None; gc.collect()
        for sid in high_det:
            sample, all_sam = dataset[sid], []
            for c in range(0, len(sample[prompt_field].detections), 300):
                sample[f"{prompt_field}_chunk"] = fo.Detections(detections=sample[prompt_field].detections[c:c+300]); sample.save()
                try: dataset.select([sid]).apply_model(sam_model, label_field=f"{temp_field}_chunk", prompt_field=f"{prompt_field}_chunk"); all_sam.extend(sample[f"{temp_field}_chunk"].detections)
                except Exception as e: print(f"  SAM failed on chunk {c//300} of {os.path.basename(sample.filepath)}, those labels keep their original geometry: {e}")
                finally: [sample.clear_field(f) for f in [f"{prompt_field}_chunk", f"{temp_field}_chunk"] if sample.has_field(f)]; torch.cuda.empty_cache() if torch.cuda.is_available() else None; gc.collect()
            sample[temp_field] = fo.Detections(detections=all_sam); sample.save()
    print("Applying SAM masks...")
    n_mismatch = n_lowq = 0
    _quality = lambda gb, sb: (lambda r: min(1.0, r if r <= 1.0 else 2.0 - r) if 0.1 <= r <= 2.0 else 0.0)((sb[2]*sb[3]) / (gb[2]*gb[3])) if gb[2]*gb[3] > 0 else 0.0
    # the sample filter admits a sample for ONE label, so the same per-label rule decides which of its labels change
    _off = lambda l: (clean_4point_seg_outliers or clean_complex_seg_outliers) and not should_check_label_for_outliers(l, clean_4point_seg_outliers, clean_complex_seg_outliers)
    for sample in view.iter_samples(progress=True, autosave=True):
        if not (sample.has_field(temp_field) and sample[temp_field] and sample.has_field(gt_field) and sample[gt_field]): continue
        is_detections = hasattr(sample[gt_field], 'detections')
        gt_labels = sample[gt_field].detections if is_detections else sample[gt_field].polylines
        sam_dets = sample[temp_field].detections
        if len(sam_dets) != len(gt_labels):
            n_mismatch += 1   # SAM returned a different count, so the pairing is unusable
            continue
        if is_detections:
            for gt_det, sam_det in zip(gt_labels, sam_dets):
                if sam_det.mask is None or _off(gt_det): continue
                if _quality(gt_det.bounding_box, _sam_mask_box(sam_det)) < sam_confidence_threshold: n_lowq += 1; continue
                gt_det.mask = sam_det.mask
        else:
            out_polys = []
            for gt_poly, sam_det in zip(gt_labels, sam_dets):
                if sam_det.mask is None or _quality(gt_poly.to_detection().bounding_box, _sam_mask_box(sam_det)) < sam_confidence_threshold:
                    n_lowq += sam_det.mask is not None
                    out_polys.append(gt_poly)
                    continue
                if getattr(gt_poly, "coco", False):   # COCO geometry is never overwritten by a script
                    out_polys.append(gt_poly); continue
                if not _off(gt_poly): gt_poly.points = sam_det.to_polyline().points   # in place: id and attributes stay
                out_polys.append(gt_poly)
            sample[gt_field] = fo.Polylines(polylines=out_polys)
    if n_mismatch: print(f"SAM refinement skipped {n_mismatch} samples where the SAM and ground-truth label counts differ")
    if n_lowq: print(f"SAM refinement kept {n_lowq} original labels below quality {sam_confidence_threshold}")
    dataset.delete_sample_field(temp_field)
    if prompt_field != gt_field: dataset.delete_sample_field(prompt_field)
    print("SAM refinement complete")

def _sam_mask_box(d):
    """Box of the mask's own extent, relative coords. SAM echoes the PROMPT box as bounding_box (fiftyone/utils/sam.py),
    so the box-area ratio in the quality gate was always 1.0; the mask, cropped to that box, is the only thing SAM decided."""
    ys, xs = np.nonzero(d.mask); b = d.bounding_box
    return [b[0], b[1], b[2] * (np.ptp(xs) + 1) / d.mask.shape[1], b[3] * (np.ptp(ys) + 1) / d.mask.shape[0]] if xs.size else [b[0], b[1], 0.0, 0.0]

def _own_dir_view(view, own_dir, what="Export", unit="samples"):
    """A merged dataset holds other dirs' samples; keep only the images that live under this input dir."""
    d = os.path.realpath(own_dir).rstrip(os.sep) + os.sep
    sub = view.select([i for i, fp in zip(*view.values(["id", "filepath"])) if os.path.realpath(fp).startswith(d)], ordered=True)   # both sides resolved: a symlinked dir matched 0 (W76)
    print(f"{what} restricted to {d}: {len(sub)} of {len(view)} {unit}")
    return sub

def _refuse_overwrite(output_dir, force_export, dataset=None):
    """True when an export must not rmtree this directory: it holds images this export reads (by the dataset's
    own filepaths), it overlaps an input dir, or it is non-empty and not forced. --force_export overrides neither overlap."""
    _out = os.path.realpath(output_dir)
    if dataset is not None:
        if not len(dataset): print(f"REFUSED: 0 samples to export; {output_dir} left untouched"); return True
        fps = dataset.values("filepath") + [p for p in (dataset.values("original_filepath") if dataset.has_sample_field("original_filepath") else []) if p]
        for fp in fps:   # the folder holding the file, not its target: rmtree deletes a symlink wherever it points
            if any(x.startswith(_out + os.sep) for x in (os.path.realpath(fp), os.path.realpath(os.path.dirname(os.path.abspath(fp))) + os.sep)):
                print(f"REFUSED: {output_dir} holds the images this export is reading (e.g. {fp}); the export deletes that path first")
                return True
    for _in in _INPUT_DIRS:
        if _overlaps(_out, _in):
            print(f"REFUSED: {output_dir} overlaps the input dir {_in}; the export deletes that path first and the images it is still reading live under it")
            return True
    if os.path.exists(output_dir) and os.listdir(output_dir) and not force_export:
        print(f"{output_dir} is not empty; refusing to overwrite, pass --force_export")
        return True
    return False

def _resolve_splits(dataset, resplit, who):
    """{split tag: view} to export. --resplit always re-splits, in memory only: the stored tags are his --to_valid choices."""
    tags = [t for t in ["train", "validation", "test"] if t in dataset.distinct("tags")]
    if resplit is None: return {t: dataset.match_tags(t) for t in tags}
    r = resplit if len(resplit) == 3 else [0.7, 0.2, 0.1]
    assert abs(sum(r) - 1.0) < 1e-6, f"--resplit ratios must sum to 1.0, got {r} (sum={sum(r)})"
    if tags: print(f"  {who}: --resplit ignores the stored {'/'.join(tags)} tags and re-splits {len(dataset)} samples at {r}; the tags stay.")
    ids = list(np.random.default_rng(42).permutation(dataset.values("id")))
    a = round(r[0] * len(ids)); b = a + round(r[1] * len(ids))
    return {t: dataset.select(p) for t, p in zip(["train", "validation", "test"], (ids[:a], ids[a:b], ids[b:]))}

def _export_splits(dataset, output_dir, resplit, force_export, who):
    """{split folder: view} in a fresh output_dir, or None when _refuse_overwrite says no. Both exports start here."""
    if _refuse_overwrite(output_dir, force_export, dataset): return None
    splits = _resolve_splits(dataset, resplit, f"{who} export")
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"\nExporting {who} to {output_dir}...")
    dataset.compute_metadata()   # no W, H means no simplify and no speck rule, so a sample without metadata exported raw
    return {{"train": "train", "validation": "valid", "test": "test"}[t]: v for t, v in splits.items() if len(v)}

def _export_samples(view, st):
    """(sample, image file, export basename) for each sample whose image file exists and has a size. Both exports name
    files here, so a COCO file_name is the YOLO image name, and both skip the same frames."""
    used_stems = {}   # keyed on the label stem: x.jpg and x.png share one x.txt
    for sample in view.iter_samples(progress=True):
        export_filepath = (sample.original_filepath if sample.has_field("original_filepath") and sample.original_filepath else sample.filepath)
        if not export_filepath or not os.path.exists(export_filepath): st["missing"] += 1; continue
        if not _wh(sample)[0]: st["unreadable"] += 1; continue   # compute_metadata could not read it
        # strip imgproc/superres suffixes, with or without an intensity number
        clean_basename = re.sub(r'_(?:contrast2?|denoise[23]?|retinex|swinir|supir|enhanced_4x)(?:_(?:contrast2?|denoise[23]?|retinex|swinir|supir))*(?:_[\d.]+)?(\.(jpg|png|jpeg|bmp|tiff))$', r'\1', os.path.basename(export_filepath))
        stem, ext = os.path.splitext(clean_basename); k, i = stem, 1
        while used_stems.get(k, export_filepath) != export_filepath:   # taken by another file: prefix the parent, then count
            i += 1; k = f"{os.path.basename(os.path.dirname(export_filepath))}_{stem}" + (f"_{i}" if i > 2 else "")
        used_stems[k] = export_filepath; st["renamed"] += k != stem
        yield sample, export_filepath, k + ext

def _export_labels(sample, label_field, final_yaml_classes, st, off, pts_of):
    """(class index, pts_of(poly), poly) for each label an export writes: pts_of's rules first, then the class list, both counted."""
    lab = sample[label_field] if sample.has_field(label_field) else None
    W, H = _wh(sample)
    for poly in (lab.to_polylines(tolerance=1e-10) if isinstance(lab, fo.Detections) else lab if isinstance(lab, fo.Polylines) else fo.Polylines()).polylines:
        if (pts := pts_of(poly, W, H, st)) is None: continue
        if poly.label not in final_yaml_classes: off[poly.label] += 1; continue
        yield final_yaml_classes.index(poly.label), pts, poly

def _export_report(dest_name, st, off):
    print(f"  Exported {st['labels']} total labels for split '{dest_name}'")
    if st["missing"]: print(f"  Skipped {st['missing']} sample(s) whose image file is gone, with all of their labels")
    if st["unreadable"]: print(f"  Skipped {st['unreadable']} sample(s) whose image file is unreadable, with all of their labels")
    if st["degenerate"]: print(f"  Dropped {st['degenerate']} label(s) left with no polygon: an empty mask, or every part under 3 points or thin (thin parts are also counted below)")
    if st["vin"]: print(f"  Simplified at eps {SIMPLIFY_EPS}: {st['vin']} vertices in, {st['vout']} out")
    if st["flat"]: print(f"  Dropped {st['flat']} thin part(s) that eps {SIMPLIFY_EPS} flattened to a line, thinner than about 1% of their own length")
    if st["specks"]: print(f"  Dropped {st['specks']} contour speck(s) under {SPECK_MIN_AREA}px and {SPECK_MIN_FRAC:.0%} of their label's largest ring")
    if st["merged"]: print(f"  Bridged {st['merged']} multi-blob label(s) into a single polygon (YOLO has one polygon per line)")
    if off: print(f"  Dropped labels whose class is not in the export list: {dict(off)}")
    if st["renamed"]: print(f"  Renamed {st['renamed']} file(s) in '{dest_name}' whose basename was already taken by another source")

def export_yolo_dataset(dataset, output_dir, final_yaml_classes, resplit, label_field, resize=None, force_export=False):
    if (splits := _export_splits(dataset, output_dir, resplit, force_export, "YOLO")) is None: return
    for dest_name, split_view in splits.items():
        os.makedirs(images_dir := os.path.join(output_dir, dest_name, "images"), exist_ok=True)
        os.makedirs(labels_dir := os.path.join(output_dir, dest_name, "labels"), exist_ok=True)
        print(f"  Exporting {len(split_view)} samples for '{dest_name}'...")
        st, off = Counter(), Counter()
        for sample, export_filepath, clean_basename in _export_samples(split_view, st):
            dest_path = os.path.join(images_dir, clean_basename)
            # a truncated file passes os.path.exists but imread returns None, killing the whole export
            if resize and (img := cv2.imread(export_filepath)) is None: st["unreadable"] += 1; continue
            if resize and (scale := resize / min(img.shape[:2])) < 1.0:
                cv2.imwrite(dest_path, cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
            else:
                shutil.copy(export_filepath, dest_path)
            with open(os.path.join(labels_dir, os.path.splitext(clean_basename)[0] + ".txt"), "w") as f:
                for c, pts, _ in _export_labels(sample, label_field, final_yaml_classes, st, off, _export_pts):
                    f.write(_yolo_line(c, pts)); st["labels"] += 1
        _export_report(dest_name, st, off)

    with open(os.path.join(output_dir, "data.yaml"), 'w') as f:
        # val always: the importer loads only the splits a yaml names and --to_valid moves frames into valid/ after an export.
        # test only when written: a bench reads a test key as a test set
        yaml.dump({"train": "train/images", "val": "valid/images", **({"test": "test/images"} if os.path.isdir(os.path.join(output_dir, "test")) else {}),
                   "nc": len(final_yaml_classes), "names": final_yaml_classes}, f, sort_keys=False)
    print("Export complete.")

def export_coco_dataset(dataset, output_dir, final_yaml_classes, resplit, label_field, force_export=False):
    """COCO layout for RF-DETR: output_dir/{train,valid,test}/{_annotations.coco.json + images}. The YOLO export's splits, files,
    names and shapes, except that a label's parts stay apart: a COCO segmentation holds several."""
    import json
    if (splits := _export_splits(dataset, output_dir, resplit, force_export, "COCO")) is None: return
    for dest_name, view in splits.items():
        os.makedirs(dest := os.path.join(output_dir, dest_name))
        print(f"  Exporting {len(view)} samples for '{dest_name}'...")
        st, off, images, anns = Counter(), Counter(), [], []
        for sample, export_filepath, name in _export_samples(view, st):
            W, H = _wh(sample)
            shutil.copy(export_filepath, os.path.join(dest, name))
            images.append({"id": len(images) + 1, "file_name": name, "height": H, "width": W})
            for c, rings, poly in _export_labels(sample, label_field, final_yaml_classes, st, off, _export_rings):
                seg = [np.round(np.array(r, np.float64) * (W, H), 2) for r in rings]
                px = np.concatenate(seg); (x0, y0), (x1, y1) = px.min(0), px.max(0)
                anns.append({"id": len(anns) + 1, "image_id": len(images), "category_id": c + 1, "iscrowd": 0,
                             "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0),
                             "segmentation": [q.ravel().tolist() for q in seg],
                             **({"score": poly.confidence} if poly.confidence is not None else {})})
                st["labels"] += 1
        # marked as ours, so a --local_coco_dirs re-import never stamps SAM3's adoptions coco=True
        with open(os.path.join(dest, "_annotations.coco.json.tmp"), "w") as f:   # closed, then one rename: a failed flush stops the export, no cut file
            json.dump({"info": {"fyal_export": True}, "images": images, "annotations": anns,
                       "categories": [{"id": i + 1, "name": n, "supercategory": None} for i, n in enumerate(final_yaml_classes)]}, f)
        os.replace(os.path.join(dest, "_annotations.coco.json.tmp"), os.path.join(dest, "_annotations.coco.json"))
        _export_report(dest_name, st, off)
    print("COCO export complete.")


if __name__ == "__main__":
    args = parse_args()
    _ALLOW_DELETE_LIVE = args.allow_delete_live
    print(f"FiftyOne database: {fo.config.database_uri} ({len(fo.list_datasets())} datasets)" + (" [--allow_delete_live]" if _ALLOW_DELETE_LIVE else ""))

    is_outlier_workflow = (args.clip_outliers or args.max_bbox_width != 1.0 or args.min_bbox_width != 0.01 or args.max_aspect_ratio != 0.0 or args.min_aspect_ratio != 0.0) and not args.find_missed_labels and not args.refine_boxes
    target_classes, final_yaml_classes, classes_were_specified = get_class_configuration(args)
    _yaml0 = get_class_configuration(args)[1]
    _export_classes = lambda: final_yaml_classes + [c for c in get_class_configuration(args)[1] if c not in final_yaml_classes + _yaml0]   # a class the pass added to data.yaml exports too
    _export = lambda view, out: export_yolo_dataset(view, out, _export_classes(), args.resplit, "ground_truth", args.resize, force_export=args.force_export)

    # several dirs make ONE dataset: named after dirs[0] it would overwrite that dir's cache
    _dirs_name = lambda ds: "+".join(os.path.basename(os.path.normpath(d)) for d in ds)
    _src = (args.local_yolo_dirs or []) + (args.local_coco_dirs or [])   # a mixed load names both, or opens the yolo-only cache
    input_persistent_name = (_dirs_name(_src) if _src else
                             f"{'-'.join(sorted(args.dataset_name)) if args.dataset_name else 'local'}-{'all' if args.all_coco_classes else f'{len(target_classes)}c'}-default")
    output_persistent_name = (os.path.basename(os.path.normpath(args.output_dir)) if args.output_dir else input_persistent_name)

    # the cache is keyed on the folder's last name, so two folders of one name share it
    _roots = tuple(os.path.realpath(d) + os.sep for d in _src + ([fo.config.dataset_zoo_dir] if _src and args.dataset_name else []))
    def _foreign(ds):
        if _roots and (_alien := sorted({d for d in map(os.path.dirname, ds.values("filepath")) if not (os.path.realpath(d) + os.sep).startswith(_roots)})):
            exit(f"'{ds.name}' holds frames from {_alien[0]} and {len(_alien) - 1} other folders, not from {' '.join(_src)}.\n"
                 f"Two folders share that name, or this folder moved. Nothing opened, nothing moved. Export '{ds.name}' if it holds review, then delete it to rebuild.")
    for n in {output_persistent_name, input_persistent_name} & set(fo.list_datasets()): _foreign(fo.load_dataset(n))   # before a delete, a paste or a clone acts on it

    if args.force_reload:
        if fo.dataset_exists(output_persistent_name):
            _del_dataset(output_persistent_name, "--force_reload")
            print(f"Deleted output dataset '{output_persistent_name}'")
        if output_persistent_name != input_persistent_name and fo.dataset_exists(input_persistent_name):
            _del_dataset(input_persistent_name, "--force_reload")
            print(f"Deleted input dataset '{input_persistent_name}'")

    if fo.dataset_exists(output_persistent_name):
        dataset = fo.load_dataset(output_persistent_name)
        print(f"Loaded cached output dataset '{output_persistent_name}'")
        if not dataset.persistent:   # a launch that died after the rename; the next last-client exit would delete his clicks
            dataset.persistent = True; print(f"  '{output_persistent_name}' was not persistent: made persistent")
        # --output_dir reloads its clone, so review on the input since the clone would be lost
        if output_persistent_name != input_persistent_name and fo.dataset_exists(input_persistent_name):
            _in = fo.load_dataset(input_persistent_name)   # a click writes the sample, never the dataset document
            _in_mod = max(filter(None, [_in.last_modified_at, _in.bounds("last_modified_at")[1]]), default=None)
            if _in_mod and dataset.created_at and _in_mod > dataset.created_at:
                exit(f"'{input_persistent_name}' was changed at {_sast(_in_mod)}, after '{output_persistent_name}' was cloned off it at {_sast(dataset.created_at)}.\n"
                     f"This run would export the older copy and silently drop that work. Delete the stale copy to export the newer review:\n"
                     # the paste refuses a clone holding review tags; os._exit(0), or a last-client exit sweeps every non-persistent dataset
                     f"  python -c \"import fiftyone as fo, os; n='{output_persistent_name}'; d=fo.load_dataset(n); "
                     f"t=[x for x in d.distinct('tags') if x.startswith('adopt_reviewed')]; "
                     f"print('REFUSED, holds review work: '+str(t)) if t else fo.delete_dataset(n); os._exit(0)\"\n"
                     f"Keep it only if you reviewed under --output_dir and the change to '{input_persistent_name}' was not yours.")
    elif output_persistent_name != input_persistent_name and fo.dataset_exists(input_persistent_name):
        dataset = fo.load_dataset(input_persistent_name)
        print(f"Loaded cached input dataset '{input_persistent_name}' → will process and save as '{output_persistent_name}'")
        # clone, not rename: a rename leaves the input cache nameless and the next run reloads from disk
        dataset = dataset.clone(output_persistent_name, persistent=True)
    else:
        _tgt = output_persistent_name
        _lk = _build_lock(_tgt)
        try:
            # built by the launch we waited for; a non-persistent one is from a launch that died mid-build
            if fo.dataset_exists(_tgt) and fo.load_dataset(_tgt).persistent:
                dataset = fo.load_dataset(_tgt)
                print(f"Loaded '{_tgt}': the launch we waited for had already built it")
            else:
                if fo.dataset_exists(_tgt): _del_dataset(_tgt, "a non-persistent set left by a launch that died mid-build")   # else the rename below raises
                print("Loading dataset from disk...")
                dataset = load_and_merge_datasets(args.dataset_name, args.local_yolo_dirs, target_classes, args.label_types, args.max_samples, args.splits_to_run, user_classes_arg=args.classes, find_missed_labels=args.find_missed_labels, local_coco_dirs=args.local_coco_dirs)
                dataset.name = _tgt
                dataset.persistent = True
                print(f"Created new persistent dataset '{dataset.name}'")
        finally: _lk.close()
    _foreign(dataset)
    if dataset.info.get("sam3_add_only") and (_ed := [f for f, on in (("--sam3_prompt/--sam3_refine without --sam3_adopt", (args.sam3_refine or args.sam3_prompt is not None) and not args.sam3_adopt),
            ("--consensus", args.consensus), ("--refine_boxes", args.refine_boxes), ("--clean_overlap_on", args.clean_overlap_on), ("--remap_classes", args.remap_classes),
            ("--min_label_area", args.min_label_area), ("--dedup_contained", args.dedup_contained), ("--bioclip_classify", args.bioclip_classify), ("--find_missed_labels", args.find_missed_labels), ("outlier filtering", is_outlier_workflow),
            ("the duplicate and similar screens", args.exact_dup or args.find_duplicates is not None or args.find_similar is not None)) if on]):   # their tick drops the frame
        exit(f"'{dataset.name}' was built --sam3_add_only: every annotation on it survives and only --sam3_adopt --review_reason more runs on it. {', '.join(_ed)} would edit them. Nothing touched.")

    if args.consensus_revert:
        if not dataset.has_sample_field("ground_truth_orig"): exit("No ground_truth_orig on this dataset; nothing to revert.")
        _rt = [t for t in dataset.distinct("tags") if t.startswith("adopt_reviewed")]
        n_rev = len(dataset.match_tags(_rt)) if _rt else 0
        # --review_reset clears the tags, not dataset.info
        _wiped = dataset.info.get("review_reset_cleared") or []
        # the rename below CONSUMES the snapshot: no undo
        _clk = dataset.info.get("screen_clicks")
        if (n_rev or _wiped or _clk) and not args.allow_revert_reviewed:
            _why = (f"has {n_rev} of {len(dataset)} images already reviewed ({', '.join(_rt)})" if n_rev else
                    f"was reviewed and then put back in the queue by --review_reset ({', '.join(_wiped)}); the tags are gone, the review is not" if _wiped else
                    f"lost {_clk} labels to your clicks and ticks since the snapshot, some on screens that tag nothing (--to_valid, --find_redundant, the duplicate screens)")
            exit(f"REFUSED: '{dataset.name}' {_why}. Reverting deletes that review, the SAM3 masks it was about, the YOLO judge's proposals, and the tags, and the snapshot is consumed so there is no undo. Nothing was touched. Export first, then pass --allow_revert_reviewed if you really mean it.")
        dataset.delete_sample_field("ground_truth"); dataset.rename_sample_field("ground_truth_orig", "ground_truth")
        # a revert starts over: stale verdicts would be voted on again
        for f in ("consensus_proposals", "consensus_worst", "yolo_a", "yolo_b", "review_reasons", "review_worst", "yolo_fill",
                  "review_sheet", "review_prior", "sam3_worst", "sam3_predictions", "sam3_disagree", "sam3_verdict", "sam3_gt_label",
                  "yolo_judge_preds", "review_pos"):
            if dataset.has_sample_field(f): dataset.delete_sample_field(f)
        for k in ("sam3_sig", "sam3_script", "sam3_prompts", "yolo_judge_sig", "sam3_worst_by", "consensus_yolo_a", "consensus_yolo_b",
                  "review_reset_cleared", "sam3_add_only", "screen_clicks"):   # the revert destroys what each records
            if k in dataset.info: del dataset.info[k]
        dataset.save()
        # a stale review tag excludes the image forever
        if _rt: dataset.untag_samples(_rt)
        print(f"Reverted ground_truth from ground_truth_orig on '{dataset.name}'; cleared {', '.join(_rt) or 'no reviewed tags'} from {n_rev} images"); exit()

    if args.dedup_contained:
        from PIL import Image
        _pts = lambda p: tuple(tuple(map(tuple, r)) for r in (p.points or []))
        drops, seen = [], set()
        for sample in dataset.iter_samples(progress=True):
            polys = _polys(sample, "ground_truth")
            if len(polys) < 2: continue
            W, H = (sample.metadata.width, sample.metadata.height) if (sample.metadata and sample.metadata.width) else Image.open(sample.filepath).size
            if not (d := _contained_dups(polys, W, H)): continue
            orig, n0 = {(_norm(p.label), _pts(p)) for p in _polys(sample, "ground_truth_orig")}, len(drops)
            for p in polys:
                # never a COCO label; skipped here, not at the write, so the CSV is honest too
                if p.id not in d or getattr(p, "coco", False): continue
                kept, r, iou, area = d[p.id]
                drops.append((sample.id, sample.filepath, _norm(p.label), p.id, getattr(p, "adopt", None) or "", area, kept, r, iou, (_norm(p.label), _pts(p)) in orig))
            if len(drops) > n0: seen.add(sample.id)   # an image whose every pair was COCO-skipped drops nothing
        by_tag = Counter(x[4] or "none" for x in drops); n_his = sum(x[9] for x in drops)
        print(f"containment dedup on '{dataset.name}': {len(drops)} pairs on {len(seen)} images, {len(drops)} polylines to drop"
              f"\n  by adopt tag of the dropped polyline: " + (", ".join(f"{k} {v}" for k, v in by_tag.most_common()) or "nothing")
              + f"\n  {n_his} of them are his own ground_truth_orig originals (exact points).")
        out = os.path.join(os.path.expanduser("~/bench_any/runs/sam3_adopt"), dataset.name, f"contained_{datetime.datetime.now():%Y-%m-%d_%H%M%S}.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", newline="") as fh:
            csv.writer(fh).writerows([["sample_id", "image", "label", "dropped_id", "adopt", "area_px", "kept_id", "inter_over_min_crop", "mask_iou", "was_his_original"], *drops])
        print(f"  CSV -> {out}")
        if drops:
            gone = {x[3] for x in drops}
            for sample in dataset.select(sorted(seen)).iter_samples(autosave=True, progress=True):
                sample["ground_truth"] = fo.Polylines(polylines=[p for p in sample["ground_truth"].polylines if p.id not in gone])
        print(f"Dropped {len(drops)} containment duplicates. Re-run to confirm 0."); exit()

    if (args.find_missed_labels or args.consensus) and dataset.has_sample_field("ground_truth"):
        existing_classes = set()
        if dataset.count("ground_truth.detections") > 0:
            existing_classes.update(dataset.distinct("ground_truth.detections.label"))
        if dataset.count("ground_truth.polylines") > 0:
            existing_classes.update(dataset.distinct("ground_truth.polylines.label"))
        if existing_classes:
            final_yaml_classes = final_yaml_classes + sorted(existing_classes - set(final_yaml_classes))   # source order kept, new names at the end
            print(f"Expanded export classes to include all existing labels: {final_yaml_classes}")

    if is_outlier_workflow:
        filter_outlier_detections(dataset, args.max_bbox_width, args.min_bbox_width, args.max_aspect_ratio, args.min_aspect_ratio, label_field='ground_truth', clean_4point_seg_outliers=args.clean_4point_seg_outliers, clean_complex_seg_outliers=args.clean_complex_seg_outliers, vertical_scaling=args.vertical_scaling, region=args.region, clip_model=args.similar_model if args.clip_outliers else None, outlier_classes=args.outlier_classes)
        print_dataset_stats(dataset, "Label counts after outlier filtering")

    if args.clean_overlap_on:
        print(f"Cleaning overlaps (priority: {args.clean_overlap_on})...")
        for sample in dataset.iter_samples(autosave=True, progress=True):
            if (labels := get_labels_from_sample(sample, "ground_truth")):
                temp_dets = [l.to_detection() if isinstance(l, fo.Polyline) else l for l in labels]
                keep_ids = {id(d) for d in filter_overlapping_classes(temp_dets, args.clean_overlap_on, args.pred_dedup_iou)}
                # no rule takes a COCO label
                filtered_labels = [labels[i] for i, d in enumerate(temp_dets)
                                   if id(d) in keep_ids or getattr(labels[i], "coco", False)]
                sample["ground_truth"] = fo.Polylines(polylines=filtered_labels) if isinstance(sample["ground_truth"], fo.Polylines) else fo.Detections(detections=filtered_labels)

    if len(dataset) == 0: exit("FATAL: Dataset is empty after loading.")

    dup = args.exact_dup or args.find_duplicates is not None or args.find_redundant is not None
    if (dup or args.find_similar is not None or args.subset or args.to_valid or args.overlap_to_valid is not None) and not args.bioclip_classify:
        # an untag alone leaves a stale group id on frames a stricter rerun drops
        if dup and "duplicate_group_id" in dataset.get_field_schema(): dataset.clear_sample_field("duplicate_group_id")
        if args.exact_dup:
            print("Finding exact duplicates by size+md5..."); dataset.untag_samples(["duplicate", "duplicate_group"])
            _by_key = defaultdict(list)
            for p in dataset.values("filepath"):
                try: _by_key[(os.path.getsize(p), hashlib.md5(open(p, 'rb').read()).hexdigest())].append(os.path.abspath(p))
                except OSError as e: print(f"  skipped {p}: {e}")
            all_dupe_groups = [g for g in _by_key.values() if len(g) > 1]
            print(f"Found {sum(len(g)-1 for g in all_dupe_groups)} duplicate files in {len(all_dupe_groups)} groups")
            for idx, group in enumerate(all_dupe_groups):
                dupe_set = set(group[1:])
                for sample in dataset.match(F("filepath").is_in(group)).iter_samples(autosave=True):
                    sample["duplicate_group_id"] = idx
                    sample.tags.extend(["duplicate_group", "duplicate"] if sample.filepath in dupe_set else ["duplicate_group"])
        elif args.to_valid or args.overlap_to_valid is not None:
            # Author 2026-09-22: he picks the partial detections for valid/, not a rule. 2026-10-01: --overlap_to_valid, his tick keeps a frame in train/
            # 2026-10-04: --border_to_valid, and --to_valid_small and --to_valid_confusion flipped to that way, "Flip tiny + person/dog"
            ov = args.overlap_to_valid  # to_valid
            flag, sub, jt = next(x[1:] for x in ((args.to_valid_confusion, "--to_valid_confusion", _confused, "confusion_judged"), (args.to_valid_small, "--to_valid_small", _small, "small_judged"),
                                 (args.border_to_valid, "--border_to_valid", _border, "border_judged"), (ov is not None, "--overlap_to_valid", lambda v: _overlap(v, ov), "overlap_judged"), (True, "--to_valid", None, None)) if x[0])  # to_valid
            if sub and not dataset.has_sample_field("ground_truth"): exit(f"No frame of {dataset.name} holds a label, so {flag} has nothing to serve.")  # to_valid
            pool = dataset  # to_valid
            if args.review_files:  # to_valid: by file name or stem, so a list of another set's paths finds its frames after a -final build moved them
                pool = dataset.select([i for i, fp in zip(*dataset.values(["id", "filepath"])) if {os.path.basename(fp), os.path.splitext(os.path.basename(fp))[0]} & args.review_files])
                print(f"--review_files: {len(pool)} frames of {dataset.name} named in its {len(args.review_files)} names")
                if not len(pool): exit(f"--review_files: none of its {len(args.review_files)} names is a frame of '{dataset.name}'. Check the set, and one image file name or stem per line, no folder.")
            # to_valid: Author 2026-09-26, "train/ only", a frame already in valid/ has nowhere to go. Both splits where a rule moved frames to valid/ first (overlap, border) or his list names them. A judged frame never again
            train = pool if args.review_files or flag in ("--overlap_to_valid", "--border_to_valid") else pool.match_tags("train")
            if jt: train = train.match_tags(jt, bool=False).match(~F("filepath").re_match(r"/valid/images/cam112-[^/]*$"))  # cam112, cam11's thermal view, never goes to train/ (Author 2026-10-04)
            view = _cam_first(train.select(sub(train)) if sub else train).set_field("tags", F("tags").filter(F().is_in(["train", "validation"])))  # to_valid sidebar: the split tags only
            if jt and not len(view): exit(f"Nothing left: no {jt.split('_')[0]} frame of {dataset.name} waits to be judged.")
            if jt and args.review_cap: view = view.limit(args.review_cap)  # Author 2026-10-10: his --review_cap, else every waiting frame
            # 12/3539 on the sidebar. Only changed rows are written, so a relaunch leaves every sample clock alone
            _ids = view.values("id"); _old = dict(zip(*dataset.values(["id", "review_pos"]))) if dataset.has_sample_field("review_pos") else {}  # to_valid
            _new = {i: f"{n + 1}/{len(_ids)}" for n, i in enumerate(_ids) if _old.get(i) != f"{n + 1}/{len(_ids)}"}  # to_valid
            if _new: dataset.set_values("review_pos", _new, key_field="id")  # to_valid
            session = fo.launch_app(view=_class_menu(_gt_only(view), args)); session.config.show_confidence = session.config.show_label = False  # to_valid
            session.color_scheme = _color_scheme(); picked, ticked = _wait_or_enter(session, dataset)
            _write_back(dataset, picked, (), args)  # to_valid: a tick MOVES the frame below, it never wipes it
            if jt:  # ticked = train/, unticked = valid/. A frame that lands where he put it is judged, a held one comes back
                at = lambda: {i: fp.split(os.sep)[-3] for i, fp in zip(*dataset.select(_ids).values(["id", "filepath"]))}
                keep, sp = set(ticked) & set(_ids), at(); up = [i for i in keep if sp[i] == "valid"]; down = [i for i in _ids if i not in keep and sp[i] == "train"]
                _, note = _stay_note(dataset, {"train": up, "valid": down}, args)
                if input(f"{len(keep)} TICKED of {len(_ids)} stay in train/. Move {len(up)} ticked from valid/ to train/ and {len(down)} unticked from train/ to valid/, image and label? " + note + "[y/N] ").strip().lower() == "y":
                    if up: _move_to_valid(dataset, up, args, "train")
                    if down: _move_to_valid(dataset, down, args, what="unticked")
                    done = [i for i, s_ in at().items() if s_ == ("train" if i in keep else "valid")]; dataset.select(done).tag_samples(jt)
                    print(f"{len(done)} of {len(_ids)} frames judged, never shown again. {len(_ids) - len(done)} stay in the queue.")
                else: print("Nothing moved, nothing judged.")
                exit()  # never on to the --to_valid prompt
            if args.review_files: ticked = set(ticked) & set(dataset.match_tags("train").values("id"))  # to_valid: a ticked frame of his list already in valid/ stays
            _, note = _stay_note(dataset, {"valid": ticked}, args)  # to_valid: counted without the frames the move leaves, they stay in train/
            if ticked and input(f"Move these {len(ticked)} TICKED frames of {len(train)} from train/ to valid/ on disk, image and label? " + note + "[y/N] ").strip().lower() == "y": _move_to_valid(dataset, list(ticked), args)
            else: print("Nothing moved.")
            exit()
        elif args.find_redundant is not None:
            iou, ham = (float(v) for v in args.find_redundant.split(":"))
            # train/ only: a frame in valid/ is spent
            train = dataset.match_tags("train")
            print(f"Finding redundant frames per camera in train/ at IoU {iou}, dHash <= {ham:.0f} of 64..."); dataset.untag_samples(["duplicate", "duplicate_group", "kept"])
            groups = _redundant_groups(train, iou, int(ham))
            for gid, ids in enumerate(groups):
                dataset.select(ids[:1]).tag_samples("kept"); dataset.select(ids[1:]).tag_samples("duplicate")
                dataset.select(ids).set_values("duplicate_group_id", [gid] * len(ids))
            print(f"Found {sum(len(g) - 1 for g in groups)} redundant frames in {len(groups)} groups of {len(train)} in train/; largest group {max(map(len, groups), default=0)}")
            # Two tags, his ruling 2026-09-21: kept and duplicate, groups side by side, kept first
            view = _cam_first(dataset.match_tags(["kept", "duplicate"]), ("duplicate_group_id", 1)).set_field("tags", F("tags").filter(F().is_in(["train", "validation", "kept", "duplicate"])))
            session = fo.launch_app(view=_class_menu(_gt_only(view), args)); session.config.show_confidence = session.config.show_label = False
            session.color_scheme = _color_scheme(); picked, ticked = _wait_or_enter(session, dataset)
            _write_back(dataset, picked, (), args)  # the redundant tick moves the frame below, it never wipes it
            # Author 2026-09-25, "Tick = move on both": a ticked kept frame takes its scene out of train/ once its whole group goes, so the prompt counts both
            go, note = _stay_note(dataset, {"valid": ticked}, args)   # counted after the frames the move leaves
            n_lead, n_whole = len(dataset.select(list(go)).match_tags("kept")), sum(set(g) <= go for g in groups)
            if ticked and input(f"Move these {len(ticked)} TICKED frames of {len(train)} in train/ to valid/ on disk, image and label? "
                                + note
                                + f"{n_lead} of them lead their group, {n_whole} of {len(groups)} groups leave train/ whole. [y/N] ").strip().lower() == "y": _move_to_valid(dataset, list(ticked), args)
            else: print("Nothing moved.")
            exit()
        elif args.find_duplicates is not None:
            print(f"Finding near-duplicates with '{args.similar_model}' at distance {args.find_duplicates}..."); dataset.untag_samples(["duplicate", "duplicate_group"])
            # embeddings cached per model; --reevaluate recomputes
            emb_field = "dup_emb_" + re.sub(r"\W", "_", args.similar_model)
            # brain reuses on schema presence alone, so a partly-filled field drops unembedded samples
            if emb_field in dataset.get_field_schema() and (args.reevaluate or dataset.exists(emb_field).count() != len(dataset)):
                dataset.delete_sample_field(emb_field)
            cached = emb_field in dataset.get_field_schema()
            if cached: print(f"  Reusing embeddings in '{emb_field}'; pass --reevaluate to recompute")
            idx = fob.compute_near_duplicates(dataset, threshold=args.find_duplicates, embeddings=emb_field,
                                              **({} if cached else {"model": args.similar_model}))
            for gid, (u, dups) in enumerate(idx.neighbors_map.items()):
                ids = [u] + [d for d, _ in dups]
                dataset.select(ids).tag_samples("duplicate_group"); dataset.select([d for d, _ in dups]).tag_samples("duplicate")
                dataset.select(ids).set_values("duplicate_group_id", [gid] * len(ids))
            print(f"Found {sum(len(d) for d in idx.neighbors_map.values())} near-duplicate images in {len(idx.neighbors_map)} groups")
        elif args.find_similar is not None:
            uq = "uniqueness_" + re.sub(r"\W", "_", args.similar_model)
            if uq in dataset.get_field_schema() and (args.reevaluate or dataset.exists(uq).count() != len(dataset)): dataset.delete_sample_field(uq)
            if uq not in dataset.get_field_schema():
                print(f"Computing image uniqueness with '{args.similar_model}'...")
                fob.compute_uniqueness(dataset, model=args.similar_model, uniqueness_field=uq)
                dataset.save()

        view = dataset.match_tags("duplicate").sort_by("duplicate_group_id") if dup else (dataset.match(F(uq) < args.find_similar) if args.find_similar is not None else dataset)
        session = fo.launch_app(view=_class_menu(_gt_only(view.sort_by(uq) if args.find_similar and not dup else view), args)); session.config.show_confidence = session.config.show_label = False
        session.color_scheme = _color_scheme()
        picked, ticked = _wait_or_enter(session, dataset); ticked = list(ticked)   # the harvested ticks: a grid Escape empties session.selected
        _write_back(dataset, picked, (), args)  # a click deletes on the duplicate and similar views too; the tick below is the sample

        if args.subset and ticked and args.output_dir:
            _export(dataset.select(ticked), f"{args.output_dir}1")
            _export(dataset.exclude(ticked), f"{args.output_dir}2")
        elif ticked and not (dup or args.find_similar is not None):
            print("--subset needs --output_dir: a tick here would delete those samples from the persistent input dataset, not split it. Nothing deleted.")
        elif ticked and _bare_bucket_argv(sys.argv[1:]):   # disk follows, or the merge builder ships the frame he dropped
            to = f"archived/dedup_{datetime.date.today()}"
            if input(f"Move these {len(ticked)} TICKED frames of {len(dataset)} to <bucket>/{to}/ on disk, image and label, and drop them from FiftyOne? [y/N] ").strip().lower() != "y": print("Nothing moved, nothing dropped."); exit()
            _move_to_valid(dataset, ticked, args, to); dataset.delete_samples(dataset.select(ticked).match(F("filepath").contains_str(f"/{to}/")))
            dataset.save()
        elif ticked:
            print(f"Dropping {len(ticked)} ticked frames from FiftyOne only. Disk unchanged: the line is not one --local_yolo_dirs bucket.")
            dataset.delete_samples(ticked)
            dataset.save()
            if args.output_dir: _export(dataset, args.output_dir)
        exit()

    if args.remap_classes:
        mapping = {_norm(k): _norm(v) for k, v in (p.split(":") for p in args.remap_classes)}
        for sample in dataset.iter_samples(autosave=True):   # a class ruling never reaches a COCO label
            for p in _polys(sample, "ground_truth"):
                if not getattr(p, "coco", False): p.label = mapping.get(p.label, p.label)
        kept = set(dataset.filter_labels("ground_truth", F("coco") == True).distinct("ground_truth.polylines.label"))   # the names COCO's labels still carry
        final_yaml_classes = list(dict.fromkeys([mapping.get(c, c) for c in final_yaml_classes] + [c for c in final_yaml_classes if c in kept]))   # COCO's old name exports too
        print(f"Final export classes: {final_yaml_classes}")

    class_thresholds_parsed = parse_class_thresholds(args.auto_approve_threshold)
    # YOLO runs at the lowest threshold; per-class filtering happens at approval
    class_dict, default_t = class_thresholds_parsed
    yolo_conf_thresh = min([*class_dict.values(), *([default_t] if default_t else [])], default=default_conf)
    filtered_view = active_learning_workflow(dataset, args.eval_model, target_classes, classes_were_specified, args.reevaluate, yolo_conf_thresh, args.eval_iou_thresh, args.nms_iou, args.pred_dedup_iou, args.splits_to_run, max_bbox_width=args.max_bbox_width, min_bbox_width=args.min_bbox_width, max_aspect_ratio=args.max_aspect_ratio, min_aspect_ratio=args.min_aspect_ratio, seg_iou=args.seg_iou, scales=args.scales, full_inference=args.full_inference, sample_num=args.sample, vertical_scaling=args.vertical_scaling, region=args.region, superres=args.superres, imgproc=args.imgproc, intensity=args.intensity, clip_model=args.similar_model if args.clip_outliers else None, find_missed_labels=args.find_missed_labels, filter_shadows=args.filter_shadows, clean_overlap_on=args.clean_overlap_on, class_thresholds=class_thresholds_parsed) if args.find_missed_labels else None

    if args.refine_boxes:
        refine_boxes_workflow(dataset, args.sam_model, args.splits_to_run, "ground_truth", args.sam_convert_to_boxes, clean_4point_seg_outliers=args.clean_4point_seg_outliers, clean_complex_seg_outliers=args.clean_complex_seg_outliers, sam_confidence_threshold=args.sam_confidence_threshold)

    if args.consensus:
        vocab = {_norm(c) for c in args.classes} if args.classes else set(dataset.distinct("ground_truth.polylines.label"))
        consensus_workflow(dataset, args, vocab, class_thresholds_parsed)
    elif args.sam3_adopt:
        sam3_adopt_workflow(dataset, args, class_thresholds_parsed)
    elif args.yolo_judge:
        yolo_judge_workflow(dataset, args)
    elif args.sam3_refine or args.sam3_prompt is not None:
        sam3_refine_workflow(dataset, args, class_thresholds_parsed)
    # export drops any class not in the list, by exact string
    if not classes_were_specified and (_live := {c for c in dataset.distinct("ground_truth.polylines.label") + dataset.distinct("ground_truth.detections.label") if c}):
        _cc = [_norm(c) for c in COCO_CLASSES]   # known names keep COCO order, only new ones are sorted onto the end
        final_yaml_classes = [c for c in dict.fromkeys(final_yaml_classes + _cc) if c in _live] + sorted(_live - set(final_yaml_classes) - set(_cc)); print(f"--classes not given: export class list is every label present: {final_yaml_classes}")

    if args.bioclip_classify:
        bioclip_classify_workflow(dataset, args, gt_field="ground_truth")
        existing_classes = set(dataset.distinct("ground_truth.detections.label")) | set(dataset.distinct("ground_truth.polylines.label"))
        existing_classes.discard(None)
        final_yaml_classes = final_yaml_classes + sorted(existing_classes - set(final_yaml_classes))
        print(f"Final export classes after BioCLIP: {final_yaml_classes}")
        if args.subset and args.output_dir:
            targets = {_norm(c) for c in args.bioclip_classes}
            sp2ids = defaultdict(list)
            for s in dataset.iter_samples():
                for l in (get_labels_from_sample(s, "ground_truth") or []):
                    name = getattr(l, "label", None)
                    if name and _norm(name) not in targets:
                        sp2ids[name].append(s.id); break
            species_labels = sorted(sp2ids)
            print(f"\n--- Subset by species ({len(species_labels)} groups): {species_labels} ---")
            for sp in species_labels:
                ids = sp2ids[sp]
                out = f"{args.output_dir.rstrip('/')}-{sp}"
                print(f"  {sp}: {len(ids)} images → {out}")
                export_yolo_dataset(dataset.select(ids), out, [sp], args.resplit, "ground_truth", args.resize, force_export=args.force_export)
            exit()

    view_to_show, done_tag = _cam_first(dataset.view()), "adopt_reviewed"
    dataset.save()

    session = None
    if not args.no_launch:
        has_gt = dataset.has_sample_field("ground_truth") and (dataset.count("ground_truth.detections") > 0 or dataset.count("ground_truth.polylines") > 0)
        # the sheet is valid only for the run that built it. `.exists()` keeps last_modified_at on samples already
        # None, the clock that attributes a label loss.
        if dataset.has_sample_field("review_sheet"):
            dataset.exists("review_sheet").set_field("review_sheet", None).save("review_sheet")  # every launch, not just adopt
        if args.consensus:
            # >= 2 is add_proposed and worse: mask_replaced/renamed were auto-applied and cover ~98%
            view_to_show = _cam_first(dataset.match(F("consensus_worst") >= 2), ("consensus_worst", -1))
            print(f"\n--- Consensus Review ---\n{len(view_to_show)} samples, worst first."
                  f"\nClick a ground_truth instance to DELETE it. Click a consensus_proposals instance to ACCEPT it into ground_truth."
                  f"\nFilter ground_truth by consensus (unsupported / disputed / rename_proposed) to work one verdict at a time;"
                  f" renamed and mask_replaced were applied automatically."
                  f"\nsingle (one agreeing vote, nothing contradicting) is NOT in this queue: filter consensus=single to see it."
                  f"\nground_truth_orig is the pre-corrector state; --consensus_revert restores it."
                  + (f"\nadded = SAM3 found it above {args.consensus_sam3_add} and it is ALREADY in ground_truth. Select it only to REJECT it." if args.consensus_sam3_add else ""))
        elif args.sam3_adopt:
            # silent SAM3 rewrites are excluded by design: the masks beat the boxes, only triggers need eyes
            fired = dataset.match(F("review_worst") > 0)
            # an add-only build promises every annotation survives
            if args.review_reason and args.review_reason != "more" and dataset.info.get("sam3_add_only"):
                sys.exit(f"'{dataset.name}' was built with --sam3_add_only, so its annotations were never up for review and only --review_reason more applies. '{args.review_reason}' would queue labels SAM3 simply did not find, and a tick would delete them.")
            if args.tick_wipes and dataset.info.get("sam3_add_only"):
                sys.exit(f"'{dataset.name}' was built with --sam3_add_only, so every annotation on it was promised to survive, and --tick_wipes deletes them from ground_truth_orig as well. The argparse check only sees the flag on the command line; the build's own record is what refuses here.")
            if args.review_reason:
                # review_reasons is written once at adopt, so the live passes ask the polylines
                if args.review_reason == "unsupported":
                    # his surviving labels SAM3 never backed, measured live off the masks
                    _uid, _ua, _ub, _uboth, _uoff, _udone = [], 0, 0, 0, 0, 0
                    _own = "/mnt/LinuxBackup2/Datasets_of_interest/Surveillance/Own/labeled/house-24Sept2024-final/"
                    for _s in dataset.select_fields(["filepath", "metadata", "ground_truth", "sam3_predictions"]).iter_samples(progress=True):
                        if not (_u := _unsupported(_s, min_conf=args.unsup_sam3_conf, hier=_hier(args), classes=_review_scope(args))): continue
                        _uid.append(_s.id); _udone += any(t.startswith(f"adopt_reviewed_{args.review_reason}") for t in _s.tags)   # the bare tag hides no gallery image (W3)
                        _a, _b = any(k == "mismatch" for _, k, _, _ in _u), any(k == "nocover" for _, k, _, _ in _u)
                        _ua += _a and not _b; _ub += _b and not _a; _uboth += _a and _b
                        _uoff += not os.path.realpath(_s.filepath).startswith(_own)
                    fired = dataset.select(_uid)
                    print(f"unsupported queue: {len(_uid)} images, {_udone} of them already tagged reviewed so you get {len(_uid) - _udone}: {_ua} class mismatch only, {_ub} no cover only, {_uboth} both."
                          + (f" {_uoff} of them live outside {_own}, which 24Sept merged in from the other house dirs." if dataset.name == "house-24Sept2024-final" else ""))
                elif args.review_reason == "more":
                    # live off the masks, confidence floor HERE, so queue, outlines and click act on one list;
                    # starts from the images the reason string lists, so it can only ever show FEWER
                    _mid, _mstale, _madd, _mrepl, _monly, _mbuck = [], 0, 0, 0, 0, Counter()
                    for _s in fired.match(F("review_reasons").contains("more")).select_fields(
                            ["filepath", "metadata", "ground_truth", "ground_truth_orig"]).iter_samples(progress=True):
                        _mstale += 1
                        _mbuck.update("unscored" if getattr(_p, "confidence", None) is None else f"{round(float(_p.confidence), 1):.1f}"
                                      for _p in _polys(_s, "ground_truth") if getattr(_p, "adopt", None) == "adopted")
                        # queue on anything the sheet will outline, not just on what a tick deletes
                        _m = _more_added(_s, args.adopt_match_iou, args.review_more_conf, classes=_review_scope(args), hier=_hier(args))
                        if not _m: continue
                        _mid.append(_s.id)
                        _a = sum(k == "added" for _, k, _ in _m)
                        _madd += _a; _mrepl += len(_m) - _a; _monly += not _a
                    fired = dataset.select(_mid)
                    print(f"more queue: {len(_mid)} images holding {_madd} additions at or above confidence {args.review_more_conf} "
                          f"and {_mrepl} replacements at any confidence, out of {_mstale} images the adopt-time reason string still lists."
                          + " That string is the only way in, and it was written at adopt and never redone, so a"
                          " SAM3 addition on an image the string does not list is not in this queue and you will"
                          " not see it."
                          + (f" {_monly} of them show replacements only, so a tick deletes nothing there." if _monly else ""))
                    print("  every adopted polyline on those " + str(_mstale) + " images by confidence: "
                          + (", ".join(f"{k}:{v}" for k, v in sorted(_mbuck.items())) or "none")
                          + "  (an unscored one is always shown, never held back)")
                elif args.review_reason == "dropped":
                    # `dropped` is `unsupported` asked at BUILD time, so it queues nothing and says so
                    _did, _dup2, _dun, _dcc, _dstale = [], 0, 0, 0, 0
                    for _s in _scan(dataset):
                        _live = _polys(_s, "ground_truth", live=True)
                        _kept = {p.id for p in _live if getattr(p, "adopt", None) == "kept_orig"}
                        if not _kept: continue
                        _dup2 += len(_kept)
                        _u = {p.id for p, _k, _, _ in _unsupported(_s, min_conf=args.unsup_sam3_conf, hier=_hier(args), classes=_review_scope(args))}
                        _c = set(_loose_class(_s, args)) | {p.id for p in _live if getattr(p, "adopt", None) == "classchange"}
                        _dun += len(_kept & _u); _dcc += len((_kept - _u) & _c); _dstale += len(_kept - _u - _c)
                    fired = dataset.select([])
                    print(f"dropped queue: nothing to show, on purpose. {_dup2} of your labels carry the build-time "
                          f"'SAM3 found nothing for this' stamp. {_dun} of them are in the unsupported pass right now, "
                          f"which asks the same question off the masks as they stand. {_dcc} are live class disputes. "
                          f"The other {_dstale} are stale: SAM3 does sit on those objects now, so the stamp is wrong "
                          f"and deleting them would throw away labels SAM3 backs.\n"
                          f"  Run --review_reason unsupported instead. It is the same pass, asked live.")
                elif args.review_reason == "rename":
                    # live, over the WHOLE dataset: a cow YOLO called a dog is where SAM3 did not argue (review_worst 0)
                    _hi, _kid, _kn, _kseen = _hier(args), [], 0, len(dataset)
                    for _s in _scan(dataset):
                        if _ka := _alt_names(_s, _hi, args.adopt_match_iou): _kid.append(_s.id); _kn += len(_ka)
                    fired = dataset.select(_kid)
                    print(f"rename queue: {_kn} objects on {len(_kid)} images of {_kseen} in the dataset that SAM3 also called "
                          f"by a coarser name, pairs from --sam3_hierarchy {' '.join(f'{k}:{v}' for k, v in sorted(_hi.items())) or '(none given, so only the pairs the build stored fire)'}.")
                else:
                    fired = (fired.match(F("ground_truth.polylines").filter(F("adopt") == args.review_reason).length() > 0)
                             if args.review_reason in SHEET_REASONS else
                             fired.match(F("review_reasons").contains(args.review_reason)))
                    # --classchange_iou widens THIS pass with the class errors adopt filed as new objects, whole dataset
                    if args.classchange_iou and args.review_reason == "classchange":   # lowiou walked the set for 0 (W47)
                        _tight, _limg, _ln, _lseen = set(fired.values("id")), set(), 0, len(dataset)
                        for _s in _scan(dataset):
                            if _lc := _loose_class(_s, args): _ln += len(_lc); _limg.add(_s.id)
                        fired = dataset.select(list(_tight | _limg))
                        print(f"classchange queue widened to IoU {args.classchange_iou}: {_ln} more objects on "
                              f"{len(_limg)} images of {_lseen} where SAM3 sits on one of your labels and "
                              f"calls it something else, {len(_limg - _tight)} of those images the tight pass never showed. "
                              f"Adopt only books a class error at {args.adopt_match_iou}, which is why these never queued.")
            # one direction at a time; adopt must match too, or the queue holds images the sheet draws nothing on
            if args.review_pair:
                _was, _, _now = args.review_pair.partition(":")
                fired = fired.match(F("ground_truth.polylines").filter(
                    (F("adopt") == args.review_reason) & (F("prev_label") == _norm(_was)) & (F("label") == _norm(_now))).length() > 0)
            elif args.review_origin:
                fired = fired.match(F("ground_truth.polylines").filter(
                    (F("adopt") == args.review_reason) & (F("prev_label").is_in(_ORIGINS(args)))).length() > 0)
            # own-dir before the tag, sort, cap and freeze, so served, ticked, tagged and outlined are one set
            _ownd = args.local_yolo_dirs[0] if args.own_dir else None
            if _ownd: fired = _own_dir_view(fired, _ownd, "Review", "queue images")
            # the list narrows the queue and its reset alike
            _rf = lambda v: v.select([i for i, fp in zip(*v.values(["id", "filepath"])) if os.path.basename(fp) in args.review_files], ordered=True) if args.review_files else v
            if args.review_files:
                _n0, fired = len(fired), _rf(fired)
                print(f"--review_files: {len(fired)} of {_n0} queue images; {len(_fr := args.review_files & set(map(os.path.basename, dataset.values('filepath'))))} of {len(args.review_files)} names are frames of '{dataset.name}'")
                # no listed name is a frame here: the wrong list or the wrong set, not a queue already done
                if not _fr: sys.exit(f"--review_files: none of its {len(args.review_files)} names is a frame of '{dataset.name}'. Check the set, and one image file name per line, no folder.")
            # the done tag carries the whole scope: reason, pair, loose IoU and origins
            done_tag = "_".join(x for x in ("adopt_reviewed", args.review_reason, (args.review_pair or "").replace(":", "_") or None,
                                            f"iou{args.classchange_iou}" if (args.classchange_iou and args.review_reason in SHEET_REASONS) else None,
                                            "from_" + "_".join(_ORIGINS(args)) if args.review_origin else None) if x)
            _sheet = args.review_reason in SHEET_REASONS and not args.review_pair
            # tag only, the annotations are not touched
            if args.review_reset:
                # the queue hides on ANY earlier scope's tag, so clear every tag asserting an origin this pass shows
                if _sheet:
                    _w = set(_ORIGINS(args)) if args.review_origin else None
                    _undo = [t for t in dataset.distinct("tags") if t != "adopt_reviewed" and t.startswith(f"adopt_reviewed_{args.review_reason}")
                             and (_w is None or (_reviewed_origins([t], args.review_reason) or _w) & _w)]
                else:
                    _undo = [done_tag] if (args.review_reason or args.review_pair) else [t for t in dataset.distinct("tags") if t.startswith("adopt_reviewed")]
                _rv = _rf(_own_dir_view(dataset, _ownd, "Reset", "samples") if _ownd else dataset)
                # the bare tag hides every sheet reason and a reset needs a reason, so split it: this reason comes
                # back, the other keeps its own tag, iou0 = hidden at every IoU. Origin scope cannot split it
                _bare = _rv.match_tags("adopt_reviewed").values("id") if _sheet else []
                if _bare and not args.review_origin:
                    _n = len(_rv.match_tags(_undo + ["adopt_reviewed"]))
                    for _r2 in sorted(SHEET_REASONS - {args.review_reason}): dataset.select(_bare).tag_samples(f"adopt_reviewed_{_r2}" + ("_iou0" if _r2 == "classchange" else ""))
                    _undo, _bare = _undo + ["adopt_reviewed"], []
                else: _n = len(_rv.match_tags(_undo)) if _undo else 0
                if _undo:
                    _rv.untag_samples(_undo)
                    # --consensus_revert's guard reads these tags, so record the reset where a tag wipe cannot reach
                    dataset.info["review_reset_cleared"] = sorted(set(dataset.info.get("review_reset_cleared") or []) | set(_undo))
                    dataset.save()
                print(f"--review_reset: put {_n} images back in the queue (cleared {', '.join(_undo) or 'nothing'}). Annotations untouched."
                      + (f"\n  {len(_bare)} stay hidden: a no-reason pass showed every origin on them. Reset without --review_origin to bring them back." if _bare else ""))
            # hide a sample when every origin this pass would show was already on screen under some scope
            if _sheet:
                _want, _keep, _hid = (set(_ORIGINS(args)) if args.review_origin else None), [], 0
                for _s in fired.select_fields(["tags", "ground_truth"] + (["sam3_predictions"] if fired.has_field("sam3_predictions") else [])).iter_samples(progress=False):
                    _seen = _reviewed_origins(_s.tags, args.review_reason)
                    _live = {_norm(pv) for p in _polys(_s, "ground_truth")
                             if getattr(p, "adopt", None) == args.review_reason and (pv := getattr(p, "prev_label", None))}
                    # a tag at a tighter IoU, or none, never showed the looser half this pass has to show; an origin tag
                    # carries no IoU at all, its pass ran tight
                    _ious = [float(g.group(1)) for t in _s.tags if t.startswith(f"adopt_reviewed_{args.review_reason}") and (g := re.search(r"_iou([0-9.]+)", t))]
                    _loose = not any(v <= args.classchange_iou for v in _ious) and _loose_class(_s, args)
                    if _seen is None:
                        if "adopt_reviewed" in _s.tags or not _loose: _hid += 1
                        else: _keep.append(_s.id)
                        continue
                    if not _live or ((_live & _want if _want is not None else _live) - _seen) or _loose: _keep.append(_s.id)
                    else: _hid += 1
                view_to_show = fired.select(_keep)
                if _hid: print(f"  {_hid} of {len(fired)} hidden: every {args.review_reason} origin on them was already on screen in an earlier pass, whatever its scope.")
            else:
                view_to_show = fired.match_tags(done_tag, bool=False)
            # inside one reason review_worst ties, so rank by what the dispute WAS, which predicts a rejection
            if args.review_reason in SHEET_REASONS:
                for _s in view_to_show.iter_samples(autosave=True, progress=False):
                    _s["review_prior"] = max((ORIGIN_REJECT_RATE.get(_norm(getattr(p, "prev_label", "") or ""), ORIGIN_REJECT_UNKNOWN)
                                              for p in _polys(_s, "ground_truth")
                                              if getattr(p, "adopt", None) == args.review_reason), default=0.0)
            view_to_show = _cam_first(view_to_show, ("review_prior" if args.review_reason in SHEET_REASONS else "review_worst", -1))
            n_fired, n_left = len(fired), len(view_to_show)
            if args.review_cap: view_to_show = view_to_show.limit(args.review_cap)
            # FREEZE the queue: a lazy view re-matches after his edits and tags images never on screen
            view_to_show = dataset.select(view_to_show.values("id"), ordered=True)
            # an empty queue opens no app window
            if dataset.has_sample_field("review_pos"): dataset.exists("review_pos").set_field("review_pos", None).save("review_pos")   # before the empty exit (W106), set rows only (W46)
            if args.review_reason and not len(view_to_show):
                print(f"Nothing left to review for '{args.review_reason}'"
                      + (f" (origin {'/'.join(sorted(_ORIGINS(args)))})" if args.review_origin else "")
                      + f" on '{dataset.name}': 0 of {n_fired} fired images unreviewed."); sys.exit(0)
            # 12/150 on the sidebar
            _pos_ids = view_to_show.values("id")
            dataset.select(_pos_ids, ordered=True).set_values(
                "review_pos", [f"{i + 1}/{len(_pos_ids)}" for i in range(len(_pos_ids))])
            # an unfilled outline over the disputed object, rebuilt every pass
            if args.review_reason in GALLERY_REASONS:
                _rw, _, _rn = (args.review_pair or "").partition(":")
                _n_sheet = _n_repl = _n_out = 0
                _scope = _review_scope(args)
                for _s in view_to_show.iter_samples(autosave=True, progress=False):
                    _d = []
                    # outline exactly the set the select writes to
                    if args.review_reason == "unsupported":
                        # what the prompt never named is not outlined, and is counted
                        _u = _unsupported(_s, min_conf=args.unsup_sam3_conf, hier=_hier(args), classes=_scope)
                        if _scope: _n_out += len(_unsupported(_s, min_conf=args.unsup_sam3_conf, hier=_hier(args))) - len(_u)
                        for p, kind, s3, iou in _u:
                            _d.append(_sheet_box(p, f"unsupported: {p.label} -> {s3} {iou:.2f}" if kind == "mismatch" else f"unsupported: {p.label} no sam3", kind=kind))
                    elif args.review_reason == "more":
                        # same call the queue and the post-session block make, so the three lists cannot drift
                        _m = _more_added(_s, args.adopt_match_iou, args.review_more_conf, classes=_scope, hier=_hier(args))
                        if _scope: _n_out += len(_more_added(_s, args.adopt_match_iou, args.review_more_conf, hier=_hier(args))) - len(_m)
                        for p, _kind, _iou in _m:
                            _d.append(_sheet_box(p, f"replaced: {p.label} {_iou:.2f}" if _kind == "replaced" else f"added: {p.label}", kind=_kind))
                        _n_repl += sum(x.kind == "replaced" for x in _d)
                    elif args.review_reason == "rename":
                        # the COARSER name SAM3 also put on the object
                        _ka = _alt_names(_s, _hier(args), args.adopt_match_iou)
                        _d += [_sheet_box(p, f"{p.label} or {_ka[p.id][0]} {_ka[p.id][1]:.2f}", kind="rename") for p in _polys(_s, "ground_truth") if p.id in _ka]
                    else:
                        # the loose half of classchange: arrow reads his class -> what SAM3 called it
                        _lc = _loose_class(_s, args)
                        for p in _polys(_s, "ground_truth"):
                            if p.id in _lc and p.points and p.points[0]:
                                _d.append(_sheet_box(p, f"{p.label} -> {_lc[p.id][0]} {_lc[p.id][1]:.2f} (loose)", kind="loose"))
                                continue
                            if getattr(p, "adopt", None) != args.review_reason or not (p.points and p.points[0]): continue
                            _pv = getattr(p, "prev_label", "") or ""
                            if args.review_pair and not (_norm(_pv) == _norm(_rw) and _norm(p.label) == _norm(_rn)): continue
                            if args.review_origin and _norm(_pv) not in _ORIGINS(args): continue
                            # a lowiou polyline kept its class, so no arrow; kind colours the box by its stamp
                            _d.append(_sheet_box(p, f"{_pv}->{p.label}" if _pv and _pv != p.label else f"{args.review_reason}: {p.label}", kind=args.review_reason))
                    _s["review_sheet"] = fo.Detections(detections=_d); _n_sheet += len(_d)
                dataset.add_dynamic_sample_fields()
                print(f"review_sheet: {_n_sheet} outline boxes drawn over "
                      + (f"the additions a select would delete ({_n_sheet - _n_repl} magenta) and the same-class replacements where a select takes SAM3's shape off and puts back the label that was there before SAM3{', or deletes both where that label is ' + ' or '.join(args.tick_wipes) + ', COCO labels excepted' if args.tick_wipes else ''} ({_n_repl} yellow)."
                         if args.review_reason == "more" else
                         "your own labels SAM3 does not back, black where it calls the object something else and orange where it found nothing there. A select deletes exactly these."
                         if args.review_reason == "unsupported" else
                         "objects SAM3 ALSO called by a coarser name, which was thrown away at build time. Select the IMAGE to take the coarse name on all of them; click a BOX to keep your own name on just that one."
                         if args.review_reason == "rename" else "the disputes on screen."))
                if _scope and args.review_reason in ("more", "unsupported"):
                    print(f"  --sam3_prompt {' '.join(args.sam3_prompt)}: this pass and its ticks act on {' and '.join(sorted(_scope))} only."
                          f" {_n_out} labels of other classes on these images were left out of the queue, off the screen and untouched.")
            trig = Counter(r for rs in dataset.values("review_reasons") for r in rs or [])
            # count the live polylines, not the stale adopt-time string
            for _r in SHEET_REASONS:
                if _r in trig: trig[_r] = len(dataset.match(F("ground_truth.polylines").filter(F("adopt") == _r).length() > 0))
            # rename has no cheap match, and its queue line already printed the live count
            trig.pop("rename", None)
            # what a dispute WAS predicts whether SAM3 is wrong; its confidence does not
            if trig.get("classchange"):
                _org, _todo = Counter(), Counter()
                for _s in (dataset.match(F("ground_truth.polylines").filter(F("adopt") == "classchange").length() > 0)
                           .select_fields(["tags", "ground_truth"]).iter_samples(progress=False)):
                    _seen, _here = _reviewed_origins(_s.tags, "classchange"), set()
                    for p in _polys(_s, "ground_truth"):
                        if getattr(p, "adopt", None) == "classchange" and (pv := getattr(p, "prev_label", None)):
                            _org[pv] += 1
                            if _seen is not None and _norm(pv) not in _seen: _here.add(pv)
                    # images, not instances: what is left to open per origin
                    _todo.update(_here)
                # silent on a pass that cannot act on it
                if _org and args.review_reason in (None, "classchange"):
                    # sorted by measured reject rate, not by count
                    _rank = sorted(_org, key=lambda k: -ORIGIN_REJECT_RATE.get(_norm(k), ORIGIN_REJECT_UNKNOWN))
                    print("classchange instances by ORIGIN class (what it was before SAM3), rejection rates"
                          " measured on the 173 disputes ruled on in house-24Sept2024-final, not on this dataset: "
                          + ", ".join(f"{k} {_org[k]} instance{'s' * (_org[k] != 1)}, {_todo[k]} image{'s' * (_todo[k] != 1)} left to open"
                                      f" ({f'~{ORIGIN_REJECT_RATE[_norm(k)]:.0%} rejected' if _norm(k) in ORIGIN_REJECT_RATE else 'unmeasured'})" for k in _rank)
                          + (f"\n  --review_origin {' '.join(_left[:2])} is one launch for what is worth your eyes; the queue is already in this order."
                             if (_left := [k for k in _rank if _todo[k]]) and not args.review_origin else ""))
            # what each name does, printed beside it
            _GLOSS = {"unsupported": "unsupported (your labels SAM3 does not back)",
                      "classchange": "classchange (SAM3 disputes what you called it)",
                      "more": "more (SAM3 found objects you had not labelled)",
                      "rename": "rename (SAM3 also had a coarser name for it)",
                      "dropped": "dropped (the build-time copy of unsupported)",
                      "lowiou": "lowiou (SAM3's mask moved off yours)"}
            _scope = ", ".join(x for x in (_GLOSS.get(args.review_reason, args.review_reason) or "every fault type",
                                           "origin " + "/".join(_ORIGINS(args)) if args.review_origin else None,
                                           args.review_pair.replace(":", "->") if args.review_pair else None) if x)
            # reference lines print only on an unscoped pass; _banner is re-printed after launch_app, last on screen
            _banner = (f"{dataset.name}  |  {_scope}  |  {n_left} LEFT"
                       + (f", showing {len(view_to_show)} (cap {args.review_cap}), {n_left - len(view_to_show)} hidden"
                          if args.review_cap and n_left > len(view_to_show) else "")
                       + f"  |  {n_fired - n_left} done of {n_fired}  |  {len(dataset)} images")
            print("\n--- SAM3 Adopt Review ---\n" + _banner
                  # Author 2026-09-15: "BE CONSISTENT, a click must mean the same thing on every pass."
                  + ({"classchange": "\nStay zoomed out and SELECT THE IMAGE to REJECT SAM3 HERE: the label goes back to what you called it,"
                                     " or is deleted if another label already covers the object. Every outlined label on the image gets this."   # W362: two on one image both go back, tested
                                     "\nOr open it: CLICK THE OUTLINE BOX to put your class back on just that one, CLICK THE POLYGON to remove SAM3; orig label stays unless you click it too.",
                      "lowiou": "\nStay zoomed out and SELECT THE IMAGE to REJECT SAM3 HERE: your own mask goes back in its place, or SAM3's is deleted where your mask is already there.",
                      # --tick_wipes changes what the tick deletes, so the screen says so before the tick
                      "more": (f"\nStay zoomed out and SELECT THE IMAGE to WIPE {' '.join(args.tick_wipes)} HERE: every label of those classes on it goes, yours too. Every magenta box goes. A yellow box, or a label of those classes SAM3 drew over yours, goes back to yours. If yours is of those classes, both go. Yours of those classes leave ground_truth_orig too, so it cannot be undone. COCO labels stay."
                               if args.tick_wipes else "\nStay zoomed out and SELECT THE IMAGE to REJECT SAM3 HERE: every magenta box goes. Under every yellow box SAM3's shape goes and the label that was there before SAM3 comes back. Everything else stays."),
                      "unsupported": "\nStay zoomed out and SELECT THE IMAGE to REJECT SAM3 HERE: your own labels SAM3 does not back go. The OUTLINED boxes are exactly what goes; everything else stays.",
                      "rename": "\nThis is the one pass where SELECT means ACCEPT, not reject, and NOTHING IS DELETED. SAM3 also called these objects by a coarser name. SELECT THE IMAGE to ACCEPT that coarser name on every outlined object at once. Leave it alone to keep your own name."
                                 "\nOr open it: CLICK THE OUTLINE BOX to keep your own name on just that one and stop being asked, CLICK THE POLYGON to remove SAM3; orig label stays unless you click it too."}
                     .get(args.review_reason, "") if args.review_reason in GALLERY_REASONS else "")
                  + (f"\nThe boxes marked (loose) are the widened half at IoU {args.classchange_iou}: SAM3's name was never"
                     " written on those, so keeping your class changes nothing on the label and only stops it being asked again."
                     if args.classchange_iou and args.review_reason == "classchange" else "")
                  # TWO VERBS, Author 2026-09-16: the polygon removes SAM3, the outline box cancels what this pass argues
                  + ("" if args.review_reason in ("classchange", "rename") else
                     "\nOr open it and CLICK THE POLYGON to remove SAM3; orig label stays unless you click it too.")
                  + " Where an image is selected AND one of its labels is clicked, the click wins."
                  + ("" if args.review_reason else
                     f"\n\nAll {len(dataset)} images hold: "
                     + ", ".join(f"{c} {v}" + ("" if v in SHEET_REASONS else " (as adopted)") for v, c in trig.most_common())
                     + f"\nMixing every fault type. --review_reason {'/'.join(sorted(ADOPT_RANK))} shows one at a time."
                     + "\nNothing here needs accepting; SAM3 output is already the annotation."
                     + "\nFilter ground_truth by adopt (classchange / lowiou / yolo_fill) to work one fault at a time; prev_label is what the annotation it replaced was called."
                     + f"\nreview_reasons: classchange = SAM3 disputes the class at IoU >= {args.eval_iou_thresh}, more = SAM3 found an object you had not labelled,"
                     + " dropped = SAM3 covered nothing there so your label was kept as it was, lowiou = the mask moved (best IoU under 0.5),"
                     + " rename = SAM3 also called the object by a coarser name and that name was dropped at build time."
                     + "\nground_truth_orig is the pre-adopt state; --consensus_revert restores it."))
        elif is_outlier_workflow: view_to_show = _cam_first(dataset.match_tags("outlier_candidate"))
        elif args.sam3_eval:
            # rank by worst disagreement, not count: one confident class swap beats five boundary quibbles
            view_to_show = _cam_first(dataset.match(F("sam3_worst") > 0), ("sam3_worst", -1))
            n_miss = 0
            for _p in ("ground_truth.polylines.sam3_verdict", "ground_truth.detections.sam3_verdict"):
                try: n_miss = n_miss or dataset.count_values(_p).get("missed", 0)
                except Exception: pass
            _who, _pf = ("YOLO", "yolo_judge_preds") if args.yolo_judge else ("SAM3", "sam3_predictions")
            print(f"\n--- {_who} Label-Error Review ---\n{len(view_to_show)} samples, worst disagreement first."
                  f"\nOn {_who}'s proposals ({_pf}): sam3_disagree, sam3_verdict, sam3_gt_label,"
                  f" sam3_seg_iou, sam3_boundary_iou, sam3_class_margin."
                  f"\nOn the annotations (ground_truth), which is what you edit: sam3_verdict, sam3_label"
                  f" (what {_who} called it instead), sam3_disagree. {n_miss} carry verdict 'missed'."
                  f"\nIn the app, filter ground_truth by sam3_verdict to work one fault class at a time:"
                  f" 'missed' for annotations {_who} saw nothing at, 'renamed' for a class {_who} disputes."
                  + (f"\nClick a ground_truth label to DELETE it, and it is PERMANENT: this pass writes no ground_truth_orig snapshot." + (" The only way back is --consensus_revert --allow_revert_reviewed, which hands the whole pre-adopt snapshot back and throws every review on this bucket away." if dataset.has_sample_field("ground_truth_orig") else " There is no snapshot to revert to on this bucket, so nothing brings it back.") + f" Clicks on {_pf} are counted and ignored." if args.yolo_judge else ""))
        elif args.find_missed_labels and args.keep_tn:
            view_to_show = _cam_first(dataset.match(~F("predictions").exists() | (F("predictions.detections").length() == 0)))
            print("\n--- Keep TN Mode ---")
            print(f"Showing {len(view_to_show)} images with NO predictions.")
            print("SELECT images that are FALSE NEGATIVES (should have been labeled but weren't).")
            print("UNSELECTED images will be exported as true negatives (TN).")
        elif args.find_missed_labels and args.auto_approve_threshold is None:
            base = filtered_view if filtered_view is not None else dataset
            view_to_show = _cam_first(base.filter_labels("predictions", F("eval") == "fp") if has_gt else base.exists("predictions"))
            print("\n--- Manual Review Mode ---\nSELECT samples (images) to approve ALL predictions in those samples.")
        if args.yolo_judge:
            # a comparison, so both layers on; sam3_predictions off, its verdicts share this pass's attribute names
            dataset.app_config.active_fields = ActiveFields(exclude=False, paths=["ground_truth", "yolo_judge_preds"])
            dataset.save()
            print("Layers: ground_truth and yolo_judge_preds, both on. ground_truth is what you click and what exports."
                  + (" sam3_predictions stays off: those polylines still carry SAM3's own verdicts under the same"
                     " attribute names, so reading them here would mix two passes." if dataset.has_sample_field("sam3_predictions") else ""))
        elif not (args.review_reason or args.sam3_eval or args.consensus or args.sam3_adopt or args.find_missed_labels or is_outlier_workflow) and dataset.has_sample_field("ground_truth"):   # the plain look only (W56); _gt_only sets its layers at launch
            print("Layers: ground_truth only. sam3_predictions is a PROPOSAL layer and never exports; "
                  "turn it on in the sidebar if you want to see it. Click a ground_truth label to DELETE it, permanent. On a bare --local_yolo_dirs line the label file on disk follows.")
        if args.review_reason or args.consensus:
            # ground_truth off, the rest named as the label fields to SHOW; yolo_judge_preds is not one of them
            _cons = ("ground_truth", "ground_truth_orig", "consensus_proposals")   # consensus builds no sheet and clicks proposals
            _show = [f for f in (_cons if args.consensus else ("ground_truth_orig", "sam3_predictions", "review_sheet")) if dataset.has_sample_field(f)]
            _prev = dataset.app_config.active_fields
            # every layer set this script writes counts as ours, listed literally, not gated on has_sample_field
            _ours = (["ground_truth", "yolo_judge_preds"], ["ground_truth"], list(_cons), _show)
            _mine = _prev is not None and (
                (bool(_prev.exclude) and list(_prev.paths or []) == ["ground_truth"])      # today's earlier form
                or (not _prev.exclude and list(_prev.paths or []) in _ours))
            _rev = "None" if (_prev is None or _mine) else f"ActiveFields(exclude={bool(_prev.exclude)}, paths={list(_prev.paths or [])})"
            dataset.app_config.active_fields = ActiveFields(exclude=False, paths=_show)
            dataset.save()
            # the revert recipe only when this run overwrites a layer set that was not ours
            if _rev != "None":
                print(f"Layers: {' / '.join(_show)} start on, the rest hidden."
                      f" This is saved on the dataset and outlives the run. It was {_rev}, and this puts that back:"
                      f"\n  python -c \"import fiftyone as fo; from fiftyone.core.odm.dataset import ActiveFields;"
                      f" d=fo.load_dataset('{dataset.name}'); d.app_config.active_fields={_rev}; d.save()\"")
        # what --sam3_prompt never named is not drawn; display only, never by confidence. Author 2026-09-17: "way too cluttered"
        _shown, _sc = view_to_show, _review_scope(args) if args.review_reason else None
        if _sc: _shown = _shown.filter_labels("sam3_predictions", F("label").is_in(sorted(_sc | set(args.sam3_prompt))), only_matches=False)
        # hide a SAM3 mask with no export twin and a dead original with no SAM3 mask on it.
        # Display only; an original under a SAM3 mask stays for click-both-to-remove, except on more.
        # Author 2026-10-04 on D: "All the old stale bounding boxes are still there from frigate". Redrawn boxes go in by his rule.
        if args.review_reason:
            _dead_s, _dead_o = [], []
            for _s in view_to_show.select_fields(["ground_truth", "ground_truth_orig", "sam3_predictions", "metadata"]):
                _g = {_vk(p) for p in _polys(_s, "ground_truth", live=True)}
                _sam = _polys(_s, "sam3_predictions", live=True)
                W, H = _wh(_s)
                if not W: continue
                # a twinless mask is not in the export. Only unsupported draws one, where it BACKS a label it judges. One over a COCO label goes
                _gp, _gb = _poly_boxes([p for p in _polys(_s, "ground_truth", live=True) if not getattr(p, "coco", False)
                                        and getattr(p, "adopt", None) in _UNSUP_LIVE and not (_sc and _norm(p.label) not in _sc)])
                _dead_s += [ObjectId(q.id) for q in _sam if _vk(q) not in _g
                            and (args.review_reason != "unsupported" or (args.unsup_sam3_conf is not None and (getattr(q, "confidence", None) is None or float(q.confidence) < args.unsup_sam3_conf))
                                 or _best_overlap(q, q.to_detection().bounding_box, _gp, _gb, W, H, boundary=False)[1] < UNSUP_COVER_IOU)]
                _sp, _sb = _poly_boxes([q for q in _sam if _vk(q) in _g])
                _dead_o += [ObjectId(o.id) for o in _polys(_s, "ground_truth_orig", live=True) if _vk(o) not in _g
                            and (args.review_reason == "more" or _best_overlap(o, o.to_detection().bounding_box, _sp, _sb, W, H, boundary=False)[1] < args.adopt_match_iou)]
            if _dead_s: _shown = _shown.filter_labels("sam3_predictions", ~F("_id").is_in(_dead_s), only_matches=False)
            if _dead_o: _shown = _shown.filter_labels("ground_truth_orig", ~F("_id").is_in(_dead_o), only_matches=False)
            print(f"Not drawn, not in the export: {len(_dead_s)} SAM3 masks, {len(_dead_o)} " + ("originals" if args.review_reason == "more" else "dead originals with no exported SAM3 mask on them"))
        # the app pages every label field whatever active_fields says, and no adopt pass draws ground_truth
        if args.sam3_adopt and args.review_reason: _shown = _shown.exclude_fields("ground_truth")
        if not (args.review_reason or args.yolo_judge or args.sam3_eval or args.consensus or args.sam3_adopt): _shown = _gt_only(_shown, *(["predictions"] if args.find_missed_labels else []))  # the plain look: only what exports is on screen
        _app = _app_copy(dataset, _shown) if args.sam3_adopt and args.review_reason else None   # W371: the editor gets what it shows
        session = fo.launch_app(view=_class_menu(_app.view() if _app is not None else _shown, args)); session.config.show_confidence = session.config.show_label = False
        session.color_scheme = _color_scheme(ADOPT_FIELDS if args.review_reason else ())
        if args.sam3_eval or args.consensus or args.sam3_adopt: session.config.show_confidence = True
        if args.consensus or args.sam3_adopt: session.config.show_label = True
        # last line before the prompt
        if locals().get("_banner"): print("\n" + _banner)
        picked, ticked = _wait_or_enter(session, dataset)   # grid Escape clears the live session.selected; use the union
        _dels, _edits, _drawn = _app_diff(dataset, _app, _shown, args.adopt_match_iou) if _app is not None else ([], [], [])   # not bool(): that counts the copy
        picked += _dels   # an editor delete is a click; one he also clicked counts once downstream
        if _dc := {p.id for _, p in _drawn} & {l["label_id"] for l in picked}:   # a drawing clicked in its own pass: the click wins, and it is no unresolved click
            _drawn, picked = [d for d in _drawn if d[1].id not in _dc], [l for l in picked if l["label_id"] not in _dc]; print(f"{len(_dc)} labels you drew and clicked in this pass: the click wins, they are not kept")

        # Tagging happens AFTER the edit blocks below, never here.

        # Gallery pass: a tick touches only this pass's fault. A rejected class goes back to his, deleted only
        # where a surviving label of that class already covers the object.
        _reason = args.review_reason if args.sam3_adopt and args.review_reason in GALLERY_REASONS else None
        tick_dead = {}   # id -> points of each label the tick removed, so a click on one is not called unresolved
        if _reason and ticked:
            _was, _, _now = (args.review_pair or "").partition(":")
            def _disputed(p):
                if getattr(p, "adopt", None) != _reason: return False
                _pv = _norm(getattr(p, "prev_label", "") or "")
                # only the origins the sheet outlined
                if args.review_pair: return _pv == _norm(_was) and _norm(p.label) == _norm(_now)
                return not args.review_origin or _pv in _ORIGINS(args)
            n_rev = n_del = n_fill = n_res = n_miss = n_bare = n_keep = n_wipe = n_worig = n_wcoco = 0
            _wipe = {_norm(c) for c in (args.tick_wipes or [])}
            for sample in dataset.select(list(ticked)).iter_samples(autosave=True, progress=False):
                polys = list(sample["ground_truth"].polylines) if sample["ground_truth"] else []
                orig = sample["ground_truth_orig"] if sample.has_field("ground_truth_orig") else None
                gts, gboxes = _poly_boxes(list(orig.polylines) if orig else [])
                W, H = _wh(sample)
                others, oboxes = _poly_boxes([p for p in polys if not _disputed(p)])
                # same predicate the sheet drew with, so the set a select acts on is the set it outlined
                _unsup = {q.id for q, _, _, _ in _unsupported(sample, min_conf=args.unsup_sam3_conf, hier=_hier(args), classes=_review_scope(args))} if _reason == "unsupported" else set()
                _more, _mrep = set(), {}
                if _reason == "more":
                    # a tick rejects every SAM3 prediction on the image; one drawn OVER him hands his back
                    for q, k, _ in _more_added(sample, args.adopt_match_iou, args.review_more_conf, classes=_review_scope(args), hier=_hier(args)):
                        if k == "added": _more.add(q.id); continue
                        qb = q.to_detection().bounding_box if (q.points and q.points[0]) else None
                        if not (qb and W and gts): continue
                        b, _bi, _ = _best_overlap(q, qb, gts, gboxes, W, H, boundary=False, prefer=_agree(_hier(args), _norm(q.label)), prefer_floor=args.adopt_match_iou)
                        if b is not None: _mrep[q.id] = b
                    # a class change to a wiped class is judged by the label it changed, as a yellow box is
                    for q in polys:
                        if getattr(q, "adopt", None) == "classchange" and _norm(q.label) in _wipe and q.points and q.points[0] and W and gts:
                            _pv = _norm(getattr(q, "prev_label", "") or "")
                            b, _bi, _ = _best_overlap(q, q.to_detection().bounding_box, gts, gboxes, W, H, boundary=False, prefer=_agree({}, _pv), prefer_floor=args.adopt_match_iou)
                            if b is not None and _bi >= args.adopt_match_iou and _norm(gts[b].label) == _pv: _mrep[q.id] = b
                    _mused = {i for i, g in enumerate(gts) for p in polys if p.id not in _mrep and _vk(p) == _vk(g)}   # his label already live
                # a rename tick takes the coarser name; his outline-box click says "my name stands" and wins
                _boxed = {getattr(q, "gt_id", None) for q in (sample["review_sheet"].detections if sample.has_field("review_sheet") and sample["review_sheet"] else [])
                          if any(l["field"] == "review_sheet" and l["label_id"] == q.id for l in picked)}
                _ren = {k: v for k, v in _promote_map(sample, args, _reason).items() if k not in _boxed}
                _loose = _loose_class(sample, args)
                if not gts: n_bare += 1
                keep, changed, used = [], False, set()
                if _reason == "lowiou" and gts and W:
                    # an annotation a surviving label already sits on is not free to be restored onto
                    for q, qbox in zip(others, oboxes):
                        if not qbox: continue
                        bq, iq, _ = _best_overlap(q, qbox, gts, gboxes, W, H, boundary=False)
                        if bq is not None and iq >= args.adopt_match_iou: used.add(bq)
                for p in polys:
                    box = p.to_detection().bounding_box if p.points and p.points[0] else None
                    # --tick_wipes, FIRST test on purpose: every branch below keeps the label. A label SAM3 drew over one is judged
                    # by that one, which the restore below hands back unless it is his own of a wiped class. COCO's stays
                    _k = gts[_mrep[p.id]] if p.id in _mrep else p
                    if _norm(_k.label) in _wipe:
                        if getattr(_k, "coco", False): n_wcoco += _k is p
                        else: n_wipe += 1; changed = True; continue
                    if _reason == "unsupported":
                        if p.id in _unsup: n_del += 1; changed = True; continue
                        keep.append(p); continue
                    if _reason == "more":
                        if p.id in _more: n_del += 1; n_fill += getattr(p, "adopt", None) == "yolo_fill"; changed = True; continue
                        if p.id in _mrep:
                            if _mrep[p.id] in _mused: n_del += 1; changed = True; continue   # a second mask over one label of his is a twin
                            _mused.add(_mrep[p.id]); _o = gts[_mrep[p.id]]
                            p.points = [list(part) for part in _o.points]
                            p.label, p.adopt = _o.label, "restored"
                            # COCO's points come back with COCO's mark, or the restored label is on trial again
                            if getattr(_o, "coco", False): p.coco = True
                            n_res += 1; changed = True
                        keep.append(p); continue
                    # the loose half of classchange: stamping it answered is the whole action
                    if p.id in _loose:
                        p.alt_name_done = True; p.clear_field("alt_name")
                        n_keep += 1; changed = True; keep.append(p); continue
                    if _reason == "rename":
                        if p.id in _ren:
                            p.alt_name_prev, p.label = p.label, _ren[p.id]
                            p.alt_name_done = True; p.clear_field("alt_name")
                            n_rev += 1; changed = True
                        keep.append(p); continue
                    if not _disputed(p): keep.append(p); continue
                    prev = _norm(getattr(p, "prev_label", "") or "")
                    if not prev: n_del += 1; changed = True; continue
                    if _reason == "lowiou":
                        # ask which annotation the moved mask claims
                        b = None if (not box or not W) else _best_overlap(p, box, gts, gboxes, W, H, boundary=False)[0]
                        if b is None or _norm(gts[b].label) != prev:
                            n_miss += 1; keep.append(p); continue
                        if b in used: n_del += 1; changed = True; continue
                        used.add(b); p.points = [list(part) for part in gts[b].points]
                        p.label, p.adopt = prev, "restored"
                        if getattr(gts[b], "coco", False): p.coco = True
                        n_res += 1; changed = True; keep.append(p); continue
                    # only a surviving label of the class he wants back is cover
                    covered, _same = False, [(q, qb) for q, qb in zip(others, oboxes) if _norm(q.label) == prev]
                    if box and W and _same:
                        b, seg_iou, _ = _best_overlap(p, box, [q for q, _ in _same], [qb for _, qb in _same], W, H, boundary=False)
                        covered = b is not None and seg_iou > COVERED_IOU
                    if covered: n_del += 1; changed = True; continue
                    p.label, p.adopt = prev, "reverted"
                    n_rev += 1; changed = True; keep.append(p)
                if _wipe and orig:
                    # the irreversible half
                    _ok = [q for q in orig.polylines if _norm(q.label) not in _wipe or getattr(q, "coco", False)]
                    if len(_ok) != len(orig.polylines):
                        n_worig += len(orig.polylines) - len(_ok); changed = True
                        sample["ground_truth_orig"] = fo.Polylines(polylines=_ok)
                if changed: sample["ground_truth"] = fo.Polylines(polylines=keep); tick_dead.update({p.id: p.points for p in polys if p.id not in {q.id for q in keep}})
            tail = f" across {len(ticked)} selected images."
            if _reason == "unsupported": print(f"{n_del} of your own labels deleted that SAM3 does not back,{tail}")
            elif _reason == "more": print(f"{n_del - n_fill} SAM3" + (f" and {n_fill} YOLO fill" if n_fill else "") + f" labels deleted, {n_res} objects put back to your own label where SAM3 had drawn over you,{tail}"
                                        + (f" {n_bare} of those images carried no annotation of yours at all, so every label on them"
                                           + (f" in {' and '.join(sorted(_review_scope(args)))}" if _review_scope(args) else "") + " was an addition." if n_bare else ""))
            elif _reason == "rename": print(f"{n_rev} labels renamed to the coarser class SAM3 also saw on them, nothing deleted,{tail}")
            elif _reason == "lowiou": print(f"{n_res} masks put back to yours, {n_del} deleted (your own mask is already on the object), {n_miss} left alone (could not find the annotation SAM3 replaced),{tail}")
            else: print(f"{n_rev} labels put back to your class, {n_del} deleted (something else already covers the object),{tail}"
                        + (f" {n_keep} more kept your class against SAM3 at the loose IoU {args.classchange_iou}; nothing was renamed and they will not be shown again." if n_keep else ""))
            if _review_scope(args) and _reason in ("more", "unsupported"):
                print(f"  --sam3_prompt {' '.join(args.sam3_prompt)}: only {' and '.join(sorted(_review_scope(args)))} was touched"
                      + (f", plus the {n_res} objects put back to your own label above" if _reason == "more" and n_res else "")
                      + ". No other label of any other class on those images changed, in ground_truth or ground_truth_orig.")
            if n_wipe or n_worig or n_wcoco: print(f"  --tick_wipes {' '.join(args.tick_wipes)}: {n_wipe} more labels deleted from ground_truth by the wipe. {n_worig} of your own labels of those classes left ground_truth_orig too,"
                                        " so --consensus_revert can no longer bring those back."
                                        + (f" {n_wcoco} COCO labels of those classes were KEPT: only your own click removes one." if n_wcoco else ""))
        elif args.sam3_adopt and ticked and not _reason:
            print(f"Ignored {len(ticked)} selected images: '{args.review_reason or 'no reason'}' has no single label carrying the fault, so click the polygon instead.")

        # --yolo_judge and the plain look: a click deletes. Author: "there should be no such thing as read only".
        # the outlier and find_missed screens read the tick below (clean, approve, FN), so it must reach them unwiped
        if not (args.consensus or args.sam3_adopt) and (picked or ticked or _EDITED):
            _tk = () if is_outlier_workflow or args.find_missed_labels else ticked
            if _tk and dataset.info.get("sam3_add_only"):   # every annotation survives a tick on such a build; a click still deletes
                print(f"Ignored {len(_tk)} ticked images: '{dataset.name}' was built --sam3_add_only, so a tick deletes nothing. Click a label to delete it."); _tk = ()
            _write_back(dataset, picked, _tk, args)
            print_dataset_stats(dataset, "Label counts after the review")

        # Per-instance review: a clicked label is a delete, a clicked proposal an accept. Any click that
        # resolves to no live label is reported, never dropped in silence.
        if (args.consensus or args.sam3_adopt) and picked:
            by, n_del, n_acc, n_dup, n_res, n_mine, n_kept, unresolved, n_tickd = defaultdict(lambda: ([], [])), 0, 0, 0, 0, 0, 0, Counter(), Counter()
            for l in picked: by[l["sample_id"]][1 if l["field"] == "consensus_proposals" else 0].append((l["field"], l["label_id"]))
            for sample in dataset.select(list(by)).iter_samples(autosave=True):
                sel, accs = by[sample.id]; polys = list(sample["ground_truth"].polylines) if sample["ground_truth"] else []
                accs = [lid for _, lid in accs]; dels = set()
                # a click on SAM3 over him hands his own back, clicking both leaves nothing.
                # His clicks are read first, or a click on SAM3 would restore a label he just deleted.
                origs = _polys(sample, "ground_truth_orig")
                obys, oboxs = _poly_boxes(origs)
                W, H = _wh(sample)
                under = {}                                  # live polyline id -> index of his label it drew over
                # "restored" and "reverted" belong here too: a tick's handback stays mapped
                for p in polys:
                    if not (obys and W and p.points and p.points[0]
                            and getattr(p, "adopt", None) in ("adopted", "restored", "reverted")): continue
                    if (b := _orig_under(p, obys, oboxs, W, H, args)) is not None: under[p.id] = b
                dead_orig, restore, reclass, kept = set(), {}, {}, set()
                # a click on an adopted object hands it back, a click on anything a tick already handed
                # back ("restored"/"reverted") deletes. The snapshot is untouched.
                _stamp = {p.id: getattr(p, "adopt", None) for p in polys}
                # classchange: the OUTLINE BOX puts his class back, the POLYGON deletes
                _prev = {p.id: (getattr(p, "prev_label", "") or "") for p in polys}
                _kalt = _promote_map(sample, args, args.review_reason)
                _klo = _loose_class(sample, args)
                for fld, lid in sorted(sel, key=lambda t: t[0] != "ground_truth_orig"):
                    if fld == "ground_truth_orig":
                        oi = [i for i, q in enumerate(origs) if q.id == lid]
                        if not oi: unresolved[fld] += 1; continue
                        dead_orig.add(oi[0]); n_mine += 1
                        # a live copy carrying HIS geometry goes too, unless it stands in for a DIFFERENT label of his
                        dels |= {p.id for p in polys
                                 if _vk(p) == _vk(origs[oi[0]]) and under.get(p.id, oi[0]) == oi[0]}
                        continue
                    hit = {lid} if fld == "ground_truth" else set()
                    if fld == "review_sheet" and sample.has_field("review_sheet") and sample["review_sheet"]:
                        hit = {getattr(q, "gt_id", None) for q in sample["review_sheet"].detections if q.id == lid}
                    elif fld == "sam3_predictions" and sample.has_field(fld) and sample[fld]:
                        # an adopted label IS its prediction's points; a copy stacked on his label may start on another vertex
                        _q = next((q for q in getattr(sample[fld], "polylines", []) if q.id == lid), None)
                        hit = {p.id for p in polys if _q is not None and _vk(p) == _vk(_q)}
                        if not hit and _q is not None and obys and W and _q.points and _q.points[0]:
                            # a tick already handed it back: find the label of his it covers, and the class must agree
                            ag = _agree(_hier(args), _norm(_q.label))
                            b, iou, _ = _best_overlap(_q, _q.to_detection().bounding_box, obys, oboxs, W, H, boundary=False, prefer=ag, prefer_floor=args.adopt_match_iou)
                            if b is not None and iou >= args.adopt_match_iou and ag(_norm(obys[b].label)):
                                hit = {pid for pid, oi in under.items() if oi == b}
                        hit = hit or {pid for pid, pts in tick_dead.items() if _q is not None and pts == _q.points}   # an addition the tick took
                    if not (hit & {p.id for p in polys}): (n_tickd if hit & tick_dead.keys() else unresolved)[fld] += 1; continue
                    hit &= {p.id for p in polys}
                    # TWO VERBS, Author 2026-09-16: the box CANCELS what this pass argues, the polygon DELETES
                    _cancel = fld == "review_sheet"
                    for pid in hit:
                        if _cancel and _kalt.get(pid): kept.add(pid)          # rename: my name stands, stop asking
                        elif _cancel and _stamp.get(pid) == "classchange" and _prev.get(pid): reclass[pid] = _prev[pid]
                        elif _cancel and pid in _klo: kept.add(pid)           # loose classchange: same, no data change
                        # on these two passes the box NEVER deletes, even after a tick answered the object
                        elif _cancel and args.review_reason in ("classchange", "rename"): kept.add(pid)
                        elif (pid in under and under[pid] not in dead_orig and _stamp.get(pid) == "adopted" and under[pid] not in restore.values()
                              and not any(q.id != pid and _vk(q) == _vk(origs[under[pid]]) for q in polys)): restore[pid] = under[pid]   # his, already exported, is not handed back twice
                        else: dels.add(pid); restore.pop(pid, None); reclass.pop(pid, None)
                for p in polys:
                    if p.id in dels: continue   # his polygon click wins over his box click on one object: counted once, as a delete
                    # adopt left alone on purpose, so a later click still hands it back
                    if p.id in kept:
                        p.alt_name_done = True; p.clear_field("alt_name"); n_kept += 1
                    if p.id in reclass:
                        # the tick's cover rule: a live label of his class already on the object makes the revert a twin
                        if W and any(q.id != p.id and q.id not in dels and _norm(q.label) == reclass[p.id] and _pair_ious(p, q, W, H, False)[0] > COVERED_IOU for q in polys): dels.add(p.id); continue
                        # reclass needs stamp "classchange", restore needs "adopted", so n_res counts each object once
                        p.label, p.adopt = reclass[p.id], "reverted"; n_res += 1
                    if p.id not in restore: continue
                    q = origs[restore[p.id]]
                    p.points = [list(part) for part in q.points]; p.label, p.adopt = q.label, "restored"; n_res += 1
                    if getattr(q, "coco", False): p.coco = True
                    # a restore on a rename pass is an answer
                    if args.review_reason == "rename": p.alt_name_done = True; p.clear_field("alt_name")
                if dead_orig:
                    sample["ground_truth_orig"] = fo.Polylines(polylines=[q for i, q in enumerate(origs) if i not in dead_orig])
                gts = [p for p in polys if p.id not in dels]; n_del += len(polys) - len(gts); _nd, _a0 = len(gts) < len(polys), n_acc
                props = _polys(sample, "consensus_proposals")
                for p in (p for p in props if p.id in accs):
                    if any(g.label == p.label and compute_iou(g.to_detection().bounding_box, p.to_detection().bounding_box) > args.pred_dedup_iou for g in gts): n_dup += 1
                    else: gts.append(fo.Polyline(label=p.label, points=p.points, closed=True, filled=True)); n_acc += 1
                # only a change counts as reviewed: tagging what was merely opened drops unfixed faults. An accept is a change
                if args.sam3_adopt and (_nd or reclass or kept or n_acc > _a0) and done_tag not in sample.tags: sample.tags.append(done_tag)
                sample["ground_truth"] = fo.Polylines(polylines=gts)
                # has_field, or a delete-only adopt pass writes an empty proposals field onto every sample it touched
                if sample.has_field("consensus_proposals"): sample["consensus_proposals"] = fo.Polylines(polylines=[p for p in props if p.id not in accs])
            # adopt has no proposals, so reporting "accepted 0" reads as a silent failure to save the work
            print(f"Consensus review: deleted {n_del} ground_truth instances, accepted {n_acc} proposals" if args.consensus
                  # n_mine is NOT a subset of n_del
                  else f"Adopt review: {n_del} objects deleted, {n_res} put back to your own label where you rejected SAM3's, "
                       f"{n_mine} of your own labels dropped from the snapshot; {len(view_to_show)} images marked reviewed."
                       + (f" {n_acc} leftover consensus proposals you clicked were ACCEPTED into ground_truth." if n_acc else ""))
            if n_dup: print(f"  {n_dup} proposals you clicked were not added: a label of the same class already covers that object.")
            if n_kept: print(f"  {n_kept} objects kept YOUR OWN name against SAM3 because you clicked their outline box. "
                             "Nothing was renamed or deleted and they will not be shown again.")
            if n_mine: print(f"  {n_mine} of those were your own annotations and they are gone from ground_truth_orig too,"
                             " so --consensus_revert can no longer bring them back. That is what clicking your own label means.")
            if n_tickd: print(f"  {sum(n_tickd.values())} clicks were on labels your tick on the same image already removed.")
            if unresolved: print(f"IGNORED {sum(unresolved.values())} clicks that match no live ground_truth label, by field: {dict(unresolved)}")
        if args.consensus and not args.sam3_adopt and ticked: _delete_picked(dataset, [], ticked)   # a consensus tick wipes too, after its clicks resolve

        if _edits or _drawn: _app_apply(dataset, _edits, _drawn, picked)

        # everything shown counts as reviewed, or the queue never drains; last write of the block on purpose
        if args.sam3_adopt:
            view_to_show.tag_samples(done_tag)
            print(f"{len(view_to_show)} images tagged {done_tag} and will not be served again. If you closed early, "
                  f"the ones you never opened are tagged too. Undo the whole pass with --review_reset.")

        # Manual approval: merge selected samples' predictions into ground_truth
        if args.find_missed_labels and args.auto_approve_threshold is None and ticked:
            approved_count = 0
            # session.view carries the UI filters (confidence slider, eval filter)
            selected_view = session.view.select(list(ticked))
            for sample in selected_view.iter_samples(autosave=True, progress=True):
                if sample.has_field("predictions") and sample["predictions"] and sample["predictions"].detections:
                    approved_count += merge_predictions_to_ground_truth(sample["predictions"].detections, sample, "ground_truth", args.pred_dedup_iou)
            dataset.save()
            print(f"\nMerged {approved_count} FP predictions from {len(ticked)} samples into ground_truth")
            print_dataset_stats(dataset, "Dataset stats after manual approval")

        # Universal FP export for prediction workflows (not outlier cleaning).
        # a SAM3 review is not an FP pass: its selections mean "SAM3 got this wrong"
        if ticked and not is_outlier_workflow and not args.keep_tn and not args.consensus and not args.sam3_adopt and not args.sam3_eval and not args.sam3_refine and not (args.find_missed_labels and args.auto_approve_threshold is None):
            fp_export = input("\nExport selected samples to -fp directory? (y/n): ").lower().strip() == 'y'
            if fp_export:
                if args.output_dir:
                    output_dir_base = args.output_dir.rstrip('/')
                elif args.local_yolo_dirs:
                    output_dir_base = args.local_yolo_dirs[0].rstrip('/') + '-clean'
                elif args.local_coco_dirs:
                    output_dir_base = args.local_coco_dirs[0].rstrip('/') + '-clean'
                else:
                    output_dir_base = input("No --output_dir and no local input dir; type an export path (blank to skip): ").strip()
            if fp_export and output_dir_base:
                fp_dataset, clean_dataset = dataset.select(list(ticked)), dataset.exclude(list(ticked))
                print(f"\n--- Exporting FP Split ---\nFalse Positives: {len(fp_dataset)} images → {output_dir_base}-fp\nClean data: {len(clean_dataset)} images → {output_dir_base}")
                _export(fp_dataset, f"{output_dir_base}-fp")
                _export(clean_dataset, output_dir_base)
                exit()

        if is_outlier_workflow and ticked:
            print(f"\n--- Cleaning {len(ticked)} selected samples ---"); removed_annotations = 0
            for sample in dataset.select(list(ticked)).iter_samples(progress=True, autosave=True):
                labels = get_labels_from_sample(sample, "ground_truth")
                if "outlier_candidate" in sample.tags: sample.tags.remove("outlier_candidate")   # an emptied frame too (W115)
                if not labels: continue
                kept_labels = [l for l in labels if "outlier" not in l.tags]
                removed_annotations += len(labels) - len(kept_labels)
                # per sample, not off the first: an empty first sample rebuilt every other as the wrong type
                sample.ground_truth = (fo.Detections(detections=kept_labels) if isinstance(sample.ground_truth, fo.Detections)
                                       else fo.Polylines(polylines=kept_labels))

            print(f"Removed {removed_annotations} outlier annotations. Cleaning complete."); print_dataset_stats(dataset, "Label counts after manual cleaning")
        elif args.find_missed_labels and args.keep_tn and args.output_dir:
            unlabeled_view = dataset.match(~F("predictions").exists() | (F("predictions.detections").length() == 0))
            autolabeled_view = dataset.match(F("predictions").exists() & (F("predictions.detections").length() > 0))
            # ticked frames are false negatives; with no tick every unlabeled frame is a true negative
            fn_dataset = dataset.select(list(set(ticked)))
            tn_dataset = dataset.select(list(set(unlabeled_view.values("id")) - set(ticked))) if ticked else unlabeled_view
            print("\n--- Exporting with keep_tn ---" if ticked else "\n--- Exporting with keep_tn (no FN selected) ---")
            if ticked: print(f"False Negatives (FN): {len(fn_dataset)} images")
            print(f"True Negatives (TN): {len(tn_dataset)} images")
            print(f"Auto-labeled: {len(autolabeled_view)} images")
            if ticked: _export(fn_dataset, f"{args.output_dir}-fn")
            _export(tn_dataset, f"{args.output_dir}-tn")
            _export(autolabeled_view, f"{args.output_dir}-autolabeled")
    if (args.consensus or args.sam3_adopt) and session is None and (args.output_dir or args.output_coco_dir):
        print("Export skipped: --sam3_adopt/--consensus with --no_launch never exports, because the export ships the state the app pass produces. Drop --sam3_adopt to export.")
    if not args.keep_tn and not (args.find_missed_labels and args.auto_approve_threshold is None and session is not None and not ticked) and not ((args.consensus or args.sam3_adopt) and session is None):
        export_view = _own_dir_view(dataset, args.local_yolo_dirs[0]) if args.own_dir else dataset
        if args.output_dir: _export(export_view, args.output_dir)
        if args.output_coco_dir: export_coco_dataset(export_view, args.output_coco_dir, _export_classes(), args.resplit, "ground_truth", force_export=args.force_export)
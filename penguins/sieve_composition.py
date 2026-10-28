#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""What does the sanity sieve actually remove, and does the published table
reproduce from the sieve the paper describes?

WHY
---
P5 claims at lines 116 and 213 that the sieve eliminates three dominant failure
modes. The paper shows only an aggregate precision delta, so a reviewer cannot
see which test does the work. Worse, `tab:sieve_ablation` matches
`~/bench_any/runs/p5_r1_v2/precision_count_raw_vs_sieve.json` cell for cell but
nothing in the repository writes that file, and the one script `build_release.py`
names as its source carries a private sieve at min_area 64 and NMS 0.8, which are
not the paper's 324 and 0.55. So the table may not have come from the sieve the
paper describes.

This runs the canonical `sanity_sieve.sieve_masks` over the same raw predictions
and checks. Reconciliation comes first; the composition breakdown is only
meaningful once the published cells reproduce.

The raw source is the single prompt "a penguin" at the generation floor. That is
what gives 912 detections on Boulders over 123 images and 1252 on SANParks over
51, matching the published raw row exactly. The union of both prompts gives 1847
and 2637, so the table is not the union.

WHAT IT REPORTS
---------------
reconcile   canonical sieve against the published precision, recall and count MAE
compose     per-reason removal counts, plus how many removed masks were real
            penguins, so the sieve's recall cost is visible next to its gain
loo         leave-one-out marginal contribution, because the tests fire in order
            and first-fires-wins flatters whichever test runs first
sweep       a_min, rho_max and area_k against precision, recall and count MAE,
            not against survivor counts, which move without meaning anything

OUTPUT
------
~/bench_any/runs/p5_sieve_composition/sieve_composition.json plus a summary block
appended to summary.md. Touches _done. Nothing polls it.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pycocotools.mask as mu  # noqa: E402

from sanity_sieve import (  # noqa: E402
    sieve_masks, class_agnostic_nms, reject_area, reject_dimension,
    reject_edge_stub, reject_edge_relsize, _bbox_area, A_MIN, RHO_MAX,
    EDGE_BORDER, EDGE_STUB_AREA_K, NMS_IOU, EDGE_RELSIZE_K,
)

R1 = Path.home() / "bench_any" / "runs" / "p5_r1_v2"
RUNS = Path.home() / "bench_any" / "runs" / "obj2_counting"
OUT = Path.home() / "bench_any" / "runs" / "p5_sieve_composition"
PUBLISHED = R1 / "precision_count_raw_vs_sieve.json"

CORPORA = {"boulders-penguin": "boulders", "sanparks-algoa": "sanparks",
           "pw-v2-200": "pw200_zeroshot"}
KEY = {"boulders-penguin": "boulders", "sanparks-algoa": "sanparks"}

# Penguin-Watch v2-200 lives under p5_r1_gold (not p5_r1_v2) with a different
# file-name prefix, so load_corpus needs the explicit path.
R1_GOLD = Path.home() / "bench_any" / "runs" / "p5_r1_gold"
PW_RAW = R1_GOLD / "pw200_zeroshot" / "pw_v2_200_sam3_raw.json"

CANON = dict(a_min=A_MIN, rho_max=RHO_MAX, border=EDGE_BORDER,
             area_k=EDGE_STUB_AREA_K, nms_iou=NMS_IOU)
ASRUN = dict(a_min=64.0, rho_max=1.0, border=EDGE_BORDER, area_k=0.0, nms_iou=0.8)

MATCH_IOU = 0.5


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "sieve_composition.log", "a") as f:
        f.write(msg.rstrip() + "\n")
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _as_rle(seg):
    r = dict(seg)
    if isinstance(r.get("counts"), str):
        r["counts"] = r["counts"].encode("ascii")
    return r


def load_corpus(corpus):
    """Raw single-prompt predictions and ground truth, both keyed by filename.

    Ground-truth masks come back alongside the boxes because the published
    artefact has to be reconciled against both matching conventions before any
    of its cells can be trusted."""
    sub = CORPORA[corpus]
    if corpus == "pw-v2-200":
        raw = json.loads(PW_RAW.read_text())
    else:
        raw = json.loads((R1 / sub / f"{sub}_sam3_raw.json").read_text())
    gt = json.loads((RUNS / f"sam2refine_{corpus}" / "gt_coco.json").read_text())
    hw = {im["file_name"]: (im["height"], im["width"]) for im in gt["images"]}
    fn_of = {im["id"]: im["file_name"] for im in gt["images"]}
    gt_boxes, gt_rles = {}, {}
    for a in gt["annotations"]:
        fn = fn_of[a["image_id"]]
        x, y, w, h = a["bbox"]
        gt_boxes.setdefault(fn, []).append((x, y, x + w, y + h))
        gt_rles.setdefault(fn, []).append(_as_rle(a["segmentation"]))
    return raw, hw, gt_boxes, gt_rles


def decode_image(anns, hw, with_masks):
    """Tight boxes, areas and scores for one image.

    Nothing is decoded densely. `mu.toBbox` gives the same tight extent as
    `sanity_sieve._mask_bbox` and `mu.area` gives the same pixel count as
    `_mask_area`, both straight off the run-length encoding. An earlier version
    called `mu.decode` on every mask and had to be killed after two minutes on
    4K frames; the sieve's three tests only ever need an area and a box extent,
    so the dense array was pure waste.

    Two box conventions leave this function and they must not be confused.
    `sbox` is inclusive, x2 = xs.max(), because that is what `sanity_sieve`
    expects and, since masks are passed as None, whatever is handed over becomes
    authoritative for the dimension and edge-stub tests. `mbox` is half-open,
    x2 = x + w, to match the ground-truth boxes read out of COCO and so keep the
    intersection-over-union used for matching self-consistent.

    with_masks=False falls back to the stored detector box, which is what the
    sweep script used. Running both says whether the published table was built
    on mask area or on box area."""
    sboxes, mboxes, areas, scores, rles = [], [], [], [], []
    for a in anns:
        bx1, by1, bx2, by2 = a["box"]
        sbox = (int(round(bx1)), int(round(by1)), int(round(bx2)), int(round(by2)))
        mbox = (float(bx1), float(by1), float(bx2), float(by2))
        rle = _as_rle(a["rle"])
        if with_masks:
            x, y, w, h = (float(v) for v in mu.toBbox(rle))
            if w > 0 and h > 0:
                sbox = (int(x), int(y), int(x + w) - 1, int(y + h) - 1)
                mbox = (x, y, x + w, y + h)
            areas.append(int(mu.area(rle)))
        else:
            areas.append(_bbox_area(*sbox))
        sboxes.append(sbox)
        mboxes.append(mbox)
        rles.append(rle)
        scores.append(float(a["score"]))
    return sboxes, mboxes, areas, scores, rles


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def iou_matrix(rles, mboxes, gt_rles, gt_boxes, mode):
    """Detections by ground truth overlap for one image, computed once.

    Every sieve configuration reuses this, so the sweep costs one matrix per
    image rather than one per configuration. `mode` selects mask overlap or box
    overlap; the published artefact is reconciled against both because P5 is a
    segmentation paper but its predecessor scripts matched on boxes, and only
    the numbers can say which convention produced the table."""
    if not rles or not gt_rles:
        return np.zeros((len(rles), len(gt_rles)), dtype=np.float64)
    crowd = [0] * len(gt_rles)
    if mode == "mask":
        m = mu.iou(rles, gt_rles, crowd)
    else:
        to_xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in mboxes]
        gt_xywh = [[g[0], g[1], g[2] - g[0], g[3] - g[1]] for g in gt_boxes]
        m = mu.iou(to_xywh, gt_xywh, crowd)
    return np.asarray(m, dtype=np.float64).reshape(len(rles), len(gt_rles))


def greedy_match(idxs, scores, M, thr=MATCH_IOU):
    """Highest score first, each prediction takes the best free ground truth.
    `idxs` indexes rows of `M`; returns the subset of `idxs` that matched."""
    order = sorted(idxs, key=lambda i: -scores[i])
    taken = set()
    matched = set()
    for i in order:
        best, best_j = thr, -1
        for j in range(M.shape[1]):
            if j in taken:
                continue
            if M[i, j] >= best:
                best, best_j = M[i, j], j
        if best_j >= 0:
            taken.add(best_j)
            matched.add(i)
    return matched


def _rescore(d, thr):
    """The raw row at one matching threshold, used only by the provenance probe."""
    tp = fp = fn_ = 0
    for _a, sboxes, scores, n_gt, M in d.values():
        matched = greedy_match(range(len(sboxes)), scores, M, thr)
        tp += len(matched)
        fp += len(sboxes) - len(matched)
        fn_ += n_gt - len(matched)
    return {"TP": tp, "FP": fp, "FN": fn_,
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn_) if (tp + fn_) else 0.0}


def score_corpus(per_image):
    """per_image: {fn: (kept_idxs, scores, n_gt, M)} -> precision, recall, MAE."""
    tp = fp = fn_ = 0
    abs_err = []
    for kept, scores, n_gt, M in per_image.values():
        matched = greedy_match(kept, scores, M)
        tp += len(matched)
        fp += len(kept) - len(matched)
        fn_ += n_gt - len(matched)
        abs_err.append(abs(len(kept) - n_gt))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn_) if (tp + fn_) else 0.0
    return {"precision": prec, "recall": rec, "TP": tp, "FP": fp, "FN": fn_,
            "count_mae": float(np.mean(abs_err)) if abs_err else 0.0,
            "n_images": len(per_image)}


# ---------------------------------------------------------------------------
# the three passes
# ---------------------------------------------------------------------------
def run_config(data, cfg, apply_nms=True):
    """One sieve configuration over a whole corpus. Returns the score block, the
    aggregate rejection counters, and the recall cost."""
    per_image = {}
    agg = {"n_in": 0, "area": 0, "dim": 0, "edge": 0, "nms": 0, "out": 0}
    removed_true = 0
    for fn, (areas, sboxes, scores, n_gt, M) in data.items():
        h, w = HW[fn]
        kept, st = sieve_masks([None] * len(sboxes), (w, h), scores=scores,
                               bboxes=sboxes, areas=areas,
                               apply_nms=apply_nms, **cfg)
        agg["n_in"] += st.n_in
        agg["area"] += st.n_reject_area
        agg["dim"] += st.n_reject_dim
        agg["edge"] += st.n_reject_edge
        agg["nms"] += st.n_reject_nms
        agg["out"] += st.n_out
        # A removed mask that matched a ground-truth penguin is a recall cost.
        pre = greedy_match(range(len(sboxes)), scores, M)
        removed_true += len(pre - set(kept))
        per_image[fn] = (list(kept), scores, n_gt, M)
    block = score_corpus(per_image)
    block["rejections"] = agg
    block["removed_true_positives"] = removed_true
    return block


# ---------------------------------------------------------------------------
# Edge-rule bake-off
# ---------------------------------------------------------------------------
# The sieve's edge test has no precedent in the literature, which is a reason to
# measure it against the conventions that do, not a reason to retreat to them.
# Four arms, each swapping only the edge test. Everything else, including the
# order the tests fire in and the class-agnostic suppression, is the canonical
# code imported from `sanity_sieve`.
#
#   none     no edge test, so the other arms are read against a real baseline
#   area     the paper's rule, reject a border-touching mask under area_k * a_min
#   quadrat  Krebs' two-sides-in, two-sides-out quadrat convention. Right and
#            bottom are the "out" sides. Unbiased in expectation and the standard
#            answer to boundary double-counting in field census work
#   relsize  the strip-transect half-in rule in adaptive form. Reject a
#            border-touching mask under half the median area of that image's
#            interior detections. Scale-free, so unlike a fixed 2592 px it does
#            not have to be justified per corpus
#   combo    the separability probe's answer. Relative size nominates a mask as
#            before, but the removal only stands if the mask is also pressed
#            flush against the frame and is small in absolute terms. Three weak
#            signals in conjunction, not a fourth constant
#   combo_sf the same conjunction with the absolute-area term dropped and the
#            contact threshold tightened instead. Both surviving terms are
#            ratios, so this arm is the only one in the bake-off that introduces
#            no pixel constant at all. Since the sieve's whole defect is that its
#            constants are absolute, an arm that fixes the edge test by adding
#            another one would be arguing against itself
#   fiftyone the authors' own relative-area border rule, the one that actually ran, from
#            `sam3_refine_workflow` in `fiftyone_yolo_active_learning.py` around
#            line 788. It is not `EDGE_MIN_AREA`: it rejects a border-touching
#            detection whose BOX area falls under 0.6 of the median box area in
#            that image, and the median deliberately includes the border-touching
#            objects themselves. Reconstructed here without the ground-truth boxes
#            that pipeline also folds into the median, because a sieve running
#            before annotation has none
ARMS = ("none", "area", "quadrat", "relsize", "combo", "combo_sf", "fiftyone")

# `sam3_refine_workflow`: threshold 0.6, and a border margin expressed as a
# fraction of the image rather than the sieve's 4 px.
FIFTYONE_K = 0.6
FIFTYONE_EPS = 0.005

# Flush contact per detection, keyed (filename, index), filled once in `main`
# because it is the only quantity in this script that needs a dense mask and so
# must never be recomputed inside an arm.
CONTACT = {}
COMBO_CONTACT_MIN = 0.85
COMBO_AREA_K = 185.0  # multiples of a_min; see the sweep, and see its caveat
# 185 * a_min is 59940 pixels, which is the very thing this audit indicts the rest
# of the sieve for: a constant calibrated at one resolution. Restating it as a
# second, tighter size ratio removes the last pixel count from the rule. Both
# constants below come off the joint sweep, where the k2 = 0.1 column holds one
# real penguin removed for every contact threshold from 0.75 to 0.95.
COMBO_SF_CONTACT_MIN = 0.80
COMBO_SF_RELSIZE_K = 0.10


def contact_ratio(rle, bbox, img_wh, border=EDGE_BORDER):
    """Longest flush run against the frame, over the mask's extent along it.

    This is the one place a mask is decoded densely, which is affordable only
    because it runs once per border-touching detection rather than once per
    configuration. A band `border` deep is collapsed rather than reading the
    single outermost line, because a box qualifies as border-touching at four
    pixels out and its mask need not reach the literal edge.

    The statistic reads backwards from the intuition it was built on. A mask that
    presses hard against the frame is more often an artefact running along the
    edge than a penguin half outside it, so a *low* contact is what argues for
    keeping a nominated mask."""
    sides = []
    x1, y1, x2, y2 = bbox
    w, h = img_wh
    b = int(border)
    if x1 < b:
        sides.append("l")
    if y1 < b:
        sides.append("t")
    if x2 > (w - 1 - b):
        sides.append("r")
    if y2 > (h - 1 - b):
        sides.append("bt")
    if not sides:
        return 0.0
    m = mu.decode(rle)
    bw, bh = max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)
    best = 0.0
    for s in sides:
        if s == "l":
            line, denom = m[:, :b].any(axis=1), bh
        elif s == "r":
            line, denom = m[:, -b:].any(axis=1), bh
        elif s == "t":
            line, denom = m[:b, :].any(axis=0), bw
        else:
            line, denom = m[-b:, :].any(axis=0), bw
        best = max(best, float(line.sum()) / denom)
    return best


def _touches_border(bbox, img_wh, border):
    x1, y1, x2, y2 = bbox
    w, h = img_wh
    return x1 < border or y1 < border or x2 > (w - 1 - border) or y2 > (h - 1 - border)


def _edge_rejects(arm, bbox, img_wh, area, median_interior, cfg, key=None,
                  median_all_box=0.0):
    if arm == "none":
        return False
    if arm == "area":
        return reject_edge_stub(bbox, img_wh, area, cfg["a_min"],
                                cfg["border"], cfg["area_k"])
    if arm == "fiftyone":
        # Carries its own border margin and its own notion of size, so it cannot
        # reuse the gate below. Both differences are the rule as written, not
        # choices made here.
        x1, y1, x2, y2 = bbox
        w, h = img_wh
        eps = cfg.get("fiftyone_eps", FIFTYONE_EPS)
        touches = (x1 <= eps * w or y1 <= eps * h
                   or x2 >= (1 - eps) * w or y2 >= (1 - eps) * h)
        if not touches or median_all_box <= 0:
            return False
        box_area = max(1, x2 - x1 + 1) * max(1, y2 - y1 + 1)
        return box_area < cfg.get("fiftyone_k", FIFTYONE_K) * median_all_box
    if not _touches_border(bbox, img_wh, cfg["border"]):
        return False
    if arm in ("combo", "combo_sf"):
        if not reject_edge_relsize(bbox, img_wh, area, median_interior,
                                   cfg["border"], cfg.get("relsize_k", EDGE_RELSIZE_K)):
            return False
        cmin = cfg.get("combo_contact_min",
                       COMBO_SF_CONTACT_MIN if arm == "combo_sf" else COMBO_CONTACT_MIN)
        if CONTACT.get(key, 0.0) < cmin:
            return False
        if arm == "combo_sf":
            # The absolute-area guard restated as a ratio. Nothing here is a pixel
            # count, so the rule reads the same at any resolution, which the arm it
            # replaces cannot claim.
            return reject_edge_relsize(
                bbox, img_wh, area, median_interior, cfg["border"],
                cfg.get("combo_sf_relsize_k", COMBO_SF_RELSIZE_K))
        return area < cfg.get("combo_area_k", COMBO_AREA_K) * cfg["a_min"]
    if arm == "quadrat":
        x1, y1, x2, y2 = bbox
        w, h = img_wh
        return x2 > (w - 1 - cfg["border"]) or y2 > (h - 1 - cfg["border"])
    if arm == "relsize":
        return reject_edge_relsize(bbox, img_wh, area, median_interior,
                                   cfg["border"], cfg.get("relsize_k", EDGE_RELSIZE_K))
    raise ValueError(arm)


def run_arm(data, arm, cfg):
    """The canonical sieve with one edge test swapped out.

    This walks the tests itself rather than calling `sieve_masks`, because the
    swap has to happen in the right position in a first-fires-wins order. Every
    decision is still made by the imported `sanity_sieve` predicates, and
    `main` asserts that arm "area" reproduces `sieve_masks` exactly, which is
    what stops this drifting into a second implementation of the sieve."""
    per_image = {}
    agg = {"n_in": 0, "area": 0, "dim": 0, "edge": 0, "nms": 0, "out": 0}
    removed_true = 0
    edge_removed_true = 0
    for fn, (areas, sboxes, scores, n_gt, M) in data.items():
        h, w = HW[fn]
        img_wh = (w, h)
        interior = [areas[i] for i in range(len(sboxes))
                    if not _touches_border(sboxes[i], img_wh, cfg["border"])]
        median_interior = float(np.median(interior)) if interior else 0.0
        # The FiftyOne rule's denominator: box areas over every detection in the
        # image, border-touching ones included, which is the point of its comment.
        all_box = [max(1, b[2] - b[0] + 1) * max(1, b[3] - b[1] + 1) for b in sboxes]
        median_all_box = float(np.median(all_box)) if all_box else 0.0

        survivors = []
        edge_rejected = []
        agg["n_in"] += len(sboxes)
        for i in range(len(sboxes)):
            if reject_area(areas[i], cfg["a_min"]):
                agg["area"] += 1
                continue
            if reject_dimension(sboxes[i], img_wh, cfg["rho_max"]):
                agg["dim"] += 1
                continue
            if _edge_rejects(arm, sboxes[i], img_wh, areas[i], median_interior,
                             cfg, key=(fn, i), median_all_box=median_all_box):
                agg["edge"] += 1
                edge_rejected.append(i)
                continue
            survivors.append(i)

        if len(survivors) > 1:
            keep = class_agnostic_nms([sboxes[i] for i in survivors],
                                      [scores[i] for i in survivors], cfg["nms_iou"])
            agg["nms"] += len(survivors) - len(keep)
            survivors = [survivors[k] for k in keep]
        agg["out"] += len(survivors)

        pre = greedy_match(range(len(sboxes)), scores, M)
        removed_true += len(pre - set(survivors))
        # How many of the edge test's own removals were real penguins. A reviewer
        # will divide this by the edge count, so it is measured, not left implicit.
        edge_removed_true += len(pre & set(edge_rejected))
        per_image[fn] = (survivors, scores, n_gt, M)
    block = score_corpus(per_image)
    block["rejections"] = agg
    block["removed_true_positives"] = removed_true
    block["edge_removed_true_positives"] = edge_removed_true
    block["edge_removal_precision"] = (
        1.0 - edge_removed_true / agg["edge"] if agg["edge"] else 0.0)
    return block


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    with_masks = os.environ.get("SIEVE_COMP_BBOX_ONLY", "0") != "1"
    log(f"=== sieve composition, mask areas={'on' if with_masks else 'off'} ===")

    published = json.loads(PUBLISHED.read_text())
    results = {"mask_areas": with_masks, "corpora": {}}

    for corpus, sub in CORPORA.items():
        raw, hw, gt_boxes, gt_rles = load_corpus(corpus)
        global HW
        HW = hw
        # Keyed on filename, and the two corpora are free to share one. Clearing
        # here is what stops a Boulders mask lending its contact to a SANParks one.
        CONTACT.clear()
        prepped = {}
        for fn, anns in raw.items():
            sboxes, mboxes, areas, scores, rles = decode_image(anns, hw[fn], with_masks)
            g_r, g_b = gt_rles.get(fn, []), gt_boxes.get(fn, [])
            h_, w_ = hw[fn]
            for i, bb in enumerate(sboxes):
                if _touches_border(bb, (w_, h_), EDGE_BORDER):
                    CONTACT[(fn, i)] = contact_ratio(rles[i], bb, (w_, h_))
            prepped[fn] = {"areas": areas, "sboxes": sboxes, "scores": scores,
                           "n_gt": len(g_r),
                           "M": {m: iou_matrix(rles, mboxes, g_r, g_b, m)
                                 for m in ("mask", "box")}}

        def data_for(mode):
            return {fn: (p["areas"], p["sboxes"], p["scores"], p["n_gt"], p["M"][mode])
                    for fn, p in prepped.items()}

        n_raw = sum(len(p["sboxes"]) for p in prepped.values())
        n_gold = sum(p["n_gt"] for p in prepped.values())
        log(f"[{corpus}] {len(prepped)} images, {n_raw} raw detections, {n_gold} gold")

        pub = published[KEY[corpus]]
        c = {"n_raw": n_raw, "n_gold": n_gold, "published": pub}

        # Reconciliation comes first and everything below depends on it. The
        # published raw row is scored under both matching conventions, because a
        # count MAE that already agrees to four decimals says the detections are
        # right and only the matcher can be responsible for a precision gap.
        gate = {}
        for mode in ("mask", "box"):
            d = data_for(mode)
            rb = score_corpus({fn: (list(range(len(v[1]))), v[2], v[3], v[4])
                               for fn, v in d.items()})
            cb = run_config(d, CANON)
            gate[mode] = {"raw": rb, "canonical": cb}
            log(f"  [{mode:4s}] raw   P={rb['precision']:.4f} R={rb['recall']:.4f} "
                f"MAE={rb['count_mae']:.4f}  (published "
                f"{pub['raw']['precision']:.4f}/{pub['raw']['recall']:.4f}/"
                f"{pub['raw']['count_mae']:.4f})")
        c["gate"] = gate

        # Neither convention reproduces the published raw row at the usual 0.5,
        # so the threshold itself is a free variable and gets swept. Boulders
        # lands exactly on mask overlap at 0.55, which is NMS_IOU, so the
        # artefact appears to have reused the suppression constant as its
        # matching threshold. That is recorded rather than adopted; the
        # regenerated table matches at the conventional 0.5.
        prov = []
        for pmode in ("mask", "box"):
            for thr in (0.3, 0.4, 0.5, 0.55, 0.6, 0.7):
                b = _rescore(data_for(pmode), thr)
                exact = (b["TP"] == pub["raw"]["TP"] and b["FP"] == pub["raw"]["FP"]
                         and b["FN"] == pub["raw"]["FN"])
                prov.append({"mode": pmode, "thr": thr, "TP": b["TP"], "FP": b["FP"],
                             "FN": b["FN"], "precision": b["precision"],
                             "recall": b["recall"], "exact": exact})
                if exact:
                    log(f"  PROVENANCE exact match: {pmode} overlap at IoU {thr}")
        c["provenance_probe"] = prov
        if not any(p["exact"] for p in prov):
            log("  PROVENANCE no exact match under any swept convention")

        agree = [m for m in ("mask", "box")
                 if abs(gate[m]["raw"]["precision"] - pub["raw"]["precision"]) < 1e-4
                 and abs(gate[m]["raw"]["recall"] - pub["raw"]["recall"]) < 1e-4]
        mode = "mask"
        c["match_mode"] = mode
        c["raw_reconciles"] = bool(agree)
        log(f"  raw row reconciles at IoU {MATCH_IOU} under: {agree or 'NEITHER'}"
            f" -> scoring with {mode} at {MATCH_IOU}")

        data = data_for(mode)
        c["raw"] = gate[mode]["raw"]

        # reconciliation: canonical constants, then the as-run sweep constants
        for name, cfg in (("canonical", CANON), ("geometric_asrun", ASRUN)):
            b = gate[mode]["canonical"] if name == "canonical" else run_config(data, cfg)
            c[name] = b
            d = b["rejections"]
            log(f"  {name:16s} kept={d['out']:5d} P={b['precision']:.4f} "
                f"R={b['recall']:.4f} MAE={b['count_mae']:.4f} | "
                f"area={d['area']} dim={d['dim']} edge={d['edge']} nms={d['nms']} "
                f"| true penguins removed={b['removed_true_positives']}")
        pk = pub["sieve"]["TP"] + pub["sieve"]["FP"]
        log(f"  published sieve kept={pk} P={pub['sieve']['precision']:.4f} "
            f"R={pub['sieve']['recall']:.4f} MAE={pub['sieve']['count_mae']:.4f}")
        c["sieve_reconciles"] = (c["canonical"]["rejections"]["out"] == pk and
                                 abs(c["canonical"]["precision"] - pub["sieve"]["precision"]) < 1e-4)
        log(f"  sieve row reconciles: {c['sieve_reconciles']}")

        # leave one out, so ordering does not get credited to the first test
        loo = {}
        for name, override in (("no_area", dict(a_min=0.0)),
                               ("no_dim", dict(rho_max=1e9)),
                               ("no_edge", dict(area_k=0.0)),
                               ("no_nms", None)):
            cfg = dict(CANON)
            if override is None:
                b = run_config(data, cfg, apply_nms=False)
            else:
                cfg.update(override)
                b = run_config(data, cfg)
            loo[name] = b
            log(f"  loo {name:10s} kept={b['rejections']['out']:5d} "
                f"P={b['precision']:.4f} R={b['recall']:.4f} MAE={b['count_mae']:.4f}")
        c["leave_one_out"] = loo

        # 1.4a: the edge rule against the conventions it has no precedent in.
        bake = {}
        for arm in ARMS:
            b = run_arm(data, arm, CANON)
            bake[arm] = b
            log(f"  arm {arm:8s} kept={b['rejections']['out']:5d} "
                f"P={b['precision']:.4f} R={b['recall']:.4f} "
                f"MAE={b['count_mae']:.4f} edge_removed={b['rejections']['edge']} "
                f"| true penguins removed={b['removed_true_positives']}")
        # The bake-off walks the tests itself so it can swap one out mid-order.
        # If the "area" arm did not land exactly where `sieve_masks` lands, the
        # walk has drifted from the sieve and every arm beside it is worthless.
        same = (bake["area"]["rejections"] == c["canonical"]["rejections"]
                and abs(bake["area"]["precision"] - c["canonical"]["precision"]) < 1e-12)
        log(f"  bake-off fidelity check vs sieve_masks: {'PASS' if same else 'FAIL'}")
        if not same:
            raise SystemExit("bake-off diverged from sanity_sieve.sieve_masks")
        c["bake_off"] = bake

        # The relative-size rule now lives in `sanity_sieve` as a selectable form
        # of test 3, so the same arm has to come back identical when the sieve
        # runs it end to end. This is what stops the bake-off number and the
        # shipped rule drifting apart.
        rel = run_config(data, dict(CANON, edge_rule="relsize"))
        rel_same = (rel["rejections"] == bake["relsize"]["rejections"]
                    and abs(rel["precision"] - bake["relsize"]["precision"]) < 1e-12)
        log(f"  relsize round-trip vs sieve_masks(edge_rule=relsize): "
            f"{'PASS' if rel_same else 'FAIL'}")
        if not rel_same:
            raise SystemExit("relsize arm diverged from sanity_sieve.sieve_masks")
        c["relsize_via_sieve_masks"] = rel

        # 1.4b: does the reported number move when the threshold moves?
        sweep = []
        for a_min in (64.0, 128.0, 256.0, 324.0, 512.0, 784.0):
            cfg = dict(CANON, a_min=a_min)
            b = run_config(data, cfg)
            sweep.append({"knob": "a_min", "value": a_min, **{k: b[k] for k in
                          ("precision", "recall", "count_mae")},
                          "kept": b["rejections"]["out"]})
        for rho in (0.8, 0.9, 0.95, 1.0):
            cfg = dict(CANON, rho_max=rho)
            b = run_config(data, cfg)
            sweep.append({"knob": "rho_max", "value": rho, **{k: b[k] for k in
                          ("precision", "recall", "count_mae")},
                          "kept": b["rejections"]["out"]})
        for k in (0.0, 4.0, 8.0, 16.0):
            cfg = dict(CANON, area_k=k)
            b = run_config(data, cfg)
            sweep.append({"knob": "area_k", "value": k, **{kk: b[kk] for kk in
                          ("precision", "recall", "count_mae")},
                          "kept": b["rejections"]["out"]})
        # The relative-size rule carries its own free constant, and it was never
        # swept. If the fraction of real penguins among its edge removals is flat
        # across k then the rule removes small penguins by construction, which the
        # paper has to say out loud. If it improves as k falls, the constant was
        # simply untuned.
        rel_sweep = []
        # The low end of this grid is also the contact term's ablation. The
        # scale-free conjunction is relsize below 0.1 and a contact floor, so
        # relsize alone at 0.1 is that rule with contact taken out.
        for k in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75):
            b = run_arm(data, "relsize", dict(CANON, relsize_k=k))
            rel_sweep.append({"knob": "relsize_k", "value": k,
                              **{kk: b[kk] for kk in
                                 ("precision", "recall", "count_mae")},
                              "kept": b["rejections"]["out"],
                              "edge_removed": b["rejections"]["edge"],
                              "edge_removed_true": b["edge_removed_true_positives"],
                              "edge_removal_precision": b["edge_removal_precision"]})
            r = rel_sweep[-1]
            log(f"  relsize_k {k:4.2f} kept={r['kept']:5d} P={r['precision']:.4f} "
                f"R={r['recall']:.4f} MAE={r['count_mae']:.4f} "
                f"edge={r['edge_removed']} of which real={r['edge_removed_true']} "
                f"edge_removal_P={r['edge_removal_precision']:.4f}")
        c["relsize_sweep"] = rel_sweep

        # The conjunction's own constants, swept over both of its ratios. The
        # absolute-area version is swept beside it rather than in place of it,
        # because the only honest way to argue that dropping a pixel constant is
        # free is to show what it was buying.
        sf_sweep = []
        for cmin in (0.75, 0.8, 0.85, 0.9, 0.95):
            for k2 in (0.1, 0.15, 0.2, 0.3, 0.5):
                b = run_arm(data, "combo_sf", dict(CANON, combo_contact_min=cmin,
                                                   combo_sf_relsize_k=k2))
                sf_sweep.append({"contact_min": cmin, "relsize_k2": k2,
                                 **{kk: b[kk] for kk in
                                    ("precision", "recall", "count_mae")},
                                 "kept": b["rejections"]["out"],
                                 "edge_removed": b["rejections"]["edge"],
                                 "edge_removed_true": b["edge_removed_true_positives"],
                                 "edge_removal_precision": b["edge_removal_precision"]})
                r = sf_sweep[-1]
                log(f"  combo_sf c={cmin:4.2f} k2={k2:4.2f} kept={r['kept']:5d} "
                    f"P={r['precision']:.4f} R={r['recall']:.4f} "
                    f"MAE={r['count_mae']:.4f} edge={r['edge_removed']} "
                    f"real={r['edge_removed_true']} "
                    f"edgeP={r['edge_removal_precision']:.4f}")
        c["combo_sf_sweep"] = sf_sweep

        # the authors' own constant, swept. 0.6 was set by eye a year ago and never
        # tested against an alternative, which is the only reason to sweep it.
        fo_sweep = []
        for k in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            b = run_arm(data, "fiftyone", dict(CANON, fiftyone_k=k))
            fo_sweep.append({"k": k, **{kk: b[kk] for kk in
                                        ("precision", "recall", "count_mae")},
                             "kept": b["rejections"]["out"],
                             "edge_removed": b["rejections"]["edge"],
                             "edge_removed_true": b["edge_removed_true_positives"],
                             "edge_removal_precision": b["edge_removal_precision"]})
            r = fo_sweep[-1]
            log(f"  fiftyone_k {k:4.2f} kept={r['kept']:5d} P={r['precision']:.4f} "
                f"R={r['recall']:.4f} MAE={r['count_mae']:.4f} "
                f"edge={r['edge_removed']} real={r['edge_removed_true']} "
                f"edgeP={r['edge_removal_precision']:.4f}")
        c["fiftyone_sweep"] = fo_sweep

        c["sweep"] = sweep
        log(f"  sweep: {len(sweep)} configurations")
        results["corpora"][corpus] = c

    (OUT / "sieve_composition.json").write_text(json.dumps(results, indent=1))
    write_summary(results)
    (OUT / "_done").touch()
    log(f"=== done in {time.time() - t0:.0f}s ===")


def write_summary(results):
    """One readable block, so the numbers can be checked without opening the JSON."""
    lines = [f"\n## Sieve composition, {time.strftime('%Y-%m-%d %H:%M')}\n"]
    for corpus, c in results["corpora"].items():
        pub, canon, asrun = c["published"], c["canonical"], c["geometric_asrun"]
        lines.append(f"\n### {corpus}, {c['n_raw']} raw detections, {c['n_gold']} gold\n")
        ex = ", ".join("{} at {}".format(p["mode"], p["thr"])
                       for p in c["provenance_probe"] if p["exact"])
        lines.append(f"Scored on {c['match_mode']} overlap at IoU {MATCH_IOU}. "
                     f"Published raw row reproduces exactly under: "
                     f"{ex or 'no swept convention'}. "
                     f"Published sieve row reconciles: {c['sieve_reconciles']}.\n")
        lines.append("| Configuration | Precision | Recall | Count MAE | Kept |")
        lines.append("|---|---|---|---|---|")
        lines.append(f"| published raw | {pub['raw']['precision']:.4f} | "
                     f"{pub['raw']['recall']:.4f} | {pub['raw']['count_mae']:.4f} | "
                     f"{pub['raw']['TP'] + pub['raw']['FP']} |")
        lines.append(f"| this run, raw | {c['raw']['precision']:.4f} | "
                     f"{c['raw']['recall']:.4f} | {c['raw']['count_mae']:.4f} | "
                     f"{c['raw']['TP'] + c['raw']['FP']} |")
        lines.append(f"| published sieve | {pub['sieve']['precision']:.4f} | "
                     f"{pub['sieve']['recall']:.4f} | {pub['sieve']['count_mae']:.4f} | "
                     f"{pub['sieve']['TP'] + pub['sieve']['FP']} |")
        lines.append(f"| canonical sieve | {canon['precision']:.4f} | "
                     f"{canon['recall']:.4f} | {canon['count_mae']:.4f} | "
                     f"{canon['rejections']['out']} |")
        lines.append(f"| geometric as-run | {asrun['precision']:.4f} | "
                     f"{asrun['recall']:.4f} | {asrun['count_mae']:.4f} | "
                     f"{asrun['rejections']['out']} |")
        d = canon["rejections"]
        lines.append(f"\nCanonical removals by reason. Area {d['area']}, "
                     f"dimension {d['dim']}, edge stub {d['edge']}, "
                     f"suppression {d['nms']}. True penguins removed "
                     f"{canon['removed_true_positives']}.\n")
        lines.append("| Left out | Kept | Precision | Recall | Count MAE |")
        lines.append("|---|---|---|---|---|")
        for name, b in c["leave_one_out"].items():
            lines.append(f"| {name} | {b['rejections']['out']} | "
                         f"{b['precision']:.4f} | {b['recall']:.4f} | "
                         f"{b['count_mae']:.4f} |")
        lines.append("\nEdge rule bake-off, all other tests canonical.\n")
        lines.append("| Edge rule | Kept | Removed by edge test | Precision | Recall "
                     "| Count MAE | True penguins removed |")
        lines.append("|---|---|---|---|---|---|---|")
        for arm in ARMS:
            b = c["bake_off"][arm]
            lines.append(f"| {arm} | {b['rejections']['out']} | "
                         f"{b['rejections']['edge']} | {b['precision']:.4f} | "
                         f"{b['recall']:.4f} | {b['count_mae']:.4f} | "
                         f"{b['removed_true_positives']} |")
        lines.append("\n| Knob | Value | Kept | Precision | Recall | Count MAE |")
        lines.append("|---|---|---|---|---|---|")
        for s in c["sweep"]:
            lines.append(f"| {s['knob']} | {s['value']} | {s['kept']} | "
                         f"{s['precision']:.4f} | {s['recall']:.4f} | "
                         f"{s['count_mae']:.4f} |")
    with open(OUT / "summary.md", "a") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

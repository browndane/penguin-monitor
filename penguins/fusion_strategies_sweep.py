#!/usr/bin/env python3
"""Multi-strategy prompt-fusion sweep on P5's REAL metric (tier-mean mAP@50 + 0.5:0.95).

WHY
---
P5's contribution reframed (2026-07-29): not "union wins" but "different fusion
strategies win different tiers." Four strategies, genuinely distinct mechanisms:

  1. Union (OR)        — pool all prompts, NMS keeps higher-conf. Maximises recall.
  2. Consensus (AND)   — keep only detections a peer prompt agrees on (box IoU>tau).
                         Maximises precision. Drop unmatched (pure AND).
  3. Mask-fuse         — on agreed pairs, fuse masks: pixel-union (broader boundary),
                         intersection (tighter), or majority-vote (2-of-3). Hybrid:
                         pass unmatched single-prompt detections through untouched.
  4. Soft-average      — on agreed pairs, average sigmoid probabilities then rethreshold.
                         Targets strict mAP@0.5:0.95 boundary metric. Needs logit raws.

All four run the 54-config sieve sweep under prep_10's exact tier-mean metric
(GT from tiers_manifest RLEs, iscrowd-ignore, areaRng=[[0,1e10]], precision
validity filter, real scores). Single-prompt "a_penguin" is the baseline
(reproduces paper 69.0/38.7 exactly).

Three prompts: a_penguin (single baseline), penguin, african_penguin. All from
the SAME SAM3 generation (dense-image detections kept).

Self-reports §20 to FINDINGS.md + sentinel. CPU-only (soft-average reads saved
logit .npy files, no GPU). tau fixed at 0.5 (the eval IoU, no tau sweep).
"""
from __future__ import annotations
import json, sys, time, copy, tempfile, os
from pathlib import Path
import numpy as np
import pycocotools.mask as mu
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PREP_DIR = Path("/home/vorse/bench_any/datasets/prep")
sys.path.insert(0, str(PREP_DIR))
import prep_10_filebased as P10

R1 = Path.home() / "bench_any" / "runs" / "p5_r1_v2"
RUNS = Path.home() / "bench_any" / "runs" / "obj2_counting"
FINDINGS = RUNS / "FINDINGS.md"
DONE = RUNS / "_fusion_sweep_done"
OUT_TXT = RUNS / "fusion_sweep_results.txt"
# One corpus takes hours, so the two run as separate concurrent processes. Set
# BENCH_FUSION_CORPUS to a corpus key to restrict this process to it; the sentinel
# and result file then carry that key so the two processes never collide.
_ONLY = os.environ.get("BENCH_FUSION_CORPUS", "").strip()
if _ONLY:
    DONE = RUNS / f"_fusion_sweep_done_{_ONLY}"
    OUT_TXT = RUNS / f"fusion_sweep_results_{_ONLY}.txt"

CORPORA = {"boulders-penguin": "Boulders", "sanparks-algoa": "SANParks"}
SUBDIR = {"boulders-penguin": "boulders", "sanparks-algoa": "sanparks"}
PROMPTS = ["a_penguin", "penguin", "african_penguin"]
SINGLE = "a_penguin"
TAU = 0.5  # box-IoU agreement threshold (fixed)

CHOSEN = {"min_area": 64, "nms_iou": 0.8, "max_dim_frac": 1.0, "score_floor": 0.25}
MIN_AREAS = [32, 64, 128]
NMS_IOUS = [0.7, 0.8, 0.9]
MAX_DIM_FRACS = [0.9, 0.95, 1.0]
SCORE_FLOORS = [0.25, 0.28]


def _raw_path(corpus, prompt):
    sub = SUBDIR[corpus]
    tag = "" if prompt == "a_penguin" else "_" + prompt
    return R1 / sub / f"{sub}_sam3_raw{tag}.json"


def _soft_path(corpus, prompt):
    """The _v2 files only. The first soft-logit run prompted SAM3 with the file
    tag 'a_penguin' instead of the text 'a penguin', so its detection set does not
    line up with the raw set and positional fusion against it is meaningless. The
    non-v2 files are still on disk, so pointing anywhere else here silently
    reintroduces that defect."""
    sub = SUBDIR[corpus]
    tag = "" if prompt == "a_penguin" else "_" + prompt
    return R1 / sub / f"{sub}_sam3_soft{tag}_v2.json", R1 / sub / f"{sub}_soft{tag}_v2_npys"


def _raw_per_prompt(corpus):
    """Load {prompt: {filename: [{rle, score, box}, ...]}} for all prompts."""
    out = {}
    for p in PROMPTS:
        f = _raw_path(corpus, p)
        if f.exists():
            out[p] = json.loads(f.read_text())
        else:
            out[p] = {}
    return out


def _iou(b1, b2):
    xa, ya, x2a, y2a = b1
    xb, yb, x2b, y2b = b2
    ix1, iy1 = max(xa, xb), max(ya, yb)
    ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = (x2a - xa) * (y2a - ya)
    a2 = (x2b - xb) * (y2b - yb)
    return inter / (a1 + a2 - inter)


def _nms(items, iou_thr):
    """items: [{rle,score,box,...}]. Greedy NMS by score. Returns keep indices."""
    if not items:
        return []
    order = list(np.argsort([a["score"] for a in items])[::-1])
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        rest = [j for j in order if _iou(items[i]["box"], items[j]["box"]) > iou_thr]
        for j in rest:
            order.remove(j)
    return keep


def _pass_sieve(items, fn_hw, fn, cfg):
    """Apply sieve (min_area, max_dim_frac, score_floor) to a list of anns for one file."""
    h, w = fn_hw[fn]
    kept = []
    for a in items:
        if a["score"] < cfg["score_floor"]:
            continue
        x1, y1, x2, y2 = a["box"]
        bw, bh = x2 - x1, y2 - y1
        area = bw * bh
        if area < cfg["min_area"]:
            continue
        if max(bw, bh) > cfg["max_dim_frac"] * max(w, h):
            continue
        kept.append(a)
    return kept


# ---- fusion strategies ----

def fuse_union(raw_pp, fn, cfg, fn_hw):
    """OR: pool all prompts' detections for this file, sieve, NMS."""
    pooled = []
    for p in PROMPTS:
        pooled.extend(raw_pp.get(p, {}).get(fn, []))
    kept = _pass_sieve(pooled, fn_hw, fn, cfg)
    idx = _nms(kept, cfg["nms_iou"])
    return [kept[i] for i in idx]


def fuse_consensus(raw_pp, fn, cfg, fn_hw):
    """AND: keep only detections a DIFFERENT prompt agrees with (box IoU>tau).
    Drop unmatched (pure consensus). Sieve then NMS within the agreed set."""
    agreed = []
    prompts_present = [p for p in PROMPTS if raw_pp.get(p, {}).get(fn)]
    for pi, pa in enumerate(prompts_present):
        a_list = _pass_sieve(raw_pp[pa].get(fn, []), fn_hw, fn, cfg)
        for a in a_list:
            # does any OTHER prompt have a detection agreeing?
            matched = False
            for pj, pb in enumerate(prompts_present):
                if pj == pi:
                    continue
                b_list = _pass_sieve(raw_pp[pb].get(fn, []), fn_hw, fn, cfg)
                for b in b_list:
                    if _iou(a["box"], b["box"]) > TAU:
                        matched = True
                        break
                if matched:
                    break
            if matched:
                agreed.append(a)
    idx = _nms(agreed, cfg["nms_iou"])
    return [agreed[i] for i in idx]


def fuse_mask(raw_pp, fn, cfg, fn_hw, mode):
    """Hybrid mask-fuse: on agreed pairs, fuse masks by `mode`
    (union/intersection/majority). Pass unmatched single-prompt detections through.
    Returns list of {rle, score, box}."""
    prompts_present = [p for p in PROMPTS if raw_pp.get(p, {}).get(fn)]
    # sieve each prompt's detections for this file
    per_prompt = {p: _pass_sieve(raw_pp.get(p, {}).get(fn, []), fn_hw, fn, cfg) for p in prompts_present}
    # greedy cross-prompt matching by box-IoU>tau, highest-score first
    used = {p: [False] * len(per_prompt[p]) for p in prompts_present}
    groups = []  # each group: list of (prompt, idx, ann) that agree
    all_by_score = []
    for p in prompts_present:
        for k, a in enumerate(per_prompt[p]):
            all_by_score.append((a["score"], p, k))
    all_by_score.sort(key=lambda x: -x[0])
    for _score, p_seed, k_seed in all_by_score:
        if used[p_seed][k_seed]:
            continue
        seed = per_prompt[p_seed][k_seed]
        used[p_seed][k_seed] = True
        group = [(p_seed, k_seed, seed)]
        for p_other in prompts_present:
            if p_other == p_seed:
                continue
            best_j, best_iou = -1, TAU
            for j, b in enumerate(per_prompt[p_other]):
                if used[p_other][j]:
                    continue
                iou = _iou(seed["box"], b["box"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0:
                used[p_other][best_j] = True
                group.append((p_other, best_j, per_prompt[p_other][best_j]))
        groups.append(group)
    # build output: fused masks for multi-member groups, pass-through for singles
    out = []
    for group in groups:
        anns = [g[2] for g in group]
        if len(anns) == 1:
            out.append(anns[0])
            continue
        # fuse masks
        masks = [mu.decode(a["rle"]) for a in anns]
        if mode == "union":
            fused = np.any(np.stack(masks), axis=0)
        elif mode == "intersection":
            fused = np.all(np.stack(masks), axis=0)
        elif mode == "majority":
            fused = (np.sum(np.stack(masks), axis=0) >= 2)
        else:
            raise ValueError(mode)
        rle = mu.encode(np.asfortranarray(fused.astype(np.uint8)))
        if isinstance(rle["counts"], bytes):
            rle["counts"] = rle["counts"].decode("utf-8")
        box = anns[0]["box"]  # keep seed's box
        score = max(a["score"] for a in anns)
        out.append({"rle": rle, "score": score, "box": box})
    idx = _nms(out, cfg["nms_iou"])
    return [out[i] for i in idx]


def fuse_soft(raw_pp, fn, cfg, fn_hw, soft_data):
    """Soft-average: on agreed pairs, average sigmoid probabilities, rethreshold at 0.5.
    Pass unmatched through. soft_data: {prompt: {fn: [{prob_npy, score, box}]}} + npy_dirs."""
    if not soft_data:
        return None  # logit raws not ready
    prompts_present, per_prompt = [], {}
    for p in PROMPTS:
        anns = raw_pp.get(p, {}).get(fn)
        if not anns or not soft_data.get("dirs", {}).get(p):
            continue
        sd = soft_data["data"].get(p, {}).get(fn, [])
        # The sieve drops arbitrary indices, so slicing a prefix off the soft list
        # pairs maps with the wrong detections. Carry each map on its own ann
        # through the sieve instead. A count disagreement means the raw and soft
        # sets came from different runs, and no index mapping between them holds,
        # so drop the prompt rather than fuse across two prediction sets.
        if len(sd) != len(anns):
            continue
        prompts_present.append(p)
        per_prompt[p] = _pass_sieve([dict(a, _soft=s) for a, s in zip(anns, sd)], fn_hw, fn, cfg)
    # greedy matching as in fuse_mask
    used = {p: [False] * len(per_prompt[p]) for p in prompts_present}
    groups = []
    all_by_score = []
    for p in prompts_present:
        for k, a in enumerate(per_prompt[p]):
            all_by_score.append((a["score"], p, k))
    all_by_score.sort(key=lambda x: -x[0])
    for _score, p_seed, k_seed in all_by_score:
        if used[p_seed][k_seed]:
            continue
        seed = per_prompt[p_seed][k_seed]
        used[p_seed][k_seed] = True
        group = [(p_seed, k_seed)]
        for p_other in prompts_present:
            if p_other == p_seed:
                continue
            best_j, best_iou = -1, TAU
            for j, b in enumerate(per_prompt[p_other]):
                if used[p_other][j]:
                    continue
                iou = _iou(seed["box"], b["box"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0:
                used[p_other][best_j] = True
                group.append((p_other, best_j))
        groups.append(group)
    out = []
    npy_dirs = soft_data["dirs"]
    def _plain(a):
        return {k: v for k, v in a.items() if k != "_soft"}
    for group in groups:
        if len(group) == 1:
            p, k = group[0]
            out.append(_plain(per_prompt[p][k]))
            continue
        # average prob maps (stored at native 288x288; upsample to full-res)
        probs = []
        for p, k in group:
            sd = per_prompt[p][k]["_soft"]
            npy_path = npy_dirs[p] / sd["prob_npy"]
            if npy_path.exists():
                pr = np.load(npy_path).astype(np.float32)
                # reference full-res size from the binary RLE of this instance
                bin_rle = per_prompt[p][k]["rle"]
                th, tw = bin_rle["size"]
                if (pr.shape[0], pr.shape[1]) != (th, tw):
                    import cv2
                    pr = cv2.resize(pr, (tw, th), interpolation=cv2.INTER_LINEAR)
                probs.append(pr)
        if not probs:
            p, k = group[0]
            out.append(_plain(per_prompt[p][k]))
            continue
        avg = np.mean(np.stack(probs), axis=0)
        fused = avg >= 0.5
        rle = mu.encode(np.asfortranarray(fused.astype(np.uint8)))
        if isinstance(rle["counts"], bytes):
            rle["counts"] = rle["counts"].decode("utf-8")
        p0, k0 = group[0]
        box = per_prompt[p0][k0]["box"]
        score = max(per_prompt[p][k]["score"] for p, k in group)
        out.append({"rle": rle, "score": score, "box": box})
    idx = _nms(out, cfg["nms_iou"])
    return [out[i] for i in idx]


def _fn_hw(corpus):
    gt = json.loads((RUNS / f"sam2refine_{corpus}" / "gt_coco.json").read_text())
    return {im["file_name"]: (im["height"], im["width"]) for im in gt["images"]}


def _grid():
    for ma in MIN_AREAS:
        for nms in NMS_IOUS:
            for mdf in MAX_DIM_FRACS:
                for sf in SCORE_FLOORS:
                    yield {"min_area": ma, "nms_iou": nms, "max_dim_frac": mdf, "score_floor": sf}


def _eval50(gt_dict, pred_list):
    return P10.eval_tier(gt_dict, pred_list) or 0.0


def _eval5095(gt_dict, pred_list):
    pred_list = copy.deepcopy(pred_list)
    gtf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(gt_dict, gtf); gtf.close()
    try:
        cg = COCO(gtf.name); dt = cg.loadRes(pred_list)
        ev = COCOeval(cg, dt, "segm")
        ev.params.maxDets = [1, 10, 100]
        ev.params.areaRng = [[0, 1e10]]; ev.params.areaRngLbl = ["all"]
        ev.evaluate(); ev.accumulate()
        pr = ev.eval["precision"][:, :, 0, 0, -1]
        v = pr[pr > -1]
        return (float(np.mean(v)) * 100.0) if len(v) else 0.0
    finally:
        os.unlink(gtf.name)


def _to_coco(fused_by_fn, man):
    """{filename: [{rle,score,box}]} -> COCO pred list via build_pred_coco."""
    return P10.build_pred_coco(fused_by_fn, man)


def _load_soft(corpus):
    """Load soft-logit index + npy dirs if they exist; return None otherwise."""
    dirs, data = {}, {}
    have = True
    for p in PROMPTS:
        idx_path, npy_dir = _soft_path(corpus, p)
        if idx_path.exists() and npy_dir.exists():
            data[p] = json.loads(idx_path.read_text())
            dirs[p] = npy_dir
        else:
            have = False
    return {"data": data, "dirs": dirs} if have else None


def main():
    t0 = time.time()
    strategies = [
        ("union", lambda raw, fn, cfg, hw: fuse_union(raw, fn, cfg, hw)),
        ("consensus", lambda raw, fn, cfg, hw: fuse_consensus(raw, fn, cfg, hw)),
        ("mask_union", lambda raw, fn, cfg, hw: fuse_mask(raw, fn, cfg, hw, "union")),
        ("mask_intersection", lambda raw, fn, cfg, hw: fuse_mask(raw, fn, cfg, hw, "intersection")),
        ("mask_majority", lambda raw, fn, cfg, hw: fuse_mask(raw, fn, cfg, hw, "majority")),
    ]
    todo = {k: v for k, v in CORPORA.items() if not _ONLY or k == _ONLY}
    if not todo:
        raise SystemExit(f"BENCH_FUSION_CORPUS={_ONLY!r} matches no corpus in {list(CORPORA)}")
    soft_avail = all(_load_soft(c) is not None for c in todo)
    if soft_avail:
        strategies.append(("soft_average", None))  # handled specially
    lines = [f"\n## 20. Multi-strategy prompt-fusion sweep, corrected inputs"
             f"{' — ' + _ONLY if _ONLY else ''} (tier-mean mAP, 2026-07-29)\n\n",
             "Supersedes the 16:55 run, which was killed at 4 of 10 cells. That run "
             "consumed an african_penguin raw set built by prompting SAM3 with the "
             "file tag 'african_penguin' rather than the text 'african penguin', "
             "which cost 36 detections on Boulders and 148 on SANParks. This run "
             "uses the corrected raw set and the aligned _v2 soft logits.\n\n",
             f"Strategies: {', '.join(s[0] for s in strategies)}. tau={TAU} (box-IoU). ",
             f"Soft-average {'available' if soft_avail else 'PENDING (logit run not done)'}. ",
             "Single 'a_penguin' is baseline (reproduces paper 69.0/38.7). "
             "Hybrid: unmatched single-prompt detections pass through (except pure consensus).\n\n"]
    summary = []
    for corpus, label in todo.items():
        man = json.loads((R1 / SUBDIR[corpus] / f"{SUBDIR[corpus]}_tiers.json").read_text())["manifest"]
        fn_hw = _fn_hw(corpus)
        raw_pp = _raw_per_prompt(corpus)
        soft_data = _load_soft(corpus)
        # baseline single
        single_by_fn = {fn: _pass_sieve(raw_pp.get(SINGLE, {}).get(fn, []), fn_hw, fn, CHOSEN) for fn in man}
        single_by_fn = {fn: [single_by_fn[fn][i] for i in _nms(single_by_fn[fn], CHOSEN["nms_iou"])] for fn in man}
        spred = _to_coco(single_by_fn, man)
        s50 = [_eval50(P10.build_gt_coco(man, t), spred) for t in ("easy", "medium", "hard")]
        s95 = [_eval5095(P10.build_gt_coco(man, t), spred) for t in ("easy", "medium", "hard")]
        lines.append(f"### {label} (baseline single @ chosen: E={s50[0]:.1f} M={s50[1]:.1f} H={s50[2]:.1f} agg50={np.mean(s50):.1f} | agg5095={np.mean(s95):.2f})\n\n")
        lines.append("| strategy | E50 | M50 | H50 | agg50 | E5095 | M5095 | H5095 | agg5095 | best cfg |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for strat_name, strat_fn in strategies:
            best = None
            for cfg in _grid():
                if strat_name == "soft_average":
                    fused = {fn: (fuse_soft(raw_pp, fn, cfg, fn_hw, soft_data) or []) for fn in man}
                else:
                    fused = {fn: strat_fn(raw_pp, fn, cfg, fn_hw) for fn in man}
                pred = _to_coco(fused, man)
                r50 = [_eval50(P10.build_gt_coco(man, t), pred) for t in ("easy", "medium", "hard")]
                agg50 = float(np.mean(r50))
                if best is None or agg50 > best["agg50"]:
                    r95 = [_eval5095(P10.build_gt_coco(man, t), pred) for t in ("easy", "medium", "hard")]
                    best = {"cfg": cfg, "r50": r50, "agg50": agg50, "r95": r95, "agg95": float(np.mean(r95))}
            b = best
            lines.append(f"| {strat_name} | {b['r50'][0]:.1f} | {b['r50'][1]:.1f} | {b['r50'][2]:.1f} | {b['agg50']:.1f} | {b['r95'][0]:.1f} | {b['r95'][1]:.1f} | {b['r95'][2]:.1f} | {b['agg95']:.2f} | ma{b['cfg']['min_area']}n{b['cfg']['nms_iou']}d{b['cfg']['max_dim_frac']}s{b['cfg']['score_floor']} |\n")
            summary.append((label, strat_name, b["agg50"], b["agg95"]))
            print(f"[{label}/{strat_name}] agg50={b['agg50']:.1f} agg5095={b['agg95']:.2f}", file=sys.stderr, flush=True)
    # cross-corpus summary
    lines.append("\n### Summary\n\n| corpus | strategy | agg50 | agg5095 |\n|---|---|---|---|\n")
    for lab, sn, a50, a95 in summary:
        lines.append(f"| {lab} | {sn} | {a50:.1f} | {a95:.2f} |\n")
    text = "".join(lines)
    OUT_TXT.write_text(text)
    with open(FINDINGS, "a") as fh:
        fh.write(text)
    DONE.touch()
    print(f"[fusion-sweep] done {time.time()-t0:.1f}s sentinel={DONE}", file=sys.stderr)


if __name__ == "__main__":
    main()

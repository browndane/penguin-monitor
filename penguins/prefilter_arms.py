# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Score every prefilter arm over one cached set of SAM 3 detections.

Four arms share the detections in `<corpus>_sam3_raw.json` and differ only in which
geometry predicate decides what reaches class-agnostic NMS: the production sanity
sieve, the geometric prefilter that ran in the evaluation pipeline, their
intersection and their union. A fifth row reproduces the geometric prefilter in the
order it actually ran, with suppression ahead of the gates, which is the row the
FINDINGS.md numbers should match.

`passes_sanity` and its constants come out of the production annotator's own source.
That file loads SAM 3 at import, so it cannot be imported, and copying the function
here would leave a second copy to drift; the definition is lifted verbatim instead.

Evaluation is prep_10's, unchanged: mask mAP@0.5 per tier via iscrowd ignore.

Usage:
    python penguins/prefilter_arms.py --corpus sanparks --out arms_sanparks.json
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

PRODUCTION = Path(
    "/home/user/Dropbox/Work/PycharmProjects/cvMLCode/12_Deep/EndtoEndProjects"
    "/Fish/YOLO/point_crowdsource_annotate/point_to_sam3_coco.py"
)
PREP_DIR = Path.home() / "bench_any/datasets/prep"
R1_DIR = Path.home() / "bench_any/runs/p5_r1_v2"
NMS_IOU = 0.55  # production, point_to_sam3_coco.py
ARMS = ["sanity", "geometric", "intersection", "union", "geometric_asrun"]


def load_production_sieve():
    """Return the production `passes_sanity` and its constants, from source."""
    tree = ast.parse(PRODUCTION.read_text())
    wanted = {"MIN_MASK_AREA", "MAX_MASK_DIM_FRACTION", "EDGE_MARGIN", "EDGE_MIN_AREA"}
    body = [
        node for node in tree.body
        if (isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in wanted for t in node.targets))
        or (isinstance(node, ast.FunctionDef) and node.name == "passes_sanity")
    ]
    names = {n.name for n in body if isinstance(n, ast.FunctionDef)}
    if "passes_sanity" not in names:
        raise SystemExit(f"passes_sanity not found in {PRODUCTION}")
    ns = {"np": np}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(PRODUCTION), "exec"), ns)
    return ns


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def iou(a, b):
    """IoU on the detector's own boxes, which are xyxy in the cached SAM 3 output."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(dets, thr=NMS_IOU):
    """Class-agnostic greedy NMS on the detector boxes, highest score first."""
    order = sorted(dets, key=lambda d: -d["score"])
    keep = []
    for d in order:
        if all(iou(d["box"], k["box"]) <= thr for k in keep):
            keep.append(d)
    return keep


def sanity_ok(det, sieve):
    mask = mu.decode(det["rle"])
    h, w = mask.shape[:2]
    return bool(sieve["passes_sanity"](mask, h, w))


def geometric_ok(det, cfg):
    """The geometric prefilter's own gates: score floor, area floor, dimension cap."""
    rle = det["rle"]
    if det["score"] < cfg["score_floor"]:
        return False
    if float(mu.area(rle)) < cfg["min_area"]:
        return False
    _, _, bw, bh = mu.toBbox(rle).tolist()
    h, w = rle["size"]
    return bw / w <= cfg["max_dim_frac"] and bh / h <= cfg["max_dim_frac"]


def apply_arm(dets, arm, sieve, cfg):
    if arm == "geometric_asrun":
        # As it ran: suppression over everything clearing the score floor, gates after.
        scored = [d for d in dets if d["score"] >= cfg["score_floor"]]
        kept = nms(scored, cfg["nms_iou"])
        return [d for d in kept if geometric_ok(d, cfg)]
    if arm == "sanity":
        gate = lambda d: sanity_ok(d, sieve)
    elif arm == "geometric":
        gate = lambda d: geometric_ok(d, cfg)
    elif arm == "intersection":
        gate = lambda d: geometric_ok(d, cfg) and sanity_ok(d, sieve)
    elif arm == "union":
        gate = lambda d: geometric_ok(d, cfg) or sanity_ok(d, sieve)
    else:
        raise ValueError(arm)
    return nms([d for d in dets if gate(d)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=["boulders", "sanparks"])
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = R1_DIR / args.corpus
    raw = json.load(open(d / f"{args.corpus}_sam3_raw.json"))
    tiers = json.load(open(d / f"{args.corpus}_tiers.json"))["manifest"]
    cfg = json.load(open(d / f"{args.corpus}_sam3_pred.json")).get("chosen_config", {})
    print(f"{args.corpus}: {sum(len(v) for v in raw.values())} raw detections "
          f"over {len(raw)} images; as-run config {cfg}", flush=True)

    sieve = load_production_sieve()
    prep10 = load_module(PREP_DIR / "prep_10_filebased.py", "prep_10_filebased")

    results = {}
    for arm in args.arms:
        kept = {f: apply_arm(dets, arm, sieve, cfg) for f, dets in raw.items()}
        n = sum(len(v) for v in kept.values())
        preds = prep10.build_pred_coco(kept, tiers)
        row = {"n_kept": n}
        for tier in prep10.TIERS:
            row[tier] = prep10.eval_tier(prep10.build_gt_coco(tiers, tier), preds)
        scored = [row[t] for t in prep10.TIERS if row[t] is not None]
        row["agg"] = float(np.mean(scored)) if len(scored) == len(prep10.TIERS) else None
        results[arm] = row
        print(f"  {arm:17s} kept {n:5d}  " + "  ".join(
            f"{t} {row[t]:5.1f}" if row[t] is not None else f"{t}  None"
            for t in prep10.TIERS
        ) + (f"  agg {row['agg']:5.1f}" if row["agg"] is not None else ""), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"corpus": args.corpus, "as_run_config": cfg, "nms_iou": NMS_IOU,
               "arms": results}, open(out, "w"), indent=2)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()

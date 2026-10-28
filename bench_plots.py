# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
# Shared, framework-agnostic paper-figure utilities for bench_any.
#
# Every non-Ultralytics tuner appends one epoch row to history.csv via
# ``append_history`` and, at end of run, calls ``plot_all`` to emit a uniform
# set of Ultralytics-style PNG figures that drop straight into a LaTeX paper:
#
#   history.csv                       one row per epoch
#   results.png                       multi-panel train/val loss + metrics
#   BoxPR_curve.png / MaskPR_curve.png precision-recall curves
#   BoxP_curve.png / BoxR_curve.png / BoxF1_curve.png   (and Mask* variants)
#   confusion_matrix.png              raw counts
#   confusion_matrix_normalized.png   row-normalized
#
# The module is intentionally dependency-light: matplotlib + numpy only. It is
# imported lazily by each wrapper so a missing matplotlib never blocks a run.

from __future__ import annotations

import csv
import json
import re
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence


# All paper-figure artifacts go under <run_dir>/graphs/ so the run base dir
# stays clean (results.csv, weights/, predictions.json, etc. live there).
GRAPHS_SUBDIR = "graphs"

# Ultralytics' own results.png saves at very high effective resolution; to make
# our cross-framework figures match (and be print-perfect at full-page width),
# we render at 500 dpi — i.e. ~4x the original 150 dpi default, ~2x the prior
# 250 dpi pass. Yes, files are larger (~few MB each) but reviewers expect this.
_SAVE_DPI = 500


def _savefig(fig, path):
    """Uniform high-DPI PNG save with tight bbox to avoid clipped labels."""
    fig.savefig(path, dpi=_SAVE_DPI, bbox_inches="tight")


def graphs_dir(run_dir: Path) -> Path:
    """Return (and create) the canonical <run_dir>/graphs/ subdir."""
    p = Path(run_dir) / GRAPHS_SUBDIR
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# history.csv writer
# ---------------------------------------------------------------------------

# Canonical column order. Unknown extra keys are appended after these.
_BASE_HISTORY_COLS = [
    "epoch", "elapsed_s", "lr",
    "train_loss", "val_loss",
    "box_mAP50", "box_mAP50_95", "box_mAP75",
    "seg_mAP50", "seg_mAP50_95", "seg_mAP75",
    "mIoU", "fg_mIoU", "pixel_acc", "fg_pixel_acc",
]


def append_history(run_dir: Path, row: dict) -> Path:
    """Append one epoch row to ``<run_dir>/graphs/history.csv``.

    The first call writes the header (canonical order, unknown keys appended).
    Subsequent calls reuse the existing header; brand-new keys appear as empty
    cells in earlier rows rather than reshuffling the file. Safe to call from
    multiple wrappers; no concurrent writers expected (one tuner process).
    """
    gd = graphs_dir(run_dir)
    path = gd / "history.csv"
    existing_cols: list[str] = []
    if path.is_file():
        try:
            with open(path, newline="") as fh:
                existing_cols = next(csv.reader(fh), []) or []
        except Exception:
            existing_cols = []
    if not existing_cols:
        seen = set()
        cols = [c for c in _BASE_HISTORY_COLS if c in row and not (c in seen or seen.add(c))]
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                cols.append(k)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            w.writerow([_fmt(row.get(c)) for c in cols])
        return path
    # Add any new keys to the header in-place by rewriting the file.
    new_keys = [k for k in row.keys() if k not in existing_cols]
    if new_keys:
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))
        header = (rows[0] if rows else []) + new_keys
        body = rows[1:] if rows else []
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for r in body:
                w.writerow(r + [""] * len(new_keys))
            w.writerow([_fmt(row.get(c)) for c in header])
    else:
        with open(path, "a", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([_fmt(row.get(c)) for c in existing_cols])
    return path


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        # Six significant figures is plenty for paper plots and keeps the file
        # human-readable.
        return f"{v:.6g}"
    return str(v)


def _read_history(run_dir: Path) -> tuple[list[str], list[dict]]:
    # Backward-compat: prefer graphs/history.csv but fall back to run_dir root.
    p = Path(run_dir) / GRAPHS_SUBDIR / "history.csv"
    if not p.is_file():
        p = Path(run_dir) / "history.csv"
    if not p.is_file():
        return [], []
    with open(p, newline="") as fh:
        rows = list(csv.DictReader(fh))
    header = list(rows[0].keys()) if rows else []
    return header, rows


def _col_floats(rows: list[dict], key: str) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    for r in rows:
        v = r.get(key, "")
        if v in ("", None):
            out.append(None)
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(None)
    return out


def _drop_none(xs: list, ys: list) -> tuple[list, list]:
    sx, sy = [], []
    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        sx.append(x); sy.append(y)
    return sx, sy


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def _import_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")  # never pop a window from a tuner subprocess.
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        print(f"[bench_plots] matplotlib unavailable: {e}")
        return None


def plot_training_curves(run_dir: Path, title: str = "") -> list[Path]:
    """Emit train/val loss and accuracy/lr curves from history.csv."""
    plt = _import_mpl()
    if plt is None:
        return []
    header, rows = _read_history(run_dir)
    if not rows:
        return []
    out: list[Path] = []
    epochs = _col_floats(rows, "epoch")
    if not any(e is not None for e in epochs):
        return []

    # 1) train + val loss on the same axes (YOLO-style).
    train_l = _col_floats(rows, "train_loss")
    val_l = _col_floats(rows, "val_loss")
    if any(v is not None for v in train_l) or any(v is not None for v in val_l):
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        ex, ey = _drop_none(epochs, train_l)
        if ey: ax.plot(ex, ey, lw=1.5, label="train")
        ex, ey = _drop_none(epochs, val_l)
        if ey: ax.plot(ex, ey, lw=1.5, label="val", linestyle="--")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        if title: ax.set_title(title)
        fig.tight_layout()
        p = graphs_dir(run_dir) / "train_val_loss.png"
        _savefig(fig, p); plt.close(fig); out.append(p)

    # 2) accuracy / score curves. Split into a multi-panel grid (one subplot
    # per metric) so each IoU/threshold variant is clearly visible instead of
    # being squashed onto a single y-axis where mAP50 dwarfs mAP75/mAP50_95.
    score_keys_all = ["box_mAP50", "box_mAP75", "box_mAP50_95",
                      "seg_mAP50", "seg_mAP75", "seg_mAP50_95",
                      "mIoU", "fg_mIoU", "pixel_acc"]
    score_keys = [k for k in score_keys_all
                  if k in header and any(v not in ("", None) for v in (r.get(k) for r in rows))]
    if score_keys:
        n = len(score_keys)
        ncols = 3 if n >= 3 else n
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.0 * ncols, 2.8 * nrows),
                                 squeeze=False)
        axes_flat = [ax for row in axes for ax in row]
        for ax, k in zip(axes_flat, score_keys):
            ys = _col_floats(rows, k)
            ex, ey = _drop_none(epochs, ys)
            if not ey:
                ax.set_visible(False); continue
            # Sparse eval cadence -> scatter so a single point stays visible.
            if len(ey) == 1:
                ax.scatter(ex, ey, s=40, color="C0")
                ax.annotate(f"{ey[0]:.3f}", (ex[0], ey[0]),
                            textcoords="offset points", xytext=(5, 5), fontsize=8)
            else:
                ax.plot(ex, ey, lw=1.5, marker="o", markersize=3, color="C0")
            best = max(ey)
            ax.set_title(f"{k}  (best={best:.3f})", fontsize=10)
            ax.set_xlabel("epoch"); ax.set_ylabel(k)
            ax.set_ylim(bottom=0.0)
            ax.grid(alpha=0.3)
        # Hide any trailing unused axes (when n is not a multiple of ncols).
        for ax in axes_flat[len(score_keys):]:
            ax.set_visible(False)
        if title:
            fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        p = graphs_dir(run_dir) / "accuracy_curve.png"
        _savefig(fig, p); plt.close(fig); out.append(p)

    # 3) lr curve (optional).
    lrs = _col_floats(rows, "lr")
    if any(v is not None for v in lrs):
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        ex, ey = _drop_none(epochs, lrs)
        ax.plot(ex, ey, lw=1.5)
        ax.set_xlabel("epoch"); ax.set_ylabel("lr"); ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        if title: ax.set_title(title)
        fig.tight_layout()
        p = graphs_dir(run_dir) / "lr_curve.png"
        _savefig(fig, p); plt.close(fig); out.append(p)

    return out


def _coco_eval(gt_coco_json: str, pred_coco_json: str, iou_type: str = "bbox"):
    """Return an accumulated COCOeval object or None on failure."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        if not Path(pred_coco_json).is_file() or Path(pred_coco_json).stat().st_size == 0:
            return None
        with open(pred_coco_json) as fh:
            preds = json.load(fh)
        if not preds:
            return None
        coco_gt = COCO(gt_coco_json)
        coco_dt = coco_gt.loadRes(pred_coco_json)
        ev = COCOeval(coco_gt, coco_dt, iou_type)
        ev.evaluate(); ev.accumulate()
        return ev
    except Exception as e:
        print(f"[bench_plots] COCOeval({iou_type}) failed: {e}")
        return None


def _ultra_curves_from_cocoeval(ev, out_dir: Path, prefix: str = "Box",
                                class_names: Optional[Sequence[str]] = None,
                                title: str = "") -> list[Path]:
    """Ultralytics-style P/R/F1/PR curves derived from a COCOeval precision
    tensor. Produces ``<prefix>PR_curve.png``, ``<prefix>P_curve.png``,
    ``<prefix>R_curve.png``, ``<prefix>F1_curve.png``."""
    plt = _import_mpl()
    if plt is None or ev is None:
        return []
    import numpy as np
    try:
        precision = ev.eval["precision"]   # [T, R, K, A, M]
        scores    = ev.eval["scores"]      # [T, R, K, A, M]
        rec_thrs  = ev.params.recThrs       # [R]
    except Exception:
        return []
    # IoU=0.5, all area, maxDet=100.
    iou_idx, area_idx, mdet_idx = 0, 0, 2
    P = precision[iou_idx, :, :, area_idx, mdet_idx]   # [R, K]
    S = scores[iou_idx, :, :, area_idx, mdet_idx]      # [R, K]
    P = np.where(P < 0, np.nan, P)
    S = np.where(S < 0, np.nan, S)
    R = rec_thrs                                       # recall values
    K = P.shape[1]
    names = list(class_names) if class_names and len(class_names) >= K else [f"class{i}" for i in range(K)]
    out: list[Path] = []
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) PR curve (precision vs recall, per class + mean).
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for k in range(K):
        ax.plot(R, P[:, k], lw=0.8, alpha=0.6, label=names[k] if K <= 10 else None)
    mean_p = np.nanmean(P, axis=1)
    map50 = float(np.nanmean(mean_p))
    ax.plot(R, mean_p, lw=2.0, color="k", label=f"all classes mAP@.5={map50:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    if K <= 10: ax.legend(frameon=False, fontsize=7, loc="lower left")
    if title: ax.set_title(f"{prefix} PR curve  {title}")
    fig.tight_layout()
    p = out_dir / f"{prefix}PR_curve.png"
    _savefig(fig, p); plt.close(fig); out.append(p)

    # 2/3/4) P vs conf, R vs conf, F1 vs conf.
    # Use the score axis from COCOeval; build per-class curves over conf.
    def _conf_curve(values, ylabel, fname, agg_label):
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        # Sort by score per class for monotonic conf axis.
        all_y = []
        for k in range(K):
            sk = S[:, k]; vk = values[:, k]
            mask = np.isfinite(sk) & np.isfinite(vk)
            if not mask.any():
                continue
            order = np.argsort(sk[mask])
            x = sk[mask][order]; y = vk[mask][order]
            ax.plot(x, y, lw=0.8, alpha=0.6, label=names[k] if K <= 10 else None)
            all_y.append((x, y))
        # Aggregate (interp to common conf grid then mean).
        if all_y:
            grid = np.linspace(0.0, 1.0, 101)
            stacks = []
            for x, y in all_y:
                if len(x) < 2: continue
                stacks.append(np.interp(grid, x, y, left=np.nan, right=np.nan))
            if stacks:
                mean_curve = np.nanmean(np.stack(stacks, axis=0), axis=0)
                best_idx = int(np.nanargmax(mean_curve)) if np.isfinite(mean_curve).any() else 0
                ax.plot(grid, mean_curve, lw=2.0, color="k",
                        label=f"{agg_label} {mean_curve[best_idx]:.2f} at {grid[best_idx]:.3f}")
        ax.set_xlabel("Confidence"); ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
        if K <= 10: ax.legend(frameon=False, fontsize=7, loc="lower left")
        if title: ax.set_title(f"{prefix} {ylabel}  {title}")
        fig.tight_layout()
        p = out_dir / fname
        _savefig(fig, p); plt.close(fig); return p

    # Recall axis is implicit in COCOeval: every row of P corresponds to a
    # recall threshold R[r], so R-values are just the recThrs broadcast.
    R_mat = np.broadcast_to(R[:, None], P.shape).copy()
    # F1 = 2 * P * R / (P + R)
    with np.errstate(invalid="ignore", divide="ignore"):
        F1 = 2.0 * P * R_mat / (P + R_mat)
    out.append(_conf_curve(P,    "Precision", f"{prefix}P_curve.png",  "all classes P"))
    out.append(_conf_curve(R_mat, "Recall",    f"{prefix}R_curve.png",  "all classes R"))
    out.append(_conf_curve(F1,    "F1",        f"{prefix}F1_curve.png", "all classes F1"))
    return [p for p in out if p is not None]


def plot_pr_curve(gt_coco_json: str, pred_coco_json: str, out_path: Path,
                  iou: float = 0.5, title: str = "") -> Optional[Path]:
    """Single-IoU mean PR curve. Kept for backward compatibility."""
    plt = _import_mpl()
    if plt is None:
        return None
    ev = _coco_eval(gt_coco_json, pred_coco_json, "bbox")
    if ev is None:
        return None
    try:
        import numpy as np
        iou_idx_map = {0.5: 0, 0.55: 1, 0.6: 2, 0.65: 3, 0.7: 4, 0.75: 5,
                       0.8: 6, 0.85: 7, 0.9: 8, 0.95: 9}
        iou_idx = iou_idx_map.get(round(iou, 2), 0)
        pr = ev.eval["precision"][iou_idx, :, :, 0, 2]
        pr_masked = np.where(pr < 0, np.nan, pr)
        pr_mean = np.nanmean(pr_masked, axis=1)
        rc = ev.params.recThrs
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        ax.plot(rc, pr_mean, lw=1.5)
        ax.set_xlabel("recall"); ax.set_ylabel(f"precision @ IoU={iou}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
        if title: ax.set_title(title)
        fig.tight_layout()
        out_path = Path(out_path)
        _savefig(fig, out_path); plt.close(fig)
        return out_path
    except Exception as e:
        print(f"[bench_plots] PR curve failed: {e}")
        return None


def plot_confusion_matrix(cm, class_names: Sequence[str], out_path: Path,
                          normalize: bool = True, title: str = "") -> Optional[Path]:
    """Confusion matrix PNG. Set ``normalize=True`` for row-normalized."""
    plt = _import_mpl()
    if plt is None:
        return None
    try:
        import numpy as np
        cm = np.asarray(cm, dtype=float)
        if cm.size == 0:
            return None
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, np.clip(row_sums, 1e-9, None))
        n = len(class_names)
        fig_sz = max(3.5, 0.45 * n + 2.5)
        fig, ax = plt.subplots(figsize=(fig_sz, fig_sz))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        if n <= 20:
            thresh = (np.nanmax(cm) + np.nanmin(cm)) / 2.0 if cm.size else 0.5
            for i in range(n):
                for j in range(n):
                    v = cm[i, j]
                    if not np.isfinite(v):
                        continue
                    ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}",
                            ha="center", va="center", fontsize=7,
                            color="white" if v > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if title:
            ax.set_title(f"{'Normalized ' if normalize else ''}Confusion Matrix  {title}")
        fig.tight_layout()
        out_path = Path(out_path)
        _savefig(fig, out_path); plt.close(fig)
        return out_path
    except Exception as e:
        print(f"[bench_plots] confusion matrix failed: {e}")
        return None


def plot_results_panel(run_dir: Path, title: str = "") -> Optional[Path]:
    """Ultralytics-style multi-panel ``results.png`` summarising training.

    Panels: train loss, val loss, lr, box_mAP50, box_mAP50_95, seg_mAP50_95
    (only those with data are drawn). Always rendered into a fixed 2x4 grid
    matching Ultralytics' layout so the LaTeX template can rely on a single
    file name.
    """
    plt = _import_mpl()
    if plt is None:
        return None
    header, rows = _read_history(run_dir)
    if not rows:
        return None
    epochs = _col_floats(rows, "epoch")
    if not any(e is not None for e in epochs):
        return None
    # Ultralytics-style layout: train/val loss, lr, plus every available
    # mAP/mIoU including the IoU=0.75 variants that were missing before.
    panels = [
        ("train_loss",     "train/loss"),
        ("val_loss",       "val/loss"),
        ("lr",             "lr"),
        ("box_mAP50",      "metrics/mAP50(B)"),
        ("box_mAP75",      "metrics/mAP75(B)"),
        ("box_mAP50_95",   "metrics/mAP50-95(B)"),
        ("seg_mAP50",      "metrics/mAP50(M)"),
        ("seg_mAP75",      "metrics/mAP75(M)"),
        ("seg_mAP50_95",   "metrics/mAP50-95(M)"),
        ("mIoU",           "metrics/mIoU"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    axes = axes.flatten()
    drew_any = False
    for ax, (key, label) in zip(axes, panels):
        ys = _col_floats(rows, key)
        ex, ey = _drop_none(epochs, ys)
        if not ey:
            ax.set_visible(False); continue
        ax.plot(ex, ey, lw=1.5, marker=".", markersize=4)
        ax.set_xlabel("epoch"); ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)
        if key == "lr":
            ax.set_yscale("log")
        drew_any = True
    # Hide any trailing unused axes (panel count may be < grid size).
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    if not drew_any:
        plt.close(fig); return None
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    p = graphs_dir(run_dir) / "results.png"
    _savefig(fig, p); plt.close(fig)
    return p


def plot_all(run_dir: Path, *, gt_coco_json: Optional[str] = None,
             pred_coco_json: Optional[str] = None,
             cm: Optional[object] = None,
             class_names: Optional[Sequence[str]] = None,
             title: str = "") -> dict:
    """One-call wrapper: emit every applicable figure for a finished run.

    Returns a dict {figure_kind: absolute_path_str} of what was actually
    produced, suitable for stuffing into bench_manifest.json so the LaTeX
    template can pick figures up automatically.
    """
    run_dir = Path(run_dir)
    out: dict[str, str] = {}
    for p in plot_training_curves(run_dir, title=title):
        out[p.stem] = str(p)
    # Ultralytics-style multi-panel summary.
    rp = plot_results_panel(run_dir, title=title)
    if rp is not None:
        out[rp.stem] = str(rp)
    gd = graphs_dir(run_dir)
    if gt_coco_json and pred_coco_json:
        # Ultralytics-style Box*/Mask* P/R/F1/PR curves from COCOeval tensors.
        ev_box = _coco_eval(gt_coco_json, pred_coco_json, "bbox")
        if ev_box is not None:
            for p in _ultra_curves_from_cocoeval(ev_box, gd, prefix="Box",
                                                 class_names=class_names, title=title):
                out[p.stem] = str(p)
        ev_seg = _coco_eval(gt_coco_json, pred_coco_json, "segm")
        if ev_seg is not None:
            for p in _ultra_curves_from_cocoeval(ev_seg, gd, prefix="Mask",
                                                 class_names=class_names, title=title):
                out[p.stem] = str(p)
    if cm is not None and class_names:
        p = plot_confusion_matrix(cm, list(class_names),
                                  gd / "confusion_matrix.png",
                                  normalize=False, title=title)
        if p is not None:
            out[p.stem] = str(p)
        p = plot_confusion_matrix(cm, list(class_names),
                                  gd / "confusion_matrix_normalized.png",
                                  normalize=True, title=title)
        if p is not None:
            out[p.stem] = str(p)
    if out:
        try:
            with open(gd / "figures.json", "w") as fh:
                json.dump(out, fh, indent=2, sort_keys=True)
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# optional W&B helper
# ---------------------------------------------------------------------------

def maybe_init_wandb(*, project_env: str = "WANDB_PROJECT",
                     name_env: str = "WANDB_NAME",
                     config: Optional[dict] = None,
                     extra_tags: Optional[Iterable[str]] = None):
    """Init W&B only if WANDB_PROJECT is set in the environment.

    Returns the active ``wandb.Run`` (or ``None``). All failures are
    swallowed: a missing wandb package or no API key must never break a tuner.
    Pair with ``wandb.log({...}, step=epoch)`` in the training loop and call
    ``run.finish()`` at the end. Falls back to offline mode automatically if
    no API key is available.
    """
    import os
    proj = os.environ.get(project_env, "")
    if not proj:
        return None
    try:
        import wandb  # type: ignore
    except Exception as e:
        print(f"[bench_plots] wandb not installed: {e}")
        return None
    try:
        # Force offline if no key, so .init() never blocks on a login prompt.
        if not os.environ.get("WANDB_API_KEY"):
            os.environ.setdefault("WANDB_MODE", "offline")
        run = wandb.init(
            project=proj,
            name=os.environ.get(name_env) or None,
            config=config or {},
            tags=list(extra_tags or []),
            reinit=True,
        )
        return run
    except Exception as e:
        print(f"[bench_plots] wandb.init failed: {e}")
        return None


def wandb_log_safe(run, payload: dict, step: Optional[int] = None) -> None:
    """``run.log(payload, step=step)`` with all exceptions swallowed."""
    if run is None:
        return
    try:
        if step is not None:
            run.log(payload, step=int(step))
        else:
            run.log(payload)
    except Exception:
        pass


def wandb_finish_safe(run) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Post-subprocess harvest: convert each framework's native per-epoch logs into
# our uniform history.csv schema + copy any native figures into <run_dir>/graphs/.
# This is called by bench_any.py after every subprocess returns, so wrappers do
# not need to be individually edited to get paper figures.
# ---------------------------------------------------------------------------
def _ultra_results_to_history(results_csv: Path) -> list[dict]:
    """Map an Ultralytics results.csv into our canonical history schema.

    Also merges the bench's sidecar ``mAP75_history.csv`` (written by
    ``install_ultra_map75_callback`` during training) so per-epoch
    box_mAP75 / seg_mAP75 / box_mAP50_95 / seg_mAP50_95 land in history.csv
    instead of being final-epoch-only points.
    """
    if not results_csv.is_file():
        return []
    try:
        with open(results_csv, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except Exception:
        return []
    # Sidecar with per-epoch mAP75 from our Ultralytics callback (and mAP50_95
    # for parity; we trust the callback's pycocotools-like values over Ultra's).
    sidecar_by_ep: dict[int, dict] = {}
    side = results_csv.parent / "mAP75_history.csv"
    if side.is_file():
        try:
            with open(side, newline="") as fh:
                for rs in csv.DictReader(fh):
                    try:
                        ep_s = int(float(rs.get("epoch") or 0))
                    except (TypeError, ValueError):
                        continue
                    sidecar_by_ep[ep_s] = rs
        except Exception:
            pass
    out = []
    for r in rows:
        # Ultralytics column conventions:
        #   epoch, time, train/box_loss, train/cls_loss, train/dfl_loss,
        #   val/box_loss, val/cls_loss, val/dfl_loss, lr/pg0, lr/pg1, lr/pg2,
        #   metrics/precision(B), metrics/recall(B),
        #   metrics/mAP50(B), metrics/mAP50-95(B),
        #   metrics/mAP50(M), metrics/mAP50-95(M).
        def gf(*keys):
            for k in keys:
                if k in r and r[k] not in ("", None):
                    try:
                        return float(r[k])
                    except (TypeError, ValueError):
                        pass
            return None

        tl = [v for v in (gf("train/box_loss"), gf("train/cls_loss"),
                          gf("train/dfl_loss"), gf("train/seg_loss"))
              if v is not None]
        vl = [v for v in (gf("val/box_loss"), gf("val/cls_loss"),
                          gf("val/dfl_loss"), gf("val/seg_loss"))
              if v is not None]
        # Only join the sidecar when this row carries a real epoch number.
        # The bench's own single-row harvest has no epoch column, and falling
        # back to a positional index there joined it to epoch 1 and reported
        # the first epoch's metrics as the run's history.
        has_epoch = r.get("epoch") not in (None, "")
        ep_int = int(float(r["epoch"])) if has_epoch else len(out) + 1
        row = {
            "epoch": ep_int,
            "elapsed_s": gf("time"),
            "lr": gf("lr/pg0", "lr/pg1"),
            "train_loss": sum(tl) if tl else None,
            "val_loss": sum(vl) if vl else None,
            "box_mAP50": gf("metrics/mAP50(B)"),
            "box_mAP50_95": gf("metrics/mAP50-95(B)"),
            "seg_mAP50": gf("metrics/mAP50(M)"),
            "seg_mAP50_95": gf("metrics/mAP50-95(M)"),
        }
        # Merge per-epoch mAP75 (and override mAP50_95 if sidecar provides it).
        side_r = sidecar_by_ep.get(ep_int) if has_epoch else None
        if side_r:
            for k in ("box_mAP75", "seg_mAP75", "box_mAP50_95", "seg_mAP50_95",
                      "box_mAP50", "seg_mAP50"):
                v = side_r.get(k)
                if v not in (None, ""):
                    try:
                        row[k] = float(v)
                    except (TypeError, ValueError):
                        pass
        out.append(row)
    # The bench harvest overwrites Ultralytics' native results.csv in the run
    # directory, so the only surviving per-epoch trace is the sidecar. Drive
    # the history from it whenever it is the longer of the two.
    if len(sidecar_by_ep) > len(out):
        out = []
        for ep_s in sorted(sidecar_by_ep):
            rs = sidecar_by_ep[ep_s]
            row = {"epoch": ep_s}
            for k in ("box_mAP50", "box_mAP75", "box_mAP50_95",
                      "seg_mAP50", "seg_mAP75", "seg_mAP50_95"):
                v = rs.get(k)
                if v not in (None, ""):
                    try:
                        row[k] = float(v)
                    except (TypeError, ValueError):
                        pass
            out.append(row)
    return out


def install_ultra_map75_callback(model, save_dir_hint: Optional[Path] = None) -> None:
    """Attach an Ultralytics ``on_fit_epoch_end`` callback that appends
    per-epoch box_mAP75 / seg_mAP75 (plus box/seg mAP50 and mAP50_95 for
    consistency) to ``<trainer.save_dir>/mAP75_history.csv``.

    Ultralytics' validator already runs every epoch and exposes per-IoU AP
    via ``metrics.box.all_ap`` (shape [n_classes, 10] over IoU=0.5..0.95).
    AP@0.75 lives at column index 5. We just dump it to a sidecar CSV so
    ``_ultra_results_to_history`` can join it back at plot time.

    Safe no-op on any error; never raises.
    """
    try:
        import numpy as _np  # noqa: F401
    except Exception:
        return

    def _cb(trainer):
        try:
            v = getattr(trainer, "validator", None)
            m = getattr(v, "metrics", None) if v is not None else None
            if m is None:
                return
            def _ap_at(obj, idx):
                a = getattr(obj, "all_ap", None)
                if a is None:
                    return None
                try:
                    import numpy as _np2
                    if a.ndim == 2 and a.shape[1] > idx:
                        return float(_np2.nanmean(a[:, idx]))
                except Exception:
                    return None
                return None
            b = getattr(m, "box", None)
            s = getattr(m, "seg", None)
            ep = int(getattr(trainer, "epoch", 0)) + 1
            row = {"epoch": ep}
            if b is not None:
                row["box_mAP50"] = float(getattr(b, "map50", 0.0) or 0.0)
                row["box_mAP50_95"] = float(getattr(b, "map", 0.0) or 0.0)
                m75 = _ap_at(b, 5)
                if m75 is not None:
                    row["box_mAP75"] = m75
            if s is not None:
                row["seg_mAP50"] = float(getattr(s, "map50", 0.0) or 0.0)
                row["seg_mAP50_95"] = float(getattr(s, "map", 0.0) or 0.0)
                m75 = _ap_at(s, 5)
                if m75 is not None:
                    row["seg_mAP75"] = m75
            save_dir = Path(getattr(trainer, "save_dir", "") or save_dir_hint or ".")
            save_dir.mkdir(parents=True, exist_ok=True)
            out = save_dir / "mAP75_history.csv"
            new = not out.is_file()
            cols = ["epoch", "box_mAP50", "box_mAP75", "box_mAP50_95",
                    "seg_mAP50", "seg_mAP75", "seg_mAP50_95"]
            with open(out, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(cols)
                w.writerow([row.get(c, "") for c in cols])
        except Exception:
            # Never break a YOLO/RT-DETR-x training run because of plotting glue.
            pass

    try:
        model.add_callback("on_fit_epoch_end", _cb)
    except Exception:
        pass


def _lightning_metrics_to_history(metrics_csv: Path) -> list[dict]:
    """Map Lightning CSVLogger metrics.csv into our history schema.

    Supports both:
      - generic Lightning loggers (train_loss_epoch / val_loss_epoch / lr / map_50_95 / miou)
      - EoMT-style columns: ``losses/train_loss_total``,
        ``metrics/val_ap_all`` (mAP50_95), ``metrics/val_ap_50_all``,
        ``metrics/val_ap_75_all``, ``lr-AdamW/<param>`` (one per param group;
        we pick the first numeric one and treat -1.0 sentinels as missing).
    """
    if not metrics_csv.is_file():
        return []
    try:
        with open(metrics_csv, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except Exception:
        return []
    # Lightning rows have one row per logged event; collapse by epoch.
    per_ep: dict[int, dict] = {}
    for r in rows:
        ep_raw = r.get("epoch")
        if ep_raw in (None, ""):
            continue
        try:
            ep = int(float(ep_raw))
        except (TypeError, ValueError):
            continue
        acc = per_ep.setdefault(ep, {"epoch": ep + 1})
        for k, v in r.items():
            if v in (None, "") or k == "epoch":
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            # torchmetrics' "no GT" sentinel — treat as missing.
            if fv == -1.0:
                continue
            lk = k.lower()
            # ---- losses ----
            # EoMT: "losses/train_loss_total" is the headline scalar.
            if lk == "losses/train_loss_total":
                acc["train_loss"] = fv
            elif lk == "losses/val_loss_total" or lk == "losses/val_loss":
                acc["val_loss"] = fv
            elif "train" in lk and "loss" in lk and ("epoch" in lk or lk == "train_loss"):
                acc.setdefault("train_loss", fv)
            elif "val" in lk and "loss" in lk and ("epoch" in lk or lk == "val_loss"):
                acc.setdefault("val_loss", fv)
            # ---- learning rate ----
            # Lightning generic "lr" / "lr-<opt>/<param>" / EoMT "lr-AdamW/<param>".
            elif lk == "lr" or lk.startswith("lr-") or lk.startswith("lr/"):
                acc.setdefault("lr", fv)
            # ---- mAP family ----
            # EoMT torchmetrics mask AP namespace.
            elif lk == "metrics/val_ap_all" or lk.endswith("/val_ap_all"):
                acc["seg_mAP50_95"] = fv
            elif lk == "metrics/val_ap_50_all" or lk.endswith("/val_ap_50_all"):
                acc["seg_mAP50"] = fv
            elif lk == "metrics/val_ap_75_all" or lk.endswith("/val_ap_75_all"):
                acc["seg_mAP75"] = fv
            # Generic Lightning detector tags.
            elif "map_50_95" in lk or "map50-95" in lk or "map_50-95" in lk:
                acc.setdefault("box_mAP50_95", fv)
            elif lk.endswith("/map_50") or lk == "map_50":
                acc.setdefault("box_mAP50", fv)
            elif lk.endswith("/map_75") or lk == "map_75":
                acc.setdefault("box_mAP75", fv)
            elif lk.endswith("/map") or lk == "map":
                acc.setdefault("box_mAP50_95", fv)
            elif "miou" in lk:
                acc.setdefault("mIoU", fv)
    # Carry forward sticky metrics (mAP often only logged at val epochs).
    out = [per_ep[k] for k in sorted(per_ep)]
    sticky_keys = ("seg_mAP50", "seg_mAP75", "seg_mAP50_95",
                   "box_mAP50", "box_mAP75", "box_mAP50_95", "mIoU")
    last: dict = {}
    for row in out:
        for k in sticky_keys:
            if k in row:
                last[k] = row[k]
            elif k in last:
                row[k] = last[k]
    return out


def _dinov3_m2f_epoch_metrics_to_history(run_dir: Path) -> list[dict]:
    """Scrape ``<run_dir>/epoch_<N>/epoch_<N>_metrics.json`` files emitted by
    the dinov3_m2f wrapper (HuggingFace Trainer route) into our history schema.

    Each JSON has keys like ``epoch_74_map``, ``epoch_74_map_50``,
    ``epoch_74_map_75``, etc.  Loss/lr are NOT in these files; we read them
    from the wrapper's HF Trainer ``trainer_state.json`` if available.
    """
    run_dir = Path(run_dir)
    per_ep: dict[int, dict] = {}
    # Scan both the live epoch_<N>/ dirs AND the bench_any _epoch_metrics/ archive
    # (the wrapper copies each epoch's tiny metrics JSON there before pruning the
    # full weights-only epoch_<N>/ dir mid-run, so the per-epoch history survives
    # on long runs that would otherwise keep only the most recent epoch dir).
    _metric_jsons = list(run_dir.glob("epoch_*/epoch_*_metrics.json")) \
                    + list(run_dir.glob("_epoch_metrics/epoch_*_metrics.json"))
    for jf in sorted(_metric_jsons):
        try:
            m = json.load(open(jf))
        except Exception:
            continue
        # Extract epoch number from filename.
        import re as _re
        mo = _re.search(r"epoch_(\d+)", jf.name)
        if not mo:
            continue
        ep = int(mo.group(1))
        prefix = f"epoch_{ep}_"
        row = {"epoch": ep + 1}
        def _gf(*keys):
            for k in keys:
                v = m.get(prefix + k)
                if v is None:
                    v = m.get(k)
                if isinstance(v, (int, float)) and float(v) >= 0:
                    return float(v)
            return None
        for hist_k, src in (
            ("seg_mAP50_95", "map"),
            ("seg_mAP50", "map_50"),
            ("seg_mAP75", "map_75"),
            ("seg_mAP_S", "map_small"),
            ("seg_mAP_M", "map_medium"),
            ("seg_mAP_L", "map_large"),
            ("seg_AR_100", "mar_100"),
        ):
            v = _gf(src)
            if v is not None:
                row[hist_k] = v
        per_ep[ep] = row
    # Merge HF trainer_state.json log_history for train_loss / lr if present.
    state_candidates = list(run_dir.rglob("trainer_state.json"))
    if state_candidates:
        try:
            st = json.load(open(max(state_candidates, key=lambda p: p.stat().st_mtime)))
            log_history = st.get("log_history") or []
            ep_to_loss: dict[int, float] = {}
            ep_to_lr: dict[int, float] = {}
            for e in log_history:
                ep_v = e.get("epoch")
                if ep_v is None:
                    continue
                ep_i = int(float(ep_v))
                if "loss" in e and isinstance(e["loss"], (int, float)):
                    ep_to_loss[ep_i] = float(e["loss"])
                if "learning_rate" in e and isinstance(e["learning_rate"], (int, float)):
                    ep_to_lr[ep_i] = float(e["learning_rate"])
            for ep_i, row in per_ep.items():
                if ep_i in ep_to_loss:
                    row.setdefault("train_loss", ep_to_loss[ep_i])
                if ep_i in ep_to_lr:
                    row.setdefault("lr", ep_to_lr[ep_i])
        except Exception:
            pass
    return [per_ep[k] for k in sorted(per_ep)]


def _dinov3_semseg_log_to_history(run_dir: Path) -> list[dict]:
    """Parse ``train.log`` (or stdout-style text) from the dinov3_semseg wrapper.

    The wrapper prints one line per epoch of the form
    ``"epoch 3: train_loss=0.1234"``.  No val metrics are logged per epoch,
    so we only get a train-loss curve from here; final mIoU comes from
    results.csv (single-row) which the plotter scatters.
    """
    run_dir = Path(run_dir)
    candidates = list(run_dir.rglob("train.log")) + list(run_dir.rglob("*.log"))
    if not candidates:
        return []
    import re as _re
    pat = _re.compile(r"epoch\s+(\d+)\s*[:/].*?train_loss\s*=\s*([0-9.eE+-]+)", _re.IGNORECASE)
    seen: dict[int, dict] = {}
    for log in candidates:
        try:
            text = log.read_text(errors="ignore")
        except Exception:
            continue
        for mo in pat.finditer(text):
            ep = int(mo.group(1))
            try:
                loss = float(mo.group(2))
            except ValueError:
                continue
            seen[ep] = {"epoch": ep, "train_loss": loss}
    return [seen[k] for k in sorted(seen)]


def _final_results_csv_to_history(results_csv: Path, total_epochs: Optional[int] = None) -> list[dict]:
    """Last-resort: turn a single-row results.csv (the bench harvest target)
    into one history row pinned to the final epoch so the accuracy_curve
    panel at least has one scatter point per metric instead of being blank."""
    if not results_csv.is_file():
        return []
    try:
        with open(results_csv, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except Exception:
        return []
    if not rows:
        return []
    last = rows[-1]
    row = {"epoch": int(total_epochs) if total_epochs else 1}
    for k in ("train_loss", "val_loss", "lr",
              "box_mAP50", "box_mAP75", "box_mAP50_95",
              "seg_mAP50", "seg_mAP75", "seg_mAP50_95",
              "mIoU", "fg_mIoU", "pixel_acc"):
        v = last.get(k)
        if v in (None, ""):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv != 0.0:  # all-zero rows aren't informative
            row[k] = fv
    return [row] if len(row) > 1 else []


def _tb_events_to_history(run_dir: Path) -> list[dict]:
    """Best-effort: scrape any TF event files under run_dir for scalars."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return []
    evt_files = list(Path(run_dir).rglob("events.out.tfevents.*"))
    if not evt_files:
        return []
    # Use the most recent event file.
    ef = max(evt_files, key=lambda p: p.stat().st_mtime)
    try:
        acc = EventAccumulator(str(ef.parent),
                               size_guidance={"scalars": 0})
        acc.Reload()
    except Exception:
        return []
    tags = acc.Tags().get("scalars", []) or []
    series: dict[str, list[tuple[int, float]]] = {}
    for t in tags:
        try:
            evts = acc.Scalars(t)
        except Exception:
            continue
        series[t] = [(int(e.step), float(e.value)) for e in evts]
    if not series:
        return []
    # Build per-step rows from interesting tags.
    key_map = {
        "train/loss": "train_loss",
        "val/loss": "val_loss",
        "test/loss": "val_loss",
        "lr": "lr",
        "train/lr": "lr",
        "test/mAP": "box_mAP50_95",
        "test/mAP_50": "box_mAP50",
        "test/mAP_75": "box_mAP75",
        "val/mAP": "box_mAP50_95",
        "val/mAP_50": "box_mAP50",
    }
    per_step: dict[int, dict] = {}
    for tag, hkey in key_map.items():
        for s, v in series.get(tag, []):
            row = per_step.setdefault(s, {"epoch": s + 1})
            row[hkey] = v
    return [per_step[k] for k in sorted(per_step)]


def _write_history_rows(run_dir: Path, rows: list[dict]) -> Optional[Path]:
    if not rows:
        return None
    gd = graphs_dir(run_dir)
    # If a history.csv already exists (e.g., wrapper wrote it live), keep it.
    out = gd / "history.csv"
    if out.is_file() and out.stat().st_size > 0:
        return out
    for r in rows:
        append_history(run_dir, r)
    return out


def _copy_native_figures(run_dir: Path, patterns=("*.png", "*.jpg")) -> list[Path]:
    """Copy/link any native framework figures (PNG/JPG) into graphs/."""
    run_dir = Path(run_dir)
    gd = graphs_dir(run_dir)
    copied: list[Path] = []
    # Look at run_dir and any common Ultralytics-style subdir but not into
    # graphs/ itself (avoid recursion).
    candidates = []
    for pat in patterns:
        for p in run_dir.glob(pat):
            candidates.append(p)
        # Ultralytics nests under <run_dir>/<run_name>/, scan one level deep.
        for sub in run_dir.iterdir():
            if not sub.is_dir() or sub.name == GRAPHS_SUBDIR:
                continue
            for p in sub.glob(pat):
                candidates.append(p)
    for src in candidates:
        # Keep only the canonical paper figures. Drop Ultralytics' own
        # val_batch*_pred.jpg (draws the class-name label via model.names[cls],
        # leaking "penguin" onto dense-scene overlays) and labels.jpg (class
        # histogram). The standardized detection-overlay is _draw_overlay in
        # penguins/penguin_video_demo.py (ID + conf, transparent background).
        name = src.name.lower()
        keep = any(tag in name for tag in (
            "results.png", "pr_curve", "p_curve", "r_curve", "f1_curve",
            "confusion_matrix",
        ))
        if not keep:
            continue
        dst = gd / src.name
        try:
            if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                shutil.copy2(src, dst)
                copied.append(dst)
        except Exception:
            pass
    return copied


def harmonize_run(run_dir: Path, framework: str, *, title: str = "",
                  gt_coco_json: Optional[str] = None,
                  pred_coco_json: Optional[str] = None,
                  class_names: Optional[Sequence[str]] = None,
                  cm: Optional[object] = None) -> dict:
    """Single post-subprocess entry point that gives any framework's run the
    same paper-ready set of artifacts under ``<run_dir>/graphs/``.

    Sources, in priority order, depending on framework:
      - Ultralytics (yolo / rtdetr): copy native PNGs + map results.csv ->
        history.csv (overwrites only if graphs/history.csv doesn't already
        exist; wrappers that already write history.csv live take precedence).
      - EoMT (Lightning): pick up version_*/metrics.csv -> history.csv.
      - RF-DETR (TB events): scrape scalars -> history.csv.
      - dinov3_* and any wrapper without native logs: leave history.csv as is
        (may be empty); still try to emit PR curve + CM if coco jsons exist.

    Returns the dict of produced figure paths (same shape as plot_all).
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return {}
    # 0) Clean up any stale PDF figures from previous runs (we now emit PNG).
    try:
        gd_existing = run_dir / GRAPHS_SUBDIR
        if gd_existing.is_dir():
            for old in gd_existing.glob("*.pdf"):
                try: old.unlink()
                except Exception: pass
    except Exception:
        pass
    # 1) Build/append history.csv from whichever native source exists.
    fw = (framework or "").lower()
    history_rows: list[dict] = []
    if fw in ("yolo", "rtdetr"):
        # Ultralytics nests results.csv one level deep usually.
        cands = list(run_dir.glob("results.csv")) + list(run_dir.glob("*/results.csv"))
        # Skip our own aggregated bench results.csv at run_dir/results.csv only
        # if it is the canonical bench harvested one (single row). Use the
        # deepest/largest match instead.
        cands = sorted(cands, key=lambda p: p.stat().st_size if p.is_file() else 0,
                       reverse=True)
        for cand in cands:
            history_rows = _ultra_results_to_history(cand)
            if len(history_rows) > 1:
                break
    elif fw == "eomt":
        # Lightning CSVLogger nests under <run_dir>/<name>/version_*/metrics.csv.
        # Use highest-numbered version_* (the most recent run).
        import re as _re_eomt
        cands = list(run_dir.rglob("metrics.csv"))
        if cands:
            def _ver_key(p):
                m = _re_eomt.search(r"version_(\d+)", str(p))
                return int(m.group(1)) if m else -1
            history_rows = _lightning_metrics_to_history(
                max(cands, key=_ver_key))
    elif fw == "rfdetr":
        history_rows = _tb_events_to_history(run_dir)
    elif fw == "dinov3_m2f":
        history_rows = _dinov3_m2f_epoch_metrics_to_history(run_dir)
    elif fw == "dinov3_semseg":
        history_rows = _dinov3_semseg_log_to_history(run_dir)
    # Other frameworks: rely on whatever the wrapper itself wrote.

    # Universal fallback: if no per-epoch source was found, pin a single
    # final-epoch row from <run_dir>/results.csv so the plots at least show
    # one labeled scatter point per metric instead of being completely blank.
    if not history_rows:
        history_rows = _final_results_csv_to_history(run_dir / "results.csv")

    _write_history_rows(run_dir, history_rows)

    # 2) Emit the standard paper figures.
    figs = plot_all(run_dir, gt_coco_json=gt_coco_json,
                    pred_coco_json=pred_coco_json,
                    cm=cm, class_names=class_names, title=title)

    # 3) For Ultralytics, copy its native PNGs *last* so they overwrite any
    # generic figure of the same name (Ultralytics' own results.png,
    # BoxPR_curve.png, confusion_matrix*.png are the gold standard).
    if fw in ("yolo", "rtdetr"):
        for p in _copy_native_figures(run_dir):
            figs[p.stem] = str(p)
    return figs


# ============================================================================
# Cross-model aggregation figures (Step 4)
# Reads the aggregated benchmark_results.csv (by column name — no positional
# indexing) and emits paper-ready comparison figures. Multi-seed rollup: rows
# sharing a model_tag base (config tag with any _<run_tag> suffix stripped) are
# aggregated to mean ± std, giving free error bars without a --seeds flag.
# ============================================================================

# Strip the optional trailing _<run_tag> token (bench_any.build_config_tag L90-92)
# so seed runs (seed1/seed2/seed3) fold back to the same model for aggregation.
_TAG_SUFFIX_RE = re.compile(r"_[A-Za-z0-9]+$")


def _model_base(model_tag: str) -> str:
    """Fold a trailing config-tag suffix off model_tag for multi-seed grouping.

    model_tag itself is just ``Path(entry).stem`` (e.g. ``yolo11s-seg``), so this is
    a no-op on it; kept for symmetry with future per-config grouping where the full
    run_name's config tag (with optional run_tag suffix) would be the group key.
    """
    return _TAG_SUFFIX_RE.sub("", str(model_tag)) if model_tag else model_tag


def _read_csv_rows(csv_path: str) -> list:
    """Read the aggregated CSV into a list of dict rows (column-name keyed)."""
    rows = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def _group_rows(rows: list, key_fn) -> dict:
    """Group rows by ``key_fn(row)`` → list of rows (mean±std aggregation source)."""
    groups: Dict[str, list] = {}
    for r in rows:
        groups.setdefault(key_fn(r), []).append(r)
    return groups


def _agg(values: list) -> tuple:
    """Mean and std of a list of numeric strings/floats; returns (mean, std).

    std is 0.0 for a single value (no error bar). Non-numeric cells are dropped.
    """
    xs = [float(v) for v in values if v not in (None, "") and _is_float(v)]
    if not xs:
        return (float("nan"), 0.0)
    mean = sum(xs) / len(xs)
    if len(xs) < 2:
        return (mean, 0.0)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return (mean, math.sqrt(var))


def _is_float(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _bar_series(rows: list, col: str) -> tuple:
    """Per-model (mean, std) for one metric column, sorted by mean descending."""
    groups = _group_rows(rows, lambda r: r.get("model_tag", "?"))
    out = []
    for tag, grp in groups.items():
        m, s = _agg([g.get(col, "") for g in grp])
        if not (math.isnan(m) or m == 0.0):
            out.append((tag, m, s))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def plot_cross_model_comparison(csv_path: str, out_dir=None, *, dpi: int = 500) -> dict:
    """benchmark_results.csv → paper-ready cross-model comparison figures.

    Figures (only emitted when their columns/rows are present):
      1. ``cross_seg_box_map`` — grouped bar: seg_mAP50_95 per model (sorted) with
         box_mAP50_95 alongside.
      2. ``cross_native_vs_sam2`` — native-vs-SAM2 paired bars (seg_mAP50_95 vs
         sam2_mAP50_95 per model). Answers Gap-1 (does SAM2 help RF-DETR?).
      3. ``cross_difficulty_tiers`` — difficulty-tier grouped bars
         (*_mAP50_95_{Easy,Medium,Hard}) per model.
      4. ``cross_fps_vs_map`` — FPS-vs-mAP scatter with model labels (Pareto front).
      5. ``cross_easy_hard_delta`` — Easy−Hard delta bars (diff_delta_box / _seg).

    Multi-seed rollup: rows sharing a model_tag base are aggregated to mean ± std
    (error bars); single-row models get zero-width error bars. Returns
    ``{fig_name: path}``.
    """
    plt = _import_mpl()
    if plt is None:
        return {}
    rows = _read_csv_rows(csv_path)
    if not rows:
        print(f"[bench_plots] cross_model: no rows in {csv_path}")
        return {}
    out_dir = Path(out_dir) if out_dir else Path(csv_path).resolve().parent / "cross_model_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    old_dpi = _SAVE_DPI
    _set_save_dpi(dpi)
    figs: Dict[str, str] = {}
    try:
        # 1) seg + box mAP50_95 grouped bars.
        seg = _bar_series(rows, "seg_mAP50_95")
        box = _bar_series(rows, "box_mAP50_95")
        if seg or box:
            fig, ax = plt.subplots(figsize=(11, 6))
            _grouped_bar(ax, seg, box, "seg_mAP50_95", "box_mAP50_95",
                         "Segmentation vs Detection mAP50-95 per model",
                         ("seg", "box"))
            p = out_dir / "cross_seg_box_map.png"; _savefig(fig, p); figs["cross_seg_box_map"] = str(p)
            plt.close(fig)

        # 2) native seg vs SAM2-prompted seg.
        native = _bar_series(rows, "seg_mAP50_95")
        sam2 = _bar_series(rows, "sam2_mAP50_95")
        if native and sam2:
            fig, ax = plt.subplots(figsize=(11, 6))
            _grouped_bar(ax, native, sam2, "seg_mAP50_95 (native)", "sam2_mAP50_95",
                         "Native mask head vs SAM2-prompted segmentation (mAP50-95)",
                         ("native", "sam2"))
            p = out_dir / "cross_native_vs_sam2.png"; _savefig(fig, p); figs["cross_native_vs_sam2"] = str(p)
            plt.close(fig)

        # 3) difficulty tiers (box + seg combined).
        tiers = ["Easy", "Medium", "Hard"]
        if any(_bar_series(rows, f"seg_mAP50_95_{t}") for t in tiers) or \
           any(_bar_series(rows, f"box_mAP50_95_{t}") for t in tiers):
            fig, ax = plt.subplots(figsize=(12, 6))
            _tier_grouped_bar(ax, rows, tiers)
            p = out_dir / "cross_difficulty_tiers.png"; _savefig(fig, p); figs["cross_difficulty_tiers"] = str(p)
            plt.close(fig)

        # 4) FPS vs mAP scatter.
        fps_rows = [_row_metric(r, "fps_eval", "seg_mAP50_95") or _row_metric(r, "fps_eval", "box_mAP50_95")
                    for r in rows]
        fps_rows = [x for x in fps_rows if x]
        if fps_rows:
            fig, ax = plt.subplots(figsize=(9, 7))
            _pareto_scatter(ax, fps_rows)
            p = out_dir / "cross_fps_vs_map.png"; _savefig(fig, p); figs["cross_fps_vs_map"] = str(p)
            plt.close(fig)

        # 5) Easy−Hard delta bars.
        dbox = _bar_series(rows, "diff_delta_box")
        dseg = _bar_series(rows, "diff_delta_seg")
        if dbox or dseg:
            fig, ax = plt.subplots(figsize=(11, 6))
            _grouped_bar(ax, dbox, dseg, "diff_delta_box", "diff_delta_seg",
                         "Easy − Hard mAP50-95 delta per model (higher = bigger gap)",
                         ("box", "seg"))
            ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
            p = out_dir / "cross_easy_hard_delta.png"; _savefig(fig, p); figs["cross_easy_hard_delta"] = str(p)
            plt.close(fig)
    finally:
        _set_save_dpi(old_dpi)
    print(f"[bench_plots] wrote {len(figs)} cross-model figures to {out_dir}")
    return figs


# ----- helpers used by plot_cross_model_comparison -----

def _set_save_dpi(dpi: int) -> None:
    global _SAVE_DPI
    _SAVE_DPI = int(dpi)


def _row_metric(row: dict, x_col: str, y_col: str) -> Optional[tuple]:
    """(model_tag, x, y) if row has positive numeric x and y, else None."""
    x, y = row.get(x_col, ""), row.get(y_col, "")
    if not (_is_float(x) and _is_float(y)):
        return None
    xf, yf = float(x), float(y)
    if xf <= 0 or yf <= 0:
        return None
    return (row.get("model_tag", "?"), xf, yf)


def _grouped_bar(ax, series_a, series_b, label_a, label_b, title, bar_labels) -> None:
    """Two-series grouped bar chart over the union of model tags."""
    tags_a = {t[0] for t in series_a}
    tags_b = {t[0] for t in series_b}
    tags = sorted(tags_a | tags_b, key=lambda t: -max(
        next((m for tag, m, _ in series_a if tag == t), 0.0),
        next((m for tag, m, _ in series_b if tag == t), 0.0)))
    if not tags:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    amap = {t[0]: (t[1], t[2]) for t in series_a}
    bmap = {t[0]: (t[1], t[2]) for t in series_b}
    import numpy as np
    x = np.arange(len(tags))
    w = 0.38
    a_vals = [amap.get(t, (0.0, 0.0))[0] for t in tags]
    a_err = [amap.get(t, (0.0, 0.0))[1] for t in tags]
    b_vals = [bmap.get(t, (0.0, 0.0))[0] for t in tags]
    b_err = [bmap.get(t, (0.0, 0.0))[1] for t in tags]
    ax.bar(x - w / 2, a_vals, w, yerr=a_err, label=bar_labels[0], capsize=3, alpha=0.9)
    ax.bar(x + w / 2, b_vals, w, yerr=b_err, label=bar_labels[1], capsize=3, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=30, ha="right")
    ax.set_ylabel("mAP50-95")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)


def _tier_grouped_bar(ax, rows, tiers) -> None:
    """Grouped bar: per model, box_mAP50_95_{tier} and seg_mAP50_95_{tier}."""
    groups = _group_rows(rows, lambda r: r.get("model_tag", "?"))
    tags = sorted(groups.keys())
    if not tags:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    import numpy as np
    n_tier = len(tiers)
    n_fam = 2  # box, seg
    x = np.arange(len(tags))
    total_w = 0.8
    bar_w = total_w / (n_tier * n_fam)
    colors_box = ["#4c72b0", "#6ba0d6", "#a8c5e6"]
    colors_seg = ["#dd8452", "#e6a878", "#f0c5a4"]
    for ti, tier in enumerate(tiers):
        box_vals = []
        seg_vals = []
        for tag in tags:
            bm, _ = _agg([g.get(f"box_mAP50_95_{tier}", "") for g in groups[tag]])
            sm, _ = _agg([g.get(f"seg_mAP50_95_{tier}", "") for g in groups[tag]])
            box_vals.append(0.0 if math.isnan(bm) else bm)
            seg_vals.append(0.0 if math.isnan(sm) else sm)
        offset = (ti - (n_tier - 1) / 2) * bar_w * n_fam
        ax.bar(x + offset - bar_w / 2, box_vals, bar_w, label=f"box-{tier}", color=colors_box[ti], alpha=0.9)
        ax.bar(x + offset + bar_w / 2, seg_vals, bar_w, label=f"seg-{tier}", color=colors_seg[ti], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=30, ha="right")
    ax.set_ylabel("mAP50-95")
    ax.set_title("Difficulty-stratified mAP50-95 (Easy / Medium / Hard) per model")
    ax.legend(loc="best", ncol=n_tier, fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def _pareto_scatter(ax, fps_rows) -> None:
    """FPS-vs-mAP scatter with model labels + Pareto-frontier highlight."""
    # Aggregate by model_tag (mean of fps, mean of map) for multi-seed rollup.
    groups: Dict[str, list] = {}
    for tag, fps, mAP in fps_rows:
        groups.setdefault(tag, []).append((fps, mAP))
    pts = []
    for tag, vals in groups.items():
        fps = sum(v[0] for v in vals) / len(vals)
        mAP = sum(v[1] for v in vals) / len(vals)
        pts.append((tag, fps, mAP))
    pts.sort(key=lambda p: -p[1])  # by fps desc (p[1] is fps; affects label draw order only)
    # Pareto front: points where no other point has both higher fps and higher mAP.
    pareto = []
    for tag, fps, mAP in pts:
        # A point is Pareto if no other has fps>= and mAP>= with one strictly >.
        on_front = not any(other[1] >= fps and other[2] >= mAP and
                           (other[1] > fps or other[2] > mAP)
                           for other in pts if other[0] != tag)
        pareto.append(on_front)
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    ax.scatter(xs, ys, c=["#c44e52" if f else "#4c72b0" for f in pareto], zorder=3, s=60)
    for tag, fps, mAP in pts:
        ax.annotate(tag, (fps, mAP), textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel("FPS (eval)")
    ax.set_ylabel("mAP50-95")
    ax.set_title("FPS vs mAP50-95 (red = Pareto front)")
    ax.grid(alpha=0.3)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Cross-model comparison figures from the aggregated benchmark CSV.")
    ap.add_argument("--csv", default="benchmark_results.csv",
                    help="Aggregated benchmark CSV (default: benchmark_results.csv).")
    ap.add_argument("--out", default=None,
                    help="Output dir (default: <csv parent>/cross_model_figures).")
    ap.add_argument("--dpi", type=int, default=_SAVE_DPI)
    ap.add_argument("--list_models", action="store_true", help="Print model_tags present in the CSV and exit.")
    args = ap.parse_args()
    if args.list_models:
        for r in _read_csv_rows(args.csv):
            print(r.get("model_tag", "?"))
    else:
        figs = plot_cross_model_comparison(args.csv, args.out, dpi=args.dpi)
        print(f"Wrote {len(figs)} figures: {', '.join(sorted(figs)) or '(none — empty/missing columns)'}")

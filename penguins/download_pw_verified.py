#!/usr/bin/env python3
"""Pull the verified Penguin-Watch subsets out of the DaneSegmentation workspace.

WHY
---
`Marine/penguin-watch/` was emptied and the old contents renamed to
`penguin-watch-unverified/`, because the citizen-science dot annotations that
back it were never checked against a mask. Five sub-colonies have since been
verified in Roboflow, and those are the ones the paper may cite. This script
fills `penguin-watch/` with them.

WHAT COUNTS AS THE RIGHT VERSION
--------------------------------
Original resolution, YOLOv11 polygons, auto-orient as the only preprocessing.
A version carrying a resize or a tag filter is rejected, so `lockb` takes
version 1 rather than the tag-filtered and stretched version 2, and `petee`
needs a fresh version because its only version stretches to 432x432.

The script self-reports: it counts images and labels on disk, samples image
dimensions to prove no resize was applied, and writes a verdict table. It
never prints per-file progress.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

WORKSPACE = "danesegmentation"
PROJECTS = {          # roboflow project id -> destination dir name
    "halfb": "HALFb",
    "halfc": "HALFc",
    "lockb": "LOCKb",
    "damoa": "DAMOa",
    "petee": "PETEe",
}
DEST = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/penguin-watch")
OUT_DIR = Path(os.path.expanduser("~/bench_any/runs/pw_verified_download"))
REPORT = OUT_DIR / "pw_verified_download_report.txt"
FORMAT = "yolov11"
IMG_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def api(path):
    key = os.environ["ROBOFLOW_API_KEY"]
    sep = "&" if "?" in path else "?"
    with urllib.request.urlopen(f"https://api.roboflow.com/{path}{sep}api_key={key}") as r:
        return json.load(r)


def is_clean(version):
    """Auto-orient only. Anything else changes the pixels or the image set."""
    pre = version.get("preprocessing") or {}
    return set(pre) == {"auto-orient"} and pre["auto-orient"] is True


def pick_version(pid):
    """Newest version that is auto-orient only, or None if a fresh one is needed."""
    vs = api(f"{WORKSPACE}/{pid}").get("versions", [])
    clean = [v for v in vs if is_clean(v)]
    if not clean:
        return None, [f"v{v['id'].rsplit('/', 1)[-1]}: {sorted((v.get('preprocessing') or {}))}"
                      for v in vs]
    best = max(clean, key=lambda v: int(v["id"].rsplit("/", 1)[-1]))
    return int(best["id"].rsplit("/", 1)[-1]), []


def generate_clean_version(project, pid):
    """Ask Roboflow for an auto-orient-only version and wait for it."""
    print(f"[{pid}] generating an auto-orient-only version", flush=True)
    project.generate_version({"augmentation": {}, "preprocessing": {"auto-orient": True}})
    for _ in range(180):                     # up to 90 minutes
        num, _bad = pick_version(pid)
        if num is not None:
            return num
        time.sleep(30)
    raise RuntimeError(f"{pid}: no auto-orient-only version appeared")


def audit(root: Path):
    """Count the export and sample image sizes, so a silent resize cannot pass."""
    imgs = [p for p in root.rglob("*") if p.suffix in IMG_EXT]
    # Roboflow ships README.dataset.txt and README.roboflow.txt beside the labels.
    labels = [p for p in root.rglob("*.txt") if not p.name.startswith("README")]
    sizes = Counter()
    if imgs:
        from PIL import Image
        step = max(1, len(imgs) // 40)
        for p in imgs[::step]:
            with Image.open(p) as im:
                sizes[im.size] += 1
    return {"images": len(imgs), "labels": len(labels),
            "sampled_sizes": {f"{w}x{h}": n for (w, h), n in sizes.most_common(6)}}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    from roboflow import Roboflow
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    ws = rf.workspace(WORKSPACE)

    results, failures = {}, []
    for pid, name in PROJECTS.items():
        target = DEST / name
        try:
            project = ws.project(pid)
            num, rejected = pick_version(pid)
            generated = False
            if num is None:
                print(f"[{pid}] every version carries extra preprocessing: {rejected}", flush=True)
                num = generate_clean_version(project, pid)
                generated = True
            print(f"[{pid}] downloading version {num} as {FORMAT}", flush=True)
            if target.exists():
                shutil.rmtree(target)
            project.version(num).download(FORMAT, location=str(target))
            info = audit(target)
            info.update(version=num, generated=generated, dest=str(target))
            results[pid] = info
            print(f"[{pid}] done: {info['images']} images", flush=True)
        except Exception as exc:                                  # noqa: BLE001
            results[pid] = {"error": f"{type(exc).__name__}: {exc}"}
            failures.append(pid)
            print(f"[{pid}] FAILED {exc}", flush=True)

    lines = ["| subset | project | version | images | labels | image sizes sampled |",
             "|---|---|---|---|---|---|"]
    for pid, name in PROJECTS.items():
        r = results.get(pid, {})
        if "error" in r:
            lines.append(f"| {name} | {pid} | - | - | - | {r['error']} |")
            continue
        tag = f"{r['version']}{' (generated)' if r['generated'] else ''}"
        lines.append(f"| {name} | {pid} | {tag} | {r['images']} | {r['labels']} | "
                     f"{', '.join(f'{k} x{v}' for k, v in r['sampled_sizes'].items())} |")
    total = sum(r.get("images", 0) for r in results.values())
    head = ("Penguin-Watch verified subsets: FAILED on " + ", ".join(failures)
            if failures else "Penguin-Watch verified subsets: OK")
    body = [head, f"destination {DEST}", f"total images {total}", ""] + lines + [""]
    REPORT.write_text("\n".join(body))
    (OUT_DIR / "pw_verified_download.json").write_text(json.dumps(results, indent=2))
    print("\n" + "\n".join(body))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

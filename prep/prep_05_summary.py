#!/usr/bin/env python
"""Script 5: Print tight summary of all 3 output datasets from their manifests.
Run after scripts 1-4 complete. Reads manifests only, no filesystem walk.
"""
import json, os, yaml
from pathlib import Path

R1_M = Path(os.path.expanduser("~/bench_any/runs/p5_r1_gold/gold_manifest.json"))
R3G_M = Path(os.path.expanduser("~/bench_any/runs/p5_r3_groupaware/group_split_manifest.json"))
R3R_M = Path(os.path.expanduser("~/bench_any/runs/p5_r3_random/random_split_manifest.json"))
R1 = Path(os.path.expanduser("~/bench_any/datasets/p5_r1_gold"))
R3G = Path(os.path.expanduser("~/bench_any/datasets/p5_r3_groupaware"))
R3R = Path(os.path.expanduser("~/bench_any/datasets/p5_r3_random"))

def load(p):
    if p.is_file():
        return json.load(open(p))
    return None

def cnt(root, split):
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    ni = len([f for f in os.listdir(img_dir) if not f.startswith(".")]) if img_dir.is_dir() else 0
    nl = len([f for f in os.listdir(lbl_dir) if not f.startswith(".")]) if lbl_dir.is_dir() else 0
    return ni, nl

print("=" * 70)
# R1 gold
m1 = load(R1_M)
print("DATASET 1: R1 GOLD")
print(f"  path: {R1}")
if m1:
    c = m1["counts"]
    print(f"  manifest: {R1_M}")
    print(f"  seed: {m1['seed']}, test_n: {m1['test_n']}")
    print(f"  test: PW={c['pw_test']}, iNat={c['inat_test']}, total={c['pw_test']+c['inat_test']}")
    for s in ["train","valid","test"]:
        ni,nl = cnt(R1,s)
        print(f"  {s}: images={ni} labels={nl}")
    print(f"  pw_test_per_subcolony: {m1.get('pw_test_per_subcolony')}")
    print(f"  excluded_boulderspenguin: {m1.get('excluded_boulderspenguin')}")
else:
    print("  MANIFEST MISSING")

print("=" * 70)
# R3 group-aware
mga = load(R3G_M)
print("DATASET 2: R3 GROUP-AWARE")
print(f"  path: {R3G}")
if mga:
    c = mga["counts"]
    print(f"  manifest: {R3G_M}")
    print(f"  seed: {mga['seed']}")
    print(f"  train_subs ({len(mga['train_subs'])}): {mga['train_subs']}")
    print(f"  val_subs ({len(mga['val_subs'])}): {mga['val_subs']}")
    print(f"  test_subs ({len(mga['test_subs'])}): {mga['test_subs']}")
    for s in ["train","valid","test"]:
        ni,nl = cnt(R3G,s)
        print(f"  {s}: images={ni} labels={nl}")
    print(f"  total_img: {c['total_img']}")
else:
    print("  MANIFEST MISSING")

print("=" * 70)
# R3 random
mrn = load(R3R_M)
print("DATASET 3: R3 RANDOM-SENSITIVITY")
print(f"  path: {R3R}")
if mrn:
    c = mrn["counts"]
    print(f"  manifest: {R3R_M}")
    print(f"  seed: {mrn['seed']}, pct: {mrn['pct']}")
    for s in ["train","valid","test"]:
        ni,nl = cnt(R3R,s)
        print(f"  {s}: images={ni} labels={nl}")
    print(f"  total_img: {c['total_img']}")
else:
    print("  MANIFEST MISSING")
print("=" * 70)

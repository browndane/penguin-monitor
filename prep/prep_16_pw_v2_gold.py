#!/usr/bin/env python
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Sample the v2 Penguin-Watch gold from the user-verified export.

Fifty images drawn from the five Roboflow-verified sites, seeded with the fifteen
frames the v1 gold already drew from those sites so the overlap gives a paired
v1-against-v2 read on identical images. The remaining thirty-five are chosen greedily
to lift the scarce tiers: easy and hard hold 4.2 and 6.0 percent of the corpus, so a
proportional draw would leave either tier too thin to read, the failure SANParks
already shows with a single easy instance.

Selection is deterministic given --seed; the seed only breaks ties among images of
equal marginal value.

    python prep_16_pw_v2_gold.py --floor 40
"""
import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

GOLD = Path.home() / "bench_any/runs/p5_r1_gold"
TIERS = GOLD / "pw_v2_tiers.json"
OLD_MANIFEST = GOLD / "gold_manifest.json"
OUT_MANIFEST = GOLD / "gold_manifest_pw_v2.json"
OUT_DS = Path.home() / "bench_any/datasets/p5_r1_gold_pw_v2"
SITES = ["DAMOa", "HALFb", "HALFc", "LOCKb", "PETEe"]


def paired_seeds(images):
    """Stems the v1 gold drew from the five verified sites."""
    old = json.loads(OLD_MANIFEST.read_text())
    seeds = []
    for p in old["test_pw"]:
        stem = re.sub(r"^PW_[A-Z]{4}[a-z]_", "", os.path.splitext(os.path.basename(p))[0])
        if stem in images:
            seeds.append(stem)
    return seeds


def marginal_value(dets, have, floor):
    """How much this image closes the remaining gap on the scarce tiers."""
    c = Counter(d["tier"] for d in dets)
    return sum(min(c[t], max(0, floor - have[t])) for t in ("easy", "hard"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--floor", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = json.loads(TIERS.read_text())
    images = data["images"]
    rng = random.Random(args.seed)

    chosen = paired_seeds(images)
    have = Counter()
    for s in chosen:
        have.update(d["tier"] for d in images[s]["dets"])

    pool = [s for s in images if s not in set(chosen) and images[s]["dets"]]
    rng.shuffle(pool)
    while len(chosen) < args.n:
        if have["easy"] >= args.floor and have["hard"] >= args.floor:
            # both scarce tiers satisfied; fill the rest at random for a natural tail
            pick = pool.pop()
        else:
            pick = max(pool, key=lambda s: marginal_value(images[s]["dets"], have, args.floor))
            pool.remove(pick)
        chosen.append(pick)
        have.update(d["tier"] for d in images[pick]["dets"])

    OUT_DS.mkdir(parents=True, exist_ok=True)
    img_dir, lbl_dir = OUT_DS / "test/images", OUT_DS / "test/labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for stem in chosen:
        rec = images[stem]
        name = f"PW_{rec['site']}_{stem}"
        dst_i, dst_l = img_dir / f"{name}.jpg", lbl_dir / f"{name}.txt"
        for dst, src in ((dst_i, rec["image"]), (dst_l, rec["label"])):
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
    (OUT_DS / "data.yaml").write_text(
        f"path: {OUT_DS}\ntrain: test/images\nval: test/images\nnc: 1\nnames: ['penguin']\n")

    per_site = Counter(images[s]["site"] for s in chosen)
    n_inst = sum(len(images[s]["dets"]) for s in chosen)
    manifest = {
        "seed": args.seed,
        "floor_per_scarce_tier": args.floor,
        "provenance": "Roboflow user-verified export, /mnt/LinuxBackup2/Datasets_of_interest/Marine/penguin-watch",
        "annotation_version": "v2",
        "n_images": len(chosen),
        "n_instances": n_inst,
        "tier_counts": dict(have),
        "corpus_tier_rate": data["totals"],
        "per_site": dict(per_site),
        "paired_with_v1": paired_seeds(images),
        "stems": chosen,
        "dataset": str(OUT_DS),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=1))
    print("images:", len(chosen), "instances:", n_inst)
    print("tiers:", dict(have))
    print("per site:", dict(per_site))
    print("paired with v1:", len(manifest["paired_with_v1"]))
    print("written:", OUT_MANIFEST, OUT_DS)


if __name__ == "__main__":
    main()

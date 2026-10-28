#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""How much of the 71 percent agreement-filter claim survives on v2 gold.

The claim in an earlier manuscript draft is that the filter removes
71 percent of the false positives that survive the sieve, with a bypass at 0.5
contributing about 11 percent of kept masks. Re-validating it on v2 needs the
Penguin-Watch citizen-scientist dots, and those are gone: `info.zip` and every
`<SITE>.json` are absent from all three mounts, though
`~/bench_any/runs/pw_rebuild/listings/penguin_watch_unverified_sites.txt` records
that they were there on 2026-07-30.

The agreement term a(m) >= k is therefore not computable. The confidence bypass
s(m) >= tau is, because it reads a score the raw predictions still carry. That
splits the claim into a part that can be tested and a part that cannot, and it
puts a hard ceiling on the part that cannot: no agreement filter with a bypass at
tau can remove a false positive scoring at or above tau, whatever the dots said.
So the fraction of surviving false positives scoring below tau is the largest
removal any k could ever produce on this corpus. If that ceiling sits under 71
percent, the published figure cannot be reproduced on v2 even in the best case
for it, and the dots would not change that.

The sieve itself is imported from `sanity_sieve` through `sieve_composition`'s
loader, so no second convention enters. Which false positives count as the
denominator, and which sieve config produced them, are both stated two ways in
the sources, so all four combinations are reported. See CONFIGS below.

CPU, a couple of minutes. Writes `~/bench_any/runs/p5_agreement_v2/bound.json`.
"""
import json
from pathlib import Path

from pycocotools import mask as mu

from sieve_composition import (
    load_corpus, decode_image, iou_matrix, greedy_match, _as_rle, MATCH_IOU,
)
from sanity_sieve import sieve_masks, EDGE_BORDER

CORPUS = "pw-v2-200"
SCORE_FLOOR = 0.25
TAU_BYPASS = 0.5
# The paper states the denominator two ways. Lines 59 and 65 say the false
# positives that remain, or survive the sieve; line 362 says pre-sieve false
# positives. The pre-sieve pool is larger and skews low-scoring, exactly what the
# area test and the suppression remove, so the two readings give different
# ceilings and both are reported rather than one being picked.
#
# The sieve config is stated two ways as well. `pw_v2_200_zeroshot.json` records
# min_area 128 as the chosen config; CLAUDE.md records the PW pipeline as run at
# --min-area 64 --nms 0.8. Both are swept, since which masks the sieve keeps sets
# the post-sieve denominator.
CONFIGS = {
    "chosen_128": dict(a_min=128.0, rho_max=1.0, border=EDGE_BORDER,
                       area_k=0.0, nms_iou=0.8),
    "asrun_64": dict(a_min=64.0, rho_max=1.0, border=EDGE_BORDER,
                     area_k=0.0, nms_iou=0.8),
}
TIERS = Path.home() / "bench_any" / "runs" / "p5_r1_gold" / "pw_v2_200_tiers.json"
OUT = Path.home() / "bench_any" / "runs" / "p5_agreement_v2"


def tiers_by_gt(gt_rles):
    """Tier per gold instance, as a list per filename aligned to `gt_rles`.

    The manifest tiers gold penguins, not images, so a false positive has no
    tier at all and only the kept true positives can be split this way. The two
    files encode the same gold from different sources, the manifest from
    `prep_09` and the ground truth from `pw_gold_to_coco.py`, so neither the
    manifest's `idx` nor the encoded string can be trusted to line them up.
    Overlap can: each gold takes the manifest instance it covers best, and
    anything under a near-identity overlap is left unassigned rather than
    guessed at."""
    if not TIERS.exists():
        return {}
    man = json.loads(TIERS.read_text()).get("manifest", {})
    out = {}
    for fn, rles in gt_rles.items():
        insts = man.get(fn, [])
        if not insts or not rles:
            continue
        mr = [_as_rle(e["rle"]) for e in insts]
        gr = [_as_rle(r) for r in rles]
        M = mu.iou(gr, mr, [0] * len(mr))
        out[fn] = [insts[int(row.argmax())]["tier"] if row.max() >= 0.9
                   else "unassigned" for row in M]
    return out


def measure(cfg, raw, hw, gt_boxes, gt_rles, tier_of):
    """Ceilings under one sieve config, on both denominators."""
    tot = dict(raw=0, floored=0, kept=0, tp=0, fp=0, fp_below_tau=0,
               kept_above_tau=0, presieve_tp=0, presieve_fp=0,
               presieve_fp_below_tau=0)
    # Tiers describe gold penguins, so this splits the kept true positives only.
    per_tier = {}
    for fn, anns in raw.items():
        if fn not in hw:
            continue
        h, w = hw[fn]
        tot["raw"] += len(anns)
        anns = [a for a in anns if float(a["score"]) >= SCORE_FLOOR]
        tot["floored"] += len(anns)
        if not anns:
            continue
        sboxes, mboxes, areas, scores, rles = decode_image(anns, (h, w), True)
        kept, _ = sieve_masks([None] * len(sboxes), (w, h), scores=scores,
                              bboxes=sboxes, areas=areas, apply_nms=True,
                              **cfg)
        M = iou_matrix(rles, mboxes, gt_rles.get(fn, []), gt_boxes.get(fn, []),
                       "mask")

        # The pre-sieve denominator: every detection above the score floor,
        # matched by the same rule. Line 362 reads on this pool, lines 59 and 65
        # on the kept one below.
        pre = greedy_match(list(range(len(scores))), scores, M, thr=MATCH_IOU)
        for i in range(len(scores)):
            hit = i in pre
            tot["presieve_tp" if hit else "presieve_fp"] += 1
            if not hit and scores[i] < TAU_BYPASS:
                tot["presieve_fp_below_tau"] += 1

        matched = greedy_match(kept, scores, M, thr=MATCH_IOU)
        for i in kept:
            hit = i in matched
            above = scores[i] >= TAU_BYPASS
            tot["kept"] += 1
            tot["tp" if hit else "fp"] += 1
            if above:
                tot["kept_above_tau"] += 1
            if not hit and not above:
                tot["fp_below_tau"] += 1
            if not hit:
                continue
            # Greedy matching hands back the set of matched predictions but not
            # the pairing, so the gold this one scored best against stands in.
            # The two differ only where a higher-scoring prediction has already
            # taken that gold, which needs an overlap above 0.5 with both.
            j = int(M[i].argmax()) if M.shape[1] else -1
            tl = tier_of.get(fn, [])
            t = tl[j] if 0 <= j < len(tl) else "unassigned"
            b = per_tier.setdefault(t, dict(tp=0, tp_above_tau=0))
            b["tp"] += 1
            if above:
                b["tp_above_tau"] += 1

    def frac(a, b):
        return (tot[a] / tot[b]) if tot[b] else None

    # Line 362 puts the filter's removals over a pre-sieve pool, which only
    # holds if the sieve's own removals count toward the 71 as well. Credit
    # them: the false positives the sieve dropped, plus those it kept below tau.
    sieve_dropped_fp = tot["presieve_fp"] - tot["fp"]
    combined = ((sieve_dropped_fp + tot["fp_below_tau"]) / tot["presieve_fp"]
                if tot["presieve_fp"] else None)

    return {
        "config": {**cfg, "score_floor": SCORE_FLOOR,
                   "tau_bypass": TAU_BYPASS, "match_iou": MATCH_IOU},
        "totals": tot,
        "max_removable_fp_fraction_post_sieve": frac("fp_below_tau", "fp"),
        "max_removable_fp_fraction_pre_sieve":
            frac("presieve_fp_below_tau", "presieve_fp"),
        "sieve_dropped_fp": sieve_dropped_fp,
        "max_removable_fp_fraction_sieve_plus_filter": combined,
        "bypass_eligible_share_of_kept": frac("kept_above_tau", "kept"),
        "per_tier_true_positives": {
            t: {**d, "bypass_eligible_share":
                (d["tp_above_tau"] / d["tp"]) if d["tp"] else None}
            for t, d in sorted(per_tier.items())},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw, hw, gt_boxes, gt_rles = load_corpus(CORPUS)
    tier_of = tiers_by_gt(gt_rles)

    runs = {name: measure(cfg, raw, hw, gt_boxes, gt_rles, tier_of)
            for name, cfg in CONFIGS.items()}
    res = {
        "corpus": CORPUS,
        "clicks_available": False,
        "runs": runs,
        # What P5 states today. The paper no longer claims the filter removes
        # 71 percent of false positives; it states a ceiling of 58.4 percent on
        # the 200-image v2 gold subset and calls it a ceiling. The 11 percent
        # bypass share is unchanged.
        "published_claims": {"max_removable_fp_fraction": 0.584,
                             "bypass_share_of_kept": 0.11},
        "reading": ("The three max_removable_fp_fraction entries are ceilings "
                    "computed without the dots, on the denominators the paper "
                    "states: post_sieve is the one to compare against the "
                    "published 58.4 percent, which the paper already frames as "
                    "a ceiling rather than an achieved removal; pre_sieve and "
                    "sieve_plus_filter are the wider denominators that credit "
                    "the sieve's own removals to the filter. bypass_eligible_"
                    "share_of_kept is an upper bound on the published 11 "
                    "percent, since a mask at or above tau is kept by the "
                    "bypass only if it did not already have k annotators."),
    }
    (OUT / "bound.json").write_text(json.dumps(res, indent=2))

    for name, r in runs.items():
        t = r["totals"]
        print(f"[{name}]  kept {t['kept']}  tp {t['tp']}  fp {t['fp']}   "
              f"pre-sieve tp {t['presieve_tp']}  fp {t['presieve_fp']}")
        print(f"  ceiling on 71%, post-sieve : "
              f"{r['max_removable_fp_fraction_post_sieve']}")
        print(f"  ceiling on 71%, pre-sieve  : "
              f"{r['max_removable_fp_fraction_pre_sieve']}")
        print(f"  ceiling, sieve plus filter : "
              f"{r['max_removable_fp_fraction_sieve_plus_filter']}")
        print(f"  bound on the 11% bypass    : "
              f"{r['bypass_eligible_share_of_kept']}")
        for tier, d in sorted(r["per_tier_true_positives"].items()):
            share = (d["tp_above_tau"] / d["tp"]) if d["tp"] else None
            print(f"    tp tier {tier:12s} n {d['tp']:5d}  "
                  f"at or above tau {share}")


if __name__ == "__main__":
    main()

# penguin-monitor

Code and evaluation protocol for *Zero-Shot Penguin Instance Segmentation at Colony Scale: Foundation Models, Crowd-Sourced Agreement, and Expert-Time Reduction in Conservation Informatics*.

Dane Brown, Karen Bradshaw, Department of Computer Science, Rhodes University. Under review.
Project page: https://browndane.github.io/penguin-monitor

## What this is

A zero-shot instance-segmentation pipeline for African Penguin colony imagery,
and the benchmark it was measured with. A text-prompted foundation model runs
once per image from the prompt "a penguin". A geometric sanity sieve filters the
output by area, edge contact and dimension fraction. An optional agreement
filter reuses the dot annotations left by Penguin-Watch citizen scientists. No
curated training mask enters the zero-shot path.

The headline comparison is against six supervised configurations trained on the
same data budget. Close to the training distribution the supervised models lead,
75.2 against 69.0 aggregate mask mean average precision at overlap 0.5. On
adversarial camera-trap imagery they collapse, one as far as 2.8, and the
zero-shot pipeline holds at 38.7.

## Layout

    bench_any.py            orchestrator: resolves a model, runs it as a subprocess, harvests the row
    bench_eval.py           metrics, including the COCO and torchmetrics paths and SAM 2 box prompting
    bench_schema.py         the CSV column contract, single source of truth
    bench_model_registry.py model name to framework and weights
    bench_plots.py          benchmark figures
    *_any_tune.py           one training wrapper per framework
    penguins/               the P5 analysis scripts
    prep/                   corpus preparation and the evaluation producers
    docs/                   the project page, plain HTML

## Running the benchmark

    python bench_any.py --list_models
    python bench_any.py --data <dataset> --models yolo11x-seg.pt rf-detr-seg-large \
        --epochs 125 --imgsz 640 --batch_sz 2 --batch_effective 8 --seg_head sam2

Each run writes a `results.csv` and a `bench_manifest.json` under
`~/bench_any/runs/<task>/<run_name>/`, and the orchestrator harvests those rows
into the aggregated report. Two invariants matter if you modify the evaluation:
`coco_eval` must keep `maxDets` at `[1, 10, 100]`, because pycocotools hard-codes
100 when it summarises and silently returns -1 otherwise; and every wrapper must
set the multiprocessing start method to `spawn` before importing torch.

## What produces which number

| Paper object | Script | Artefact |
|---|---|---|
| Stratified per-tier tables | `prep/prep_10_filebased.py` | `ood_sam2_refine_results.json`, `*_reeval*.json` |
| Sieve ablation, precision and count | `penguins/sanity_sieve.py`, `penguins/union_sieve_sweep.py` | `precision_count_raw_vs_sieve.json` |
| Zero-shot mask table | per-run `results.csv` under `runs/segment/` | `bench_manifest.json` beside each |
| Prompt sensitivity | `penguins/plural_prompt_sweep.py` | `prompt_sensitivity.txt` |
| Fusion strategies | `penguins/fusion_strategies_sweep.py` | `fusion_sweep_results_<corpus>.txt` |
| Annotation effort, mask IoU column | `penguins/mask_iou_routes.py`, `penguins/mask_iou_route2.py` | `mask_iou_routes.json` |
| Corpus descriptors | `penguins/dataset_counts_audit.py` | `dataset_counts.json` |
| Figures 1 to 3 | `penguins/p5_figures.py` | the PDFs and PNGs in the paper directory |

The mask IoU column is worth a note, because it is the one number in the paper
that was recomputed rather than reproduced. Five defensible readings of "mean
mask IoU against gold" were tried against the published column and none
reproduced it, so the column was recomputed under one stated protocol: the mean
over gold instances, with a gold instance receiving no mask scored zero,
averaged over the two corrected gold corpora. That protocol cannot be gamed by
predicting fewer and safer masks. Under it the sieve does not raise mask
fidelity, and the paper says so.

## Machine-specific paths

This is research code that ran on one machine. `PATHS.md` lists every absolute
path still in the source, with file and line. The two a reader hits first are
already environment variables: `SAM2_ROOT` for the SAM 2 checkout and
`P5_PAPER_DIR` for the directory the figure scripts write into.

## Data

Gold corpora, the Penguin-Watch subsets and model weights are not in this
repository. The corpus descriptors and counts are in `prep/` and
`penguins/dataset_counts_audit.py`; contact the authors for access.

## Licence

MIT, see `LICENSE`. The evaluation depends on third-party model weights carrying
their own licences.

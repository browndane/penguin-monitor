# Penguin Segmentation and Counting at Colony Scale using a Geometric Sieve on Foundation-Model Masks

Code and evaluation protocol for *Penguin Segmentation and Counting at Colony Scale using a Geometric Sieve on Foundation-Model Masks*, by Dane Brown and Karen Bradshaw, Department of Computer Science, Rhodes University. Submitted to Ecological Informatics.

## About

A geometric sieve and a citizen-science agreement filter that turn the raw masks of a
text-prompted foundation model into penguin instance masks and counts, with the
difficulty-stratified benchmark used to test them. The foundation model, SAM 3, runs
once per image on the prompt "a penguin", with no training imagery. The sieve filters
its output by area, edge contact and dimension fraction. The optional agreement
filter keeps masks that Penguin-Watch citizen scientists clicked inside. The results
are in the paper. The table further down maps each paper object to the script that
produces it.

## Data

The `data/` folder holds the gold labels, model predictions and score files, with the images: all 123 Boulders Beach photographs, all 200 Penguin-Watch frames and 650 of the 856 iNaturalist test photographs. The other 206 iNaturalist photographs carry no open licence, and `data/inaturalist/images_not_included.csv` links each one. The data README maps each file to the table or figure it backs. SANParks data are held under permit and are not included.

## Layout

    bench_any.py            orchestrator: resolves a model, runs it as a subprocess, harvests the row
    bench_eval.py           metrics, including the COCO and torchmetrics paths and SAM 2 box prompting
    bench_schema.py         the CSV column contract, single source of truth
    bench_model_registry.py model name to framework and weights
    bench_plots.py          benchmark figures
    *_any_tune.py           one training wrapper per framework
    penguins/               the P5 analysis scripts
    prep/                   corpus preparation and the evaluation producers
    fiftyone_yolo_active_learning.py  the sieve caller, with the relative border test as run
    annotate/sam3_coco_annotator.py   the SAM 3 annotator behind the prompt sweeps

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

## Paper results and their scripts

| Paper object | Script | Artefact |
|---|---|---|
| Stratified per-tier tables | `prep/prep_10_filebased.py` | `ood_sam2_refine_results.json`, `*_reeval*.json` |
| Sieve ablation, precision and count | `penguins/sanity_sieve.py`, `penguins/union_sieve_sweep.py` | `precision_count_raw_vs_sieve.json` |
| Zero-shot mask table | per-run `results.csv` under `runs/segment/` | `bench_manifest.json` beside each |
| Prompt sensitivity | `penguins/plural_prompt_sweep.py` | `prompt_sensitivity.txt` |
| Fusion strategies | `penguins/fusion_strategies_sweep.py` | `fusion_sweep_results_<corpus>.txt` |
| Annotation effort, mask IoU column | `penguins/mask_iou_routes.py`, `penguins/mask_iou_route2.py` | `mask_iou_routes.json` |
| Corpus descriptors | `penguins/dataset_counts_audit.py` | `dataset_counts.json` |
| Fig. 2 and Fig. 3 | `penguins/p5_figures.py` | the PDFs and PNGs in the paper directory |
| Fig. S1 | `penguins/plot_training_dynamics.py` | `fig4_training_dynamics.png` |
| Fig. S2 | `penguins/fig_pr_overlay.py` | `fig_pr_overlay.png` |

The mask IoU column is worth a note, because it is the one number in the paper
that was recomputed rather than reproduced. Five defensible readings of "mean
mask IoU against gold" were tried against an earlier draft's column and none
reproduced it, so the column was recomputed under one stated protocol: the mean
over gold instances, with a gold instance receiving no mask scored zero,
averaged over the two corrected gold corpora. That protocol cannot be gamed by
predicting fewer and safer masks. Under it the sieve does not raise mask
fidelity, and the paper says so.

## Running on another machine

This is research code that ran on one machine. `PATHS.md` lists every absolute
path still in the source, with file and line. The two a reader hits first are
already environment variables: `SAM2_ROOT` for the SAM 2 checkout and
`P5_PAPER_DIR` for the directory the figure scripts write into.

## Licence

MIT, see `LICENSE`. The evaluation depends on third-party model weights carrying
their own licences.

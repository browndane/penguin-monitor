# Data for "Penguin Segmentation and Counting at Colony Scale using a Geometric Sieve on Foundation-Model Masks"

Dane Brown and Karen Bradshaw, Department of Computer Science, Rhodes University. Submitted to Ecological Informatics. Code: https://github.com/browndane/penguin-monitor

This archive holds the gold labels, model predictions and score files needed to re-score the numbers reported in the article and its Supplementary material (SI). It covers three datasets: Boulders Beach, iNaturalist and Penguin Watch.

SANParks data are excluded because they are held under a South African National Parks permit. The 123 Boulders Beach photographs are included in `boulders/images/`, at full resolution, with file names matching `boulders/gold_labels/`. Penguin Watch and iNaturalist images are public at their sources and are not redistributed here.

## Licences

- Boulders Beach photographs: CC BY 4.0.
- iNaturalist subset: CC BY-NC.
- Penguin Watch dot annotations: CC BY 4.0.

## How to re-score

Mask scores use pycocotools `COCOeval` with `iouType="segm"`. Load a gold COCO file with `COCO()`, then pass the prediction list to `loadRes()`. Predictions are COCO result lists with compressed run-length (RLE) masks. Per-tier scores mark every gold instance outside the tier as `iscrowd`, so it is ignored, using the tier files below. Count scores use greedy matching at mask overlap 0.5.

Pairings:
- Boulders predictions pair with `boulders/gold_labels_coco.json`.
- iNaturalist supervised predictions and `sam3_native_preds.json` pair with `inaturalist/gold/inat_test_gt_coco.json` (iNaturalist image ids).
- iNaturalist SAM 3 prompt files and `sam3_inat_test_coco.json` pair with `inaturalist/gold/inat_test_gt_rle.json` (the same 856 images, ids 1 to 856).
- Penguin Watch arms pair with `penguin_watch_200/gold_labels_coco.json`.

## Files

### boulders/ (123 images, 769 gold penguins)

- `gold_labels/`: 123 gold label files, YOLO segmentation text (class, then normalised polygon vertices), one per image. Tables 1, 2 and 3, Fig. 2.
- `gold_labels_coco.json`: the same gold in COCO JSON with RLE masks. Every Boulders score.
- `tiers/boulders_tiers.json`: per-instance difficulty tier (Easy, Medium, Hard), JSON. Table 2 and Table S2.
- `tiers/boulders_tiers_tercile.json`, `tiers/boulders_tiers_variant{A,B,C}.json`: alternative tierings, JSON. Tables S8 and S9, Fig. S3.
- `sam3/boulders_sam3_raw.json`, `_raw_penguin.json`, `_raw_african_penguin.json`: SAM 3 predictions before the sieve for the prompts "a penguin", "penguin" and "African penguin", COCO results. Table 3, Table S5.
- `sam3/boulders_sam3_pred.json`, `_pred_penguin.json`, `_pred_african_penguin.json`: the same predictions after the sieve, COCO results. Table 2, Table 3, Fig. 2, Table S5.
- `sam3/boulders_sieve_sweep*.json`: sieve parameter sweep per prompt, JSON. The 54-cell sweep and recall floor of 0.95 in the SI count-error paragraph.
- `sam3/boulders_reeval_sam3.json`, `sam3/boulders_reeval_sam3_raw.json`: per-tier mask mAP50 of SAM 3 with and without the sieve, JSON. Table 2.
- `sam3/manifest.json`: settings of the SAM 3 Boulders run (prompt, thresholds, sieve parameters), JSON.
- `sam3/sieve_decomposition_2026-08-11.json`: sieve tests applied one at a time (precision, recall, count error, mask mAP50), JSON. Results text after Table 3 (precision 79.9% to 84.0%, count error 1.18 to 0.80).
- `sam3/nms_toggle_2026-08-11.json`: non-maximum suppression on and off for three detectors, JSON. SI section "Non-maximum suppression toggle".
- `sam3/nms_toggle_2026-08-11_preds/`: the six prediction sets behind that toggle, COCO results.
- `supervised/<model>_pred.json`: supervised predictions on Boulders for RF-DETR-seg Large and XLarge, EoMT-Large, RT-DETR-v3, RT-DETR-v4 and YOLOv11x-seg, COCO results. Table 2, Table S2, Tables S8 and S9.
- `supervised/<model>_reeval.json`: per-tier and aggregate mask mAP50 for each prediction file, JSON. Same tables.
- `supervised/yolo11x-seg_pred.json` and `_reeval.json`: an earlier YOLOv11x-seg run with a shared training recipe. Its score is not printed and is kept for completeness. The printed YOLO rows use `yolo11x-seg-native` and `yolo11x-seg-combined-native`.
- `supervised/yolo11x-seg-native_5tier.json`: five-tier scores, JSON. Fig. S3.
- `sam2_control/`: SAM 2 box-prompt refinement control. `_control_reeval` scores the model's own masks, `_sam2_reeval` scores SAM 2 masks on the same boxes, `_sam2_pred` holds the SAM 2 predictions. JSON and COCO results. SI text on SAM 2 refinement (for example 64.66 to 70.70 for the corrected YOLOv11x-seg run).
- `counting/gold_table_fixed05.txt`: count error decomposition at a fixed score of 0.5, plain-text table. Fig. 2.
- `counting/prompt_union_probe.txt`: count error for single prompts and their union over score thresholds, plain-text table. Table S6, Boulders values.
- `counting/fusion_sweep_results_boulders-penguin.txt`: prompt fusion strategies, Markdown table. Table S5.

### penguin_watch_200/ (200 images from five sites, 3114 gold penguins)

- `gold_labels/`: 200 gold label files, YOLO segmentation text. Table 1, Table 4.
- `gold_labels_coco.json`: the same gold in COCO JSON. Table 4.
- `gold_manifest_pw_v2_200.json`: image list and sampling design (40 images per site, tiers ranked within site), JSON.
- `pw_v2_200_tiers.json`: per-instance difficulty tier, JSON. Table 4.
- `agreement_bound.json`: bounds on the agreement filter, JSON. Results text on the agreement filter (58.4% and 83.8%, run `asrun_64`).
- `three_arm/sam2_vs_sam3_3arm_2026-08-11.json`: scores of the three arms, JSON. Table 4.
- `three_arm/arm1_sam2_clicks_masks.json`: arm 1, SAM 2 masks from volunteer clicks, COCO results. Table 4.
- `three_arm/arm2_sam3_agreement.json`: arm 2, SAM 3 masks kept by the annotator agreement filter, COCO results. Table 4.
- `three_arm/arm3_pw_v2_200_sam3_raw.json`: arm 3, SAM 3 predictions before the sieve (3952 masks), COCO results. Applying the sieve gives the 3557 masks scored in Table 4.
- `three_arm/agreement_stage/`: five per-site COCO files after the agreement filter, the input to arm 2.

### inaturalist/ (856-image test split, 5695 gold penguins)

- `gold/inat_test_gt_coco.json`: gold for the test split, COCO JSON with polygons. Table 1, Table S3, Fig. S2.
- `gold/inat_test_gt_rle.json`: the same gold with RLE masks and ids 1 to 856. Table S6 and the counting results.
- `gold/inat_v1_50_tiers.json`: tiers of the 50-image v1 subset (394 instances), JSON. Table S7.
- `sam3/sam3_native_preds.json`, `sam3/sam3_native_mask_results.json`: SAM 3 predictions and their scores, COCO results and JSON. Table S3, SAM 3 row.
- `sam3/sam3_inat_test_coco.json`: SAM 3 "a penguin" predictions on the RLE gold, COCO JSON. Counting results.
- `sam3/prompts/`: SAM 3 predictions for five prompts, COCO JSON. Table S6.
- `supervised/<model>/results.csv`: final test scores (box and mask mAP50 and mAP50:95) for RF-DETR-seg Large and XLarge, EoMT-Large, DINOv3-M2F-ViT-H/16+, YOLOv11x-seg and RT-DETR-v4, CSV. Table S3.
- `supervised/yolo11x-seg-shared-flipfix/results.csv`: corrected YOLOv11x-seg run, CSV. The 87.6 mask mAP50 in the main text.
- `supervised/dinov3-m2f-vith16plus/all_results.json`: DINOv3-M2F-ViT-H/16+ test scores, JSON. Table S3.
- `supervised/*/predictions.json`, `supervised/*/sam2_predictions.json`: predictions for EoMT-Large, YOLOv11x-seg and DINOv3 (SAM 2 only), COCO results. Table S3, Fig. S2.
- `counting/threshold_offtest.txt`: score threshold chosen on validation and applied to test, plain text. Main-text count error 0.813 and signed error -0.005.
- `counting/prompt_sensitivity.txt`: count error per SAM 3 prompt, plain-text table. Table S6.
- `counting/count_decompose_agg.csv`, `counting/perimage_*.csv`: count error decomposition, aggregate and per image, CSV. Fig. 3.
- `percorpus_v1/`: v1 per-tier scores for Penguin Watch and iNaturalist, JSON. Table S7.
- `corpus_diversity/`: held-out gold (COCO JSON), per-tier and final scores (CSV) and DINOv3 predictions (COCO results). Table S4.

### training_curves/

- Per-epoch validation history for the YOLO, RF-DETR, EoMT and DINOv3 runs, CSV. Fig. S1.

### summaries/

- `relsize_stratified.json`: per-tier scores under each border rule, JSON. Table 2 border row, Table S1.
- `relsize_tercile.json`: per-tier scores under the fixed and the equal-frequency (tercile) tiers, with and without the relative border test, JSON. Table 2 Agg_t cells of the border-test row (90.0 and 77.9).
- `sieve_composition.json`: masks removed by each sieve test, JSON. Table 3.
- `precision_count_raw_vs_sieve.json`: false positives before and after the sieve, JSON. Table 5 (184 to 175).
- `mask_iou_column_final.json`: mean mask overlap per annotation route, JSON. Table 5 (0.85 and 0.83).
- `expanded_scores.json`, `boulders_expanded.tex`: Boulders per-tier scores for all models, JSON and LaTeX table body. Table 2, Table S2, Tables S8 and S9, Fig. S3.
- `ood_sam2_refine_results.json`, `rtdetr_bestckpt_sam2refine_results.json`, `flipfix_yolo11x_sam2refine_results.json`: SAM 2 refinement summaries, JSON. SI text on SAM 2 refinement.

## Changes made to these copies

- File paths are shortened: the home directory reads `/home/user` and dataset roots read `data/`.
- SANParks entries were removed from `boulders/sam3/manifest.json`, `boulders/counting/gold_table_fixed05.txt`, `boulders/counting/prompt_union_probe.txt` and every file in `summaries/`. The pooled value in `mask_iou_column_final.json` averages Boulders with SANParks, so only its Boulders part can be re-scored here.
- Free-text run notes were removed: `phase` in `manifest.json`, `hypothesis` and `note` in `nms_toggle_2026-08-11.json`, `metric` in `sam3_native_mask_results.json`, `reading` in `agreement_bound.json`, `v1_paired_note`, `gold_density.note` and `sidecar` in `gold_manifest_pw_v2_200.json`, and two edit-history notes in `mask_iou_column_final.json`. No number, mask, box, score or id was changed.
- In some RLE mask strings one character is written as a JSON `\u` escape. Any JSON parser decodes it to the original character, so every mask is unchanged.
- The fields named `paper` and `paper_in_dist_mAP50` are reference values written by earlier scoring scripts. They are not computed from the files here and are not the article's numbers.

## Not included

- SANParks labels, tiers, predictions and scores (permit).
- Penguin Watch and iNaturalist images, which are public at their sources.
- Large prediction files whose scores are already in a shipped score file: iNaturalist RF-DETR-seg predictions and their SAM 2 versions, the RT-DETR-v4 iNaturalist SAM 2 predictions, the Boulders RF-DETR-seg SAM 2 control predictions and the YOLO corpus-diversity predictions.
- The iNaturalist DINOv3-M2F-ViT-H/16+ own-mask predictions, removed by an automated string screen. Its Table S3 row is backed by `all_results.json` and `results.csv`.
- Per-image score arrays for soft-average prompt fusion, so that row of Table S5 cannot be re-scored here.
- Run logs, scripts, backups and superseded runs.

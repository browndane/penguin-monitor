# Machine-specific paths still in the source

Every absolute path left in the released code. The SAM 2 checkout and the
figure output directory are environment variables, `SAM2_ROOT` and
`P5_PAPER_DIR`; the rest are literals to edit for your own machine.

| File | Line | Source |
|---|---|---|
| `bench_any.py` | 10 | `# nohup python bench_any.py --data /mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/RoboflowDatasets/SEGM` |
| `bench_any.py` | 599 | `"/home/vorse/sam2/checkpoints/sam2.1_hiera_large.pt"),` |
| `bench_any.py` | 603 | `"/home/vorse/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"),` |
| `bench_eval.py` | 1110 | `sam2_root = os.environ.get("SAM2_ROOT", "/home/vorse/sam2")` |
| `bench_eval.py` | 1135 | `# Extract relative path: /home/vorse/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml` |
| `dinov3_semseg_any_tune.py` | 655 | `f"file in /home/vorse/bench_any/dinov3_weights/ as {DINO_MODEL}*.pth, or pass "` |
| `penguins/bootstrap_ci.py` | 36 | `# All paths are under /home/vorse/bench_any/runs/segment/ (or detect for RT-DETR).` |
| `penguins/bootstrap_ci.py` | 43 | `BENCH = "/home/vorse/bench_any/runs"` |
| `penguins/c3_diagnostic.py` | 20 | `GT = "/home/vorse/bench_any/runs/segment/african-penguin_eomt-large_b2_e150_i640_be64_fair_grid/_bench_gt_coco` |
| `penguins/c3_diagnostic.py` | 21 | `PRED = "/home/vorse/bench_any/runs/segment/african-penguin_eomt-large_b2_e150_i640_be64_fair_grid/_bench_predi` |
| `penguins/dataset_counts_audit.py` | 21 | `ROOT = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine")` |
| `penguins/download_pw_verified.py` | 42 | `DEST = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/penguin-watch")` |
| `penguins/fusion_strategies_sweep.py` | 37 | `PREP_DIR = Path("/home/vorse/bench_any/datasets/prep")` |
| `penguins/gt_scaffold.py` | 49 | `--video /mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/demo_videos/penguin_african_easy.mp4 \` |
| `penguins/inat_sam2_count.py` | 25 | `D = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin")` |
| `penguins/mask_iou_route2.py` | 28 | `IMG_ROOT = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin")` |
| `penguins/mask_iou_route2.py` | 29 | `SAM2_CKPT = "/home/vorse/sam2/checkpoints/sam2.1_hiera_large.pt"` |
| `penguins/mask_iou_route2.py` | 39 | `sys.path.insert(0, "/home/vorse/sam2")` |
| `penguins/mask_iou_route2.py` | 43 | `initialize_config_dir(config_dir="/home/vorse/sam2/sam2/configs", version_base="1.2")` |
| `penguins/nms_toggle_refit.py` | 37 | `BENCH = "/home/vorse/bench_any/runs"` |
| `penguins/nms_toggle_refit.py` | 42 | `DATASET = "/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/african-penguin-inaturalist-sam-veryclean"` |
| `penguins/ood_sam2_refine.py` | 57 | `SAM2_CK = "/home/vorse/sam2/checkpoints/checkpoints/sam2.1_hiera_large.pt"` |
| `penguins/ood_sam2_refine.py` | 58 | `SAM2_CFG = "/home/vorse/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"` |
| `penguins/penguin_video_demo.py` | 14 | `--weights /home/vorse/bench_any/runs/segment/penguin-seg-combined_rf-detr-seg-large/checkpoint_best_total.pth ` |
| `penguins/penguin_video_demo.py` | 15 | `--video   /mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/demo_videos/penguin_underwater.mp4 \` |
| `penguins/penguin_video_demo.py` | 75 | `DEFAULT_WEIGHTS = "/home/vorse/bench_any/runs/segment/african-penguin_rf-detr-seg-xlarge_b1_e150_i640_be64_fai` |
| `penguins/penguin_video_demo.py` | 76 | `DEFAULT_VIDEO = "/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/demo_videos/penguin_underwater.mp4"` |
| `penguins/plural_prompt_sweep.py` | 31 | `D = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin")` |
| `penguins/plural_prompt_sweep.py` | 32 | `ANNOT = Path("/home/vorse/Dropbox/Work/PycharmProjects/cvMLCode/12_Deep/"` |
| `penguins/point_noise_study.py` | 50 | ```/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/boulders-penguin/train``),` |
| `penguins/point_noise_study.py` | 87 | `SAM2_ROOT = "/home/vorse/sam2"` |
| `penguins/point_noise_study.py` | 95 | `"/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/"` |
| `penguins/post_loop_recall.py` | 20 | `YOLO_RD = "/home/vorse/bench_any/runs/segment/african-penguin_yolo11x-seg_b0.98_e150_i640_be64_fair_grid/test"` |
| `penguins/post_loop_recall.py` | 21 | `RF_RD   = "/home/vorse/bench_any/runs/segment/african-penguin_rf-detr-seg-xlarge_b1_e150_i640_be64_fair_grid"` |
| `penguins/prompt_union_probe.py` | 37 | `D = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin")` |
| `penguins/run_point_noise.py` | 28 | `"--corpus", "/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/boulders-penguin/train",` |
| `penguins/sam2_refine_timing.py` | 25 | `_SAM2_ROOT = "/home/vorse/sam2"` |
| `penguins/sam2_video_propagate.py` | 28 | `--video /mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/demo_videos/penguin_african_easy.mp4 \` |
| `penguins/sam2_video_propagate.py` | 64 | `"SAM2_CKPT", "/home/vorse/sam2/checkpoints/sam2.1_hiera_large.pt")` |
| `penguins/sam2_video_propagate.py` | 66 | `"SAM2_CONFIG", "/home/vorse/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")` |
| `penguins/sam2_video_propagate.py` | 67 | `SAM2_CONFIGS_ROOT = "/home/vorse/sam2/sam2/configs"` |
| `penguins/sam2_video_propagate.py` | 137 | `sam2_root = "/home/vorse/sam2"` |
| `penguins/sensitivity_table.py` | 28 | `D = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin")` |
| `penguins/union_sam2refine.py` | 24 | `D = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin")` |
| `penguins/union_sam2refine.py` | 28 | `CK = "/home/vorse/sam2/checkpoints/checkpoints/sam2.1_hiera_large.pt"` |
| `penguins/union_sam2refine.py` | 29 | `CFG = "/home/vorse/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"` |
| `penguins/union_sieve_sweep.py` | 37 | `PREP_DIR = Path("/home/vorse/bench_any/datasets/prep")` |
| `prep/prep_01_pw_coco_to_yolo.py` | 31 | `SRC = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/penguin-watch")` |
| `prep/prep_02_r1_gold_sampler.py` | 24 | `INAT = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/african-penguin-inaturalist-sam-veryclean")` |
| `prep/prep_04_recover_skipped.py` | 9 | `os.environ.setdefault("HF_HOME", "/mnt/LinuxBackup/sam3")` |
| `prep/prep_04_recover_skipped.py` | 10 | `os.environ.setdefault("HF_HUB_CACHE", "/mnt/LinuxBackup/sam3/hub")` |
| `prep/prep_04_recover_skipped.py` | 11 | `os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/LinuxBackup/sam3/hub")` |
| `prep/prep_04_recover_skipped.py` | 16 | `BOU = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/Karen/BouldersBeachDec25/train/images")` |
| `prep/prep_06_export_roboflow.py` | 14 | `OUT = Path("/home/vorse/bench_any/datasets/p5_gold200_roboflow")` |
| `prep/prep_07_upload_roboflow.py` | 13 | `IMG = Path("/home/vorse/bench_any/datasets/p5_gold200_roboflow/images")` |
| `prep/prep_07_upload_roboflow.py` | 14 | `MANIFEST = Path("/home/vorse/bench_any/datasets/p5_gold200_roboflow/upload_manifest.json")` |
| `prep/prep_08_load_roboflow_verified.py` | 22 | `DOWNLOAD = Path("/home/vorse/bench_any/datasets/p5_gold200_roboflow_download")` |
| `prep/prep_08_load_roboflow_verified.py` | 23 | `MANIFEST = Path("/home/vorse/bench_any/datasets/p5_gold200_roboflow/upload_manifest.json")` |
| `prep/prep_10b_debug_medium.py` | 13 | `TIERS_MAN = "/home/vorse/bench_any/runs/p5_r1_gold/gold200_tiers.json"` |
| `prep/prep_10c_diag_medium.py` | 23 | `TIERS_MAN = "/home/vorse/bench_any/runs/p5_r1_gold/gold200_tiers.json"` |
| `prep/prep_10d_per_corpus.py` | 13 | `TIERS_MAN = "/home/vorse/bench_any/runs/p5_r1_gold/gold200_tiers.json"` |
| `prep/prep_11_filebased.py` | 24 | `target_sizes=[(H,W)]). HF_HOME=/mnt/LinuxBackup/sam3.` |
| `prep/prep_11_filebased.py` | 31 | `os.environ.setdefault("HF_HOME", "/mnt/LinuxBackup/sam3")` |
| `prep/prep_11_filebased.py` | 32 | `os.environ.setdefault("HF_HUB_CACHE", "/mnt/LinuxBackup/sam3/hub")` |
| `prep/prep_11_filebased.py` | 33 | `os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/LinuxBackup/sam3/hub")` |
| `prep/prep_11_generate_preds.py` | 18 | `os.environ.setdefault("HF_HOME", "/mnt/LinuxBackup/sam3")` |
| `prep/prep_11_generate_preds.py` | 19 | `os.environ.setdefault("HF_HUB_CACHE", "/mnt/LinuxBackup/sam3/hub")` |
| `prep/prep_11_generate_preds.py` | 20 | `os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/LinuxBackup/sam3/hub")` |
| `prep/prep_11_generate_preds.py` | 28 | `YOLO_CKPT = "/home/vorse/bench_any/runs/segment/penguin-seg-combined_yolo11x-seg/weights/best.pt"` |
| `prep/prep_11_generate_preds.py` | 29 | `RTDETR_CKPT = "/home/vorse/bench_any/runs/detect/penguin-seg-combined_rtdetr-x/weights/best.pt"` |
| `prep/prep_11_run_all.py` | 7 | `os.environ.setdefault("HF_HOME", "/mnt/LinuxBackup/sam3")` |
| `prep/prep_11_run_all.py` | 8 | `os.environ.setdefault("HF_HUB_CACHE", "/mnt/LinuxBackup/sam3/hub")` |
| `prep/prep_11_run_all.py` | 9 | `os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/LinuxBackup/sam3/hub")` |
| `prep/prep_11b_soft_logit.py` | 30 | `os.environ.setdefault("HF_HOME", "/mnt/LinuxBackup/sam3")` |
| `prep/prep_11b_soft_logit.py` | 31 | `os.environ.setdefault("HF_HUB_CACHE", "/mnt/LinuxBackup/sam3/hub")` |
| `prep/prep_11b_soft_logit.py` | 33 | `sys.path.insert(0, "/home/vorse/bench_any/datasets/prep")` |
| `prep/prep_11b_soft_logit.py` | 46 | `DS_ROOT = "/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin"` |
| `prep/prep_11b_verify_boxprompt.py` | 4 | `os.environ.setdefault("HF_HOME", "/mnt/LinuxBackup/sam3")` |
| `prep/prep_11b_verify_boxprompt.py` | 5 | `os.environ.setdefault("HF_HUB_CACHE", "/mnt/LinuxBackup/sam3/hub")` |
| `prep/prep_11b_verify_boxprompt.py` | 6 | `os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/LinuxBackup/sam3/hub")` |
| `prep/prep_11b_verify_boxprompt.py` | 17 | `RTDETR_CKPT = "/home/vorse/bench_any/runs/detect/penguin-seg-combined_rtdetr-x/weights/best.pt"` |
| `prep/prep_13_supervised_reeval.py` | 98 | `"images": "/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/boulders-penguin/train/images",` |
| `prep/prep_13_supervised_reeval.py` | 102 | `"images": "/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/sanparks-algoa/train/images",` |
| `prep/prep_13_supervised_reeval.py` | 348 | `sys.path.insert(0, "/home/vorse/bench_any/eomt")` |
| `prep/prep_13_supervised_reeval.py` | 475 | `DINOv3 (default HF cache) and SAM3 (/mnt/LinuxBackup/sam3) don't collide."""` |
| `prep/prep_13_supervised_reeval.py` | 478 | `cache = "/mnt/LinuxBackup/sam3/hub"` |
| `sam2_backfill_rfdetr.py` | 16 | `"SAM2_CKPT", "/home/vorse/sam2/checkpoints/sam2.1_hiera_large.pt")` |
| `sam2_backfill_rfdetr.py` | 18 | `"SAM2_CONFIG", "/home/vorse/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")` |
| `yolo_any_tune.py` | 4 | `# USAGE (Advanced): python3 yolo_any_tune.py --data /mnt/LinuxBackup/FoundationDatasets/oiv7-yolo-allfindrefin` |

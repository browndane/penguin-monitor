# Machine-specific paths still in the source

Every absolute path left in the released code. The SAM 2 checkout and the
figure output directory are environment variables, `SAM2_ROOT` and
`P5_PAPER_DIR`; the rest are literals to edit for your own machine.

| File | Line | Source |
|---|---|---|
| `bench_any.py` | 11 | `# nohup python bench_any.py --data /mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/RoboflowDatasets/SEGM` |
| `bench_any.py` | 785 | `"/home/user/sam2/checkpoints/sam2.1_hiera_large.pt"),` |
| `bench_any.py` | 789 | `"/home/user/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"),` |
| `bench_eval.py` | 1120 | `sam2_root = os.environ.get("SAM2_ROOT", "/home/user/sam2")` |
| `bench_eval.py` | 1145 | `# Extract relative path: /home/user/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml` |
| `dinov3_semseg_any_tune.py` | 659 | `f"file in /home/user/bench_any/dinov3_weights/ as {DINO_MODEL}*.pth, or pass "` |
| `penguins/bootstrap_ci.py` | 37 | `# All paths are under /home/user/bench_any/runs/segment/ (or detect for RT-DETR).` |
| `penguins/bootstrap_ci.py` | 44 | `BENCH = "/home/user/bench_any/runs"` |
| `penguins/c3_diagnostic.py` | 21 | `GT = "/home/user/bench_any/runs/segment/african-penguin_eomt-large_b2_e150_i640_be64_fair_grid/_bench_gt_coco` |
| `penguins/c3_diagnostic.py` | 22 | `PRED = "/home/user/bench_any/runs/segment/african-penguin_eomt-large_b2_e150_i640_be64_fair_grid/_bench_predi` |
| `penguins/dataset_counts_audit.py` | 22 | `ROOT = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine")` |
| `penguins/download_pw_verified.py` | 43 | `DEST = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/penguin-watch")` |
| `penguins/fusion_strategies_sweep.py` | 38 | `PREP_DIR = Path("/home/user/bench_any/datasets/prep")` |
| `penguins/gt_scaffold.py` | 50 | `--video /mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/demo_videos/penguin_african_easy.mp4 \` |
| `penguins/inat_sam2_count.py` | 26 | `D = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin")` |
| `penguins/nms_toggle_refit.py` | 38 | `BENCH = "/home/user/bench_any/runs"` |
| `penguins/nms_toggle_refit.py` | 43 | `DATASET = "/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/african-penguin-inaturalist-sam-veryclean"` |
| `penguins/ood_sam2_refine.py` | 58 | `SAM2_CK = "/home/user/sam2/checkpoints/checkpoints/sam2.1_hiera_large.pt"` |
| `penguins/ood_sam2_refine.py` | 59 | `SAM2_CFG = "/home/user/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"` |
| `penguins/penguin_video_demo.py` | 15 | `--weights /home/user/bench_any/runs/segment/penguin-seg-combined_rf-detr-seg-large/checkpoint_best_total.pth ` |
| `penguins/penguin_video_demo.py` | 16 | `--video   /mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/demo_videos/penguin_underwater.mp4 \` |
| `penguins/penguin_video_demo.py` | 76 | `DEFAULT_WEIGHTS = "/home/user/bench_any/runs/segment/african-penguin_rf-detr-seg-xlarge_b1_e150_i640_be64_fai` |
| `penguins/penguin_video_demo.py` | 77 | `DEFAULT_VIDEO = "/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/demo_videos/penguin_underwater.mp4"` |
| `penguins/plural_prompt_sweep.py` | 32 | `D = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin")` |
| `penguins/plural_prompt_sweep.py` | 33 | `ANNOT = Path("/home/user/Dropbox/Work/PycharmProjects/cvMLCode/12_Deep/"` |
| `penguins/point_noise_study.py` | 51 | ```/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/boulders-penguin/train``),` |
| `penguins/point_noise_study.py` | 88 | `SAM2_ROOT = "/home/user/sam2"` |
| `penguins/point_noise_study.py` | 96 | `"/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/"` |
| `penguins/relsize_stratified_recompute.py` | 43 | `PREP_DIR = Path("/home/user/bench_any/datasets/prep")` |
| `penguins/sam2_refine_timing.py` | 26 | `_SAM2_ROOT = "/home/user/sam2"` |
| `penguins/sam2_video_propagate.py` | 29 | `--video /mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/demo_videos/penguin_african_easy.mp4 \` |
| `penguins/sam2_video_propagate.py` | 65 | `"SAM2_CKPT", "/home/user/sam2/checkpoints/sam2.1_hiera_large.pt")` |
| `penguins/sam2_video_propagate.py` | 67 | `"SAM2_CONFIG", "/home/user/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")` |
| `penguins/sam2_video_propagate.py` | 68 | `SAM2_CONFIGS_ROOT = "/home/user/sam2/sam2/configs"` |
| `penguins/sam2_video_propagate.py` | 138 | `sam2_root = "/home/user/sam2"` |
| `prep/prep_01_pw_coco_to_yolo.py` | 33 | `SRC = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/penguin-watch")` |
| `prep/prep_02_r1_gold_sampler.py` | 25 | `INAT = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/african-penguin-inaturalist-sam-veryclean")` |
| `prep/prep_04_recover_skipped.py` | 10 | `os.environ.setdefault("HF_HOME", "/mnt/LinuxBackup/sam3")` |
| `prep/prep_04_recover_skipped.py` | 11 | `os.environ.setdefault("HF_HUB_CACHE", "/mnt/LinuxBackup/sam3/hub")` |
| `prep/prep_04_recover_skipped.py` | 12 | `os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/LinuxBackup/sam3/hub")` |
| `prep/prep_04_recover_skipped.py` | 17 | `BOU = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/Partner/BouldersBeachDec25/train/images")` |
| `prep/prep_06_export_roboflow.py` | 15 | `OUT = Path("/home/user/bench_any/datasets/p5_gold200_roboflow")` |
| `prep/prep_07_upload_roboflow.py` | 14 | `IMG = Path("/home/user/bench_any/datasets/p5_gold200_roboflow/images")` |
| `prep/prep_07_upload_roboflow.py` | 15 | `MANIFEST = Path("/home/user/bench_any/datasets/p5_gold200_roboflow/upload_manifest.json")` |
| `prep/prep_08_load_roboflow_verified.py` | 23 | `DOWNLOAD = Path("/home/user/bench_any/datasets/p5_gold200_roboflow_download")` |
| `prep/prep_08_load_roboflow_verified.py` | 24 | `MANIFEST = Path("/home/user/bench_any/datasets/p5_gold200_roboflow/upload_manifest.json")` |
| `prep/prep_10b_debug_medium.py` | 14 | `TIERS_MAN = "/home/user/bench_any/runs/p5_r1_gold/gold200_tiers.json"` |
| `prep/prep_10c_diag_medium.py` | 24 | `TIERS_MAN = "/home/user/bench_any/runs/p5_r1_gold/gold200_tiers.json"` |
| `prep/prep_10d_per_corpus.py` | 14 | `TIERS_MAN = "/home/user/bench_any/runs/p5_r1_gold/gold200_tiers.json"` |
| `prep/prep_11_filebased.py` | 25 | `target_sizes=[(H,W)]). HF_HOME=/mnt/LinuxBackup/sam3.` |
| `prep/prep_11_filebased.py` | 32 | `os.environ.setdefault("HF_HOME", "/mnt/LinuxBackup/sam3")` |
| `prep/prep_11_filebased.py` | 33 | `os.environ.setdefault("HF_HUB_CACHE", "/mnt/LinuxBackup/sam3/hub")` |
| `prep/prep_11_filebased.py` | 34 | `os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/LinuxBackup/sam3/hub")` |
| `prep/prep_11_generate_preds.py` | 19 | `os.environ.setdefault("HF_HOME", "/mnt/LinuxBackup/sam3")` |
| `prep/prep_11_generate_preds.py` | 20 | `os.environ.setdefault("HF_HUB_CACHE", "/mnt/LinuxBackup/sam3/hub")` |
| `prep/prep_11_generate_preds.py` | 21 | `os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/LinuxBackup/sam3/hub")` |
| `prep/prep_11_generate_preds.py` | 29 | `YOLO_CKPT = "/home/user/bench_any/runs/segment/penguin-seg-combined_yolo11x-seg/weights/best.pt"` |
| `prep/prep_11_generate_preds.py` | 30 | `RTDETR_CKPT = "/home/user/bench_any/runs/detect/penguin-seg-combined_rtdetr-x/weights/best.pt"` |
| `prep/prep_11_run_all.py` | 8 | `os.environ.setdefault("HF_HOME", "/mnt/LinuxBackup/sam3")` |
| `prep/prep_11_run_all.py` | 9 | `os.environ.setdefault("HF_HUB_CACHE", "/mnt/LinuxBackup/sam3/hub")` |
| `prep/prep_11_run_all.py` | 10 | `os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/LinuxBackup/sam3/hub")` |
| `prep/prep_11b_soft_logit.py` | 31 | `os.environ.setdefault("HF_HOME", "/mnt/LinuxBackup/sam3")` |
| `prep/prep_11b_soft_logit.py` | 32 | `os.environ.setdefault("HF_HUB_CACHE", "/mnt/LinuxBackup/sam3/hub")` |
| `prep/prep_11b_soft_logit.py` | 34 | `sys.path.insert(0, "/home/user/bench_any/datasets/prep")` |
| `prep/prep_11b_soft_logit.py` | 47 | `DS_ROOT = "/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin"` |
| `prep/prep_11b_verify_boxprompt.py` | 5 | `os.environ.setdefault("HF_HOME", "/mnt/LinuxBackup/sam3")` |
| `prep/prep_11b_verify_boxprompt.py` | 6 | `os.environ.setdefault("HF_HUB_CACHE", "/mnt/LinuxBackup/sam3/hub")` |
| `prep/prep_11b_verify_boxprompt.py` | 7 | `os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/LinuxBackup/sam3/hub")` |
| `prep/prep_11b_verify_boxprompt.py` | 18 | `RTDETR_CKPT = "/home/user/bench_any/runs/detect/penguin-seg-combined_rtdetr-x/weights/best.pt"` |
| `prep/prep_13_supervised_reeval.py` | 123 | `"images": "/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/boulders-penguin/train/images",` |
| `prep/prep_13_supervised_reeval.py` | 127 | `"images": "/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/sanparks-algoa/train/images",` |
| `prep/prep_13_supervised_reeval.py` | 383 | `sys.path.insert(0, "/home/user/bench_any/eomt")` |
| `prep/prep_13_supervised_reeval.py` | 632 | `DINOv3 (default HF cache) and SAM3 (/mnt/LinuxBackup/sam3) don't collide."""` |
| `prep/prep_13_supervised_reeval.py` | 635 | `cache = "/mnt/LinuxBackup/sam3/hub"` |
| `sam2_backfill_rfdetr.py` | 17 | `"SAM2_CKPT", "/home/user/sam2/checkpoints/sam2.1_hiera_large.pt")` |
| `sam2_backfill_rfdetr.py` | 19 | `"SAM2_CONFIG", "/home/user/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")` |
| `yolo_any_tune.py` | 5 | `# USAGE (Advanced): python3 yolo_any_tune.py --data /mnt/LinuxBackup/FoundationDatasets/oiv7-yolo-allfindrefin` |

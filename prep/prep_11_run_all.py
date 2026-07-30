"""Driver: run all 3 pred rows on p5_gold200 sequentially, log progress to /tmp.
Run via: ~/cvML/bin/python prep_11_run_all.py  (background, no --row arg)
Each row calls the matching row_* function from prep_11_generate_preds.py."""
import sys, time
sys.argv = ["prep_11_run_all.py"]  # clear argparse
import os
os.environ.setdefault("HF_HOME", "/mnt/LinuxBackup/sam3")
os.environ.setdefault("HF_HUB_CACHE", "/mnt/LinuxBackup/sam3/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/LinuxBackup/sam3/hub")
import fiftyone as fo
import prep_11_generate_preds as P

ds = fo.load_dataset("p5_gold200")
t0 = time.time()
print(f"[{time.strftime('%H:%M:%S')}] START all 3 pred rows on {ds.name} (n={len(ds)})", flush=True)

for name, fn in [("yolo11x", P.row_yolo11x), ("sam3", P.row_sam3), ("rtdetr_sam3", P.row_rtdetr_sam3)]:
    print(f"\n[{time.strftime('%H:%M:%S')}] === row {name} ===", flush=True)
    try:
        fn(ds)
        print(f"[{time.strftime('%H:%M:%S')}] row {name} DONE ({time.time()-t0:.0f}s elapsed)", flush=True)
    except Exception as e:
        import traceback
        print(f"[{time.strftime('%H:%M:%S')}] row {name} FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()

print(f"\n[{time.strftime('%H:%M:%S')}] ALL ROWS DONE total {time.time()-t0:.0f}s", flush=True)

# quick summary of field counts
for f in ["yolo11x_pred", "sam3_pred", "rtdetr_sam3_pred"]:
    try:
        n = sum(1 for s in ds.iter_samples() if s.get(f) and s.get(f).detections)
        tot = sum(len(s.get(f).detections) for s in ds.iter_samples() if s.get(f) and s.get(f).detections)
        print(f"  {f}: {n}/200 imgs with dets, {tot} total dets", flush=True)
    except Exception as e:
        print(f"  {f}: summary err {e}", flush=True)

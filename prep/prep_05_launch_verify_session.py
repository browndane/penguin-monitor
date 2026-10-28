# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Launch FiftyOne session for user to verify/fix SAM3 autolabels on p5_gold200_candidate.

Creates a saved view of zero-mask samples, then launches the app on port 5151 and blocks.
Run under ~/cvML (fiftyone 1.19.0).
"""
import os, fiftyone as fo

ds = fo.load_dataset("p5_gold200_candidate")
zero_fps = [s.filepath for s in ds
            if s["ground_truth"] is None or len(getattr(s["ground_truth"], "polylines", [])) == 0]
zero_view = ds.match({"filepath": {"$in": zero_fps}})
ds.save_view("needs_review_zero_mask", zero_view)
print(f"saved view 'needs_review_zero_mask': {len(zero_view)} samples")
print(f"dataset total: {len(ds)} | GT polylines: {ds.count('ground_truth.polylines')}")

session = fo.launch_app(ds, port=5151, auto=False)
print("FiftyOne session running at http://localhost:5151 - dataset 'p5_gold200_candidate'")
print("Saved view 'needs_review_zero_mask' has the zero-mask frames to eyeball.")
session.wait()

# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Upload p5_gold200_roboflow export to two Roboflow projects under anonymous-workspace.

SANParks frames (1_XXXXXXXX (n).jpg) -> sanparks-algoa
Boulders frames (IMG_XXXX.jpg)       -> boulders-penguin

Uses single_upload (returns a dict) so we capture image_id + duplicate/annotation status.
Resumable: skips a basename if already in the manifest with an image id.
Run with the anonymous-workspace key as ROBOFLOW_API_KEY.
"""
import os, json, time, roboflow
from pathlib import Path

IMG = Path("/home/user/bench_any/datasets/p5_gold200_roboflow/images")
MANIFEST = Path("/home/user/bench_any/datasets/p5_gold200_roboflow/upload_manifest.json")
done = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}

rf = roboflow.Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
ws = rf.workspace("anonymous-workspace")
projects = {
    "sanparks": ws.project("sanparks-algoa"),
    "boulders": ws.project("boulders-penguin"),
}

imgs = sorted(IMG.glob("*.jpg"))
print(f"{len(imgs)} images to upload; {sum(1 for b in done if b in [p.name for p in imgs])} already in manifest", flush=True)
n_ok = n_skip = n_err = n_dup = 0
for i, img in enumerate(imgs, 1):
    bn = img.name
    if bn in done and done[bn].get("image_id"):
        n_skip += 1
        continue
    corp = "sanparks" if bn.startswith("1_") else "boulders"
    ann = img.with_suffix(img.suffix + ".json")
    p = projects[corp]
    try:
        res = p.single_upload(image_path=str(img), annotation_path=str(ann),
                              split="train", num_retry_uploads=2, is_prediction=False)
        imgid = (res.get("image") or {}).get("id")
        dup = (res.get("image") or {}).get("duplicate", False)
        annwarn = (res.get("annotation") or {}).get("warn")
        done[bn] = {"project": corp, "image_id": imgid, "duplicate": dup, "ann_warn": annwarn}
        n_ok += 1
        if dup: n_dup += 1
        if i % 10 == 0 or i == len(imgs):
            MANIFEST.write_text(json.dumps(done, indent=2))
            print(f"  [{i}/{len(imgs)}] {bn} -> {corp} id={imgid} dup={dup} (ok={n_ok} skip={n_skip} dup={n_dup} err={n_err})", flush=True)
    except Exception as e:
        n_err += 1
        done[bn] = {"project": corp, "error": f"{type(e).__name__}: {str(e)[:160]}"}
        print(f"  [{i}/{len(imgs)}] {bn} -> {corp} ERR: {type(e).__name__}: {str(e)[:120]}", flush=True)
        MANIFEST.write_text(json.dumps(done, indent=2))
        time.sleep(1)
MANIFEST.write_text(json.dumps(done, indent=2))
print(f"DONE: ok={n_ok} (dup={n_dup}) skip={n_skip} err={n_err} / {len(imgs)}", flush=True)

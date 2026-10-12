"""Stage A: per-pedestrian, per-frame cue extraction for JAAD (runs on a Kaggle GPU, resumable).

For every annotated pedestrian track (ground-truth boxes and IDs) and every frame it is visible:
  bbox        x1, y1, x2, y2 in pixels (ground truth)
  kpts        17 COCO keypoints (x, y, conf) from YOLO11m-pose, matched to the GT box by IoU
  seg         7 scene-context features from SegFormer-B2 (Cityscapes), see SEG_FEATURES
Per frame of each video: ego-vehicle action and traffic annotations (JAAD vehicle/traffic files).
A low-resolution label map per frame is also saved for the demo and for re-deriving seg features.

JAAD behaviour labels (looking, walking, ...) are NOT extracted as inputs: they are human hindsight
labels and would leak. Head orientation is derived from pose instead.

Output: one pickle per video in OUT_DIR, written atomically, so a rerun skips finished videos.
Previous outputs attached as Kaggle inputs (any /kaggle/input/*/cache/*.pkl) are copied in first.
"""
import contextlib
import glob
import io
import json
import os
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np

CFG = dict(
    SMOKE_VIDEOS=[],          # e.g. ["video_0001", "video_0002", "video_0003"]; empty = all videos
    SHARD=0, NUM_SHARDS=1,    # run NUM_SHARDS kernels in parallel; this one takes videos i % NUM_SHARDS == SHARD
    TIME_BUDGET_H=8.0,        # stop cleanly before Kaggle's 9 h limit so outputs are saved
    BATCH=8,
    SEG_MODEL="nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    SEG_W=1024, SEG_H=576,    # 1920x1080 -> 1024x576 (aspect kept); logits come out at 1/4 of this
    POSE_MODEL="yolo11m-pose.pt",
    POSE_IMGSZ=1280,
    POSE_CONF=0.2,
    MATCH_IOU=0.3,
    MAP_DOWNSAMPLE=2,         # saved label map = logits resolution / 2  (128 x 72)
    OUT_DIR="/kaggle/working/cache",
)
CFG.update({"SHARD": 1, "NUM_SHARDS": 2})  # set by kaggle/push_stage_a.py  (the push script replaces this line with CFG.update({...}))

ROAD, SIDEWALK = 0, 1        # Cityscapes train ids
SEG_FEATURES = ["dist_road_h", "dist_sidewalk_h", "on_road", "on_sidewalk",
                "road_frac_patch", "sidewalk_frac_patch", "road_toward_centre"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find_jaad():
    hits = glob.glob("/kaggle/input/**/JAAD/repo/jaad_data.py", recursive=True)
    if not hits:
        sys.exit("JAAD repo not found under /kaggle/input (attach the ped-intent-jaad-download output)")
    repo = os.path.dirname(hits[0])
    clips = glob.glob(os.path.join(os.path.dirname(repo), "**", "video_0001.mp4"), recursive=True)
    if not clips:
        sys.exit("JAAD clips not found next to the repo")
    work = "/kaggle/tmp/jaad"           # jaad_data writes a data_cache, so it needs a writable copy
    if not os.path.exists(work):
        shutil.copytree(repo, work)
        os.symlink(os.path.dirname(clips[0]), os.path.join(work, "JAAD_clips"))
    return work


def load_db(work):
    sys.path.insert(0, work)
    from jaad_data import JAAD
    with contextlib.redirect_stdout(io.StringIO()):
        return JAAD(data_path=work).generate_database()


def iou(a, b):
    """a: (4,), b: (M,4) -> (M,)"""
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def seg_features(lab, dist_road, dist_side, box, sx, sy):
    """lab: label map (h, w) at logits resolution; sx, sy: original px per label-map px."""
    h_map, w_map = lab.shape
    x1, y1, x2, y2 = box
    bh = max(y2 - y1, 1.0)
    fx = int(np.clip((x1 + x2) / 2 / sx, 0, w_map - 1))
    fy = int(np.clip(y2 / sy, 0, h_map - 1))
    win = lab[max(fy - 1, 0):fy + 2, max(fx - 1, 0):fx + 2]
    # patch around the feet: twice the box width, a quarter box height above and below the foot point
    w = x2 - x1
    px1, px2 = int(np.clip((x1 - w / 2) / sx, 0, w_map - 1)), int(np.clip((x2 + w / 2) / sx, 1, w_map))
    py1, py2 = int(np.clip((y2 - bh / 4) / sy, 0, h_map - 1)), int(np.clip((y2 + bh / 4) / sy, 1, h_map))
    patch = lab[py1:max(py2, py1 + 1), px1:max(px2, px1 + 1)]
    cx = w_map // 2
    row = lab[fy, min(fx, cx):max(fx, cx) + 1]
    return np.array([
        dist_road[fy, fx] * sy / bh,                 # distance to nearest road pixel, in box heights
        dist_side[fy, fx] * sy / bh,
        float(np.mean(win == ROAD) > 0.5),
        float(np.mean(win == SIDEWALK) > 0.5),
        float(np.mean(patch == ROAD)),
        float(np.mean(patch == SIDEWALK)),
        float(np.mean(row == ROAD)) if row.size else 0.0,  # road between the person and the ego lane
    ], dtype=np.float32)


def main():
    t0 = time.time()
    subprocess.run("pip install -q ultralytics", shell=True, check=True)
    import cv2
    import torch
    from transformers import SegformerForSemanticSegmentation
    from ultralytics import YOLO

    os.makedirs(CFG["OUT_DIR"], exist_ok=True)
    for p in glob.glob("/kaggle/input/**/cache/*.pkl", recursive=True):
        dst = os.path.join(CFG["OUT_DIR"], os.path.basename(p))
        if not os.path.exists(dst):
            shutil.copy(p, dst)
    work = find_jaad()
    db = load_db(work)
    vids = CFG["SMOKE_VIDEOS"] or sorted(db)
    vids = [v for i, v in enumerate(vids) if i % CFG["NUM_SHARDS"] == CFG["SHARD"]]
    todo = [v for v in vids if not os.path.exists(os.path.join(CFG["OUT_DIR"], v + ".pkl"))]
    log(f"videos: {len(vids)} total, {len(vids) - len(todo)} already done, {len(todo)} to do")

    dev = "cuda"
    seg = SegformerForSemanticSegmentation.from_pretrained(CFG["SEG_MODEL"]).to(dev).half().eval()
    pose = YOLO(CFG["POSE_MODEL"])
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    timings = {}

    for vi, vid in enumerate(todo):
        if time.time() - t0 > CFG["TIME_BUDGET_H"] * 3600:
            log("time budget reached; stopping so outputs are saved. Rerun to resume.")
            break
        tv = time.time()
        v = db[vid]
        nf = v["num_frames"]
        cap = cv2.VideoCapture(os.path.join(work, "JAAD_clips", vid + ".mp4"))
        cap_nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        peds = v["ped_annotations"]
        # frame -> list of (pid, index into that pedestrian's arrays)
        at = {}
        for pid, p in peds.items():
            for j, f in enumerate(p["frames"]):
                at.setdefault(f, []).append((pid, j))
        out_p = {pid: dict(frames=np.asarray(p["frames"], np.int32),
                           bbox=np.asarray(p["bbox"], np.float32),
                           occlusion=np.asarray(p["occlusion"], np.int8),
                           kpts=np.zeros((len(p["frames"]), 17, 3), np.float32),
                           pose_iou=np.zeros(len(p["frames"]), np.float32),
                           seg=np.zeros((len(p["frames"]), len(SEG_FEATURES)), np.float32))
                 for pid, p in peds.items()}
        maps = []
        f = 0
        done = False
        while not done:
            frames = []
            while len(frames) < CFG["BATCH"]:
                ok, img = cap.read()
                if not ok:
                    done = True
                    break
                frames.append(img)
            if not frames:
                break
            with torch.no_grad():
                x = torch.from_numpy(np.stack([cv2.resize(im, (CFG["SEG_W"], CFG["SEG_H"]))[:, :, ::-1]
                                               for im in frames])).to(dev)
                x = ((x.permute(0, 3, 1, 2).float() / 255 - mean) / std).half()
                labs = seg(pixel_values=x).logits.argmax(1).to(torch.uint8).cpu().numpy()
            res = pose.predict(frames, imgsz=CFG["POSE_IMGSZ"], conf=CFG["POSE_CONF"], half=True,
                               classes=[0], verbose=False)
            H, W = frames[0].shape[:2]
            for b, (lab, r) in enumerate(zip(labs, res)):
                fi = f + b
                sx, sy = W / lab.shape[1], H / lab.shape[0]
                dist_road = cv2.distanceTransform((lab != ROAD).astype(np.uint8), cv2.DIST_L2, 3)
                dist_side = cv2.distanceTransform((lab != SIDEWALK).astype(np.uint8), cv2.DIST_L2, 3)
                d = CFG["MAP_DOWNSAMPLE"]
                maps.append(lab[::d, ::d])
                pb = r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.zeros((0, 4))
                pk = r.keypoints.data.cpu().numpy() if len(pb) else np.zeros((0, 17, 3))
                for pid, j in at.get(fi, []):
                    box = out_p[pid]["bbox"][j]
                    out_p[pid]["seg"][j] = seg_features(lab, dist_road, dist_side, box, sx, sy)
                    if len(pb):
                        ious = iou(box, pb)
                        k = int(ious.argmax())
                        if ious[k] >= CFG["MATCH_IOU"]:
                            out_p[pid]["kpts"][j] = pk[k]
                            out_p[pid]["pose_iou"][j] = ious[k]
            f += len(frames)
        cap.release()
        frame_info = dict(
            vehicle=np.array([v["vehicle_annotations"].get(i, -1) for i in range(nf)], np.int8),
            traffic=np.array([[v["traffic_annotations"].get(i, {}).get(k, 0)
                               for k in ("ped_crossing", "ped_sign", "stop_sign", "traffic_light")]
                              for i in range(nf)], np.int8),
            road_type=v["traffic_annotations"].get("road_type", -1),
        )
        rec = dict(vid=vid, num_frames=nf, frames_read=f, cap_frames=cap_nf, width=v["width"],
                   height=v["height"], peds=out_p, frame_info=frame_info,
                   label_maps=np.stack(maps) if maps else None, seg_features=SEG_FEATURES, cfg=CFG)
        tmp = os.path.join(CFG["OUT_DIR"], vid + ".pkl.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(rec, fh, protocol=4)
        os.replace(tmp, tmp[:-4])
        dt = time.time() - tv
        timings[vid] = dict(sec=round(dt, 1), frames=f, fps=round(f / max(dt, 1e-6), 1))
        matched = np.mean([o["pose_iou"].astype(bool).mean() for o in out_p.values()]) if out_p else 0
        log(f"[{vi + 1}/{len(todo)}] {vid}: {f} frames (annot {nf}, container {cap_nf}), "
            f"{len(peds)} peds, pose matched {matched:.0%}, {f / dt:.1f} fps")

    have = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(CFG["OUT_DIR"], "*.pkl")))
    json.dump(dict(done=len(have), wanted=len(vids), complete=set(vids) <= set(have), timings=timings,
                   cfg=CFG, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                   elapsed_h=round((time.time() - t0) / 3600, 2)),
              open(os.path.join(CFG["OUT_DIR"], "..", f"stage_a_manifest_shard{CFG['SHARD']}.json"), "w"), indent=1)
    log(f"done: {len(have)}/{len(vids)} videos cached")


if __name__ == "__main__":
    main()

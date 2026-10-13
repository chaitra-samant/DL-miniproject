"""Stage A: per-frame and per-object features for accident anticipation (resumable, one .npz per video).

For every frame of every video:
  det       (T, N, 8)  x1, y1, x2, y2, conf, class index (see CLASSES), ByteTrack id (-1 if none), valid
  obj_emb   (T, N, D)  DINOv2-small patch tokens averaged inside each box (float16)
  frame_emb (T, D)     DINOv2-small CLS token (float16)
  ctx       (T, N, C)  segmentation-context features, see CTX_FEATURES
  labels    (T, h, w)  SegFormer-B2 (Cityscapes) label map, downsampled (uint8)
N = top-N detections per frame by confidence among road-user classes (19, as in UString / STAGNet).
Detection-only motion features (velocity, scale change) are derived in Stage B from det + track ids.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

CFG = dict(
    YOLO="yolo11m.pt", YOLO_IMGSZ=1280, YOLO_CONF=0.25, TOP_N=19,
    SEG="nvidia/segformer-b2-finetuned-cityscapes-1024-1024", SEG_W=1024, SEG_H=576, LABEL_DOWNSAMPLE=2,
    EMB="facebook/dinov2-small", EMB_W=644, EMB_H=364,  # multiples of the 14-px patch, ~16:9
    BATCH=10, TIME_BUDGET_H=8.0, MAX_FRAMES=0,
)
# COCO ids of road users -> our class index
COCO = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 7: 5}
CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
ROAD, SIDEWALK = 0, 1
CTX_FEATURES = ["road_frac_box", "drivable_overlap_bottom", "foot_on_road", "foot_on_sidewalk",
                "foot_road_signed_dist", "ego_lane_offset", "abs_ego_lane_offset", "bottom_y", "height_frac"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ego_lane_x(lab):
    """x (fraction of width) of the road centre in a band at 65-85% of the image height, middle half of the
    width (above the ego vehicle's hood, which often fills the bottom rows). 0.5 when no road is visible."""
    h, w = lab.shape
    band = lab[int(h * 0.65):int(h * 0.85), int(w * 0.25):int(w * 0.75)] == ROAD
    if band.sum() < 10:
        return 0.5
    return float((np.nonzero(band)[1] + int(w * 0.25)).mean() / w)


def context(lab, d_in, d_out, boxes, W, H, ego_x):
    """Segmentation-context features for boxes (n, 4) in pixels; lab is the label map (h, w)."""
    h, w = lab.shape
    sx, sy = w / W, h / H
    out = np.zeros((len(boxes), len(CTX_FEATURES)), np.float32)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        a, b = int(np.clip(x1 * sx, 0, w - 1)), int(np.clip(np.ceil(x2 * sx), 1, w))
        c, d = int(np.clip(y1 * sy, 0, h - 1)), int(np.clip(np.ceil(y2 * sy), 1, h))
        box = lab[c:max(d, c + 1), a:max(b, a + 1)]
        # the box's bottom edge usually lies on the object's own pixels (wheels, shadow), so ground contact is
        # measured on a strip straddling the bottom edge and at a point just below it
        gap = max(1, int(min(0.03 * h, 0.15 * (d - c))))   # small for distant objects
        bottom = lab[max(d - max((d - c) // 10, 1), 0):min(d + gap, h), a:max(b, a + 1)]
        fx, fy = int(np.clip((x1 + x2) / 2 * sx, 0, w - 1)), int(np.clip(y2 * sy + gap, 0, h - 1))
        win = lab[max(fy - 2, 0):fy + 3, max(fx - 2, 0):fx + 3]
        on_road = float(np.mean(win == ROAD) > 0.3)
        # signed distance to the road boundary, + inside the road, - outside; in units of image height
        dist = (d_in[fy, fx] if on_road else -d_out[fy, fx]) / h
        off = (x1 + x2) / 2 / W - ego_x
        out[i] = [np.mean(box == ROAD), np.mean(bottom == ROAD), on_road, float(np.mean(win == SIDEWALK) > 0.3),
                  np.clip(dist, -1, 1), off, abs(off), y2 / H, (y2 - y1) / H]
    return out


def process_video(path, models, dev, out_path):
    import cv2
    import torch
    yolo, seg, emb, mean, std = models
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, img = cap.read()
        if not ok or (CFG["MAX_FRAMES"] and len(frames) >= CFG["MAX_FRAMES"]):
            break
        frames.append(img)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    T, N = len(frames), CFG["TOP_N"]
    if T == 0:
        return None
    H, W = frames[0].shape[:2]
    det = np.zeros((T, N, 8), np.float32); det[..., 6] = -1
    D = emb.config.hidden_size
    obj_emb = np.zeros((T, N, D), np.float16)
    frame_emb = np.zeros((T, D), np.float16)
    ctx = np.zeros((T, N, len(CTX_FEATURES)), np.float32)
    ds = CFG["LABEL_DOWNSAMPLE"]
    labels = []
    half = dev == "cuda"
    for t, img in enumerate(frames):
        r = yolo.track(img, persist=t > 0,  # new ByteTrack state for each video
                        tracker="bytetrack.yaml", imgsz=CFG["YOLO_IMGSZ"], conf=CFG["YOLO_CONF"],
                       classes=list(COCO), half=half, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            b = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy(); cl = r.boxes.cls.cpu().numpy().astype(int)
            ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else np.full(len(b), -1)
            order = np.argsort(-cf)[:N]
            k = len(order)
            det[t, :k, :4] = b[order]; det[t, :k, 4] = cf[order]
            det[t, :k, 5] = [COCO[c] for c in cl[order]]; det[t, :k, 6] = ids[order]; det[t, :k, 7] = 1
    for s in range(0, T, CFG["BATCH"]):
        batch = frames[s:s + CFG["BATCH"]]
        with torch.no_grad():
            x = torch.from_numpy(np.stack([cv2.resize(im, (CFG["SEG_W"], CFG["SEG_H"]))[:, :, ::-1] for im in batch])).to(dev)
            x = ((x.permute(0, 3, 1, 2).float() / 255 - mean) / std).to(next(seg.parameters()).dtype)
            labs = seg(pixel_values=x).logits.argmax(1).to(torch.uint8).cpu().numpy()
            e = torch.from_numpy(np.stack([cv2.resize(im, (CFG["EMB_W"], CFG["EMB_H"]))[:, :, ::-1] for im in batch])).to(dev)
            e = ((e.permute(0, 3, 1, 2).float() / 255 - mean) / std).to(next(emb.parameters()).dtype)
            hs = emb(pixel_values=e).last_hidden_state.float()                     # (B, 1 + gh*gw, D)
        gh, gw = CFG["EMB_H"] // 14, CFG["EMB_W"] // 14
        cls_tok = hs[:, 0].cpu().numpy()
        patches = hs[:, 1:].reshape(len(batch), gh, gw, D).cpu().numpy()
        for j, lab in enumerate(labs):
            t = s + j
            frame_emb[t] = cls_tok[j]
            d_in = cv2.distanceTransform((lab == ROAD).astype(np.uint8), cv2.DIST_L2, 3)
            d_out = cv2.distanceTransform((lab != ROAD).astype(np.uint8), cv2.DIST_L2, 3)
            ex = ego_lane_x(lab)
            valid = det[t, :, 7] > 0
            if valid.any():
                bx = det[t, valid, :4]
                ctx[t, valid] = context(lab, d_in, d_out, bx, W, H, ex)
                for n, (x1, y1, x2, y2) in zip(np.nonzero(valid)[0], bx):
                    a, b2 = int(np.clip(x1 / W * gw, 0, gw - 1)), int(np.clip(np.ceil(x2 / W * gw), 1, gw))
                    c, d2 = int(np.clip(y1 / H * gh, 0, gh - 1)), int(np.clip(np.ceil(y2 / H * gh), 1, gh))
                    obj_emb[t, n] = patches[j, c:max(d2, c + 1), a:max(b2, a + 1)].mean((0, 1))
            labels.append(lab[::ds, ::ds])
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(tmp, det=det, obj_emb=obj_emb, frame_emb=frame_emb, ctx=ctx.astype(np.float16),
                        labels=np.stack(labels), fps=fps, size=np.array([W, H]), n_frames=T)
    os.replace(tmp, out_path)
    return dict(frames=T, fps=fps, W=W, H=H, mean_objects=float(det[..., 7].sum(1).mean()),
                road_frac=float(np.mean(np.stack(labels) == ROAD)))


def load_models(dev):
    import torch
    from transformers import AutoModel, SegformerForSemanticSegmentation
    from ultralytics import YOLO
    seg = SegformerForSemanticSegmentation.from_pretrained(CFG["SEG"]).to(dev).eval()
    emb = AutoModel.from_pretrained(CFG["EMB"]).to(dev).eval()
    if dev == "cuda":
        seg, emb = seg.half(), emb.half()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    return YOLO(CFG["YOLO"]), seg, emb, mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="text file: one '<video_id> <path>' per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--cfg", default="{}", help="JSON overrides for CFG")
    a = ap.parse_args()
    CFG.update(json.loads(a.cfg))
    import torch
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    items = [l.split(maxsplit=1) for l in open(a.videos) if l.strip()]
    items = [(v, p.strip()) for i, (v, p) in enumerate(items) if i % a.num_shards == a.shard]
    os.makedirs(a.out, exist_ok=True)
    todo = [(v, p) for v, p in items if not os.path.exists(os.path.join(a.out, v + ".npz"))]
    log(f"shard {a.shard}/{a.num_shards}: {len(items)} videos, {len(items) - len(todo)} done, {len(todo)} to do, device {dev}")
    models = load_models(dev)
    stats = {}
    for i, (vid, path) in enumerate(todo):
        if time.time() - t0 > CFG["TIME_BUDGET_H"] * 3600:
            log("time budget reached; stopping cleanly (rerun to resume)")
            break
        ts = time.time()
        try:
            s = process_video(path, models, dev, os.path.join(a.out, vid + ".npz"))
        except Exception as e:  # keep going; the failure is recorded
            s = dict(error=repr(e)[:300])
        if s is not None:
            s["sec"] = round(time.time() - ts, 2)
        stats[vid] = s
        if i < 5 or i % 100 == 0:
            log(f"[{i + 1}/{len(todo)}] {vid}: {s}")
    json.dump(dict(stats=stats, cfg=CFG, elapsed_h=round((time.time() - t0) / 3600, 3)),
              open(os.path.join(a.out, f"_extract_log_shard{a.shard}.json"), "w"), indent=1)
    done = sum(os.path.exists(os.path.join(a.out, v + ".npz")) for v, _ in items)
    log(f"shard {a.shard}: {done}/{len(items)} videos have features")


if __name__ == "__main__":
    main()

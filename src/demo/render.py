"""End-to-end demo clips for mined events.

Pipeline per clip (no ground-truth boxes): YOLO11m-pose + ByteTrack -> tracked pedestrians with
keypoints; SegFormer -> scene-context features; JAAD vehicle/traffic annotations -> ego cues (the
demo has no ego-motion sensor). Every tracked person gets P(cross) per frame from the trained
models (mean over seeds). The target is the track that best overlaps the annotated pedestrian; if
the tracker misses them, the clip falls back to the annotated box and says so on screen.

Each clip: boxes coloured by P(cross), a fan of K predicted future paths for the target, a
probability timeline, and a freeze about 1 s before the event with "naive vs ours" panels and a
cue-removal bar chart. Also: a counterfactual side-by-side (all cues vs no pose) and a 4-frame
storyboard PNG per clip. Videos are H.264 / yuv420p so they play in browsers.
"""
import glob
import json
import os
import subprocess

import cv2
import numpy as np
import torch

from src.data.samples import track_features
from src.features.stage_a import ROAD, SEG_FEATURES, iou, seg_features
from src.models.nets import OFFSET_SCALE, CueNet

OUT_W, OUT_H, TL_H = 960, 540, 140
FREEZE_FRAMES = 75


# ---------------------------------------------------------------- models
def load_models(runs_dir, exp, dev):
    ms = []
    for f in sorted(glob.glob(os.path.join(runs_dir, f"{exp}_s*", "model.pt"))):
        ck = torch.load(f, map_location=dev, weights_only=False)
        s = ck["spec"]
        m = CueNet(ck["dims"], s["groups"], rnn=s.get("rnn", "lstm"), fusion=s.get("fusion", "gated"),
                   K=s.get("K", 1), dec_cond=s.get("dec_cond", False)).to(dev).eval()
        m.load_state_dict(ck["state"])
        ms.append((m, ck["norm"], sorted(set(s["groups"]) | {"ego"})))
    return ms


def run_models(ms, feats, dev, traj=False):
    """feats: dict group -> (N, T, d). Returns mean P over seeds, and seed-0 trajectory output."""
    ps, tr = [], None
    with torch.no_grad():
        for i, (m, st, gs) in enumerate(ms):
            x = {g: torch.from_numpy((feats[g] - st[g][0]) / st[g][1]).float().to(dev) for g in gs}
            o = m(x, traj=traj and i == 0)
            ps.append(torch.sigmoid(o["logit"]).cpu().numpy())
            if traj and i == 0:
                tr = (o["traj"].cpu().numpy(), torch.softmax(o["mode_logits"], -1).cpu().numpy())
    return np.mean(ps, 0), tr


# ---------------------------------------------------------------- drawing helpers
def p_color(p):  # green (unlikely to cross) -> red (likely to cross), BGR
    return (int(60 + 40 * (1 - p)), int(200 * (1 - p)), int(230 * p + 20))


def put(img, text, org, scale=0.55, color=(255, 255, 255), thick=1, bg=True):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    if bg:
        cv2.rectangle(img, (org[0] - 3, org[1] - h - 4), (org[0] + w + 3, org[1] + 4), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def timeline(frames, pm, pb, cur, event, w=OUT_W, h=TL_H, labels=("ours", "bbox-only")):
    img = np.full((h, w, 3), 24, np.uint8)
    x0, x1, y0, y1 = 50, w - 15, 12, h - 28
    X = lambda f: int(x0 + (f - frames[0]) / max(frames[-1] - frames[0], 1) * (x1 - x0))
    Y = lambda p: int(y1 - p * (y1 - y0))
    cv2.line(img, (x0, Y(0.5)), (x1, Y(0.5)), (90, 90, 90), 1)
    for p in (0, 0.5, 1):
        put(img, f"{p:.1f}", (8, Y(p) + 4), 0.4, bg=False)
    for series, col in ((pb, (170, 170, 170)), (pm, (0, 165, 255))):
        pts = [(X(f), Y(p)) for f, p in zip(frames, series) if p is not None and not np.isnan(p) and f <= cur]
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts)], False, col, 2, cv2.LINE_AA)
    if event is not None and frames[0] <= event <= frames[-1]:
        cv2.line(img, (X(event), y0), (X(event), y1), (80, 80, 255), 1)
        put(img, "crossing starts", (min(X(event) + 4, w - 130), y0 + 12), 0.4, (120, 120, 255), bg=False)
    cv2.line(img, (X(cur), y0), (X(cur), y1), (255, 255, 255), 1)
    put(img, f"P(cross): {labels[0]} (orange) vs {labels[1]} (grey)", (x0, h - 8), 0.45, bg=False)
    return img


def cue_bars(deltas, w=420, h=190):
    img = np.full((h, w, 3), 30, np.uint8)
    put(img, "why: change in P(cross) when a cue is removed", (8, 20), 0.45, bg=False)
    cx = w // 2 + 40
    for i, (name, d) in enumerate(deltas.items()):
        y = 45 + i * 32
        put(img, name, (8, y + 12), 0.5, bg=False)
        L = int(np.clip(d, -1, 1) * (w - cx - 15) / 0.5) if not np.isnan(d) else 0
        L = int(np.clip(L, -(cx - 110), w - cx - 15))
        cv2.rectangle(img, (cx, y), (cx + L, y + 18), (0, 165, 255) if d >= 0 else (200, 120, 60), -1)
        put(img, f"{d:+.2f}", (cx + L + (6 if L >= 0 else -52), y + 14), 0.42, bg=False)
    cv2.line(img, (cx, 38), (cx, h - 10), (200, 200, 200), 1)
    put(img, "shows model reliance, not the person's reason", (8, h - 6), 0.38, (170, 170, 170), bg=False)
    return img


# ---------------------------------------------------------------- per-clip processing
def frame_info_from_db(v):
    nf = v["num_frames"]
    return dict(vehicle=np.array([v["vehicle_annotations"].get(i, -1) for i in range(nf)], np.int8),
                traffic=np.array([[v["traffic_annotations"].get(i, {}).get(k, 0)
                                   for k in ("ped_crossing", "ped_sign", "stop_sign", "traffic_light")]
                                  for i in range(nf)], np.int8))


def analyse_clip(clip, db, clips_dir, pose_path, seg_model, dev, seg_size=(1024, 576)):
    """Read frames, track people, compute per-track cue arrays. Returns frames and a fake cache record."""
    from ultralytics import YOLO
    v = db[clip["vid"]]
    ped = v["ped_annotations"][clip["pid"]]
    t_first, t_last = ped["frames"][0], ped["frames"][-1]
    event = clip["event_frame"] if clip["y"] == 1 else None
    start = max(t_first, clip["first_frame"] - 45)
    end = min(t_last, (event if event is not None else clip["last_frame"]) + 45, v["num_frames"] - 1)
    cap = cv2.VideoCapture(os.path.join(clips_dir, clip["vid"] + ".mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    tracker = YOLO(pose_path)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    frames, tracks = [], {}
    gt = dict(frames=[], bbox=[], kpts=[], pose_iou=[], seg=[])
    for f in range(start, end + 1):
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
        H, W = img.shape[:2]
        with torch.no_grad():
            x = torch.from_numpy(cv2.resize(img, seg_size)[:, :, ::-1].copy()).to(dev)[None]
            x = ((x.permute(0, 3, 1, 2).float() / 255 - mean) / std).to(next(seg_model.parameters()).dtype)
            lab = seg_model(pixel_values=x).logits.argmax(1)[0].to(torch.uint8).cpu().numpy()
        sx, sy = W / lab.shape[1], H / lab.shape[0]
        dr = cv2.distanceTransform((lab != ROAD).astype(np.uint8), cv2.DIST_L2, 3)
        ds = cv2.distanceTransform((lab != 1).astype(np.uint8), cv2.DIST_L2, 3)
        r = tracker.track(img, persist=True, tracker="bytetrack.yaml", classes=[0], conf=0.2, imgsz=1280,
                          half=dev == "cuda", verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.zeros((0, 4))
        ids = r.boxes.id.cpu().numpy().astype(int) if len(boxes) and r.boxes.id is not None else np.full(len(boxes), -1)
        kp = r.keypoints.data.cpu().numpy() if len(boxes) else np.zeros((0, 17, 3))
        for b, tid, k in zip(boxes, ids, kp):
            if tid < 0:
                continue
            t = tracks.setdefault(int(tid), dict(frames=[], bbox=[], kpts=[], pose_iou=[], seg=[]))
            t["frames"].append(f); t["bbox"].append(b); t["kpts"].append(k); t["pose_iou"].append(1.0)
            t["seg"].append(seg_features(lab, dr, ds, b, sx, sy))
        if f in ped["frames"]:   # annotated target, for matching and the fallback
            gb = np.asarray(ped["bbox"][ped["frames"].index(f)], np.float32)
            j = int(iou(gb, boxes).argmax()) if len(boxes) else -1
            ok_pose = j >= 0 and iou(gb, boxes)[j] >= 0.3
            gt["frames"].append(f); gt["bbox"].append(gb)
            gt["kpts"].append(kp[j] if ok_pose else np.zeros((17, 3), np.float32))
            gt["pose_iou"].append(1.0 if ok_pose else 0.0)
            gt["seg"].append(seg_features(lab, dr, ds, gb, sx, sy))
    cap.release()
    to_np = lambda t: {k: np.asarray(v, np.float32 if k != "frames" else np.int32) for k, v in t.items()}
    peds = {f"t{k}": to_np(t) for k, t in tracks.items() if len(t["frames"]) >= 16}
    # choose the target track: highest mean IoU with the annotated box over shared frames
    best, best_iou = None, 0.0
    for name, t in peds.items():
        common = [(i, gt["frames"].index(f)) for i, f in enumerate(t["frames"]) if f in gt["frames"]]
        if common:
            m = np.mean([iou(t["bbox"][i], np.asarray([gt["bbox"][j]]))[0] for i, j in common])
            if m > best_iou:
                best, best_iou = name, m
    fallback = best is None or best_iou < 0.4
    if fallback:
        peds["target"] = to_np(gt)
        best = "target"
    rec = dict(peds=peds, width=W, height=H, frame_info=frame_info_from_db(v))
    return dict(frames=frames, start=start, rec=rec, target=best, fallback=fallback, event=event,
                freeze=max(start + 16, (event - 30) if event is not None else clip["last_frame"]))


def per_frame_probs(rec, models, dev, obs=16):
    """P(cross) per track per frame (window = that track's last `obs` detections)."""
    centers = {}
    for pid, p in rec["peds"].items():
        for f, b in zip(p["frames"], p["bbox"]):
            centers.setdefault(int(f), []).append((pid, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)))
    out, feats_all = {}, {}
    for pid, p in rec["peds"].items():
        feats = track_features(rec, pid, centers)
        feats_all[pid] = feats
        n = len(p["frames"])
        if n < obs:
            continue
        idx = np.arange(obs, n + 1)
        win = {g: np.stack([feats[g][i - obs:i] for i in idx]) for g in feats}
        out[pid] = {name: dict(zip(p["frames"][idx - 1].tolist(), run_models(ms, win, dev)[0].tolist()))
                    for name, ms in models.items() if ms}
    return out, feats_all


def render(clip, A, models, dev, out_path, story_path, prob_key="main", base_key="bbox_lstm",
           cf_key=None, explain_groups=("pose", "seg", "ego", "social")):
    rec, tgt, frames = A["rec"], A["target"], A["frames"]
    fr_ids = list(range(A["start"], A["start"] + len(frames)))
    probs, feats = per_frame_probs(rec, models, dev)
    tp = probs.get(tgt, {})
    pm = [tp.get(prob_key, {}).get(f) for f in fr_ids]
    pb = [tp.get(base_key, {}).get(f) for f in fr_ids]
    tframes = list(rec["peds"][tgt]["frames"])
    freeze = min(max(A["freeze"], tframes[min(15, len(tframes) - 1)]), tframes[-1])
    # explanation at the freeze frame
    j = tframes.index(freeze) if freeze in tframes else len(tframes) - 1
    win = {g: feats[tgt][g][max(0, j - 15):j + 1][None] for g in feats[tgt]}
    deltas, fan = {}, None
    if win["bbox"].shape[1] == 16:
        p_main, tr = run_models(models[prob_key], win, dev, traj=True)
        for g in explain_groups:
            ms = models.get(f"main_no_{g}")
            if ms:
                deltas[g] = float(p_main[0] - run_models(ms, win, dev)[0][0])
        if tr is not None:
            boxes, mp = tr
            W, H = rec["width"], rec["height"]
            last = rec["peds"][tgt]["bbox"][j]
            fan = (last + boxes[0] / OFFSET_SCALE * np.array([W, H, W, H]), mp[0], freeze)
    pmf = tp.get(prob_key, {}).get(freeze, np.nan)
    pbf = tp.get(base_key, {}).get(freeze, np.nan)

    sx = OUT_W / rec["width"]
    writer = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                               "-s", f"{OUT_W}x{OUT_H + TL_H}", "-r", "30", "-i", "-", "-c:v", "libx264",
                               "-pix_fmt", "yuv420p", "-crf", "30", "-preset", "veryfast", "-movflags",
                               "+faststart", out_path], stdin=subprocess.PIPE)
    story = {}
    title = ("SURPRISE: bbox-only model was confidently wrong" if clip["kind"] == "success"
             else "FAILURE CASE: our model is wrong here")
    after_k = (len(frames) + fr_ids.index(freeze)) // 2 if freeze in fr_ids else len(frames) - 1
    for k, (f, img) in enumerate(zip(fr_ids, frames)):
        im = cv2.resize(img, (OUT_W, OUT_H))
        for pid, p in rec["peds"].items():
            if f not in p["frames"]:
                continue
            b = (p["bbox"][list(p["frames"]).index(f)] * sx).astype(int)
            pr = probs.get(pid, {}).get(prob_key, {}).get(f)
            col = p_color(pr) if pr is not None else (160, 160, 160)
            cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), col, 4 if pid == tgt else 2)
            if pr is not None:
                put(im, f"{pr:.0%}", (b[0], max(b[1] - 6, 14)), 0.5, col)
        if fan is not None and f >= fan[2]:
            fb, mp, _ = fan
            for kk in range(len(fb)):
                c = ((fb[kk][:, 0] + fb[kk][:, 2]) / 2 * sx, (fb[kk][:, 1] + fb[kk][:, 3]) / 2 * sx)
                pts = np.stack(c, 1).astype(int)
                cv2.polylines(im, [pts], False, (255, 255, 0), max(1, int(1 + 4 * mp[kk])), cv2.LINE_AA)
        put(im, title, (10, 24), 0.6)
        put(im, f"{clip['vid']}  detector+tracker: YOLO11m-pose + ByteTrack"
            + ("  (tracker lost target: annotated box used)" if A["fallback"] else ""), (10, 50), 0.45)
        if A["event"] is not None and f >= A["event"]:
            put(im, "ground truth: crossing", (10, OUT_H - 14), 0.55, (120, 120, 255))
        tl = timeline(fr_ids, pm, pb, f, A["event"])
        frame = np.vstack([im, tl])
        writer.stdin.write(frame.tobytes())
        if k == 15:
            story["start"] = frame
        if f == freeze:
            panel = frame.copy()
            ov = panel[70:70 + 200, 10:OUT_W - 10]
            panel[70:70 + 200, 10:OUT_W - 10] = (ov * 0.25).astype(np.uint8)
            verdict = lambda p: "will cross" if p >= 0.5 else "will not cross"
            put(panel, "Naive predictor (bbox-only LSTM) expects:", (20, 100), 0.55, bg=False)
            put(panel, f"{pbf:.0%} crossing -> {verdict(pbf)}", (20, 130), 0.75, (190, 190, 190), 2, bg=False)
            put(panel, "Our model (pose + scene + ego + social cues):", (20, 175), 0.55, bg=False)
            put(panel, f"{pmf:.0%} crossing -> {verdict(pmf)}", (20, 205), 0.75, (0, 165, 255), 2, bg=False)
            if deltas:
                bars = cue_bars(deltas)
                panel[75:75 + bars.shape[0], OUT_W - 15 - bars.shape[1]:OUT_W - 15] = bars
            put(panel, "about 1 s before the event (paused)", (20, 255), 0.45, (200, 200, 200), bg=False)
            for _ in range(FREEZE_FRAMES):
                writer.stdin.write(panel.tobytes())
            story["freeze"] = panel
        if k == after_k:
            story["after"] = frame
        if k == len(frames) // 4:
            story.setdefault("mid", frame)
    writer.stdin.close(); writer.wait()
    keys = [k for k in ("start", "mid", "freeze", "after") if k in story]
    tiles = [cv2.resize(story[k], (OUT_W // 2, (OUT_H + TL_H) // 2)) for k in keys]
    while len(tiles) < 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(story_path, np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    return dict(p_main_freeze=float(pmf), p_base_freeze=float(pbf), deltas=deltas, fallback=A["fallback"],
                frames=len(frames), freeze_frame=int(freeze))


def render_counterfactual(clip, A, models, dev, out_path, removed="pose"):
    """Side by side: all cues vs the model trained without one cue group, same tracked inputs."""
    rec, tgt, frames = A["rec"], A["target"], A["frames"]
    fr_ids = list(range(A["start"], A["start"] + len(frames)))
    probs, _ = per_frame_probs(rec, {"main": models["main"], "cf": models[f"main_no_{removed}"]}, dev)
    tp = probs.get(tgt, {})
    a = [tp.get("main", {}).get(f) for f in fr_ids]
    b = [tp.get("cf", {}).get(f) for f in fr_ids]
    w, h = 640, 360
    sx = w / rec["width"]
    writer = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                               "-s", f"{2 * w}x{h + TL_H}", "-r", "30", "-i", "-", "-c:v", "libx264",
                               "-pix_fmt", "yuv420p", "-crf", "30", "-preset", "veryfast", "-movflags",
                               "+faststart", out_path], stdin=subprocess.PIPE)
    for f, img in zip(fr_ids, frames):
        halves = []
        for key, label, series in (("main", "all cues", a), ("cf", f"without {removed}", b)):
            im = cv2.resize(img, (w, h))
            p = rec["peds"][tgt]
            if f in p["frames"]:
                bb = (p["bbox"][list(p["frames"]).index(f)] * sx).astype(int)
                pr = tp.get(key, {}).get(f)
                col = p_color(pr) if pr is not None else (160, 160, 160)
                cv2.rectangle(im, tuple(bb[:2]), tuple(bb[2:]), col, 3)
                if pr is not None:
                    put(im, f"{pr:.0%}", (bb[0], max(bb[1] - 6, 14)), 0.5, col)
            put(im, label, (10, 24), 0.65)
            tl = timeline(fr_ids, series, [None] * len(fr_ids), f, A["event"], w=w, labels=(label, "-"))
            halves.append(np.vstack([im, tl]))
        writer.stdin.write(np.hstack(halves).tobytes())
    writer.stdin.close(); writer.wait()


def run_demo(selection, db, clips_dir, runs_dir, out_dir, dev, seg_name, pose_path, explain_groups, log=print):
    from transformers import SegformerForSemanticSegmentation
    os.makedirs(out_dir, exist_ok=True)
    seg = SegformerForSemanticSegmentation.from_pretrained(seg_name).to(dev).eval()
    if dev == "cuda":
        seg = seg.half()
    names = ["main", "bbox_lstm"] + [f"main_no_{g}" for g in explain_groups]
    models = {n: load_models(runs_dir, n, dev) for n in names}
    results = []
    for i, clip in enumerate(selection):
        tag = f"clip{i + 1}_{clip['kind']}_{clip['vid']}_{clip['pid']}"
        A = analyse_clip(clip, db, clips_dir, pose_path, seg, dev)
        info = render(clip, A, models, dev, os.path.join(out_dir, tag + ".mp4"),
                      os.path.join(out_dir, tag + "_storyboard.png"), explain_groups=explain_groups)
        info.update(tag=tag, **{k: clip[k] for k in ("vid", "pid", "y", "p_base", "p_main", "kind")})
        if i == 0 and models.get("main_no_pose"):
            render_counterfactual(clip, A, models, dev, os.path.join(out_dir, tag + "_counterfactual_no_pose.mp4"))
        results.append(info)
        log(f"  rendered {tag}: freeze P ours {info['p_main_freeze']:.2f} vs bbox-only {info['p_base_freeze']:.2f}"
            f"{' (fallback to annotated box)' if info['fallback'] else ''}")
    json.dump(results, open(os.path.join(out_dir, "demo_info.json"), "w"), indent=1, default=float)
    return results

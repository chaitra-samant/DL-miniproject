"""Demo clips for the selected videos (results/tables/demo_selection.json), from cached outputs only:
Stage A detections + road masks, and the saved per-frame risk and object attention of `main` and `det_graph`
(seed 0, the split in which the video was a test video).

Each clip is a side-by-side: left = detection-only baseline (det_graph), right = main model (+ segmentation
context). Per side: translucent road mask, boxes coloured by the model's attention weight (which object the
risk pooling attended to), risk-over-time curve with the 0.5 threshold and the accident marker, and a
"warned X s before the accident" caption. A 4-frame storyboard PNG is written per clip.
Attention shows what the model relied on, not what caused the crash.
"""
import argparse
import glob
import json
import os
import subprocess

import cv2
import numpy as np

W1, H1, TL = 640, 360, 150      # one panel


def find_preds(runs, exp, scheme, split, vid):
    f = os.path.join(runs, exp, scheme, f"{split}_s0", "preds.npz")
    z = np.load(f, allow_pickle=True)
    i = int(np.where(z["vid"] == vid)[0][0])
    return z["p"][i].astype(float), (z["attn"][i].astype(float) if "attn" in z.files else None), int(z["toa"][i]), int(z["y"][i])


def put(img, text, org, scale=0.5, color=(255, 255, 255), thick=1):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img, (org[0] - 3, org[1] - h - 4), (org[0] + w + 3, org[1] + 4), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def curve(p, t, toa, y, th=0.5, label=""):
    img = np.full((TL, W1, 3), 22, np.uint8)
    x0, x1, y0, y1 = 40, W1 - 12, 14, TL - 26
    X = lambda f: int(x0 + f / (len(p) - 1) * (x1 - x0))
    Y = lambda v: int(y1 - v * (y1 - y0))
    cv2.line(img, (x0, Y(th)), (x1, Y(th)), (80, 80, 200), 1)
    put(img, f"threshold {th}", (x1 - 110, Y(th) - 4), 0.38, (150, 150, 230))
    for v in (0, 1):
        cv2.putText(img, str(v), (12, Y(v) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    if y > 0:
        cv2.line(img, (X(toa), y0), (X(toa), y1), (0, 0, 255), 2)
        put(img, "accident", (min(X(toa) + 4, W1 - 80), y0 + 12), 0.4, (120, 120, 255))
    pts = np.array([(X(f), Y(v)) for f, v in enumerate(p[:t + 1])])
    if len(pts) > 1:
        cv2.polylines(img, [pts], False, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.line(img, (X(t), y0), (X(t), y1), (255, 255, 255), 1)
    put(img, f"P(accident) - {label}", (x0, TL - 8), 0.45)
    return img


def panel(img, det, lab, attn, p, t, toa, y, title, warn_text):
    im = cv2.resize(img, (W1, H1))
    road = cv2.resize((lab == 0).astype(np.uint8), (W1, H1), interpolation=cv2.INTER_NEAREST).astype(bool)
    im[road] = (0.65 * im[road] + 0.35 * np.array([180, 60, 200])).astype(np.uint8)
    H, Wd = img.shape[:2]
    sx, sy = W1 / Wd, H1 / H
    a = attn[t] if attn is not None else np.zeros(det.shape[1])
    amax = max(a.max(), 1e-6)
    for n in np.where(det[t, :, 7] > 0)[0]:
        x1, y1, x2, y2 = det[t, n, :4]
        w = a[n] / amax
        col = (int(255 * (1 - w)), int(255 * (1 - w)), 255) if w > 0.05 else (200, 200, 200)
        cv2.rectangle(im, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), col, 1 + int(3 * w))
    put(im, title, (8, 20), 0.55)
    put(im, f"risk {p[t]:.2f}", (8, 44), 0.5, (0, 200, 255))
    if warn_text:
        put(im, warn_text, (8, H1 - 12), 0.5, (120, 255, 120) if "before" in warn_text else (120, 120, 255))
    return np.vstack([im, curve(p, t, toa, y, label=title)])


def warn_caption(p, toa, y, th=0.5, fps=10.0):
    if y > 0:
        hit = np.where(p[:toa] >= th)[0]
        return f"warned {(toa - hit[0]) / fps:.1f} s before the accident" if len(hit) else "no warning before the accident"
    hit = np.where(p >= th)[0]
    return f"FALSE ALARM at {hit[0] / fps:.1f} s" if len(hit) else "no alarm (correct: no accident)"


def render(clip, video_path, feat_path, runs, out_dir):
    z = np.load(feat_path)
    det, labels = z["det"], z["labels"]
    pm, am, toa, y = find_preds(runs, "main", clip["scheme"], clip["split"], clip["vid"])
    pb, ab, _, _ = find_preds(runs, "det_graph", clip["scheme"], clip["split"], clip["vid"])
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    T = min(len(frames), len(pm))
    tag = f"{clip['kind'].replace(' ', '_')}_{clip['vid']}"
    out = os.path.join(out_dir, tag + ".mp4")
    w, h = 2 * W1, H1 + TL + 40
    wr = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                           "-r", "10", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30", "-preset", "veryfast",
                           "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    cb, cm = warn_caption(pb, toa, y), warn_caption(pm, toa, y)
    header = np.full((40, w, 3), 0, np.uint8)
    put(header, f"{clip['vid']}  |  {clip['kind']}  |  boxes: brighter red = more attention  |  purple: road (SegFormer)", (8, 26), 0.5)
    keep = {}
    if y > 0:   # early, 1 s before the accident, 0.3 s before it, end
        story_t = sorted({int(T * 0.2), max(0, toa - 10), max(0, toa - 3), T - 1})
    else:
        story_t = [int(T * 0.25), int(T * 0.5), int(T * 0.75), T - 1]
    for t in range(T):
        lab = labels[min(t, len(labels) - 1)]
        left = panel(frames[t], det, lab, ab, pb, t, toa, y, "detection-only (det_graph)", cb if t >= T - 1 or (y and t >= toa) else "")
        right = panel(frames[t], det, lab, am, pm, t, toa, y, "main (+ road context)", cm if t >= T - 1 or (y and t >= toa) else "")
        fr = np.vstack([header, np.hstack([left, right])])
        wr.stdin.write(fr.tobytes())
        if t in story_t:
            keep[t] = fr
    # hold the last frame for 2 s so the captions can be read
    for _ in range(20):
        wr.stdin.write(fr.tobytes())
    wr.stdin.close(); wr.wait()
    tiles = [cv2.resize(keep[t], (w // 2, h // 2)) for t in story_t if t in keep]
    while len(tiles) < 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(os.path.join(out_dir, tag + "_storyboard.png"), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    return dict(tag=tag, main=cm, baseline=cb, toa=toa, y=y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True); ap.add_argument("--runs", required=True)
    ap.add_argument("--feat_dir", required=True); ap.add_argument("--video_list", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    paths = dict(l.split(maxsplit=1) for l in open(a.video_list) if l.strip())
    info = []
    for c in json.load(open(a.selection))["clips"]:
        info.append(render(c, paths[c["vid"]].strip(), os.path.join(a.feat_dir, c["vid"] + ".npz"), a.runs, a.out))
        print(info[-1], flush=True)
    json.dump(info, open(os.path.join(a.out, "demo_info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

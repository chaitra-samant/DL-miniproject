"""Sanity checks on Stage A pickles: frame counts, pose matching, seg feature ranges, one rendered map.

usage: python src/features/check_cache.py CACHE_DIR [OUT_PNG_DIR]
"""
import glob
import os
import pickle
import sys

import numpy as np

CITYSCAPES_COLORS = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
    [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]], np.uint8)


def main(cache, out_dir=None):
    files = sorted(glob.glob(os.path.join(cache, "*.pkl")))
    print(f"{len(files)} cached videos in {cache}")
    for fp in files:
        r = pickle.load(open(fp, "rb"))
        peds = r["peds"]
        n_obs = sum(len(p["frames"]) for p in peds.values())
        matched = sum(int((p["pose_iou"] > 0).sum()) for p in peds.values())
        seg = np.concatenate([p["seg"] for p in peds.values()]) if peds else np.zeros((0, 7))
        print(f"\n{r['vid']}: frames read {r['frames_read']} / annotated {r['num_frames']} / container "
              f"{r['cap_frames']}  {r['width']}x{r['height']}  peds {len(peds)}  ped-frames {n_obs}")
        print(f"  pose matched {matched}/{n_obs} ({matched / max(n_obs, 1):.0%}); "
              f"label maps {None if r['label_maps'] is None else r['label_maps'].shape}")
        for name, col in zip(r["seg_features"], seg.T):
            print(f"  {name:22s} min {col.min():6.2f}  mean {col.mean():6.2f}  max {col.max():6.2f}")
        fi = r["frame_info"]
        print(f"  vehicle codes {np.unique(fi['vehicle']).tolist()}  ped_crossing frames "
              f"{int(fi['traffic'][:, 0].sum())}  road_type {fi['road_type']}")
        if out_dir and r["label_maps"] is not None:
            import cv2
            os.makedirs(out_dir, exist_ok=True)
            k = len(r["label_maps"]) // 2
            img = CITYSCAPES_COLORS[np.clip(r["label_maps"][k], 0, 18)][:, :, ::-1]
            img = cv2.resize(img, (r["width"] // 2, r["height"] // 2), interpolation=cv2.INTER_NEAREST)
            for pid, p in peds.items():
                hit = np.where(p["frames"] == k)[0]
                if not len(hit):
                    continue
                j = hit[0]
                x1, y1, x2, y2 = (p["bbox"][j] / 2).astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(img, f"{pid} road_d={p['seg'][j][0]:.1f}", (x1, max(y1 - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                for x, y, c in p["kpts"][j]:
                    if c > 0.3:
                        cv2.circle(img, (int(x / 2), int(y / 2)), 3, (0, 255, 255), -1)
            path = os.path.join(out_dir, f"{r['vid']}_frame{k}.png")
            cv2.imwrite(path, img)
            print(f"  wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

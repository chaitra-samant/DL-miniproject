# Official CCD (github.com/Cogito2012/CarCrashDataset) from its Google Drive folder:
# videos/Crash-1500.zip, videos/Normal.zip, videos/Crash-1500.txt, vgg16_features/{train,test}.txt.
# Counts videos and frames from the files themselves and writes MANIFEST.json.
import json, os, subprocess, zipfile, glob, time
import cv2
subprocess.run("pip install -q gdown", shell=True, check=True)
import gdown
root = "/kaggle/working/CCD"
os.makedirs(root, exist_ok=True)
files = {"Crash-1500.txt": "13OgrD0-8cKG0X00MlA6JXr0G_JJmHYXg", "train.txt": "1mnsL-TnWQlQ4cfgViiLZe2Y9ycpF8IeU",
         "test.txt": "1FGsR58h_0kfOgbecvOhpcXF-9tWIjsG3", "ytb_list.txt": "1xETn9viyjaEt4giYXZQI9Vs-oxEIJv4h",
         "Crash-1500.zip": "1fmcwGhr8JT9YfLUrlcvuCZ3ychi2eyFP", "Normal.zip": "11ErpWQmmV5au2JOQVxwYl3ebtuugdlan"}
log = {}
for name, fid in files.items():
    dst = os.path.join("/kaggle/tmp" if name.endswith(".zip") else root, name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    t = time.time()
    try:
        gdown.download(id=fid, output=dst, quiet=True)
        log[name] = dict(ok=os.path.exists(dst), bytes=os.path.getsize(dst) if os.path.exists(dst) else 0, sec=round(time.time() - t))
    except Exception as e:
        log[name] = dict(ok=False, error=repr(e)[:300])
    print(name, log[name], flush=True)
    if name.endswith(".zip") and log[name].get("ok"):
        # each zip into its own folder: both archives use the same file names (000001.mp4, ...)
        with zipfile.ZipFile(dst) as z:
            print(name, "first entries:", z.namelist()[:3], "count:", len(z.namelist()), flush=True)
            z.extractall(os.path.join(root, "videos", name[:-4]))
        os.remove(dst)
vids = sorted(glob.glob(os.path.join(root, "videos", "**", "*.mp4"), recursive=True))
counts = {}
for v in vids:
    cap = cv2.VideoCapture(v)
    counts[os.path.relpath(v, root)] = dict(frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps=cap.get(cv2.CAP_PROP_FPS),
                                           w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
dirs = {}
for k in counts:
    dirs.setdefault(os.path.dirname(k), 0); dirs[os.path.dirname(k)] += 1
json.dump(dict(downloads=log, n_videos=len(vids), per_folder=dirs, videos=counts), open(os.path.join(root, "MANIFEST.json"), "w"), indent=1)
print("videos:", len(vids), dirs)

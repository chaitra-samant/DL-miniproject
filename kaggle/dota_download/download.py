# STAGNet's released DoTA clip set ("DoTA Videos.rar", github.com/Vipooshan1998/STAGNet, Drive folder
# 1twxueRZRn2Pos2QZRa_7bzrImcebKB0c), built from DoTA (Yao et al., TPAMI 2023) by their
# generate_videos_for_stagnet.py. Also fetches DoTA's official annotations for source-video grouping.
import json, os, subprocess, glob, time, shutil
import cv2
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True)
print(sh("pip install -q gdown rarfile").stderr[-300:])
import gdown
root = "/kaggle/working/DoTA_STAGNet"; os.makedirs(root, exist_ok=True)
rar = "/kaggle/tmp/dota_videos.rar"; os.makedirs("/kaggle/tmp", exist_ok=True)
t = time.time(); gdown.download(id="1BeaMNudLXPf51mp6M3go95g1kC4xJUDW", output=rar, quiet=True)
log = dict(rar_bytes=os.path.getsize(rar) if os.path.exists(rar) else 0, download_sec=round(time.time() - t))
print(log, flush=True)
tool = None
for cmd, name in (("which unrar", "unrar"), ("which 7z", "7z"), ("which bsdtar", "bsdtar")):
    if sh(cmd).stdout.strip(): tool = name; break
if tool is None:
    print(sh("apt-get -qq update && apt-get -qq install -y unrar").stderr[-300:])
    tool = "unrar" if sh("which unrar").stdout.strip() else None
log["extract_tool"] = tool
cmd = {"unrar": f"unrar x -o+ -inul {rar} {root}/", "7z": f"7z x -y -o{root} {rar}", "bsdtar": f"bsdtar -xf {rar} -C {root}"}.get(tool)
r = sh(cmd) if cmd else None
log["extract_rc"] = r.returncode if r else None
log["extract_err"] = (r.stderr[-500:] if r else "no extractor")
os.remove(rar)
vids = sorted(p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True) if p.lower().endswith((".avi", ".mp4")))
others = sorted({os.path.splitext(p)[1] for p in glob.glob(os.path.join(root, "**", "*"), recursive=True) if os.path.isfile(p)})
info = {}
for v in vids:
    cap = cv2.VideoCapture(v)
    info[os.path.relpath(v, root)] = dict(frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps=cap.get(cv2.CAP_PROP_FPS),
                                         w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
folders = {}
for k in info: folders[os.path.dirname(k)] = folders.get(os.path.dirname(k), 0) + 1
anno = os.path.join(root, "dota_official"); os.makedirs(anno, exist_ok=True)
for f in ("metadata_train.json", "metadata_val.json", "DoTA_annotations.zip"):
    sh(f"wget -q -O {anno}/{f} https://raw.githubusercontent.com/MoonBlvd/Detection-of-Traffic-Anomaly/master/dataset/{f}")
json.dump(dict(log=log, n_videos=len(vids), folders=folders, file_types=others,
               txt_files=sorted(os.path.relpath(p, root) for p in glob.glob(os.path.join(root, "**", "*.txt"), recursive=True))[:20],
               videos=info), open(os.path.join(root, "MANIFEST.json"), "w"), indent=1)
print("videos", len(vids), folders, others)

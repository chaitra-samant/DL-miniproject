import subprocess, shutil, sys, urllib.request, json, os
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as e: return f"ERR {e}"
out = {}
out["python"] = sys.version.split()[0]
out["gpu"] = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
try:
    import torch; out["torch"] = torch.__version__; out["cuda"] = torch.cuda.is_available()
except Exception as e: out["torch"] = str(e)
for m in ["ultralytics", "transformers", "cv2", "timm"]:
    try: mod = __import__(m); out[m] = getattr(mod, "__version__", "ok")
    except Exception as e: out[m] = f"missing ({e.__class__.__name__})"
out["ffmpeg"] = sh("ffmpeg -version | head -1")
t, u, f = shutil.disk_usage("/kaggle/working"); out["working_free_gb"] = round(f/1e9, 1)
for name, url in [("yorku", "https://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip"), ("github", "https://github.com"), ("hf", "https://huggingface.co")]:
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30); out[name] = r.status
    except Exception as e: out[name] = f"ERR {e}"
print(json.dumps(out, indent=1))
json.dump(out, open("/kaggle/working/probe.json", "w"), indent=1)

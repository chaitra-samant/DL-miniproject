# Downloads official JAAD clips (YorkU server) and annotations (github.com/ykotseruba/JAAD, branch JAAD_2.0).
# Output (/kaggle/working/JAAD) is consumed by later kernels via kernel_sources.
import os, subprocess, zipfile, hashlib, json, time
root = "/kaggle/working/JAAD"
os.makedirs(root, exist_ok=True)
def sh(c): print("+", c, flush=True); subprocess.run(c, shell=True, check=True)
sh(f"git clone -q --depth 1 -b JAAD_2.0 https://github.com/ykotseruba/JAAD {root}/repo && rm -rf {root}/repo/.git")
z = "/tmp/JAAD_clips.zip"
t = time.time()
sh(f"wget -q -O {z} https://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip")
print("download s:", round(time.time() - t), "bytes:", os.path.getsize(z), flush=True)
with zipfile.ZipFile(z) as f:
    f.extractall(root)
os.remove(z)
clips = sorted(p for d, _, fs in os.walk(root) for p in [os.path.join(d, x) for x in fs] if p.endswith(".mp4"))
print("mp4 count:", len(clips))
json.dump({"n_clips": len(clips), "first": clips[:2], "source": "https://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip"},
          open(f"{root}/MANIFEST.json", "w"), indent=1)

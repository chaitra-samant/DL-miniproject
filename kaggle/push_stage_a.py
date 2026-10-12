"""Build and push the Stage A kernel from src/features/stage_a.py.

usage: python kaggle/push_stage_a.py smoke|full [--shards N] [--resume-from KERNEL_SLUG ...]
  smoke: 3 videos, to measure speed and check outputs before the full run
  full:  every JAAD video, split over N kernels that run at the same time (Kaggle allows 2
         concurrent sessions); resumable (attach earlier Stage A outputs with --resume-from)
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER = "chaitrasamant"

mode = sys.argv[1]
shards = int(sys.argv[sys.argv.index("--shards") + 1]) if "--shards" in sys.argv else 1
resume = sys.argv[sys.argv.index("--resume-from") + 1:] if "--resume-from" in sys.argv else []
kaggle = os.path.join(ROOT, ".venv", "Scripts", "kaggle")
src = open(os.path.join(ROOT, "src", "features", "stage_a.py"), encoding="utf8").read()
marker = "# @CFG_OVERRIDE"
assert marker in src

for shard in range(shards if mode == "full" else 1):
    slug = "ped-intent-stage-a-smoke" if mode == "smoke" else f"ped-intent-stage-a-jaad-{shard}"
    override = ({"SMOKE_VIDEOS": ["video_0001", "video_0002", "video_0003"]} if mode == "smoke"
                else {"SHARD": shard, "NUM_SHARDS": shards})
    out = os.path.join(ROOT, "kaggle", "stage_a_" + slug.split("stage-a-")[1])
    os.makedirs(out, exist_ok=True)
    code = src.replace(marker, f"CFG.update({json.dumps(override)})  # set by kaggle/push_stage_a.py")
    open(os.path.join(out, "stage_a.py"), "w", encoding="utf8").write(code)
    json.dump({
        "id": f"{USER}/{slug}", "title": slug, "code_file": "stage_a.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [f"{USER}/ped-intent-jaad-download"] + [f"{USER}/{r}" for r in resume],
    }, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=1)
    subprocess.run([kaggle, "kernels", "push", "-p", out], check=True)

"""Build the Kaggle notebooks (notebooks/02_train_eval_ablate.ipynb, notebooks/03_demo.ipynb) from this repo's
src/ and configs/ (embedded as a zip, so a run uses exactly the committed code), and optionally push one.

usage: python kaggle/build_notebooks.py [--push 02|03] [--resume-from SLUG]
Notebook 01 (feature extraction) is the script kernel built by kaggle/build_extract.py.
"""
import base64
import io
import json
import os
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER = "chaitrasamant"
SLUGS = {"02": "accident-02-train-eval-ablate", "03": "accident-03-demo"}

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for top in ("src", "configs"):
        for d, _, fs in os.walk(os.path.join(ROOT, top)):
            for f in fs:
                if f.endswith((".py", ".yaml")):
                    z.write(os.path.join(d, f), os.path.relpath(os.path.join(d, f), ROOT))
BUNDLE = base64.b64encode(buf.getvalue()).decode()
COMMIT = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


UNPACK = code("import base64, io, zipfile, os, sys, glob, json, subprocess\n"
              f"BUNDLE = '{BUNDLE}'\n"
              "CODE = '/kaggle/tmp/code'; os.makedirs(CODE, exist_ok=True)\n"
              "zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))).extractall(CODE)\n"
              "sys.path.insert(0, CODE)\n"
              "def find(p):\n    h = glob.glob(p, recursive=True)\n    return h[0] if h else None\n"
              "FEAT = os.path.dirname(find('/kaggle/input/**/features/ccd_pos_000001.npz'))\n"
              "CCD_ROOT = os.path.dirname(find('/kaggle/input/**/CCD/Crash-1500.txt'))\n"
              "DOTA_META = os.path.dirname(find('/kaggle/input/**/dota_official/metadata_train.json'))\n"
              "print(FEAT, CCD_ROOT, DOTA_META, len(glob.glob(FEAT + '/*.npz')), 'feature files')")

nb02 = [
    md("# 02 Train, evaluate and ablate (Stage B)\n\n"
       f"Built from repo commit `{COMMIT}`. Trains every experiment in `configs/experiments.yaml` on the cached Stage A "
       "features (notebook 01), on two GPUs, resumable. Then builds all result tables and figures with "
       "`src/analysis/report.py`. Every number shown is produced by this run."),
    UNPACK,
    code("OUT = '/kaggle/working'\n"
         "os.makedirs(OUT + '/runs', exist_ok=True)\n"
         "# resume: copy finished runs from an attached earlier run of this notebook\n"
         "for f in glob.glob('/kaggle/input/**/runs/*/*/*/metrics.json', recursive=True):\n"
         "    src = os.path.dirname(f); dst = os.path.join(OUT, 'runs', *src.split('/runs/')[1].split('/'))\n"
         "    if not os.path.exists(dst): subprocess.run(['cp', '-r', src, dst])\n"
         "import torch\n"
         "n = max(1, torch.cuda.device_count())\n"
         "env = dict(os.environ, PYTHONPATH=CODE)\n"
         "procs = [subprocess.Popen([sys.executable, '-u', '-m', 'src.train.run', '--feat_dir', FEAT, '--ccd_root', CCD_ROOT,\n"
         "                           '--dota_meta', DOTA_META, '--out', OUT, '--worker', str(i), '--nworkers', str(n), '--budget_h', '7.6'],\n"
         "                          cwd=CODE, env=dict(env, CUDA_VISIBLE_DEVICES=str(i)), stdout=open(f'{OUT}/train_gpu{i}.log', 'w'),\n"
         "                          stderr=subprocess.STDOUT) for i in range(n)]\n"
         "for p in procs: p.wait()\n"
         "for i in range(n): print(''.join(open(f'{OUT}/train_gpu{i}.log').readlines()[-15:]))"),
    code("r = subprocess.run([sys.executable, '-m', 'src.analysis.report', '--runs', OUT + '/runs', '--out', OUT + '/results'],\n"
         "                   cwd=CODE, env=dict(os.environ, PYTHONPATH=CODE), capture_output=True, text=True)\n"
         "print(r.stdout[-6000:]); print(r.stderr[-3000:])\n"
         "for f in glob.glob(OUT + '/leakage_*.json'): print(os.path.basename(f), open(f).read())"),
    md("## Results tables"),
    code("from IPython.display import Markdown, Image, display\n"
         "for n in ('main_comparable', 'main_and_ablations_clean', 'cross_dataset'):\n"
         "    f = f'{OUT}/results/tables/{n}.md'\n"
         "    if os.path.exists(f): display(Markdown(f'### {n}\\n' + open(f).read()))\n"
         "import pandas as pd\n"
         "for n in ('early_late_summary.csv', 'errors_ccd_source.csv', 'errors_dota_source.csv'):\n"
         "    f = f'{OUT}/results/tables/{n}'\n"
         "    if os.path.exists(f): display(Markdown(f'### {n}')); display(pd.read_csv(f))\n"
         "for f in sorted(glob.glob(OUT + '/results/figures/*.png')): display(Image(f, width=1000))"),
]

nb03 = [
    md("# 03 Demo clips\n\n"
       f"Built from repo commit `{COMMIT}`. Renders the clips chosen by the fixed rule in `configs/experiments.yaml` "
       "(`demo`) from cached outputs of notebooks 01 and 02: largest earlier-warning margins of the main model over the "
       "detection-only model, plus a false alarm and a miss. Left: detection-only (`det_graph`); right: main model with "
       "road context. Box colour = attention weight, i.e. what the model relied on, not what caused the crash."),
    UNPACK,
    code("RUNS = os.path.dirname(find('/kaggle/input/**/runs/main'))\n"
         "SEL = find('/kaggle/input/**/results/tables/demo_selection.json')\n"
         "ccd_v = find('/kaggle/input/**/CCD/videos'); dota_v = find('/kaggle/input/**/DoTA_STAGNet/videos')\n"
         "paths = {}\n"
         "for p in glob.glob(ccd_v + '/Crash-1500/*.mp4'): paths['ccd_pos_' + os.path.basename(p)[:-4]] = p\n"
         "for p in glob.glob(ccd_v + '/Normal/*.mp4'): paths['ccd_neg_' + os.path.basename(p)[:-4]] = p\n"
         "for p in glob.glob(dota_v + '/*/positive/*.avi'): paths['dota_pos_' + os.path.basename(p)[:-4]] = p\n"
         "for p in glob.glob(dota_v + '/*/negative/*.avi'): paths['dota_neg_' + os.path.basename(p)[:-4]] = p\n"
         "open('/kaggle/tmp/videos.txt', 'w').write('\\n'.join(f'{k} {v}' for k, v in paths.items()))\n"
         "r = subprocess.run([sys.executable, '-m', 'src.demo.render', '--selection', SEL, '--runs', RUNS, '--feat_dir', FEAT,\n"
         "                    '--video_list', '/kaggle/tmp/videos.txt', '--out', '/kaggle/working/demo'],\n"
         "                   cwd=CODE, env=dict(os.environ, PYTHONPATH=CODE), capture_output=True, text=True)\n"
         "print(r.stdout[-3000:]); print(r.stderr[-2000:])"),
    code("from IPython.display import Markdown, Image, Video, display\n"
         "for f in sorted(glob.glob('/kaggle/working/demo/*.mp4')):\n"
         "    display(Markdown(f'**{os.path.basename(f)}** ({os.path.getsize(f) / 1e6:.1f} MB)'))\n"
         "    display(Video(f, embed=True, width=960, html_attributes='controls muted loop'))\n"
         "for f in sorted(glob.glob('/kaggle/working/demo/*_storyboard.png')):\n"
         "    display(Markdown(f'**{os.path.basename(f)}**')); display(Image(f, width=960))"),
]


def write(nb_cells, name):
    nb = {"cells": nb_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    p = os.path.join(ROOT, "notebooks", name)
    json.dump(nb, open(p, "w"), indent=1)
    return nb


os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
nbs = {"02": write(nb02, "02_train_eval_ablate.ipynb"), "03": write(nb03, "03_demo.ipynb")}
print(f"wrote notebooks 02 and 03 from commit {COMMIT}")

if "--push" in sys.argv:
    which = sys.argv[sys.argv.index("--push") + 1]
    resume = [sys.argv[sys.argv.index("--resume-from") + 1]] if "--resume-from" in sys.argv else []
    out = os.path.join(ROOT, "kaggle", "nb" + which)
    os.makedirs(out, exist_ok=True)
    fname = {"02": "02_train_eval_ablate.ipynb", "03": "03_demo.ipynb"}[which]
    json.dump(nbs[which], open(os.path.join(out, fname), "w"), indent=1)
    sources = [f"{USER}/accident-stage-a", f"{USER}/accident-ccd-download", f"{USER}/accident-dota-stagnet-download"]
    if which == "03":
        sources.append(f"{USER}/{SLUGS['02']}")
    json.dump({"id": f"{USER}/{SLUGS[which]}", "title": SLUGS[which], "code_file": fname, "language": "python",
               "kernel_type": "notebook", "is_private": True, "enable_gpu": which == "02", "enable_internet": True,
               "dataset_sources": [], "competition_sources": [], "kernel_sources": sources + [f"{USER}/{r}" for r in resume]},
              open(os.path.join(out, "kernel-metadata.json"), "w"), indent=1)
    subprocess.run([os.path.join(ROOT, ".venv", "Scripts", "kaggle"), "kernels", "push", "-p", out], check=True)

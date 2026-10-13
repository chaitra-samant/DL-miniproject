"""Whole JAAD pipeline in one Kaggle session (2 x T4), resumable, with a time budget.

  python -m src.pipeline all           # every stage, skipping work already done
  python -m src.pipeline train_worker  # internal: one training worker per GPU

Stages: stage_a (2 GPU workers) -> samples -> train (2 GPU workers) -> analysis -> mining -> demo.
If feature extraction is not complete within its budget, later stages are skipped (results on a
partial dataset would not be comparable); rerunning with the saved cache attached resumes it.
"""
import glob
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import traceback

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T0 = time.time()

P = dict(
    cfg=os.path.join(ROOT, "configs", "jaad.yaml"),
    work=os.environ.get("PIPE_WORK", "/kaggle/working"),
    budget_total_h=float(os.environ.get("PIPE_BUDGET_H", "8.3")),
    stage_a_budget_h=float(os.environ.get("PIPE_STAGE_A_H", "5.5")),
    n_gpus=int(os.environ.get("PIPE_GPUS", "0")),   # 0 = use every visible GPU
    epochs=int(os.environ.get("PIPE_EPOCHS", "0")),  # >0 only for local tests
)
W = P["work"]
CACHE, RUNS, RES = os.path.join(W, "cache"), os.path.join(W, "runs"), os.path.join(W, "results")
TABLES, FIGS, DEMO = os.path.join(RES, "tables"), os.path.join(RES, "figures"), os.path.join(RES, "demo")


def n_workers():
    if P["n_gpus"]:
        return P["n_gpus"]
    import torch
    return max(1, torch.cuda.device_count())


def dev():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{(time.time() - T0) / 3600:5.2f} h]", *a, flush=True)


def status(**kw):
    os.makedirs(RES, exist_ok=True)
    f = os.path.join(RES, "STATUS.json")
    s = json.load(open(f)) if os.path.exists(f) else {}
    s.update(kw)
    json.dump(s, open(f, "w"), indent=1, default=str)


def cfg():
    return yaml.safe_load(open(P["cfg"]))


def jaad_paths():
    from src.features import stage_a
    work = stage_a.find_jaad()
    return work, os.path.join(work, "JAAD_clips")


def spawn(module_args, gpu, env_extra, logfile):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=ROOT, **env_extra)
    return subprocess.Popen([sys.executable, "-u", "-m"] + module_args, cwd=ROOT, env=env,
                            stdout=open(logfile, "w"), stderr=subprocess.STDOUT)


def wait_all(procs, logs):
    for p in procs:
        p.wait()
    for lf in logs:
        print(f"----- tail of {os.path.basename(lf)} -----")
        print("".join(open(lf).readlines()[-12:]), flush=True)
    return [p.returncode for p in procs]


# ------------------------------------------------------------------ stages
def stage_a():
    from src.features import stage_a as A
    work, _ = jaad_paths()
    db = A.load_db(work)                    # builds the annotation cache once, before the workers start
    os.makedirs(CACHE, exist_ok=True)
    for p in glob.glob("/kaggle/input/**/cache/*.pkl", recursive=True):   # resume from an attached run
        d = os.path.join(CACHE, os.path.basename(p))
        if not os.path.exists(d):
            shutil.copy(p, d)
    want = sorted(db)
    have = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(CACHE, "*.pkl"))}
    log(f"stage A: {len(have & set(want))}/{len(want)} videos cached")
    if set(want) <= have:
        return True
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)
    from transformers import SegformerForSemanticSegmentation
    from ultralytics import YOLO
    SegformerForSemanticSegmentation.from_pretrained(A.CFG["SEG_MODEL"])       # download once
    YOLO(os.path.join(ROOT, A.CFG["POSE_MODEL"]))
    A.CFG["POSE_MODEL"] = os.path.join(ROOT, A.CFG["POSE_MODEL"])
    n = n_workers()
    logs = [os.path.join(W, f"stage_a_gpu{i}.log") for i in range(n)]
    procs = [spawn(["src.features.stage_a"], i, {"STAGE_A_CFG": json.dumps(dict(
        SHARD=i, NUM_SHARDS=n, TIME_BUDGET_H=P["stage_a_budget_h"], OUT_DIR=CACHE,
        POSE_MODEL=A.CFG["POSE_MODEL"]))}, logs[i]) for i in range(n)]
    rc = wait_all(procs, logs)
    have = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(CACHE, "*.pkl"))}
    done = set(want) <= have
    status(stage_a=dict(cached=len(have & set(want)), wanted=len(want), complete=done, returncodes=rc))
    log(f"stage A finished: {len(have & set(want))}/{len(want)} cached, complete={done}")
    return done


def samples():
    from src.data.samples import load_or_build
    work, _ = jaad_paths()
    data, info = load_or_build(CACHE, work, cfg(), os.path.join(W, "samples.pkl"))
    os.makedirs(TABLES, exist_ok=True)
    json.dump(info, open(os.path.join(TABLES, "dataset_stats.json"), "w"), indent=1)
    log("samples:", json.dumps(info["stats"]))
    if info["missing_videos"]:
        log(f"WARNING: {len(info['missing_videos'])} split videos have no cache")
    return data


def run_list():
    c = cfg()
    return [(e, s) for e in c["experiments"] for s in (c["seeds"] if e["type"] == "net" else [0])]


def train_worker():
    from src.train.train import train_one
    i, n = int(os.environ["WORKER"]), int(os.environ["NWORKERS"])
    data = pickle.load(open(os.path.join(W, "samples.pkl"), "rb"))[0]
    c = cfg()
    for k, (spec, seed) in enumerate(run_list()):
        if k % n != i:
            continue
        out = os.path.join(RUNS, f"{spec['name']}_s{seed}")
        if os.path.exists(os.path.join(out, "metrics.json")):
            continue
        if time.time() - float(os.environ["PIPE_T0"]) > P["budget_total_h"] * 3600 - 1.5 * 3600:
            log("training budget reached; leaving remaining runs for a rerun")
            break
        try:
            tc = dict(c["train"], **({"epochs": P["epochs"], "patience": P["epochs"]} if P["epochs"] else {}))
            train_one(data, spec, seed, out, tc, dev=dev(), log=log)
        except Exception:
            traceback.print_exc()


def train():
    n = n_workers()
    logs = [os.path.join(W, f"train_gpu{i}.log") for i in range(n)]
    procs = [spawn(["src.pipeline", "train_worker"], i, {"WORKER": str(i), "NWORKERS": str(n),
                                                          "PIPE_T0": str(T0)}, logs[i]) for i in range(n)]
    wait_all(procs, logs)
    done = sum(os.path.exists(os.path.join(RUNS, f"{e['name']}_s{s}", "metrics.json")) for e, s in run_list())
    status(train=dict(runs_done=done, runs_wanted=len(run_list())))
    log(f"training: {done}/{len(run_list())} runs done")


def analysis(data):
    from src.eval import analysis as An
    c = cfg()
    order = [e["name"] for e in c["experiments"]]
    df = An.load_runs(RUNS)
    ti, tt, tb = An.tables(df, order, TABLES)
    main_groups = next(e for e in c["experiments"] if e["name"] == "main")["groups"]
    An.figures(df, order, RUNS, data, main_groups, FIGS)
    An.error_analysis(RUNS, data, TABLES)
    log("intent table:\n" + ti.to_string(index=False))
    log("trajectory table:\n" + tt.to_string(index=False))


def mining(data):
    from src.mining.surprise import mine
    c = cfg()
    summary, sel = mine(RUNS, data, c, [e["name"] for e in c["experiments"] if e["type"] == "net"], TABLES)
    log("surprise mining:\n" + summary.to_string(index=False))
    return sel


def demo(sel):
    import torch
    from src.data.samples import load_jaad_db
    from src.demo.render import run_demo
    from src.features.stage_a import CFG as ACFG
    work, clips = jaad_paths()
    db = load_jaad_db(work)
    res = run_demo(sel, db, clips, RUNS, DEMO, dev(), ACFG["SEG_MODEL"],
                   os.path.join(ROOT, ACFG["POSE_MODEL"]), cfg()["explain_groups"], log=log)
    status(demo=dict(clips=len(res)))


def step(name, fn, *a):
    log(f"=== {name} ===")
    try:
        r = fn(*a)
        status(**{f"{name}_ok": True})
        return r
    except Exception as e:
        traceback.print_exc()
        status(**{f"{name}_ok": False, f"{name}_error": repr(e)})
        return None


def main_all():
    status(started=time.strftime("%Y-%m-%d %H:%M:%S"))
    if not step("stage_a", stage_a):
        log("Feature extraction is incomplete; stopping before training. Rerun with this output attached.")
        return
    data = step("samples", samples)
    if data is None:
        return
    step("train", train)
    step("analysis", analysis, data)
    sel = step("mining", mining, data)
    if sel:
        step("demo", demo, sel)
    shutil.copy(P["cfg"], os.path.join(RES, "config_used.yaml"))
    status(finished=time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_h=round((time.time() - T0) / 3600, 2))
    log("pipeline finished")


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    {"all": main_all, "train_worker": train_worker}[sys.argv[1]]()

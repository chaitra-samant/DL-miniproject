"""Stage B: train and evaluate the experiments in configs/experiments.yaml on cached Stage A features.

python -m src.train.run --feat_dir F --ccd_root C --dota_meta D --out O [--worker i --nworkers n]
Each (experiment, scheme, split, seed) writes O/runs/<exp>/<scheme>/<split>_s<seed>/{metrics.json, preds.npz}
and is skipped when it already exists (resumable). Runs are ordered by priority: baselines and main first.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import yaml
from sklearn.model_selection import GroupShuffleSplit

from src.data import loader
from src.eval.metrics import evaluation, threshold_sweep
from src.models.nets import build, exp_loss

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def norm_stats(arrs, idx):
    st = {}
    m = arrs["mask"][idx]
    for g in ("app", "geo", "seg"):
        x = arrs[g][idx][m].astype(np.float32)
        st[g] = (x.mean(0), x.std(0) + 1e-4)
    x = arrs["frame"][idx].reshape(-1, arrs["frame"].shape[-1]).astype(np.float32)
    st["frame"] = (x.mean(0), x.std(0) + 1e-4)
    return st


def batch(arrs, idx, st, dev):
    b = {}
    m = torch.from_numpy(arrs["mask"][idx]).to(dev)
    for g in ("app", "geo", "seg"):
        x = (arrs[g][idx].astype(np.float32) - st[g][0]) / st[g][1]
        b[g] = torch.from_numpy(x).to(dev) * m[..., None]
    b["frame"] = torch.from_numpy((arrs["frame"][idx].astype(np.float32) - st["frame"][0]) / st["frame"][1]).to(dev)
    b["geo_raw"] = torch.from_numpy(arrs["geo"][idx].astype(np.float32)).to(dev)
    b["mask"] = m
    return b


def predict(model, arrs, idx, st, dev, bs=64, keep_attn=False):
    model.eval()
    P, A = [], []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            o = model(batch(arrs, idx[i:i + bs], st, dev))
            P.append(torch.softmax(o["logits"], -1)[..., 1].cpu().numpy())
            if keep_attn and o["attn"] is not None:
                A.append(o["attn"].cpu().numpy().astype(np.float16))
    return np.concatenate(P), (np.concatenate(A) if A else None)


def train_eval(arrs, index, spec, tr, te, seed, tc, dev, groups):
    set_seed(seed)
    tr_idx = np.where(tr)[0]
    gss = GroupShuffleSplit(n_splits=1, test_size=tc["val_fraction"], random_state=seed)
    a, b = next(gss.split(tr_idx, groups=np.asarray(groups)[tr_idx]))
    fit_idx, val_idx, te_idx = tr_idx[a], tr_idx[b], np.where(te)[0]
    st = norm_stats(arrs, fit_idx)
    model = build(spec).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    y, toa = index["y"], index["toa"]
    best, best_state, bad, t0 = -1.0, None, 0, time.time()
    rng = np.random.default_rng(seed)
    for ep in range(tc["epochs"]):
        model.train()
        perm = rng.permutation(fit_idx)
        for i in range(0, len(perm), tc["batch_size"]):
            j = np.sort(perm[i:i + tc["batch_size"]])
            o = model(batch(arrs, j, st, dev))
            loss = exp_loss(o["logits"], torch.from_numpy(y[j]).to(dev), torch.from_numpy(toa[j]).to(dev))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        pv, _ = predict(model, arrs, val_idx, st, dev)
        ap = evaluation(pv, y[val_idx], toa[val_idx])[0] if y[val_idx].sum() and (1 - y[val_idx]).sum() else 0.0
        if ap > best + 1e-4:
            best, bad = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    model.load_state_dict(best_state)
    p, attn = predict(model, arrs, te_idx, st, dev, keep_attn=spec["name"] in ("main", "det_graph"))
    AP, mTTA, R80 = evaluation(p, y[te_idx], toa[te_idx])
    res = dict(AP=AP, mTTA=mTTA, TTA_R80=R80, val_AP=best, epochs=ep + 1, train_sec=round(time.time() - t0, 1),
               n_train=int(len(fit_idx)), n_val=int(len(val_idx)), n_test=int(len(te_idx)),
               test_pos=int(y[te_idx].sum()))
    return res, p, attn, te_idx, model, st


def run_list(cfg):
    out = []
    for e in cfg["experiments"]:
        for grp in e["run_on"]:
            for scheme in cfg["schemes"][grp]:
                out.append((e, scheme))
    # priority: baselines + main on every scheme first, then ablations
    prio = {"frame_lstm": 0, "det_lstm": 0, "det_graph": 0, "main": 0}
    return sorted(out, key=lambda r: (prio.get(r[0]["name"], 1), r[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True); ap.add_argument("--ccd_root", required=True)
    ap.add_argument("--dota_meta", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--budget_h", type=float, default=8.0); ap.add_argument("--override", default="{}")
    a = ap.parse_args()
    t_start = time.time()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "experiments.yaml")))
    tc = dict(cfg["train"], **json.loads(a.override))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = {}
    jobs = [(e, s, sp, seed) for e, s in run_list(cfg) for seed in cfg["seeds"] for sp in range(5)]
    exps = {e["name"]: e for e in cfg["experiments"]}
    jobs += [(exps[n], f"cross_{src}2{dst}", 0, seed) for src, dst in cfg["cross"]["directions"]
             for n in cfg["cross"]["experiments"] for seed in cfg["seeds"]]
    k = 0
    for e, scheme, sp, seed in jobs:
        for ds in (scheme.split("_")[1].split("2") if scheme.startswith("cross") else [scheme.split("_")[0]]):
            if ds not in data and not (scheme.startswith("cross") and "both" in data):
                idx = loader.build_index(a.feat_dir, ds, ccd_root=a.ccd_root, dota_meta_dir=a.dota_meta)
                arrs, _ = loader.load_all(idx)
                data[ds] = (idx, arrs)
                json.dump(loader.leakage_report(idx), open(os.path.join(a.out, f"leakage_{ds}.json"), "w"), indent=1)
        if scheme.startswith("cross"):
            if "both" not in data:   # one concatenated copy serves both directions; per-dataset copies are freed
                (i1, a1), (i2, a2) = data.pop("ccd"), data.pop("dota")
                arrs = {}
                for key in list(a1):
                    arrs[key] = np.concatenate([a1.pop(key), a2.pop(key)])
                n1, n2 = len(i1["y"]), len(i2["y"])
                is_ccd = np.r_[np.ones(n1, bool), np.zeros(n2, bool)]
                idx = dict(y=np.concatenate([i1["y"], i2["y"]]), toa=np.concatenate([i1["toa"], i2["toa"]]),
                           vids=i1["vids"] + i2["vids"],
                           groups=dict(source=list(i1["groups"]["source"]) + list(i2["groups"]["source"])),
                           schemes={"cross_ccd2dota": [("all", is_ccd, ~is_ccd)], "cross_dota2ccd": [("all", ~is_ccd, is_ccd)]})
                data["both"] = (idx, arrs)
            idx, arrs = data["both"]
        else:
            idx, arrs = data[scheme.split("_")[0]]
        splits = idx["schemes"][scheme]
        if sp >= len(splits):
            continue
        k += 1
        if k % a.nworkers != a.worker:
            continue
        sname, tr, te = splits[sp]
        d = os.path.join(a.out, "runs", e["name"], scheme, f"{sname}_s{seed}")
        if os.path.exists(os.path.join(d, "metrics.json")):
            continue
        if time.time() - t_start > a.budget_h * 3600:
            print("budget reached; remaining runs are left for a rerun", flush=True)
            break
        os.makedirs(d, exist_ok=True)
        groups = idx["groups"]["source"] if scheme.endswith("source") else (
            idx["groups"]["clip"] if scheme == "dota_stagnet" else idx["groups"]["source"])
        res, p, attn, te_idx, model, st = train_eval(arrs, idx, dict(e), tr, te, seed, tc, dev, groups)
        res.update(exp=e["name"], scheme=scheme, split=sname, seed=seed)
        json.dump(res, open(os.path.join(d, "metrics.json"), "w"), indent=1)
        sweep = threshold_sweep(p, idx["y"][te_idx], idx["toa"][te_idx])
        json.dump(sweep, open(os.path.join(d, "sweep.json"), "w"))
        np.savez_compressed(os.path.join(d, "preds.npz"), p=p.astype(np.float16), vid=np.array(idx["vids"])[te_idx],
                            y=idx["y"][te_idx], toa=idx["toa"][te_idx],
                            **({"attn": attn} if attn is not None else {}))
        if e["name"] in ("main", "det_graph") and seed == 0:
            torch.save(dict(state=model.state_dict(), spec=e, norm=st), os.path.join(d, "model.pt"))
        print(f"{time.strftime('%H:%M:%S')} {e['name']:20s} {scheme:13s} {sname:8s} s{seed}: AP {res['AP']:.3f} "
              f"mTTA {res['mTTA']:.2f} TTA@R80 {res['TTA_R80']:.2f} ({res['epochs']} ep, {res['train_sec']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

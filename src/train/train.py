"""Train and evaluate one experiment (config spec + seed) on cached samples. Writes, per run:
  metrics.json  intent and trajectory test metrics, epochs, val loss
  preds.npz     test intent probabilities and gate weights, test trajectory predictions (all modes)
  model.pt      weights + feature normalisation (used by mining and the demo)
"""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.models.nets import OFFSET_SCALE, ConstantVelocity, build_model


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def norm_stats(train, groups):
    st = {}
    for g in groups:
        x = np.concatenate([train["intent"][g].reshape(-1, train["intent"][g].shape[-1]),
                            train["traj"][g].reshape(-1, train["traj"][g].shape[-1])])
        st[g] = (x.mean(0).astype(np.float32), (x.std(0) + 1e-6).astype(np.float32))
    return st


def to_t(block, idx, groups, st, dev):
    return {g: torch.from_numpy((block[g][idx] - st[g][0]) / st[g][1]).to(dev) for g in groups}


def traj_targets(T, idx):
    meta = [T["meta"][i] for i in idx]
    wh = np.array([[m[4], m[5], m[4], m[5]] for m in meta], np.float32)
    y = (T["y"][idx] - T["last"][idx][:, None, :]) / wh[:, None, :] * OFFSET_SCALE
    return torch.from_numpy(y), wh


def traj_loss(out, y):
    err = ((out["traj"] - y.unsqueeze(1)) ** 2).mean((2, 3))            # (B, K)
    best = err.argmin(1)
    loss = err.gather(1, best[:, None]).mean()
    if out["traj"].shape[1] > 1:
        loss = loss + 0.1 * F.cross_entropy(out["mode_logits"], best)
    return loss


def intent_metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return dict(acc=accuracy_score(y, yhat), f1=f1_score(y, yhat, zero_division=0),
                auc=roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
                precision=precision_score(y, yhat, zero_division=0), recall=recall_score(y, yhat, zero_division=0))


def traj_metrics(pred, gt, prefix=""):
    """pred, gt: (N, 45, 4) pixel boxes. PIEPredict metrics + centre ADE/FDE."""
    m = {}
    se = (pred - gt) ** 2
    for h in (15, 30, 45):
        m[f"{prefix}mse_{h}"] = float(se[:, :h].mean())
    c = lambda b: np.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2], -1)
    cp, cg = c(pred), c(gt)
    m[f"{prefix}cmse"] = float(((cp - cg) ** 2).mean())
    m[f"{prefix}cfmse"] = float(((cp[:, -1] - cg[:, -1]) ** 2).mean())
    d = np.linalg.norm(cp - cg, axis=-1)
    m[f"{prefix}ade"] = float(d.mean())
    m[f"{prefix}fde"] = float(d[:, -1].mean())
    return m


def run_cv(data, out_dir):
    T = data["test"]["traj"]
    pred = ConstantVelocity().predict(T["obs"], T["y"].shape[1])
    m = dict(exp="const_velocity", seed=0, **traj_metrics(pred, T["y"]))
    json.dump(m, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(out_dir, "preds.npz"), traj=pred[:, None].astype(np.float32))
    return m


def predict(model, block, groups, st, dev, traj=True, bs=512):
    model.eval()
    P, Gt, Tr, ML = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(block[groups[0]]), bs):
            idx = np.arange(i, min(i + bs, len(block[groups[0]])))
            x = to_t(block, idx, sorted(set(groups) | {"ego"}), st, dev)
            o = model(x, traj=traj)
            P.append(torch.sigmoid(o["logit"]).cpu().numpy())
            if o["gates"] is not None:
                Gt.append(o["gates"].cpu().numpy())
            if traj:
                Tr.append(o["traj"].cpu().numpy()); ML.append(o["mode_logits"].cpu().numpy())
    cat = lambda L: np.concatenate(L) if L else None
    return cat(P), cat(Gt), cat(Tr), cat(ML)


def train_one(data, spec, seed, out_dir, tcfg, dev="cuda", log=print):
    os.makedirs(out_dir, exist_ok=True)
    if spec["type"] == "const_velocity":
        return run_cv(data, out_dir)
    set_seed(seed)
    groups = spec["groups"]
    feat_groups = sorted(set(groups) | {"ego"})        # ego is also needed by the PIEPredict-style decoder
    tr, va, te = data["train"], data["val"], data["test"]
    st = norm_stats(tr, feat_groups)
    dims = {g: tr["traj"][g].shape[-1] for g in feat_groups}
    model = build_model(spec, dims).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    yi = tr["intent"]["y"]
    pos_w = torch.tensor((len(yi) - yi.sum()) / max(yi.sum(), 1), device=dev)
    bs = tcfg["batch_size"]
    nT, nI = len(tr["traj"]["y"]), len(yi)
    best, best_state, bad, t0 = float("inf"), None, 0, time.time()
    rng = np.random.default_rng(seed)
    for ep in range(tcfg["epochs"]):
        model.train()
        pt, pi = rng.permutation(nT), rng.permutation(nI)
        steps = max(int(np.ceil(nT / bs)), int(np.ceil(nI / bs)))
        for s in range(steps):
            it = pt[(np.arange(s * bs, (s + 1) * bs)) % nT]
            ii = pi[(np.arange(s * bs, (s + 1) * bs)) % nI]
            y, _ = traj_targets(tr["traj"], it)
            o = model(to_t(tr["traj"], it, feat_groups, st, dev), traj=True)
            lt = traj_loss(o, y.to(dev))
            oi = model(to_t(tr["intent"], ii, feat_groups, st, dev), traj=False)
            li = F.binary_cross_entropy_with_logits(oi["logit"], torch.from_numpy(yi[ii]).to(dev), pos_weight=pos_w)
            loss = lt + tcfg["intent_weight"] * li
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        # validation
        model.eval()
        with torch.no_grad():
            vt = np.arange(len(va["traj"]["y"]))
            vl_t = 0.0
            for i in range(0, len(vt), 512):
                idx = vt[i:i + 512]
                y, _ = traj_targets(va["traj"], idx)
                vl_t += traj_loss(model(to_t(va["traj"], idx, feat_groups, st, dev)), y.to(dev)).item() * len(idx)
            vl_t /= max(len(vt), 1)
            vi = np.arange(len(va["intent"]["y"]))
            oi = model(to_t(va["intent"], vi, feat_groups, st, dev), traj=False)
            vl_i = F.binary_cross_entropy_with_logits(oi["logit"], torch.from_numpy(va["intent"]["y"]).to(dev),
                                                      pos_weight=pos_w).item()
        vl = vl_t + tcfg["intent_weight"] * vl_i
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                break
    model.load_state_dict(best_state)
    # test
    pi_, gates, _, _ = predict(model, te["intent"], groups, st, dev, traj=False)
    _, _, tr_all, ml = predict(model, te["traj"], groups, st, dev, traj=True)
    T = te["traj"]
    wh = np.array([[m[4], m[5], m[4], m[5]] for m in T["meta"]], np.float32)
    boxes = T["last"][:, None, None, :] + tr_all / OFFSET_SCALE * wh[:, None, None, :]   # (N, K, 45, 4)
    top = boxes[np.arange(len(boxes)), ml.argmax(1)]
    m = dict(exp=spec["name"], seed=seed, epochs=ep + 1, val_loss=best, train_sec=round(time.time() - t0, 1),
             **{f"intent_{k}": float(v) for k, v in intent_metrics(te["intent"]["y"].astype(int), pi_).items()},
             **traj_metrics(top, T["y"]))
    if boxes.shape[1] > 1:
        c = lambda b: np.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2], -1)
        ade = np.linalg.norm(c(boxes) - c(T["y"])[:, None], axis=-1).mean(-1)   # (N, K)
        m.update(traj_metrics(boxes[np.arange(len(boxes)), ade.argmin(1)], T["y"], prefix=f"best{boxes.shape[1]}_"))
    json.dump(m, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(out_dir, "preds.npz"), intent_p=pi_,
                        gates=gates if gates is not None else np.zeros(0), traj=boxes.astype(np.float32),
                        mode_logits=ml)
    torch.save(dict(state=model.state_dict(), spec=spec, dims=dims, norm=st), os.path.join(out_dir, "model.pt"))
    log(f"  {spec['name']} seed {seed}: {ep + 1} epochs, AUC {m['intent_auc']:.3f}, F1 {m['intent_f1']:.3f}, "
        f"MSE@45 {m['mse_45']:.0f}, {m['train_sec']:.0f}s")
    return m

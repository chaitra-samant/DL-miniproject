"""Aggregate runs into results tables (CSV + Markdown), figures and an error analysis.
Nothing here is typed by hand: every number comes from runs/*/metrics.json and preds.npz.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INTENT = ["intent_acc", "intent_f1", "intent_auc", "intent_precision", "intent_recall"]
TRAJ = ["mse_15", "mse_30", "mse_45", "cmse", "cfmse", "ade", "fde"]
VEHICLE = ["stopped", "moving_slow", "moving_fast", "decelerating", "accelerating"]


def load_runs(runs_dir):
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(runs_dir, "*", "metrics.json"))]
    return pd.DataFrame(rows)


def mean_std(df, cols, order):
    out = []
    for exp in order:
        d = df[df.exp == exp]
        if d.empty:
            continue
        r = {"experiment": exp, "seeds": len(d)}
        for c in cols:
            if c in d and d[c].notna().any():
                r[c] = f"{d[c].mean():.3f} ± {d[c].std(ddof=0):.3f}" if c.startswith("intent") else \
                       f"{d[c].mean():.0f} ± {d[c].std(ddof=0):.0f}"
            else:
                r[c] = "n/a"
        out.append(r)
    return pd.DataFrame(out)


def tables(df, order, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df.sort_values(["exp", "seed"]).to_csv(os.path.join(out_dir, "runs_raw.csv"), index=False)
    t_int = mean_std(df, INTENT, [e for e in order if e != "const_velocity"])
    t_traj = mean_std(df, TRAJ, order)
    bk = [c for c in df.columns if c.startswith("best") and c.endswith(("ade", "fde", "mse_45"))]
    t_bk = mean_std(df, bk, order) if bk else pd.DataFrame()
    for name, t in (("intent", t_int), ("trajectory", t_traj), ("trajectory_best_of_k", t_bk)):
        t.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
        with open(os.path.join(out_dir, f"{name}.md"), "w", encoding="utf8") as f:
            f.write(t.to_markdown(index=False) if not t.empty else "(no runs)")
    return t_int, t_traj, t_bk


def figures(df, order, runs_dir, data, groups_main, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    exps = [e for e in order if e in set(df.exp) and e != "const_velocity"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, met in zip(ax, ["intent_auc", "intent_f1"]):
        m = [df[df.exp == e][met].mean() for e in exps]
        s = [df[df.exp == e][met].std(ddof=0) for e in exps]
        a.bar(range(len(exps)), m, yerr=s, color="#4c72b0", capsize=3)
        a.set_xticks(range(len(exps))); a.set_xticklabels(exps, rotation=60, ha="right", fontsize=8)
        a.set_title(f"JAAD test, {met} (mean ± std over seeds)"); a.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "intent_by_experiment.png"), dpi=130); plt.close(fig)

    fig, a = plt.subplots(figsize=(6, 4))
    for e in [x for x in ("const_velocity", "bbox_lstm", "piepredict_style", "main") if x in set(df.exp)]:
        d = df[df.exp == e]
        a.errorbar([0.5, 1.0, 1.5], [d[f"mse_{h}"].mean() for h in (15, 30, 45)],
                   yerr=[d[f"mse_{h}"].std(ddof=0) for h in (15, 30, 45)], marker="o", label=e, capsize=3)
    a.set_xlabel("prediction horizon (s)"); a.set_ylabel("bbox MSE (px²)"); a.legend(); a.set_title("Trajectory error vs horizon")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "traj_mse_vs_horizon.png"), dpi=130); plt.close(fig)

    gates = [np.load(p)["gates"] for p in sorted(glob.glob(os.path.join(runs_dir, "main_s*", "preds.npz")))]
    gates = [g for g in gates if g.size]
    if gates:
        g = np.mean(gates, 0)                             # (N, T, G)
        y = data["test"]["intent"]["y"].astype(int)
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
        for a, lab in zip(ax, (1, 0)):
            mg = g[y == lab].mean(0)
            for k, name in enumerate(groups_main):
                a.plot(np.arange(1, mg.shape[0] + 1), mg[:, k], label=name)
            a.set_title(f"gate weight over the observation, {'crossers' if lab else 'non-crossers'}")
            a.set_xlabel("observed frame")
        ax[0].set_ylabel("mean gate weight"); ax[1].legend()
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "gate_weights_over_time.png"), dpi=130); plt.close(fig)


def error_analysis(runs_dir, data, out_dir, exp="main"):
    """Per-condition intent accuracy and trajectory error for `exp` (mean over seeds), and the
    most confident wrong intent predictions."""
    ps = [np.load(p) for p in sorted(glob.glob(os.path.join(runs_dir, f"{exp}_s*", "preds.npz")))]
    if not ps:
        return None
    I = data["test"]["intent"]
    p = np.mean([x["intent_p"] for x in ps], 0)
    y = I["y"].astype(int)
    ego_last = I["ego"][:, -1]
    H = np.array([m[5] for m in I["meta"]], np.float32)
    h_px = I["bbox"][:, -1, 3] * H - I["bbox"][:, -1, 1] * H
    veh = np.where(ego_last[:, :5].sum(1) > 0, ego_last[:, :5].argmax(1), -1)
    df = pd.DataFrame(dict(vid=[m[0] for m in I["meta"]], pid=[m[1] for m in I["meta"]],
                           first_frame=[m[2] for m in I["meta"]], last_frame=[m[3] for m in I["meta"]],
                           tte=[m[6] for m in I["meta"]], y=y, p=p, correct=((p >= 0.5) == y),
                           vehicle=[VEHICLE[v] if v >= 0 else "n/a" for v in veh],
                           crosswalk=ego_last[:, 5] > 0, height_px=h_px))
    df["height_bin"] = pd.cut(df.height_px, [0, 50, 100, 200, 2000], labels=["<50", "50-100", "100-200", ">200"])
    rows = []
    for col in ("vehicle", "crosswalk", "height_bin", "y"):
        for k, d in df.groupby(col, observed=True):
            rows.append(dict(factor=col, value=str(k), windows=len(d), accuracy=round(d.correct.mean(), 3)))
    by = pd.DataFrame(rows)
    by.to_csv(os.path.join(out_dir, f"error_intent_by_condition_{exp}.csv"), index=False)
    wrong = df[~df.correct].assign(confidence=lambda d: (d.p - 0.5).abs()).sort_values("confidence", ascending=False)
    wrong.head(15).to_csv(os.path.join(out_dir, f"error_intent_worst_{exp}.csv"), index=False)

    T = data["test"]["traj"]
    c = lambda b: np.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2], -1)
    top = lambda x: x["traj"][np.arange(len(x["traj"])), x["mode_logits"].argmax(1)]
    # per-seed final displacement error of the top mode, averaged over seeds (not an ensemble)
    fde = np.mean([np.linalg.norm(c(top(x))[:, -1] - c(T["y"])[:, -1], axis=-1) for x in ps], 0)
    hh = T["last"][:, 3] - T["last"][:, 1]
    tdf = pd.DataFrame(dict(fde=fde, height_bin=pd.cut(hh, [0, 50, 100, 200, 2000],
                                                       labels=["<50", "50-100", "100-200", ">200"])))
    tby = tdf.groupby("height_bin", observed=True).fde.agg(["count", "mean", "median"]).round(1).reset_index()
    tby.to_csv(os.path.join(out_dir, f"error_traj_fde_by_height_{exp}.csv"), index=False)
    return by, tby

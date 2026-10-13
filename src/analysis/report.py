"""Aggregate Stage B runs into tables (CSV + Markdown), figures, the early-vs-late analysis, the error analysis,
and the demo clip selection. Nothing here is typed by hand; everything is read from runs/*/*/*/.

python -m src.analysis.report --runs O/runs --out results
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.eval.metrics import first_alarm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICS = ["AP", "mTTA", "TTA_R80"]
SCHEME_LABEL = {"ccd_official": "CCD official split", "dota_stagnet": "DoTA, STAGNet folds (by clip)",
                "ccd_source": "CCD, folds by YouTube source", "dota_source": "DoTA, folds by YouTube source",
                "cross_ccd2dota": "train CCD -> test DoTA", "cross_dota2ccd": "train DoTA -> test CCD"}


def load_runs(runs):
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(runs, "*", "*", "*", "metrics.json"))]
    return pd.DataFrame(rows)


def summarise(df):
    """Mean over splits within a seed, then mean +- std over seeds."""
    per_seed = df.groupby(["exp", "scheme", "seed"])[METRICS].mean().reset_index()
    g = per_seed.groupby(["exp", "scheme"])[METRICS]
    out = g.mean().join(g.std(ddof=0), rsuffix="_std").join(per_seed.groupby(["exp", "scheme"]).size().rename("seeds"))
    splits = df.groupby(["exp", "scheme"])["split"].nunique().rename("splits")
    return out.join(splits).reset_index()


def fmt(row, m):
    return f"{row[m]:.3f} ± {row[m + '_std']:.3f}" if m == "AP" else f"{row[m]:.2f} ± {row[m + '_std']:.2f}"


def table(summ, exps, schemes, path):
    rows = []
    for e in exps:
        r = {"model": e}
        for s in schemes:
            hit = summ[(summ.exp == e) & (summ.scheme == s)]
            for m in METRICS:
                r[f"{s} {m}"] = fmt(hit.iloc[0], m) if len(hit) else "not run"
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(path + ".csv", index=False)
    with open(path + ".md", "w", encoding="utf8") as f:
        f.write(t.to_markdown(index=False))
    return t


def preds(runs, exp, scheme):
    out = []
    for f in sorted(glob.glob(os.path.join(runs, exp, scheme, "*", "preds.npz"))):
        z = np.load(f, allow_pickle=True)
        split, seed = os.path.basename(os.path.dirname(f)).rsplit("_s", 1)
        out.append(dict(split=split, seed=int(seed), p=z["p"].astype(np.float32), vid=z["vid"], y=z["y"], toa=z["toa"],
                        attn=z["attn"] if "attn" in z.files else None))
    return out


def sweep_figure(runs, schemes, exps, path):
    fig, axes = plt.subplots(2, len(schemes), figsize=(5 * len(schemes), 7), squeeze=False)
    for j, s in enumerate(schemes):
        for e in exps:
            sw = [json.load(open(f)) for f in glob.glob(os.path.join(runs, e, s, "*", "sweep.json"))]
            if not sw:
                continue
            d = pd.DataFrame([r for x in sw for r in x]).groupby("threshold").mean().reset_index()
            axes[0, j].plot(d.threshold, d.mean_tta_s, marker=".", label=e)
            axes[1, j].plot(d.recall, d.precision, marker=".", label=e)
        axes[0, j].set_title(SCHEME_LABEL.get(s, s)); axes[0, j].set_xlabel("threshold"); axes[0, j].set_ylabel("mean TTA (s before accident)")
        axes[1, j].set_xlabel("recall (videos)"); axes[1, j].set_ylabel("precision (videos)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def early_late(runs, cfg, out_dir):
    """Per positive test video and per (split, seed): first alarm of main vs det_graph at a fixed threshold."""
    el = cfg["early_late"]
    th, tie, fps = el["threshold"], el["tie_seconds"], 10.0
    rows = []
    for s in el["scheme"]:
        A = {(r["split"], r["seed"]): r for r in preds(runs, "main", s)}
        B = {(r["split"], r["seed"]): r for r in preds(runs, "det_graph", s)}
        for key in sorted(set(A) & set(B)):
            a, b = A[key], B[key]
            assert (a["vid"] == b["vid"]).all()
            for i in np.where(a["y"] > 0)[0]:
                fa, fb = first_alarm(a["p"][i], a["toa"][i], th), first_alarm(b["p"][i], b["toa"][i], th)
                ta = (a["toa"][i] - fa) / fps if fa is not None else None
                tb = (b["toa"][i] - fb) / fps if fb is not None else None
                if ta is None and tb is None: cat = "both miss"
                elif tb is None: cat = "only main warns"
                elif ta is None: cat = "only baseline warns"
                elif ta - tb > tie: cat = "main earlier"
                elif tb - ta > tie: cat = "baseline earlier"
                else: cat = "same time"
                rows.append(dict(scheme=s, split=key[0], seed=key[1], vid=a["vid"][i], tta_main=ta, tta_base=tb,
                                 margin=(ta or 0) - (tb or 0) if ta is not None and tb is not None else np.nan, category=cat,
                                 maxp_main=float(a["p"][i][:int(a["toa"][i])].max())))
    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame()
    df.to_csv(os.path.join(out_dir, "early_late_per_video.csv"), index=False)
    share = df.groupby(["scheme", "seed"]).category.value_counts(normalize=True).unstack(fill_value=0)
    summ = share.groupby("scheme").agg(["mean", "std"]).round(3)
    summ.columns = [f"{a} ({b})" for a, b in summ.columns]
    summ.to_csv(os.path.join(out_dir, "early_late_summary.csv"))
    return df, summ


def errors(runs, scheme, out_dir, th=0.5, fps=10.0):
    """main model: false alarms on negatives, late detections (< 0.5 s), misses; mean over seeds and splits."""
    rows, examples = [], []
    for r in preds(runs, "main", scheme):
        for i, (p, y, t, v) in enumerate(zip(r["p"], r["y"], r["toa"], r["vid"])):
            fa = first_alarm(p, t if y > 0 else len(p), th)
            if y == 0:
                kind = "false alarm" if fa is not None else "correct rejection"
            elif fa is None:
                kind = "miss"
            else:
                kind = "late (< 0.5 s)" if (t - fa) / fps < 0.5 else "timely"
            rows.append(dict(seed=r["seed"], split=r["split"], vid=v, y=int(y), kind=kind, maxp=float(p[:int(t)].max()),
                             tta=(t - fa) / fps if (y > 0 and fa is not None) else np.nan))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    summary = df.groupby(["seed", "y"]).kind.value_counts(normalize=True).groupby(["y", "kind"]).agg(["mean", "std"]).round(3)
    summary.to_csv(os.path.join(out_dir, f"errors_{scheme}.csv"))
    df.to_csv(os.path.join(out_dir, f"errors_per_video_{scheme}.csv"), index=False)
    return summary


def demo_selection(el_df, err_dfs, cfg, out_dir):
    """Fixed rule from configs/experiments.yaml (demo): largest earlier-warning margins of main over det_graph
    (seed 0), plus the most confident false alarm and the clearest miss of main (seed 0)."""
    d = cfg["demo"]
    sel = []
    e0 = el_df[(el_df.seed == 0) & (el_df.category == "main earlier")].sort_values("margin", ascending=False)
    for _, r in e0.drop_duplicates("vid").head(d["top_k_earlier"]).iterrows():
        sel.append(dict(kind="earlier warning", vid=r.vid, scheme=r.scheme, split=r.split, margin_s=float(r.margin)))
    for scheme, df in err_dfs.items():
        df0 = df[df.seed == 0]
        for kind, n, asc in (("false alarm", d["false_alarms"], False), ("miss", d["misses"], True)):
            cand = df0[df0.kind == kind].sort_values("maxp", ascending=asc).head(n)
            for _, r in cand.iterrows():
                sel.append(dict(kind=kind, vid=r.vid, scheme=scheme, split=r.split, maxp=float(r.maxp)))
    # one false alarm and one miss overall (the first scheme that has them)
    out, seen = [], set()
    for s in sel:
        if s["kind"] in ("false alarm", "miss") and s["kind"] in seen:
            continue
        seen.add(s["kind"]) if s["kind"] != "earlier warning" else None
        out.append(s)
    json.dump(dict(rule=d, clips=out), open(os.path.join(out_dir, "demo_selection.json"), "w"), indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "experiments.yaml")))
    tdir, fdir = os.path.join(a.out, "tables"), os.path.join(a.out, "figures")
    os.makedirs(tdir, exist_ok=True); os.makedirs(fdir, exist_ok=True)
    df = load_runs(a.runs)
    df.to_csv(os.path.join(tdir, "runs_raw.csv"), index=False)
    summ = summarise(df)
    summ.to_csv(os.path.join(tdir, "summary_all.csv"), index=False)
    base = ["frame_lstm", "det_lstm", "det_graph", "main"]
    abl = [e["name"] for e in cfg["experiments"]]
    print(table(summ, base, cfg["schemes"]["comparable"], os.path.join(tdir, "main_comparable")).to_string(index=False))
    print(table(summ, abl, cfg["schemes"]["clean"], os.path.join(tdir, "main_and_ablations_clean")).to_string(index=False))
    print(table(summ, cfg["cross"]["experiments"], ["cross_ccd2dota", "cross_dota2ccd"], os.path.join(tdir, "cross_dataset")).to_string(index=False))
    sweep_figure(a.runs, cfg["schemes"]["clean"] + cfg["schemes"]["comparable"], base, os.path.join(fdir, "threshold_sweep.png"))
    el_df, el_summ = early_late(a.runs, cfg, tdir)
    print(el_summ.to_string())
    err = {}
    for s in cfg["schemes"]["clean"]:
        print(errors(a.runs, s, tdir))
        e = os.path.join(tdir, f"errors_per_video_{s}.csv")
        if os.path.exists(e):
            err[s] = pd.read_csv(e)
    if not el_df.empty:
        print(json.dumps(demo_selection(el_df, err, cfg, tdir), indent=1))


if __name__ == "__main__":
    main()

"""Surprise mining on the JAAD test intent windows.

A window is surprising when the bbox-only baseline (bbox_lstm, probability averaged over seeds) is
confidently wrong: P(cross) < low but the pedestrian crosses, or > high but they do not.
For ALL surprising windows we report the hit rate (correct at 0.5) of every model, per seed.
Per-event cue explanations: delta_g = P_main - P_main_without_g (seed means), and the main model's
mean gate weights. These show what the model relied on, not why the person behaved as they did.
Demo clips are chosen by a fixed rule: top-k tracks by |P_main - P_base| where main is right,
plus the fail_k most confident tracks where main is wrong.
"""
import glob
import json
import os

import numpy as np
import pandas as pd


def seed_probs(runs_dir, exp):
    fs = sorted(glob.glob(os.path.join(runs_dir, f"{exp}_s*", "preds.npz")))
    return np.stack([np.load(f)["intent_p"] for f in fs]) if fs else None   # (seeds, N)


def mine(runs_dir, data, cfg, exps, out_dir):
    I = data["test"]["intent"]
    y = I["y"].astype(int)
    sc = cfg["surprise"]
    base = seed_probs(runs_dir, "bbox_lstm")
    main = seed_probs(runs_dir, "main")
    pb, pm = base.mean(0), main.mean(0)
    surp = ((pb < sc["low"]) & (y == 1)) | ((pb > sc["high"]) & (y == 0))
    tracks = {(I["meta"][i][0], I["meta"][i][1]) for i in np.where(surp)[0]}
    rows = []
    for e in exps:
        P = seed_probs(runs_dir, e)
        if P is None:
            continue
        hits = [((p[surp] >= 0.5) == y[surp]).mean() if surp.any() else float("nan") for p in P]
        allacc = [((p >= 0.5) == y).mean() for p in P]
        rows.append(dict(model=e, surprising_windows=int(surp.sum()), surprising_tracks=len(tracks),
                         hit_rate_mean=round(float(np.mean(hits)), 3), hit_rate_std=round(float(np.std(hits)), 3),
                         hit_rate_seed_mean_prob=round(float(((P.mean(0)[surp] >= 0.5) == y[surp]).mean()), 3)
                         if surp.any() else float("nan"),
                         accuracy_all_windows=round(float(np.mean(allacc)), 3)))
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "surprise_summary.csv"), index=False)

    gates = [np.load(f)["gates"] for f in sorted(glob.glob(os.path.join(runs_dir, "main_s*", "preds.npz")))]
    g = np.mean(gates, 0).mean(1) if gates and gates[0].size else None     # (N, G)
    groups = next(e for e in cfg["experiments"] if e["name"] == "main")["groups"]
    ev = pd.DataFrame(dict(idx=np.arange(len(y)), vid=[m[0] for m in I["meta"]], pid=[m[1] for m in I["meta"]],
                           first_frame=[m[2] for m in I["meta"]], last_frame=[m[3] for m in I["meta"]],
                           tte=[m[6] for m in I["meta"]], event_frame=[m[7] for m in I["meta"]],
                           y=y, p_base=pb, p_main=pm, surprising=surp,
                           main_correct=((pm >= 0.5) == y)))
    for gname in cfg["explain_groups"]:
        P = seed_probs(runs_dir, f"main_no_{gname}")
        if P is not None:
            ev[f"delta_{gname}"] = pm - P.mean(0)
    if g is not None:
        for k, gname in enumerate(groups):
            ev[f"gate_{gname}"] = g[:, k]
    ev["margin"] = (ev.p_main - ev.p_base).abs()
    ev[ev.surprising].to_csv(os.path.join(out_dir, "surprise_events.csv"), index=False)

    def pick(df, k, key):
        df = df.sort_values(key, ascending=False).drop_duplicates(["vid", "pid"])
        return df.head(k)
    s = ev[ev.surprising]
    good = pick(s[s.main_correct], sc["demo_top_k"], "margin")
    bad = s[~s.main_correct].assign(conf=lambda d: (d.p_main - 0.5).abs())
    if len(bad) < sc["demo_fail_k"]:   # fall back to any wrong window if too few surprising failures
        bad = ev[~ev.main_correct].assign(conf=lambda d: (d.p_main - 0.5).abs())
    bad = pick(bad, sc["demo_fail_k"], "conf")
    sel = [dict(kind="success", **r) for r in good.to_dict("records")] + \
          [dict(kind="failure", **r) for r in bad.to_dict("records")]
    sel = [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in sel]
    json.dump(dict(rule=f"top {sc['demo_top_k']} surprising tracks by |p_main - p_base| with main correct, "
                        f"plus {sc['demo_fail_k']} most confident main failures", clips=sel),
              open(os.path.join(out_dir, "demo_selection.json"), "w"), indent=1)
    return summary, sel

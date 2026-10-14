"""Shortcut diagnostic: can the video label be predicted from information that says nothing about an upcoming
accident? If a classifier on the FIRST frame only (or on detection counts) separates positives from negatives,
then AP on that dataset mostly measures source/quality differences, not anticipation.

Probes (logistic regression, 5-fold GroupKFold with the same groups as the clean split; ROC-AUC on held-out folds):
  first_frame_emb   DINOv2 CLS embedding of frame 0
  first10_frame_emb mean DINOv2 CLS embedding of frames 0-9 (at least 2 s before any CCD accident, toa >= 30)
  n_objects_first10 mean number of detections in frames 0-9
python -m src.analysis.shortcut --feat_dir F --ccd_root C --dota_meta D --out results/tables/shortcut_probe.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import loader


def probe(X, y, groups):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True); ap.add_argument("--ccd_root", required=True)
    ap.add_argument("--dota_meta", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for ds in ("ccd", "dota"):
        idx = loader.build_index(a.feat_dir, ds, ccd_root=a.ccd_root, dota_meta_dir=a.dota_meta)
        F, N = [], []
        for f in idx["files"]:
            z = np.load(f)
            F.append(z["frame_emb"][:10].astype(np.float32)); N.append(z["det"][:10, :, 7].sum(1).mean())
        F, N = np.stack(F), np.array(N)[:, None]
        y, g = idx["y"], np.asarray(idx["groups"]["source"])
        for name, X in (("first_frame_emb", F[:, 0]), ("first10_frame_emb", F.mean(1)), ("n_objects_first10", N)):
            rows.append(dict(dataset=ds, probe=name, roc_auc=round(probe(X, y, g), 4), videos=len(y), positives=int(y.sum()),
                             min_toa_frames=int(idx["toa"][y > 0].min())))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()

"""Accident-anticipation metrics: AP, mTTA, TTA@R80.

`evaluation` reproduces UString's evaluation (github.com/Cogito2012/UString, MIT licence, Bao et al. 2020),
as also used unchanged by STAGNet (eval_utils.py), so our numbers follow the same definition as the
published ones. Two properties of that definition, stated so the report can be precise:
  * a video counts as a positive prediction at threshold Th if ANY scored frame reaches Th; for positive
    videos only frames before the time of accident (toa) are scored;
  * TTA per video is (1 - first_alarm_frame / toa), averaged over true-positive videos, then multiplied by
    the clip length in seconds (T / fps), i.e. it is relative to toa but expressed in clip seconds.
`threshold_sweep` adds per-threshold precision, recall and absolute mean TTA (seconds before toa) for plots.
"""
import numpy as np


def _scored(all_pred, all_labels, toas):
    out = []
    for p, y, t in zip(all_pred, all_labels, toas):
        out.append(p[:int(t)] if y > 0 else p)
    return out


def evaluation(all_pred, all_labels, time_of_accidents, fps=10.0):
    """all_pred (N, T) frame-level accident probabilities; all_labels (N,); time_of_accidents (N,) frame index."""
    preds_eval = _scored(all_pred, all_labels, time_of_accidents)
    min_pred = min(np.min(p) for p in preds_eval)
    n_frames = sum(len(p) for p in preds_eval)
    total_seconds = all_pred.shape[1] / fps
    Precision, Recall, Time = np.zeros(n_frames), np.zeros(n_frames), np.zeros(n_frames)
    cnt = 0
    for Th in np.arange(max(min_pred, 0), 1.0, 0.001):
        Tp = Tp_Fp = time = counter = 0.0
        for i in range(len(preds_eval)):
            tp = np.where(preds_eval[i] * all_labels[i] >= Th)
            Tp += float(len(tp[0]) > 0)
            if len(tp[0]) > 0:
                time += tp[0][0] / float(time_of_accidents[i])
                counter += 1
            Tp_Fp += float(len(np.where(preds_eval[i] >= Th)[0]) > 0)
        if Tp_Fp == 0:
            continue
        Precision[cnt] = Tp / Tp_Fp
        if np.sum(all_labels) == 0:
            continue
        Recall[cnt] = Tp / np.sum(all_labels)
        if counter == 0:
            continue
        Time[cnt] = 1 - time / counter
        cnt += 1
    idx = np.argsort(Recall)
    Precision, Recall, Time = Precision[idx], Recall[idx], Time[idx]
    _, rep = np.unique(Recall, return_index=1)
    rep = rep[1:]
    new_Time, new_Precision = np.zeros(len(rep)), np.zeros(len(rep))
    for i in range(len(rep) - 1):
        new_Time[i] = np.max(Time[rep[i]:rep[i + 1]])
        new_Precision[i] = np.max(Precision[rep[i]:rep[i + 1]])
    new_Time[-1] = Time[rep[-1]]
    new_Precision[-1] = Precision[rep[-1]]
    new_Recall = Recall[rep]
    AP = 0.0
    if new_Recall[0] != 0:
        AP += new_Precision[0] * (new_Recall[0] - 0)
    for i in range(1, len(new_Precision)):
        AP += (new_Precision[i - 1] + new_Precision[i]) * (new_Recall[i] - new_Recall[i - 1]) / 2
    mTTA = np.mean(new_Time) * total_seconds
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    TTA_R80 = sort_time[np.argmin(np.abs(sort_recall - 0.8))] * total_seconds
    return float(AP), float(mTTA), float(TTA_R80)


def first_alarm(pred, toa, th):
    """Frame index of the first frame (before toa) with pred >= th, or None."""
    hit = np.where(pred[:int(toa)] >= th)[0]
    return int(hit[0]) if len(hit) else None


def threshold_sweep(all_pred, all_labels, toas, fps=10.0, ths=np.round(np.arange(0.05, 1.0, 0.05), 2)):
    """Per threshold: precision, recall (video level, as in `evaluation`), false-alarm rate on negative videos,
    and mean absolute time-to-accident in seconds over detected positives."""
    rows = []
    pos = np.asarray(all_labels) > 0
    for th in ths:
        alarms = np.array([first_alarm(p, t if y > 0 else len(p), th) is not None
                           for p, y, t in zip(all_pred, all_labels, toas)])
        tta = [(t - first_alarm(p, t, th)) / fps for p, y, t in zip(all_pred, all_labels, toas)
               if y > 0 and first_alarm(p, t, th) is not None]
        tp = (alarms & pos).sum()
        rows.append(dict(threshold=float(th), precision=float(tp / max(alarms.sum(), 1)), recall=float(tp / max(pos.sum(), 1)),
                         false_alarm_rate=float((alarms & ~pos).sum() / max((~pos).sum(), 1)),
                         mean_tta_s=float(np.mean(tta)) if tta else float("nan")))
    return rows

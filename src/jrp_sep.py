import sys
import os
from itertools import combinations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, r"D:\EMG\src")
from features import extract_features
from data import load_cropped_raw

X, y = load_cropped_raw()


def binary_rp(x, m=3, tau=2, eps=0.25):
    T = len(x)
    L = T - (m - 1) * tau
    idx = np.arange(m)[None, :] * tau + np.arange(L)[:, None]
    E = x[idx]
    D = np.sqrt(((E[:, None, :] - E[None, :, :]) ** 2).sum(-1))
    D = D / (D.max() + 1e-9)
    return (1.0 - D / eps) > 0.5


def _run_lens(arr):
    d = np.diff(arr)
    starts = np.r_[0, np.where(d == 1)[0] + 1]
    ends = np.r_[np.where(d == -1)[0] + 1, len(arr)]
    return np.diff(np.concatenate([[0], np.where(arr)[0] + 1]))

def line_features(b):
    rr = b.mean()
    bsum = b.sum() + 1e-9

    dline = np.diag(b, 1)
    dsel = dline[dline]
    dcount = np.diff(np.concatenate([[0], np.where(dline)[0] + 1, [0]]))
    dlens = dcount[dcount > 1]  # contiguous runs >= 2
    deter = (dlens.sum() / bsum) if dlens.size else 0.0

    vline = b[:, 0]
    vcount = np.diff(np.concatenate([[0], np.where(vline)[0] + 1, [0]]))
    vlens = vcount[vcount > 1]
    lam = (vlens.sum() / bsum) if vlens.size else 0.0
    if vlens.size:
        vc = np.bincount(vlens).astype(float)
        p = vc / vc.sum()
        ent = -(p * np.log(p + 1e-12)).sum()
    else:
        ent = 0.0
    return [rr, deter, lam, ent]


def jrqa_features(X, pairs=None):
    if pairs is None:
        pairs = list(combinations(range(X.shape[1]), 2))
    rps = np.stack([np.stack([binary_rp(s[c]) for c in range(X.shape[1])]) for s in X])
    out = []
    for a, b in pairs:
        J = rps[:, a] & rps[:, b]
        feats = np.array([line_features(J[i]) for i in range(len(J))])
        out.append(feats)
    return np.hstack(out)


def main():
    print("computing JRQA features...")
    Fjrqa = jrqa_features(X)
    print("JRQA matrix:", Fjrqa.shape)
    Fjrqa_s = StandardScaler().fit_transform(Fjrqa)

    Fcls = extract_features(X)
    Fcls_s = StandardScaler().fit_transform(Fcls)

    print("\n=== 5-fold CV (RF, macro-F1) ===")
    for name, F in [
        ("classical-80", Fcls_s),
        ("JRQA-112", Fjrqa_s),
        ("both-192", np.hstack([Fcls_s, Fjrqa_s])),
    ]:
        rf = RandomForestClassifier(n_estimators=300, random_state=0)
        scores = cross_val_score(rf, F, y, cv=5, scoring="f1_macro")
        print(f"{name:14s} {scores.mean():.4f} +/- {scores.std():.4f}")


if __name__ == "__main__":
    main()

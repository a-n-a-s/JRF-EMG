import os
import sys
import time
from itertools import combinations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_cropped_raw
from features import extract_features
from jrp_sep import binary_rp, line_features
from rp_samples import recurrence_dist

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG, exist_ok=True)

FOLDS, SEED = 5, 42  # identical to the TileParallelCNN Kaggle run


def rqa_features(X, chans=None):
    """Per-channel RQA-lite: RR, DET, LAM, entropy, max-line (from rp_sep)."""
    if chans is None:
        chans = range(X.shape[1])
    out = []
    for s in X:
        feats = []
        for c in chans:
            R = recurrence_dist(s[c])
            b = (R > 0.5).astype(int)
            rr = b.mean()
            dline = np.diag(b, 1)
            dlens, run = [], 0
            for v in dline:
                if v:
                    run += 1
                elif run >= 2:
                    dlens.append(run); run = 0
                else:
                    run = 0
            if run >= 2:
                dlens.append(run)
            deter = (sum(dlens) / (b.sum() + 1e-9)) if dlens else 0.0
            vline = b[:, 0]
            vlens, run = [], 0
            for v in vline:
                if v:
                    run += 1
                elif run >= 2:
                    vlens.append(run); run = 0
                else:
                    run = 0
            if run >= 2:
                vlens.append(run)
            lam = (sum(vlens) / (b.sum() + 1e-9)) if vlens else 0.0
            ent = 0.0
            if vlens:
                vc = np.bincount(vlens).astype(float)
                p = vc / vc.sum()
                ent = -(p * np.log(p + 1e-12)).sum()
            maxdl = max(dlens) if dlens else 0
            feats += [rr, deter, lam, ent, maxdl / len(b)]
        out.append(feats)
    return np.array(out)


def jrqa_features(X, pairs=None):
    """JRQA-lite: RR/DET/LAM/entropy per electrode pair (from jrp_sep)."""
    if pairs is None:
        pairs = list(combinations(range(X.shape[1]), 2))
    rps = np.stack([np.stack([binary_rp(s[c]) for c in range(X.shape[1])]) for s in X])
    out = []
    for a, b in pairs:
        J = rps[:, a] & rps[:, b]
        feats = np.array([line_features(J[i]) for i in range(len(J))])
        out.append(feats)
    return np.hstack(out)


def cv_rf(F, y, folds=FOLDS, seed=SEED):
    """Stratified shuffled CV on the exact folds used for the TileParallelCNN."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs, f1s = [], []
    for tr, te in skf.split(F, y):
        sc = StandardScaler().fit(F[tr])
        rf = RandomForestClassifier(n_estimators=300, random_state=seed)
        rf.fit(sc.transform(F[tr]), y[tr])
        p = rf.predict(sc.transform(F[te]))
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro"))
    return np.mean(accs), np.mean(f1s), np.std(f1s)


def main():
    X, y = load_cropped_raw()
    print(f"data: {X.shape}  folds={FOLDS}  seed={SEED}")

    print("classical-80 ...")
    Fcls = extract_features(X)

    print("RQA-40 (single-channel RPs) ...")
    Frqa = rqa_features(X)

    print("JRQA-112 (joint RPs, 28 pairs x 4) ...")
    t0 = time.time()
    Fjrqa = jrqa_features(X)
    print(f"  done in {time.time()-t0:.0f}s")

    print("portrait tiles -> mean/diag summaries ...")
    # JRP portrait signal-processing descriptor: per-pair RR (28) + diag RR (8) = 36 dims
    P = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                             "jrp_portraits_m3t2e0.5_tile50.npy")) / 255.0
    tile = 50
    pairs = list(combinations(range(8), 2))
    Fport = np.empty((len(P), 36))
    for k, (a, b) in enumerate(pairs):
        Fport[:, k] = P[:, a * tile:(a + 1) * tile, b * tile:(b + 1) * tile].mean(axis=(1, 2))
    for c in range(8):
        Fport[:, 28 + c] = P[:, c * tile:(c + 1) * tile, c * tile:(c + 1) * tile].mean(axis=(1, 2))

    methods = {
        "classical-80": Fcls,
        "RQA-40": Frqa,
        "JRQA-112": Fjrqa,
        "classical+JRQA-192": np.hstack([Fcls, Fjrqa]),
        "all (class+RQA+JRQA)": np.hstack([Fcls, Frqa, Fjrqa]),
        "portrait-36 (JRP mean)": Fport,
        "portrait+classical-116": np.hstack([Fport, Fcls]),
    }

    print(f"\n{'method':26s} {'acc':>8s} {'f1':>8s} {'f1 std':>8s}")
    rows = []
    for name, F in methods.items():
        acc, f1, s = cv_rf(F, y)
        print(f"{name:26s} {acc:8.4f} {f1:8.4f} {s:8.4f}")
        rows.append((name, acc, f1, s))

    print("\n--- CNN comparisons (same folds) ---")
    print("TileParallelCNN   0.8242   0.8182  0.0326  (Kaggle run)")
    print("Raw ChannelCNN    0.79-0.82        (earlier scratch models)")

    with open(os.path.join(FIG, "fair_comparison.csv"), "w") as f:
        f.write("method,acc,f1,f1_std\n")
        for name, acc, f1, s in rows:
            f.write(f"{name},{acc:.4f},{f1:.4f},{s:.4f}\n")
    print(f"\nsaved {os.path.normpath(os.path.join(FIG, 'fair_comparison.csv'))}")


if __name__ == "__main__":
    main()

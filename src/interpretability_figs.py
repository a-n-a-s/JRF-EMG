import os
import sys
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import GESTURES, load_cropped_raw
from recurrence import embed, graded_rp

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(FIG, exist_ok=True)

TILE, G, M, TAU, EPS = 50, 8, 3, 2, 0.5
P = np.load(os.path.join(DATA, "jrp_portraits_m3t2e0.5_tile50.npy")) / 255.0
y = np.load(os.path.join(DATA, "labels.npy"))
n_gest = len(GESTURES)

mean_portraits = np.stack([P[y == c].mean(axis=0) for c in range(n_gest)])
global_mean = P.mean(axis=0)

pairs = list(combinations(range(G), 2))
pair_names = [f"ch{a+1}-{b+1}" for a, b in pairs]

sig = np.empty((n_gest, len(pairs)))
for k, (a, b) in enumerate(pairs):
    tile = P[:, a * TILE:(a + 1) * TILE, b * TILE:(b + 1) * TILE]
    for c in range(n_gest):
        sig[c, k] = tile[y == c].mean()

CMAP = "magma"
VD, VMAX = 0.10, 0.40

# ------------------------------------------------------------------
# FIG 1: pipeline schematic -- raw window -> portrait -> fingerprint
# ------------------------------------------------------------------
X, yraw = load_cropped_raw()
c_thumb = int(np.nonzero(yraw == 4)[0][0])
win = X[c_thumb]
port = mean_portraits[4]

fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 0.9, 1.1, 0.9], wspace=0.35)

ax = fig.add_subplot(gs[0])
for ch in range(G):
    v = win[ch]
    v = (v - v.mean()) / (v.std() + 1e-8)
    ax.plot(v + ch * 3.0, lw=0.6, color=plt.cm.turbo(ch / 7))
ax.set_yticks([ch * 3.0 for ch in range(G)])
ax.set_yticklabels([f"ch{ch+1}" for ch in range(G)], fontsize=8)
ax.set_xticks([])
ax.set_ylim(-2, 24)
ax.set_title("(a) raw window\n[8 x 150]", fontsize=11)

ax = fig.add_subplot(gs[1])
e = embed(win[4], M, TAU)
R = graded_rp(e, eps=EPS)
ax.imshow(R, cmap=CMAP, vmin=0, vmax=1)
ax.set_title("(b) self-RP of ch5\n(recurrence of thumb muscle)", fontsize=11)
ax.axis("off")

ax = fig.add_subplot(gs[2])
ax.imshow(port, cmap=CMAP, vmin=0, vmax=0.8)
for k in range(1, G):
    ax.axhline(k * TILE - 0.5, color="w", lw=0.4, alpha=0.6)
    ax.axvline(k * TILE - 0.5, color="w", lw=0.4, alpha=0.6)
ax.set_title("(c) JRP portrait (thumb)\ndiag = self-RP, off-diag = JRP", fontsize=11)
ax.axis("off")

ax = fig.add_subplot(gs[3])
fp = sig[4]
colors = [plt.cm.magma((v - VD) / (VMAX - VD)) for v in fp]
ax.barh(range(28)[::-1], fp, color=colors, edgecolor="none", height=0.8)
ax.set_yticks(range(28)[::-1])
ax.set_yticklabels(pair_names[::-1], fontsize=6.5)
ax.set_xlim(VD, VMAX)
ax.set_xlabel("mean JRP density", fontsize=9)
ax.set_title("(d) synergy fingerprint\n(thumb: ch5 pairs high)", fontsize=11)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

fig.suptitle("From raw sEMG to a muscle-synergy fingerprint (thumb gesture)",
             fontsize=14, y=0.98)
fig.savefig(os.path.join(FIG, "interpretability_pipeline.png"),
            dpi=130, bbox_inches="tight")
plt.close(fig)
print("saved", os.path.join(FIG, "interpretability_pipeline.png"))

# ------------------------------------------------------------------
# FIG 2: differential portraits (gesture minus global mean)
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
vmax = max(np.abs(mean_portraits[c] - global_mean).max() for c in range(n_gest))
for c in range(n_gest):
    ax = axes[c // 4, c % 4]
    diff = mean_portraits[c] - global_mean
    im = ax.imshow(diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    for k in range(1, G):
        ax.axhline(k * TILE - 0.5, color="k", lw=0.3, alpha=0.5)
        ax.axvline(k * TILE - 0.5, color="k", lw=0.3, alpha=0.5)
    ax.set_title(GESTURES[c], fontsize=13)
    ax.axis("off")
axes[1, 3].axis("off")
fig.suptitle("Differential fingerprints: mean portrait minus global mean\n"
             "(red = above-average co-activation, blue = below)",
             fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "interpretability_differential.png"),
            dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved", os.path.join(FIG, "interpretability_differential.png"))

# ------------------------------------------------------------------
# FIG 3: circular chord diagrams -- electrodes on a forearm ring
# ------------------------------------------------------------------
def chord_ax(ax, fp, title, cmap=plt.cm.magma, vmin=VD, vmax=VMAX):
    n = G
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    r = 1.0
    xs = r * np.cos(angles)
    ys = r * np.sin(angles)

    ax.add_patch(plt.Circle((0, 0), r, fill=False, color="0.7", lw=1.2))
    for i in range(n):
        ax.scatter(xs[i], ys[i], s=260, color="0.15", zorder=5)
        ax.text(xs[i] * 1.28, ys[i] * 1.28, f"ch{i+1}",
                ha="center", va="center", fontsize=9, fontweight="bold")

    for k, (a, b) in enumerate(pairs):
        d = fp[k]
        if d < vmin:
            continue
        color = cmap((d - vmin) / (vmax - vmin))
        xa, ya = xs[a], ys[a]
        xb, yb = xs[b], ys[b]
        cx, cy = (xa + xb) / 2 * 0.52, (ya + yb) / 2 * 0.52
        verts = [(xa, ya), (cx, cy), (xb, yb)]
        path = Path(verts, [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        lw = 0.6 + 4.2 * (d - vmin) / (vmax - vmin)
        ax.add_patch(PathPatch(path, fill=False, lw=lw, color=color,
                               alpha=0.85, zorder=3))
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12)

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for c in range(n_gest):
    chord_ax(axes[c // 4, c % 4], sig[c], GESTURES[c])
axes[1, 3].axis("off")
fig.suptitle("Muscle-synergy chords: electrodes on the forearm ring, "
             "chord thickness/color = joint-recurrence density",
             fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "interpretability_chords.png"),
            dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved", os.path.join(FIG, "interpretability_chords.png"))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
chord_ax(axes[0], sig[4], "thumb")
chord_ax(axes[1], sig[5], "rest")
chord_ax(axes[2], sig[6], "victory")
fig.suptitle("Chord diagrams: thick bright chords = strongly co-activating "
             "electrode pairs", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "interpretability_chords_highlight.png"),
            dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved", os.path.join(FIG, "interpretability_chords_highlight.png"))

# ------------------------------------------------------------------
# FIG 4: dendrogram of the 7 fingerprints
# ------------------------------------------------------------------
from scipy.cluster.hierarchy import linkage, dendrogram

Z = linkage(sig, method="ward")
fig, ax = plt.subplots(figsize=(10, 4.5))
dendrogram(Z, labels=GESTURES, ax=ax, color_threshold=0.6 * Z[:, 2].max(),
           leaf_font_size=11)
ax.set_ylabel("ward distance between fingerprints")
ax.set_title("Hierarchical clustering of the 7 gesture fingerprints "
             "(28-dim synergy signatures)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "interpretability_dendrogram.png"),
            dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved", os.path.join(FIG, "interpretability_dendrogram.png"))

# ------------------------------------------------------------------
# FIG 5: per-pair mean +/- std with gesture rank (thumb shown)
# ------------------------------------------------------------------
pairs_ = np.asarray(pairs)
std = np.empty((n_gest, len(pairs)))
for k, (a, b) in enumerate(pairs):
    tile = P[:, a * TILE:(a + 1) * TILE, b * TILE:(b + 1) * TILE]
    for c in range(n_gest):
        std[c, k] = tile[y == c].std()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
c = 4
order = np.argsort(sig[c])[::-1]
ax = axes[0]
labels = [pair_names[k] for k in order]
vals = [sig[c, k] for k in order]
errs = [std[c, k] for k in order]
ax.errorbar(vals, range(28), xerr=errs, fmt="o", ms=5, color="0.2",
            ecolor="0.6", elinewidth=1, capsize=2)
ax.set_yticks(range(28))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("mean JRP density (mean +/- std over windows)")
ax.set_title(f"thumb: ranked electrode pairs (ch5-centered at top)",
             fontsize=12)
ax.invert_yaxis()

ax = axes[1]
c = 5
order = np.argsort(sig[c])[::-1]
labels = [pair_names[k] for k in order]
vals = [sig[c, k] for k in order]
errs = [std[c, k] for k in order]
ax.errorbar(vals, range(28), xerr=errs, fmt="o", ms=5, color="0.2",
            ecolor="0.6", elinewidth=1, capsize=2)
ax.set_yticks(range(28))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("mean JRP density (mean +/- std over windows)")
ax.set_title("rest: ranked electrode pairs (uniformly low)", fontsize=12)
ax.invert_yaxis()

fig.suptitle("Per-sample variability of the synergy fingerprint "
             "(thumb vs rest)", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "interpretability_variability.png"),
            dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved", os.path.join(FIG, "interpretability_variability.png"))

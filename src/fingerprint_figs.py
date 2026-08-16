import os
import sys
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import GESTURES

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(FIG, exist_ok=True)

TILE = 50
G = 8
P = np.load(os.path.join(DATA, "jrp_portraits_m3t2e0.5_tile50.npy")) / 255.0
y = np.load(os.path.join(DATA, "labels.npy"))

n_gest = len(GESTURES)

# ---- 1. per-gesture mean portrait (the fingerprint figure) ----
mean_portraits = np.stack([P[y == c].mean(axis=0) for c in range(n_gest)])

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for c in range(n_gest):
    ax = axes[c // 4, c % 4]
    im = ax.imshow(mean_portraits[c], cmap="magma", vmin=0, vmax=0.8)
    for k in range(1, G):  # tile gridlines so the 8x8 block structure reads
        ax.axhline(k * TILE - 0.5, color="w", lw=0.4, alpha=0.6)
        ax.axvline(k * TILE - 0.5, color="w", lw=0.4, alpha=0.6)
    ax.set_title(GESTURES[c], fontsize=14)
    ax.axis("off")
axes[1, 3].axis("off")
fig.suptitle("Mean recurrence-portrait fingerprint per gesture\n"
             "(diagonal = self-RP of each electrode, off-diagonal = joint RP of each electrode pair)",
             fontsize=16, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fingerprint_mean_portraits.png"), dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved", os.path.join(FIG, "fingerprint_mean_portraits.png"))

# ---- 2. synergy signature: 28-pair mean RR x gesture ----
pairs = list(combinations(range(G), 2))
t = np.arange(TILE)
sig = np.empty((n_gest, len(pairs)))
for k, (a, b) in enumerate(pairs):
    tile = P[:, a * TILE:(a + 1) * TILE, b * TILE:(b + 1) * TILE]
    for c in range(n_gest):
        sig[c, k] = tile[y == c].mean()

pair_names = [f"ch{p[0]+1}-{p[1]+1}" for p in pairs]

fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(sig, cmap="magma", aspect="auto")
ax.set_yticks(range(n_gest))
ax.set_yticklabels(GESTURES, fontsize=10)
ax.set_xticks(range(len(pairs)))
ax.set_xticklabels(pair_names, rotation=90, fontsize=7)
ax.set_xlabel("electrode pair (joint recurrence density)")
ax.set_title("Muscle-synergy signature: mean JRP density per electrode pair per gesture")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("mean recurrence density")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fingerprint_synergy_heatmap.png"), dpi=120, bbox_inches="tight")
print("saved", os.path.join(FIG, "fingerprint_synergy_heatmap.png"))

# ---- 3. per-gesture top-5 pairs (interpretability table) ----
print("\n===== PER-GESTURE ELECTRODE-PAIR SIGNATURES =====")
for c in range(n_gest):
    order = np.argsort(sig[c])[::-1]
    top = [(pair_names[k], sig[c][k]) for k in order[:5]]
    bot = [(pair_names[k], sig[c][k]) for k in order[-3:]]
    print(f"\n{GESTURES[c]:16s}  (mean overall density = {sig[c].mean():.3f})")
    print("  highest co-activation:      " + "  ".join(f"{n}={d:.3f}" for n, d in top))
    print("  lowest co-activation:       " + "  ".join(f"{n}={d:.3f}" for n, d in bot))

with open(os.path.join(FIG, "fingerprint_synergy.csv"), "w") as f:
    f.write("pair," + ",".join(GESTURES) + "\n")
    for k, pn in enumerate(pair_names):
        f.write(pn + "," + ",".join(f"{sig[c, k]:.4f}" for c in range(n_gest)) + "\n")
print("\nsaved", os.path.join(FIG, "fingerprint_synergy.csv"))

# ---- 4. fingerprint distinctiveness: cosine similarity between mean portraits ----
vecs = mean_portraits.reshape(n_gest, -1)
vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
sim = vecs @ vecs.T  # cosine similarity

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(sim, cmap="coolwarm", vmin=0, vmax=1)
ax.set_xticks(range(n_gest))
ax.set_xticklabels(GESTURES, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(n_gest))
ax.set_yticklabels(GESTURES, fontsize=9)
for i in range(n_gest):
    for j in range(n_gest):
        ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Cosine similarity between mean recurrence-portrait fingerprints")
fig.colorbar(im, ax=ax, label="cosine similarity")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fingerprint_similarity.png"), dpi=120, bbox_inches="tight")
print("saved", os.path.join(FIG, "fingerprint_similarity.png"))

# ---- 5. distinctiveness on the 28-dim synergy signature (no diagonal) ----
# the honest interpretable descriptor: joint-recurrence density per electrode pair
sig_n = sig / (np.linalg.norm(sig, axis=1, keepdims=True) + 1e-9)
sim_sig = sig_n @ sig_n.T

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(sim_sig, cmap="coolwarm", vmin=0, vmax=1)
ax.set_xticks(range(n_gest))
ax.set_xticklabels(GESTURES, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(n_gest))
ax.set_yticklabels(GESTURES, fontsize=9)
for i in range(n_gest):
    for j in range(n_gest):
        ax.text(j, i, f"{sim_sig[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Cosine similarity of 28-dim muscle-synergy signatures (pairs only)")
fig.colorbar(im, ax=ax, label="cosine similarity")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fingerprint_synergy_similarity.png"), dpi=120, bbox_inches="tight")
print("saved", os.path.join(FIG, "fingerprint_synergy_similarity.png"))

# ---- 6. within-vs-between gesture signature distance (separability) ----
print("\n===== SYNERGY-SIGNATURE SEPARABILITY (28-dim, pairs only) =====")
for i in range(n_gest):
    off = [sim_sig[i, j] for j in range(n_gest) if j != i]
    print(f"  {GESTURES[i]:16s} self={sim_sig[i,i]:.3f}  vs-others max={max(off):.3f} mean={np.mean(off):.3f}")

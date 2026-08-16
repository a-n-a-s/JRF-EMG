import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_cropped_raw, GESTURES

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG, exist_ok=True)


def embed(x, m=3, tau=2):
    T = len(x)
    L = T - (m - 1) * tau
    idx = np.arange(m)[None, :] * tau + np.arange(L)[:, None]
    return x[idx]  # [L, m]


def recurrence_dist(x, m=3, tau=2, eps=0.25):
    """Graded recurrence matrix: normalized distance between embedded states.

    Returns 1 - normalized_distance so that 0 = far, 1 = recurrent (close).
    """
    E = embed(x, m, tau)  # [L, m]
    D = np.linalg.norm(E[:, None, :] - E[None, :, :], axis=2)  # [L, L]
    D = D / (D.max() + 1e-9)
    R = np.clip(1.0 - D / eps, 0.0, 1.0)  # graded: 1 = within eps
    return R


if __name__ == "__main__":
    X, y = load_cropped_raw()

    n_gest = len(GESTURES)
    fig, axes = plt.subplots(n_gest, 2, figsize=(8, 3.5 * n_gest))

    for i, g in enumerate(GESTURES):
        idx = np.where(y == i)[0]
        s = X[idx[0]]  # pick first sample of this gesture
        R = recurrence_dist(s[0])  # channel 1

        axw = axes[i, 0]
        axw.plot(s[0])
        axw.set_title(f"{g}  (ch1 raw)")

        axr = axes[i, 1]
        im = axr.imshow(R, origin="lower", cmap="magma", aspect="auto", vmin=0, vmax=1)
        axr.set_title(f"{g}  (ch1 recurrence plot)")

    fig.tight_layout()
    out = os.path.join(FIG, "rp_samples.png")
    fig.savefig(out, dpi=110)
    print("saved", out)

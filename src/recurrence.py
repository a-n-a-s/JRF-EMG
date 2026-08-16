import numpy as np
from scipy import ndimage


def embed(x, m=3, tau=2):
    """Time-delay embedding: [T] -> [L, m] with L = T - (m-1)*tau."""
    T = len(x)
    L = T - (m - 1) * tau
    idx = np.arange(m)[None, :] * tau + np.arange(L)[:, None]
    return x[idx]


def graded_rp(x, m=3, tau=2, eps=0.5):
    """Graded recurrence plot: 1 = recurrent, 0 = far, scaled by eps.

    Distances are normalized by the max pairwise distance, then mapped via
    clip(1 - D / eps, 0, 1) so that eps is the fraction of the max distance
    considered 'recurrent'.
    """
    E = embed(x, m, tau)
    D = np.sqrt(((E[:, None, :] - E[None, :, :]) ** 2).sum(-1))
    D = D / (D.max() + 1e-9)
    return np.clip(1.0 - D / eps, 0.0, 1.0).astype(np.float32)


def resize(M, tile):
    """Bilinear resize a square matrix to tile x tile."""
    return ndimage.zoom(M, tile / M.shape[0], order=1, prefilter=False).astype(np.float32)


def recurrence_portrait(s, m=3, tau=2, eps=0.5, tile=50):
    """Build the 8x8 recurrence-portrait fingerprint from one [8, T] window.

    Cell (i,j) = JRP(ch_i, ch_j), diagonal = RP(ch_i). Symmetric grid of
    tile x tile blocks -> [8*tile, 8*tile] single-channel image.
    """
    C = s.shape[0]
    rps = [resize(graded_rp(s[c], m, tau, eps), tile) for c in range(C)]
    grid = np.empty((C, tile, C, tile), dtype=np.float32)
    for i in range(C):
        for j in range(C):
            R = rps[i] if i == j else (rps[i] * rps[j])
            grid[i, :, j, :] = R
    return grid.reshape(C * tile, C * tile)


def portraits(X, m=3, tau=2, eps=0.5, tile=50):
    """[N, C, T] -> [N, C*tile, C*tile] portrait images (float32, 0..1)."""
    out = np.empty((len(X), X.shape[1] * tile, X.shape[1] * tile), dtype=np.float32)
    for n in range(len(X)):
        out[n] = recurrence_portrait(X[n], m, tau, eps, tile)
    return out

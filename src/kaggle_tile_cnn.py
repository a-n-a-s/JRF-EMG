# %%
# -*- coding: utf-8 -*-
"""
Tile-Parallel CNN - Gesture Classification from Recurrence Portraits
=====================================================================
Kaggle-ready training script (cell-wise; run each `# %%` block in order).
Implements Option 1 (Tile-Parallel CNN) from 2d_cnn.md: a shared tile
encoder applied to each 50x50 tile, a learned 8x8 positional embedding,
then a spatial aggregator over the 8x8 grid.

The 400x400 portrait is sliced on-the-fly in the forward pass into
64 tiles of 50x50 (8 diagonal self-RPs + 56 joint-RP tiles). No dataset
rebuild needed - reads the same .npy files as the plain CNN version.

Dataset inputs (upload to Kaggle as a Dataset):
  jrp_portraits_m3t2e0.5_tile50.npy   -> [N, 400, 400] uint8 portrait images
  labels.npy                          -> [N] int64 class labels 0..6
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

# %% ---------------------------------------------------------------- config

class Config:
    # local path to the uploaded .npy files (no Kaggle input dir)
    DATA_DIR = os.environ.get("EMG_DATA_DIR", os.path.join(os.getcwd(), "data"))

    # If you unzip into /kaggle/input and the folder is e.g. "jrp-emg":
    # DATA_DIR = "/kaggle/input/jrp-emg"

    PORTRAITS = "jrp_portraits_m3t2e0.5_tile50.npy"
    LABELS = "labels.npy"

    GESTURES = [
        "index_finger", "middle_finger", "ring_finger", "little_finger",
        "thumb", "rest", "victory_gesture",
    ]

    # training
    FOLDS = 5
    EPOCHS = 120
    BATCH = 128
    LR = 5e-4
    PATIENCE = 20
    SEED = 42
    VAL_RATIO = 0.15          # split off from train within each fold
    LOG_EVERY = 2             # print progress every N epochs

    # model (tile-parallel)
    TILE = 50                 # tile side; portrait = 8 x 8 grid of tiles
    GRID = 8                  # electrodes
    TILE_D = 32               # tile encoder output dim
    AGG_C = 64                # aggregator channels
    HIDDEN = 128
    DROPOUT = 0.4

    SAVE_MODEL = "jrp_tile_cnn.pt"


CFG = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# %% ------------------------------------------------------------- load data

def load_data(cfg):
    P = np.load(os.path.join(cfg.DATA_DIR, cfg.PORTRAITS))
    y = np.load(os.path.join(cfg.DATA_DIR, cfg.LABELS))
    P = P.astype(np.float32) / 255.0
    P = P[:, None]  # [N, 1, 400, 400]
    print("portraits:", P.shape, "labels:", y.shape)
    for c, g in enumerate(cfg.GESTURES):
        print(f"  {g:16s} {int((y == c).sum()):5d}")
    return P, y


X, y = load_data(CFG)

# %% ---------------------------------------------------------------- model

class TileEncoder(nn.Module):
    """Small CNN shared across all 64 tiles. Tile (50x50) -> d-vector."""

    def __init__(self, cin=1, d=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, d, 3, padding=1), nn.BatchNorm2d(d), nn.ReLU(),
        )  # 50 -> 25 -> 12 -> 12; GAP below

    def forward(self, tiles):
        return self.net(tiles).mean(dim=(2, 3))   # [B, d]


class TileParallelCNN(nn.Module):
    """Shared tile encoder + learned 8x8 positional embedding + 8x8 aggregator.

    The portrait is sliced in the forward pass into 64 tiles of TILE x TILE,
    encoded by the shared TileEncoder, given their grid position via a learned
    embedding, and aggregated with a small 2D conv over the 8x8 grid.
    """

    def __init__(self, num_classes=7, tile=50, grid=8, tile_d=32, agg_c=64,
                 hidden=128, dropout=0.4):
        super().__init__()
        self.tile = tile
        self.grid = grid
        self.tile_enc = TileEncoder(1, tile_d)
        self.pos_emb = nn.Parameter(torch.zeros(1, grid * grid, tile_d))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.agg = nn.Sequential(
            nn.Conv2d(tile_d, agg_c, 3, padding=1), nn.BatchNorm2d(agg_c), nn.ReLU(),
            nn.Conv2d(agg_c, agg_c, 3, padding=1), nn.BatchNorm2d(agg_c), nn.ReLU(),
        )  # on the 8x8 grid
        self.head = nn.Sequential(
            nn.Linear(agg_c, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):                      # [B, 1, 400, 400]
        B, _, H, W = x.shape
        G, T = self.grid, self.tile
        # slice into (B, G, G, T, T) tiles
        tiles = (x.view(B, G, T, G, T)
                  .permute(0, 1, 3, 2, 4)
                  .reshape(B, G * G, 1, T, T))            # [B, 64, 1, 50, 50]
        f = self.tile_enc(tiles.reshape(-1, 1, T, T))     # [B*64, tile_d]
        f = f.view(B, G * G, -1) + self.pos_emb           # [B, 64, tile_d]
        g = f.transpose(1, 2).view(B, -1, G, G)           # [B, tile_d, 8, 8]
        h = self.agg(g).mean(dim=(2, 3))                  # [B, agg_c]
        return self.head(h)


model = TileParallelCNN(
    num_classes=len(CFG.GESTURES), tile=CFG.TILE, grid=CFG.GRID,
    tile_d=CFG.TILE_D, agg_c=CFG.AGG_C, hidden=CFG.HIDDEN, dropout=CFG.DROPOUT,
).to(device)
n_params = sum(p.numel() for p in model.parameters())
print("TileParallelCNN params:", f"{n_params:,}")
print(model)

# %% ---------------------------------------------------------- train helper

def train_fold(model, Xtr, ytr, Xva, yva, cfg):
    """Weighted CE + cosine LR + best-val-F1 checkpointing + early stop.

    Logs every `cfg.LOG_EVERY` epochs: train loss, val macro-F1, elapsed s.
    """
    weights = torch.tensor(
        len(ytr) / (len(cfg.GESTURES) * np.bincount(ytr, minlength=len(cfg.GESTURES))),
        dtype=torch.float32,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.EPOCHS)
    n = len(Xtr)
    best_f1, best_state, patience = -1, None, 0
    t0 = time.time()

    for epoch in range(cfg.EPOCHS):
        model.train()
        perm = torch.randperm(n)
        loss_sum, cnt = 0.0, 0
        for i in range(0, n, cfg.BATCH):
            idx = perm[i : i + cfg.BATCH]
            xb = torch.from_numpy(Xtr[idx]).to(device)
            yb = torch.from_numpy(ytr[idx]).long().to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb, weight=weights)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(xb)
            cnt += len(xb)
        sched.step()

        va_f1 = evaluate(model, Xva, yva)
        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch % cfg.LOG_EVERY == 0 or patience >= cfg.PATIENCE:
            star = " *" if va_f1 == best_f1 else ""
            print(f"    epoch {epoch:3d}  loss={loss_sum/cnt:.4f}  val_f1={va_f1:.4f}"
                  f"  (best={best_f1:.4f}){star}  {time.time()-t0:.0f}s")
        if patience >= cfg.PATIENCE:
            print(f"    early stop @ epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return best_f1


@torch.no_grad()
def evaluate(model, X, y, batch=64):
    model.eval()
    preds = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i : i + batch]).to(device)
        preds.append(model(xb).argmax(1).cpu().numpy())
    return f1_score(y, np.concatenate(preds), average="macro")

# %% ------------------------------------------------------------- 5-fold CV

def run_cv(X, y, cfg, save_best=True):
    skf = StratifiedKFold(n_splits=cfg.FOLDS, shuffle=True, random_state=cfg.SEED)
    fold_f1, fold_acc = [], []
    y_all, p_all = [], []
    best_state, best_val_global = None, -1.0

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        print(f"\n===== fold {fold} =====")
        Xtr_f, Xva_f, ytr_f, yva_f = split_val(
            X[tr_idx], y[tr_idx], cfg.VAL_RATIO, cfg.SEED + fold
        )

        torch.manual_seed(cfg.SEED + fold)
        model = TileParallelCNN(
            num_classes=len(cfg.GESTURES), tile=cfg.TILE, grid=cfg.GRID,
            tile_d=cfg.TILE_D, agg_c=cfg.AGG_C, hidden=cfg.HIDDEN, dropout=cfg.DROPOUT,
        ).to(device)

        t0 = time.time()
        best_val = train_fold(model, Xtr_f, ytr_f, Xva_f, yva_f, cfg)

        preds = predict(model, X[te_idx])
        f1 = f1_score(y[te_idx], preds, average="macro")
        acc = accuracy_score(y[te_idx], preds)
        fold_f1.append(f1)
        fold_acc.append(acc)
        y_all.append(y[te_idx])
        p_all.append(preds)
        print(f"  fold {fold}: val_f1={best_val:.4f} test_f1={f1:.4f} acc={acc:.4f} ({time.time()-t0:.0f}s)")

        if save_best and best_val > best_val_global:
            best_val_global = best_val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)
    print("\n========== RESULT (out-of-fold) ==========")
    print(f"TileParallelCNN {cfg.FOLDS}-fold: acc={np.mean(fold_acc):.4f}+/-{np.std(fold_acc):.4f}  "
          f"f1={np.mean(fold_f1):.4f}+/-{np.std(fold_f1):.4f}")
    print(f"\n  bar to beat: classical-80 = 0.7235 | classical+JRQA = 0.7546")
    return y_all, p_all, best_state


def split_val(X, y, val_ratio, seed):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=val_ratio, stratify=y, random_state=seed)


@torch.no_grad()
def predict(model, X, batch=64):
    model.eval()
    preds = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i : i + batch]).to(device)
        preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


if __name__ == "__main__":
    y_oof, p_oof, best_state = run_cv(X, y, CFG, save_best=True)

# %% ------------------------------------------------ per-class breakdown

if __name__ == "__main__":
    print("\n===== PER-CLASS REPORT (out-of-fold) =====")
    print(classification_report(y_oof, p_oof, target_names=CFG.GESTURES, digits=4))
    print("\nper-class macro F1:")
    for c, g in enumerate(CFG.GESTURES):
        f1 = f1_score(y_oof == c, p_oof == c)
        print(f"  {g:16s} {f1:.4f}")

# %% ------------------------------------------------ confusion matrix plot

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_oof, p_oof, labels=range(len(CFG.GESTURES)))
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar(label="count")
    plt.xticks(range(len(CFG.GESTURES)), CFG.GESTURES, rotation=45, ha="right")
    plt.yticks(range(len(CFG.GESTURES)), CFG.GESTURES)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title("TileParallelCNN confusion matrix (out-of-fold)")
    for i in range(len(CFG.GESTURES)):
        for j in range(len(CFG.GESTURES)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig("tile_cnn_confusion.png", dpi=120)
    print("saved tile_cnn_confusion.png")

# %% ------------------------------------------------------ save best model

if __name__ == "__main__":
    if best_state is not None:
        torch.save(best_state, CFG.SAVE_MODEL)
        print("saved best model ->", CFG.SAVE_MODEL)

    print("\nDONE.")

import os
import numpy as np
from sklearn.model_selection import train_test_split

BASE = r"D:\EMG\archive\Electro-Myography-EMG-Dataset\raw_emg_data_cropped_and_arranged"

GESTURES = [
    "index_finger",
    "middle_finger",
    "ring_finger",
    "little_finger",
    "thumb",
    "rest",
    "victory_gesture",
]

LABEL_MAP = {g: i for i, g in enumerate(GESTURES)}


def load_cropped_raw(base=BASE, gestures=GESTURES):
    X, y = [], []
    for g in gestures:
        arrs = []
        for ch in range(1, 9):
            f = os.path.join(base, g, f"electrode_{ch}.csv")
            arrs.append(np.loadtxt(f, delimiter=",", ndmin=2))
        X.append(np.stack(arrs, axis=1))
        y.append(np.full(len(arrs[0]), LABEL_MAP[g], dtype=np.int64))
    X = np.concatenate(X, axis=0)  # [N, 8, 150]
    y = np.concatenate(y, axis=0)
    return X, y


def split_data(X, y, val_frac=0.15, test_frac=0.15, seed=42):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_frac, stratify=y, random_state=seed
    )
    Xtr, Xva, ytr, yva = train_test_split(
        Xtr, ytr, test_size=val_frac / (1 - test_frac), stratify=ytr, random_state=seed
    )
    return Xtr, Xva, Xte, ytr, yva, yte


def fit_channel_stats(Xtr):
    mean = Xtr.mean(axis=(0, 2), keepdims=True)  # per-channel
    std = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
    return mean, std


def zscore(X, mean, std):
    return (X - mean) / std


def class_weights(ytr, num_classes):
    counts = np.bincount(ytr, minlength=num_classes).astype(np.float64)
    return torch_tensor_weights(counts)


def torch_tensor_weights(counts):
    import torch

    total = counts.sum()
    w = total / (len(counts) * counts)
    return torch.tensor(w, dtype=torch.float32)


if __name__ == "__main__":
    X, y = load_cropped_raw()
    print("X", X.shape, "y", y.shape)
    print("per-class:", {GESTURES[c]: int((y == c).sum()) for c in range(len(GESTURES))})
    Xtr, Xva, Xte, ytr, yva, yte = split_data(X, y)
    print("split sizes:", Xtr.shape[0], Xva.shape[0], Xte.shape[0])

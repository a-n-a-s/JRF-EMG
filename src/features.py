import numpy as np


FEATURE_ORDER = [
    "standard_deviation",
    "root_mean_square",
    "minimum",
    "maximum",
    "zero_crossings",
    "average_amplitude_change",
    "amplitude_first_burst",
    "mean_absolute_value",
    "waveform_length",
    "willison_amplitude",
]


def extract_features(X, fs=1.0):
    """Classical EMG features per channel, in the dataset's column order.

    X: [N, C, T] raw windows -> F: [N, 10 * C].

    Column layout matches the Kaggle 'extracted_features_and_labeled_dataset'
    readme: feature-major blocks. First 8 cols = standard_deviation across the
    8 electrodes, next 8 = root_mean_square, ... so block f of feature f_* is
    columns f*C : f*C + C.
    """
    N, C, T = X.shape
    feats = np.empty((N, C, 10), dtype=np.float64)
    for c in range(C):
        x = X[:, c]                       # [N, T]
        feats[:, c, 0] = x.std(axis=1)
        feats[:, c, 1] = np.sqrt(np.mean(x ** 2, axis=1))
        feats[:, c, 2] = x.min(axis=1)
        feats[:, c, 3] = x.max(axis=1)
        feats[:, c, 4] = _zero_crossings(x)
        feats[:, c, 5] = _average_amplitude_change(x)
        feats[:, c, 6] = _amplitude_first_burst(x)
        feats[:, c, 7] = np.mean(np.abs(x), axis=1)
        feats[:, c, 8] = _waveform_length(x)
        feats[:, c, 9] = _willison_amplitude(x)
    # [N, 10, C] -> [N, 10*C] feature-major
    return feats.transpose(0, 2, 1).reshape(N, 10 * C)


def _zero_crossings(x):
    signs = np.sign(x)
    return np.sum(np.abs(np.diff(signs, axis=1)) > 0, axis=1)


def _average_amplitude_change(x):
    d = np.diff(x, axis=1)
    return np.mean(np.abs(d), axis=1)


def _amplitude_first_burst(x):
    d = np.abs(np.diff(x, axis=1))
    idx = np.argmax(d, axis=1)
    return np.take_along_axis(d, idx[:, None], axis=1)[:, 0]


def _waveform_length(x):
    return np.sum(np.abs(np.diff(x, axis=1)), axis=1)


def _willison_amplitude(x, threshold=0.01):
    d = np.diff(x, axis=1)
    return np.sum(np.abs(d) > threshold, axis=1)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"D:\EMG\src")
    from data import load_cropped_raw, GESTURES

    X, y = load_cropped_raw()
    F = extract_features(X)
    print("X", X.shape, "-> F", F.shape)
    for i, f in enumerate(FEATURE_ORDER):
        col = F[:, i::10]
        print(f"{f:26s} mean={col.mean():12.4g} std={col.std():12.4g}")

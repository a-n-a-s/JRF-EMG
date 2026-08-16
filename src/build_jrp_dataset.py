import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_cropped_raw, GESTURES
from recurrence import recurrence_portrait

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT, exist_ok=True)


def main(args):
    t0 = time.time()
    X, y = load_cropped_raw()
    print(f"raw: X {X.shape} y {y.shape}")

    H = W = X.shape[1] * args.tile  # 8 electrodes -> 8x8 grid of tiles
    P = np.empty((len(X), H, W), dtype=np.uint8)
    for n in range(len(X)):
        if n % 500 == 0:
            print(f"  [{n}/{len(X)}] {time.time()-t0:.0f}s")
        p = recurrence_portrait(X[n], m=args.m, tau=args.tau, eps=args.eps, tile=args.tile)
        P[n] = (p * 255).astype(np.uint8)

    xpath = os.path.join(OUT, f"jrp_portraits_m{args.m}t{args.tau}e{args.eps}_tile{args.tile}.npy")
    ypath = os.path.join(OUT, "labels.npy")
    np.save(xpath, P)
    np.save(ypath, y.astype(np.int64))
    print(f"saved {os.path.normpath(xpath)}: {P.shape} uint8 ({os.path.getsize(xpath)/1e6:.0f} MB)")
    print(f"saved {os.path.normpath(ypath)}: {y.shape}")
    print(f"done in {time.time()-t0:.0f}s")
    print(f"per-class:", {g: int((y == i).sum()) for i, g in enumerate(GESTURES)})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tile", type=int, default=50)
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--tau", type=int, default=2)
    p.add_argument("--eps", type=float, default=0.5)
    main(p.parse_args())

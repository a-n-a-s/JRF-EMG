# JRP Portraits as Muscle-Synergy Fingerprints for Interpretable EMG Gesture Recognition

Joint Recurrence Plot (JRP) portraits for interpretable surface-EMG gesture recognition.
Each 8-channel sEMG window becomes a single 8x8 image whose **diagonal blocks** are each
electrode's own recurrence plot and whose **off-diagonal blocks** are the joint recurrence
plot of every electrode pair (simultaneous co-activation). Collapsing the portrait yields a
28-dimensional **muscle-synergy fingerprint** per gesture.

> Honest framing: the portrait does not beat classical features on accuracy (classical-80
> macro-F1 0.955 vs tile-parallel CNN 0.818). Its contribution is **interpretability** — a
> supervision-free, anatomically consistent visualization of muscle synergy.

## Paper

`paper/paper.md` — full draft with embedded figures and results.

## Repository layout

```
├── paper/
│   └── paper.md            # manuscript (figures referenced as ../figures/)
├── src/
│   ├── recurrence.py           # embed, graded RP, recurrence_portrait (core contribution)
│   ├── build_jrp_dataset.py    # window -> 400x400 portrait dataset builder
│   ├── data.py                 # raw data loader, GESTURES order, split helpers
│   ├── features.py             # classical time-domain features
│   ├── rp_samples.py           # recurrence_dist helper (used by fair_comparison)
│   ├── jrp_sep.py              # binary_rp, line_features (JRQA metrics)
│   ├── fair_comparison.py      # same-fold baseline comparison -> figures/fair_comparison.csv
│   ├── fingerprint_figs.py     # mean portraits, synergy heatmap, similarity figures
│   ├── interpretability_figs.py# pipeline, chord, differential, variability figures
│   └── kaggle_tile_cnn.py      # tile-parallel CNN (Kaggle-ready, self-contained)
├── figures/                    # paper figures + result tables
├── requirements.txt
```

## Getting the data

The portrait dataset `data/jrp_portraits_m3t2e0.5_tile50.npy` (4579 x 400 x 400, ~700 MB)
is **not committed** (too large for GitHub). Regenerate it from the raw sEMG windows:

1. Download the source EMG dataset and update `BASE` in `src/data.py`
   (currently `.../raw_emg_data_cropped_and_arranged`).
2. Build the portraits:
   ```
   python src/build_jrp_dataset.py
   ```
   This writes `data/jrp_portraits_m3t2e0.5_tile50.npy` and `data/labels.npy`
   (embedding m=3, tau=2, graded recurrence eps=0.5, 50x50 tiles).

## Reproducing the experiments

```bash
# same-fold baseline comparison (classical / RQA / JRQA / portrait features)
python src/fair_comparison.py

# interpretability figures (pipeline, chords, differential, variability)
python src/interpretability_figs.py

# mean-portrait fingerprints and synergy heatmap
python src/fingerprint_figs.py

# tile-parallel CNN (train on Kaggle GPU; self-contained, reads .npy directly)
python src/kaggle_tile_cnn.py
```

## Key results (5-fold StratifiedKFold, shuffle=True, seed=42)

| Method | Accuracy | Macro-F1 |
|---|---|---|
| classical-80 | 0.9515 | **0.9546** |
| classical+JRQA-192 | 0.9493 | 0.9531 |
| all (class+RQA+JRQA) | 0.9472 | 0.9512 |
| RQA-40 | 0.7803 | 0.7669 |
| JRQA-112 | 0.7401 | 0.7253 |
| portrait-36 (JRP mean) | 0.6665 | 0.6454 |
| TileParallelCNN | 0.8242 | 0.8182 |

## Interpretability findings

- **Thumb** uniquely activates channel-5-centered pairs (ch5-6, ch2-5, ch3-5) — the thenar compartment.
- **Little finger** is channel-7 centered (ch2-7, ch3-7, ch4-7) — ulnar side.
- **Rest** is uniformly dark (density ~0.116 vs ~0.28 active); **victory** is the sparsest active gesture.

See `figures/fingerprint_synergy.csv` for the full 28-pair x 7-gesture table.

## Citation

```bibtex
@misc{emg_jrp_portraits,
  title  = {JRP Portraits as Muscle-Synergy Fingerprints for Interpretable EMG Gesture Recognition},
  year   = {2026}
}
```

# Wildfire Early Warning: Confound-Controlled Multi-Sensor Fusion Study

Code accompanying the paper *"Weather Alone Predicts Early Wildfire Ignition as Well as
Multi-Sensor Fusion: A Confound-Controlled Empirical Study"* (submitted to *Computers &
Geosciences*).

This repository contains the full data-collection, feature-extraction, and modeling
pipeline used in the paper, along with a script to generate a small **synthetic** sample
dataset so the pipeline can be run and inspected end-to-end without access to the real,
private dataset.

## Data availability

The real dataset (452 wildfire events and 452 matched negatives across six California
fire seasons, 2019–2024, with associated weather and air-quality features) is currently
maintained in a private repository and is **not publicly released** at this time. It is
available from the corresponding author upon reasonable request.

In compliance with this journal's code-availability policy, this repository instead
includes:
- The complete, runnable pipeline code (data collection, event construction, feature
  extraction, model training and evaluation).
- A script (`scripts/generate_synthetic_data.py`) that generates a small **synthetic**
  sample dataset matching the real dataset's exact schema, so every stage of the
  pipeline downstream of raw data collection can be run and verified independently of
  the real data.

Synthetic data is randomly generated and does not represent real wildfire events,
weather conditions, or air-quality readings. It exists solely to demonstrate expected
input/output shapes and to let others verify the code runs correctly.

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── raw/                         # FIRMS/weather/OpenAQ pulls (real or synthetic)
│   ├── processed/                   # extracted features + five sensor-ablation arms
│   └── results/                     # baseline, GRU-D, and confound-check outputs
├── scripts/
│   ├── generate_synthetic_data.py   # produces a synthetic sample in data/raw/
│   ├── 01_collect_firms.py          # NASA FIRMS VIIRS fire detection collection
│   ├── 02_build_fire_events.py      # spatial (DBSCAN) + temporal clustering into discrete events
│   ├── 03_sample_negatives.py       # rejection-sampled non-fire negatives + spatial-cluster train/test split
│   ├── 04_collect_weather.py        # Open-Meteo hourly weather archive pull
│   ├── 05_collect_openaq.py         # OpenAQ air-quality pull
│   ├── 06_build_features.py         # leakage-safe, per-horizon feature extraction + five arms
│   └── 07_train_evaluate.py         # baselines, GRU-D, and location-confound check
└── notebooks/                       # exploratory notebooks (optional)
```

## Requirements

See `requirements.txt`. Core dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`,
`torch`, `scipy`, `requests`.

```bash
pip install -r requirements.txt
```

## Collecting fresh real data

`scripts/01_collect_firms.py` and `scripts/05_collect_openaq.py` require free API keys.
Set them as environment variables before running:

```bash
export FIRMS_MAP_KEY="your_firms_key_here"      # https://firms.modaps.eosdis.nasa.gov/api/
export OPENAQ_API_KEY="your_openaq_key_here"    # https://explore.openaq.org/register
```

`scripts/04_collect_weather.py` (Open-Meteo) requires no key.

## Reproducing the pipeline on synthetic data

```bash
cd scripts
python generate_synthetic_data.py --n_events 60 --seed 42 --out_dir ../data/raw
python 06_build_features.py --in_dir ../data/raw --out_dir ../data/processed
python 07_train_evaluate.py --in_dir ../data/processed --raw_dir ../data/raw --out_dir ../data/results
```

Add `--skip_grud` to the last command to skip GRU-D training (e.g., no GPU available;
GRU-D also runs on CPU, just more slowly). This exact sequence has been tested end-to-end
against synthetic data and completes successfully, producing `baseline_results.csv`,
`grud_results.csv`, and `location_confound_check.csv` in the output directory.

Running against the synthetic sample will execute successfully and produce output in the
same format as the real experiments, but the resulting numbers are meaningless (the
underlying data is random) — this is intended only to verify the code runs correctly,
not to reproduce the paper's actual findings. Reproducing the paper's real results
requires the private dataset, available from the corresponding author on request.

## Running the full real-data pipeline

```bash
cd scripts
python 01_collect_firms.py --out_dir ../data/raw
python 02_build_fire_events.py --in_dir ../data/raw --out_dir ../data/raw
python 03_sample_negatives.py --in_dir ../data/raw --out_dir ../data/raw
python 04_collect_weather.py --in_dir ../data/raw --out_dir ../data/raw
python 05_collect_openaq.py --in_dir ../data/raw --out_dir ../data/raw
python 06_build_features.py --in_dir ../data/raw --out_dir ../data/processed
python 07_train_evaluate.py --in_dir ../data/processed --raw_dir ../data/raw --out_dir ../data/results
```

## Citation

If you use this code, please cite the associated paper (citation details to be added
upon publication).

## License

See `LICENSE`.

"""
03_sample_negatives.py

Generates rejection-sampled non-fire negative examples matched to the
fire events, applies a spatial-cluster-based (not season-based) train/test
split to avoid location leakage across seasons, and builds the unified
query_points.csv table used as the join key for weather/air-quality pulls.

Usage:
    python 03_sample_negatives.py --in_dir ../data/raw --out_dir ../data/raw
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BBOX_TUPLE = (-124, 36, -119, 42)  # west, south, east, north
NEG_RADIUS_DEG = 0.3
NEG_MIN_GAP_DAYS = 14
NEG_MAX_ATTEMPTS = 50
TEST_SIZE = 0.25
RANDOM_SEED = 42


def sample_negative_global(all_detections, bbox_tuple, season, radius_deg=NEG_RADIUS_DEG,
                            min_gap_days=NEG_MIN_GAP_DAYS, max_attempts=NEG_MAX_ATTEMPTS, rng=None):
    """
    Sample a single non-fire space-time point uniformly across the region,
    rejecting (not clipping) any candidate within radius_deg/min_gap_days of
    a real detection. Sampling uniformly across the region rather than
    jittering around a specific fire event avoids two failure modes:
    (1) candidates falling inside a long-duration fire's own footprint, and
    (2) an artificial spike of negatives on the region boundary from clipping
    out-of-bounds coordinates instead of rejecting them.
    """
    if rng is None:
        rng = np.random
    west, south, east, north = bbox_tuple
    for _ in range(max_attempts):
        lat = rng.uniform(south, north)
        lon = rng.uniform(west, east)

        season_start = pd.Timestamp(f"{season}-08-01")
        season_end = pd.Timestamp(f"{season}-10-31")
        rand_days = rng.integers(0, (season_end - season_start).days) if hasattr(rng, "integers") else rng.randint(0, (season_end - season_start).days)
        candidate_time = season_start + pd.Timedelta(days=int(rand_days))

        nearby = all_detections[
            (np.abs(all_detections["latitude"] - lat) < radius_deg)
            & (np.abs(all_detections["longitude"] - lon) < radius_deg)
            & (np.abs((all_detections["acq_datetime"] - candidate_time).dt.total_seconds()) < min_gap_days * 86400)
        ]
        if len(nearby) == 0:
            return {"lat": lat, "lon": lon, "pseudo_time": candidate_time, "season": season}
    return None


def build_negatives(fire_events_final, firms_clean, bbox_tuple=BBOX_TUPLE, seed=RANDOM_SEED):
    np.random.seed(seed)
    negatives, failed = [], 0
    for _, row in fire_events_final.iterrows():
        neg = sample_negative_global(firms_clean, bbox_tuple, row["season"])
        if neg:
            negatives.append(neg)
        else:
            failed += 1
    negatives_df = pd.DataFrame(negatives)
    print(f"Generated {len(negatives_df)} negative samples (target was {len(fire_events_final)}, {failed} failed)")
    return negatives_df


def apply_spatial_split(fire_events_final, negatives_df, test_size=TEST_SIZE, seed=RANDOM_SEED):
    """
    Split by spatial_cluster (physical location), not by season. A
    season-based split leaks locations across train/test because
    fire-prone terrain reburns across years -- verified here by checking
    zero cluster overlap between the resulting train/test sets.
    """
    cluster_seasons = fire_events_final.groupby("spatial_cluster")["season"].nunique()
    leaking_clusters = cluster_seasons[cluster_seasons > 1]
    print(f"{len(leaking_clusters)} spatial clusters have events spanning multiple seasons "
          f"(expected -- real terrain reburns; this is why we split by location, not season)")

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(fire_events_final, groups=fire_events_final["spatial_cluster"]))

    fire_events_final = fire_events_final.copy()
    fire_events_final["split"] = "train"
    fire_events_final.iloc[test_idx, fire_events_final.columns.get_loc("split")] = "test"

    train_clusters = set(fire_events_final[fire_events_final["split"] == "train"]["spatial_cluster"])
    test_clusters = set(fire_events_final[fire_events_final["split"] == "test"]["spatial_cluster"])
    overlap = train_clusters & test_clusters
    print(f"Cluster overlap between train/test: {len(overlap)} (must be 0)")
    assert len(overlap) == 0, "Spatial cluster leakage between train and test detected!"
    print(fire_events_final["split"].value_counts())

    train_frac = (fire_events_final["split"] == "train").mean()
    negatives_df = negatives_df.copy()
    negatives_df["split"] = np.random.RandomState(seed).choice(
        ["train", "test"], size=len(negatives_df), p=[train_frac, 1 - train_frac]
    )
    print(negatives_df["split"].value_counts())

    return fire_events_final, negatives_df


def build_query_points(fire_events_final, negatives_df):
    query_points = pd.concat([
        fire_events_final[["lat", "lon", "split"]].assign(
            point_type="positive", ref_time=fire_events_final["ignition_time"], season=fire_events_final["season"]
        ),
        negatives_df[["lat", "lon", "split"]].assign(
            point_type="negative", ref_time=negatives_df["pseudo_time"], season=negatives_df["season"]
        ),
    ], ignore_index=True)
    print(f"{len(query_points)} total query points")
    print(query_points.groupby(["split", "point_type"]).size())
    return query_points


def main():
    parser = argparse.ArgumentParser(description="Sample negatives, apply spatial split, build query points.")
    parser.add_argument("--in_dir", type=str, default="../data/raw")
    parser.add_argument("--out_dir", type=str, default="../data/raw")
    args = parser.parse_args()

    fire_events_final = pd.read_csv(os.path.join(args.in_dir, "fire_events_final.csv"), parse_dates=["ignition_time", "last_detection_time"])
    firms_clean = pd.read_csv(os.path.join(args.in_dir, "firms_clean.csv"), parse_dates=["acq_datetime"])

    negatives_df = build_negatives(fire_events_final, firms_clean)
    fire_events_final, negatives_df = apply_spatial_split(fire_events_final, negatives_df)
    query_points = build_query_points(fire_events_final, negatives_df)

    os.makedirs(args.out_dir, exist_ok=True)
    fire_events_final.to_csv(os.path.join(args.out_dir, "fire_events_final.csv"), index=False)
    negatives_df.to_csv(os.path.join(args.out_dir, "negative_samples.csv"), index=False)
    query_points.to_csv(os.path.join(args.out_dir, "query_points.csv"), index=False)
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()

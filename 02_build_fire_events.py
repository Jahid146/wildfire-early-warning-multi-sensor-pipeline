"""
02_build_fire_events.py

Filters raw FIRMS detections to presumed vegetation fires, clusters them
spatially (DBSCAN) and temporally (gap threshold) into discrete fire
events, and applies a quality filter to exclude single-pixel/transient
detections.

Usage:
    python 02_build_fire_events.py --in_dir ../data/raw --out_dir ../data/raw
"""

import argparse
import os

import pandas as pd
from sklearn.cluster import DBSCAN

DBSCAN_EPS_DEG = 0.05
DBSCAN_MIN_SAMPLES = 3
GAP_THRESHOLD_DAYS = 7
MIN_DETECTIONS = 3
MIN_DURATION_DAYS = 0.5


def filter_vegetation_fires(firms_df):
    """Keep presumed vegetation fires (type == 0) at nominal or high confidence."""
    firms_clean = firms_df[
        (firms_df["type"] == 0) & (firms_df["confidence"].isin(["n", "h"]))
    ].copy()

    firms_clean["acq_datetime"] = pd.to_datetime(
        firms_clean["acq_date"] + " " + firms_clean["acq_time"].astype(str).str.zfill(4),
        format="%Y-%m-%d %H%M",
    )
    return firms_clean


def spatial_cluster(firms_clean, eps=DBSCAN_EPS_DEG, min_samples=DBSCAN_MIN_SAMPLES):
    coords = firms_clean[["latitude", "longitude"]].values
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    firms_clean = firms_clean.copy()
    firms_clean["spatial_cluster"] = clustering.labels_
    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    print(f"Found {n_clusters} spatial fire clusters (excluding noise)")
    return firms_clean


def temporal_split(firms_clean, gap_threshold_days=GAP_THRESHOLD_DAYS):
    """
    Within each spatial cluster, split detections into distinct events
    wherever a gap of more than gap_threshold_days separates consecutive
    detections -- the same location can experience multiple independent
    ignitions across or within fire seasons.
    """
    event_rows = []
    event_id = 0

    for cluster_id, group in firms_clean[firms_clean["spatial_cluster"] != -1].groupby("spatial_cluster"):
        group = group.sort_values("acq_datetime")
        gaps = group["acq_datetime"].diff().dt.total_seconds() / 86400
        sub_event = (gaps > gap_threshold_days).cumsum()
        group = group.assign(sub_event=sub_event)

        for sub_id, sub_group in group.groupby("sub_event"):
            event_rows.append({
                "event_id": event_id,
                "spatial_cluster": cluster_id,
                "ignition_time": sub_group["acq_datetime"].min(),
                "last_detection_time": sub_group["acq_datetime"].max(),
                "lat": sub_group["latitude"].mean(),
                "lon": sub_group["longitude"].mean(),
                "n_detections": len(sub_group),
                "duration_days": (sub_group["acq_datetime"].max() - sub_group["acq_datetime"].min()).total_seconds() / 86400,
            })
            event_id += 1

    fire_events = pd.DataFrame(event_rows)
    print(f"Distinct fire events after temporal split: {len(fire_events)}")
    return fire_events


def quality_filter(fire_events, min_detections=MIN_DETECTIONS, min_duration_days=MIN_DURATION_DAYS):
    fire_events_final = fire_events[
        (fire_events["n_detections"] >= min_detections) & (fire_events["duration_days"] >= min_duration_days)
    ].copy()
    fire_events_final["season"] = fire_events_final["ignition_time"].dt.year
    print(f"Final fire events: {len(fire_events_final)}")
    print(fire_events_final["season"].value_counts().sort_index())
    return fire_events_final


def main():
    parser = argparse.ArgumentParser(description="Build discrete fire events from raw FIRMS detections.")
    parser.add_argument("--in_dir", type=str, default="../data/raw")
    parser.add_argument("--out_dir", type=str, default="../data/raw")
    args = parser.parse_args()

    firms_df = pd.read_csv(os.path.join(args.in_dir, "firms_raw.csv"))
    print(f"Loaded {len(firms_df)} raw FIRMS detections")

    firms_clean = filter_vegetation_fires(firms_df)
    print(f"After filtering: {len(firms_clean)} detections (from {len(firms_df)} raw)")

    firms_clean = spatial_cluster(firms_clean)
    fire_events = temporal_split(firms_clean)
    fire_events_final = quality_filter(fire_events)

    os.makedirs(args.out_dir, exist_ok=True)
    fire_events_final.to_csv(os.path.join(args.out_dir, "fire_events_final.csv"), index=False)
    firms_clean.to_csv(os.path.join(args.out_dir, "firms_clean.csv"), index=False)
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()

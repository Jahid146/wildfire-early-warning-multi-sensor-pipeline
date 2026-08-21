"""
generate_synthetic_data.py

Generates a small SYNTHETIC sample dataset that mirrors the exact schema of
the real (private) dataset used in this project, so that the pipeline code
in this repository can be run end-to-end for testing and reproduction
purposes without access to the real, private data.

Values in the generated files are randomly sampled and DO NOT represent
real wildfire events, real weather, or real air-quality readings. They
exist only to demonstrate the expected input/output shape of each stage
of the pipeline.

Usage:
    python generate_synthetic_data.py --n_events 20 --seed 42 --out_dir ../data/synthetic_sample
"""

import argparse
import os
import numpy as np
import pandas as pd


def generate_fire_events(n_events, rng):
    """Mirrors fire_events_final.csv"""
    seasons = rng.choice([2019, 2020, 2021, 2022, 2023, 2024], size=n_events)
    rows = []
    for i in range(n_events):
        season = seasons[i]
        ignition_time = pd.Timestamp(f"{season}-08-01") + pd.Timedelta(
            days=int(rng.integers(0, 90)), hours=int(rng.integers(0, 24)), minutes=int(rng.integers(0, 60))
        )
        duration_days = float(rng.uniform(0.5, 90))
        n_detections = int(rng.integers(3, 500))
        rows.append({
            "event_id": i,
            "spatial_cluster": int(rng.integers(0, n_events // 2 + 1)),
            "ignition_time": ignition_time,
            "last_detection_time": ignition_time + pd.Timedelta(days=duration_days),
            "lat": float(rng.uniform(36, 42)),
            "lon": float(rng.uniform(-124, -119)),
            "n_detections": n_detections,
            "duration_days": duration_days,
            "season": season,
            "split": rng.choice(["train", "test"], p=[0.75, 0.25]),
        })
    return pd.DataFrame(rows)


def generate_negative_samples(n_negatives, rng):
    """Mirrors negative_samples.csv"""
    seasons = rng.choice([2019, 2020, 2021, 2022, 2023, 2024], size=n_negatives)
    rows = []
    for i in range(n_negatives):
        season = seasons[i]
        pseudo_time = pd.Timestamp(f"{season}-08-01") + pd.Timedelta(days=int(rng.integers(0, 90)))
        rows.append({
            "lat": float(rng.uniform(36, 42)),
            "lon": float(rng.uniform(-124, -119)),
            "pseudo_time": pseudo_time,
            "season": season,
            "split": rng.choice(["train", "test"], p=[0.75, 0.25]),
        })
    return pd.DataFrame(rows)


def build_query_points(fire_events, negatives):
    """Mirrors query_points.csv, the join key for weather/OpenAQ pulls."""
    pos = fire_events[["lat", "lon", "split"]].assign(
        point_type="positive", ref_time=fire_events["ignition_time"], season=fire_events["season"]
    )
    neg = negatives[["lat", "lon", "split"]].assign(
        point_type="negative", ref_time=negatives["pseudo_time"], season=negatives["season"]
    )
    query_points = pd.concat([pos, neg], ignore_index=True)
    query_points.index.name = "query_point_idx"
    return query_points.reset_index()


def generate_weather(query_points, lookback_days, rng):
    """
    Mirrors weather_raw.csv: hourly weather per query point, lookback window
    before ref_time.

    IMPORTANT: real Open-Meteo data is always returned on exact-hour marks
    (:00 minutes), regardless of the query timestamp's own precision. This
    generator floors ref_time to the hour before building the grid to match
    that real-world behavior. Anchoring the grid to ref_time's raw (un-
    floored) value would reproduce the exact timestamp-granularity bug
    documented in the paper: positive events (sub-hour satellite timestamps)
    would get weather rows at odd minute marks while negatives (exact-
    midnight pseudo-timestamps) would get rows at :00, causing every
    positive sequence to spuriously mismatch an hourly grid and appear
    "fully missing" to any downstream sequence model.
    """
    rows = []
    for _, row in query_points.iterrows():
        ref_time_floor = pd.Timestamp(row["ref_time"]).floor("h")
        hours = pd.date_range(ref_time_floor - pd.Timedelta(days=lookback_days), ref_time_floor, freq="h", inclusive="left")
        for t in hours:
            rows.append({
                "time": t,
                "temperature_2m": float(rng.normal(22, 8)),
                "relative_humidity_2m": float(np.clip(rng.normal(35, 15), 1, 100)),
                "wind_speed_10m": float(np.clip(rng.normal(8, 5), 0, None)),
                "precipitation": float(max(0, rng.exponential(0.2) - 0.15)),
                "query_point_idx": row["query_point_idx"],
                "point_type": row["point_type"],
            })
    return pd.DataFrame(rows)


def generate_openaq(query_points, lookback_days, coverage_rate, rng):
    """Mirrors openaq_raw.csv: sparse, hourly-ish air-quality readings for a coverage_rate subset of points."""
    covered_idx = query_points.sample(frac=coverage_rate, random_state=int(rng.integers(0, 1e6))).index
    rows = []
    for i in covered_idx:
        row = query_points.loc[i]
        ref_time = pd.Timestamp(row["ref_time"])
        n_obs = int(rng.integers(5, 48))
        for _ in range(n_obs):
            t = ref_time - pd.Timedelta(hours=float(rng.uniform(0, lookback_days * 24)))
            parameter = rng.choice(["pm25", "o3"], p=[0.4, 0.6])
            value = float(rng.uniform(5, 40)) if parameter == "pm25" else float(rng.uniform(10, 60))
            rows.append({
                "sensor_id": int(rng.integers(1000, 9999)),
                "value": value,
                "parameter": parameter,
                "datetime": t,
                "query_point_idx": row["query_point_idx"],
                "point_type": row["point_type"],
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic sample dataset matching the real schema.")
    parser.add_argument("--n_events", type=int, default=20, help="Number of synthetic fire events (and matched negatives).")
    parser.add_argument("--lookback_days", type=int, default=7, help="Lookback window for weather/OpenAQ pulls.")
    parser.add_argument("--openaq_coverage", type=float, default=0.26, help="Fraction of query points with OpenAQ coverage.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="../data/synthetic_sample")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    fire_events = generate_fire_events(args.n_events, rng)
    negatives = generate_negative_samples(args.n_events, rng)
    query_points = build_query_points(fire_events, negatives)
    weather_df = generate_weather(query_points, args.lookback_days, rng)
    openaq_df = generate_openaq(query_points, args.lookback_days, args.openaq_coverage, rng)

    fire_events.to_csv(os.path.join(args.out_dir, "fire_events_final.csv"), index=False)
    negatives.to_csv(os.path.join(args.out_dir, "negative_samples.csv"), index=False)
    query_points.to_csv(os.path.join(args.out_dir, "query_points.csv"), index=False)
    weather_df.to_csv(os.path.join(args.out_dir, "weather_raw.csv"), index=False)
    openaq_df.to_csv(os.path.join(args.out_dir, "openaq_raw.csv"), index=False)

    print(f"Synthetic sample written to {args.out_dir}:")
    print(f"  fire_events_final.csv : {len(fire_events)} rows")
    print(f"  negative_samples.csv  : {len(negatives)} rows")
    print(f"  query_points.csv      : {len(query_points)} rows")
    print(f"  weather_raw.csv       : {len(weather_df)} rows")
    print(f"  openaq_raw.csv        : {len(openaq_df)} rows ({args.openaq_coverage:.0%} coverage)")
    print("\nNOTE: all values are randomly generated and do not represent real events, weather, or air quality.")


if __name__ == "__main__":
    main()

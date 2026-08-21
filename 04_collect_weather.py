"""
04_collect_weather.py

Pulls hourly historical weather (Open-Meteo ERA5/ERA5-Land reanalysis) for
each query point, covering a lookback window before its reference
timestamp. No API key required; gap-free gridded reanalysis, so no
station-coverage gaps unlike ground-based sensor networks.

Usage:
    python 04_collect_weather.py --in_dir ../data/raw --out_dir ../data/raw
"""

import argparse
import os
import time

import pandas as pd
import requests

LOOKBACK_DAYS = 7


def pull_openmeteo_weather(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone": "UTC",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        hourly = resp.json().get("hourly", {})
        return pd.DataFrame(hourly) if hourly else pd.DataFrame()
    except Exception as e:
        print(f"    FAILED for ({lat},{lon}): {e}")
        return pd.DataFrame()


def collect_weather(query_points, lookback_days=LOOKBACK_DAYS, sleep_s=0.2):
    weather_frames = []
    for idx, row in query_points.iterrows():
        lat, lon = round(row["lat"], 4), round(row["lon"], 4)
        start = (row["ref_time"] - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end = row["ref_time"].strftime("%Y-%m-%d")

        df = pull_openmeteo_weather(lat, lon, start, end)
        if len(df) > 0:
            df["query_point_idx"] = idx
            df["point_type"] = row["point_type"]
            weather_frames.append(df)
        time.sleep(sleep_s)

        if idx % 50 == 0:
            print(f"  {idx}/{len(query_points)} processed, {len(weather_frames)} successful pulls so far")

    weather_df = pd.concat(weather_frames, ignore_index=True) if weather_frames else pd.DataFrame()
    return weather_df


def main():
    parser = argparse.ArgumentParser(description="Collect hourly weather for all query points.")
    parser.add_argument("--in_dir", type=str, default="../data/raw")
    parser.add_argument("--out_dir", type=str, default="../data/raw")
    parser.add_argument("--lookback_days", type=int, default=LOOKBACK_DAYS)
    args = parser.parse_args()

    query_points = pd.read_csv(os.path.join(args.in_dir, "query_points.csv"), parse_dates=["ref_time"])
    weather_df = collect_weather(query_points, lookback_days=args.lookback_days)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "weather_raw.csv")
    weather_df.to_csv(out_path, index=False)
    n_covered = weather_df["query_point_idx"].nunique() if len(weather_df) > 0 else 0
    print(f"Total weather rows: {len(weather_df)} covering {n_covered}/{len(query_points)} query points")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

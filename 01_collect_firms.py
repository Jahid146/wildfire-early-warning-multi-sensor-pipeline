"""
01_collect_firms.py

Collects raw VIIRS wildfire hotspot detections from NASA FIRMS for the
configured region and fire seasons.

Requires a free FIRMS MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/
Set it as an environment variable before running:
    export FIRMS_MAP_KEY="your_key_here"

Usage:
    python 01_collect_firms.py --out_dir ../data/raw
"""

import argparse
import os
import time
from datetime import datetime, timedelta

import pandas as pd

BBOX = "-124,36,-119,42"  # Northern California: west,south,east,north
SEASONS = [
    {"start": "2019-08-01", "end": "2019-10-31"},
    {"start": "2020-08-01", "end": "2020-10-31"},
    {"start": "2021-08-01", "end": "2021-10-31"},
    {"start": "2022-08-01", "end": "2022-10-31"},
    {"start": "2023-08-01", "end": "2023-10-31"},
    {"start": "2024-08-01", "end": "2024-10-31"},
]


def check_data_availability(map_key, source="VIIRS_SNPP_SP"):
    """Verify the FIRMS product covers the configured date range before pulling."""
    url = f"https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{map_key}/{source}"
    avail_df = pd.read_csv(url)
    print(avail_df)
    return avail_df


def pull_firms(bbox, start_date, end_date, map_key, source="VIIRS_SNPP_SP", day_range=5):
    """
    Pull FIRMS detections in day_range-day chunks (day_range=5 is the
    confirmed maximum window accepted by the Standard Processing product's
    Area API; requesting more raises a 400 error).
    """
    all_rows = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{source}/{bbox}/{day_range}/{date_str}"
        try:
            df = pd.read_csv(url)
            if len(df) > 0:
                all_rows.append(df)
            print(f"  {date_str}: {len(df)} detections")
        except Exception as e:
            print(f"  {date_str}: FAILED - {e}")
        current += timedelta(days=day_range)
        time.sleep(1)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Collect raw FIRMS VIIRS fire detections.")
    parser.add_argument("--out_dir", type=str, default="../data/raw")
    parser.add_argument("--bbox", type=str, default=BBOX)
    parser.add_argument("--day_range", type=int, default=5)
    args = parser.parse_args()

    map_key = os.environ.get("FIRMS_MAP_KEY")
    if not map_key:
        raise EnvironmentError(
            "FIRMS_MAP_KEY environment variable not set. "
            "Register for a free key at https://firms.modaps.eosdis.nasa.gov/api/"
        )

    os.makedirs(args.out_dir, exist_ok=True)

    check_data_availability(map_key)

    firms_frames = []
    for season in SEASONS:
        print(f"Pulling FIRMS {season['start']} to {season['end']}")
        df = pull_firms(args.bbox, season["start"], season["end"], map_key, day_range=args.day_range)
        firms_frames.append(df)

    firms_df = pd.concat(firms_frames, ignore_index=True)
    out_path = os.path.join(args.out_dir, "firms_raw.csv")
    firms_df.to_csv(out_path, index=False)
    print(f"\nTotal FIRMS detections: {len(firms_df)}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

"""
05_collect_openaq.py

Pulls ground-level air-quality measurements from OpenAQ v3 for the nearest
station within a fixed radius of each query point. Requires a free OpenAQ
API key: https://explore.openaq.org/register

Set it as an environment variable before running:
    export OPENAQ_API_KEY="your_key_here"

Note the OpenAQ v3 measurements endpoint parameter names are
`datetime_from`/`datetime_to`, not `date_from`/`date_to` -- the latter are
silently ignored by the API (no error, wrong data returned), which is a
real, easy-to-miss failure mode worth flagging for anyone adapting this
code to a similar API.

Usage:
    python 05_collect_openaq.py --in_dir ../data/raw --out_dir ../data/raw
"""

import argparse
import os
import time

import pandas as pd
import requests

LOOKBACK_DAYS = 7
REQUEST_DELAY = 1.1  # derived from OpenAQ's confirmed 60 requests/60s rate limit, plus margin
SEARCH_RADIUS_M = 25000


def request_with_retry(url, headers, params=None, max_retries=5, timeout=15):
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = 2 ** attempt
                print(f"    429 hit, backing off {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"    FAILED after {max_retries} attempts: {e}")
                return None
            time.sleep(2 ** attempt)
    return None


def get_locations_near(lat, lon, headers, radius=SEARCH_RADIUS_M):
    resp = request_with_retry(
        "https://api.openaq.org/v3/locations", headers,
        params={"coordinates": f"{lat},{lon}", "radius": radius, "limit": 5},
    )
    return resp.json().get("results", []) if resp else []


def pull_openaq_measurements(sensor_id, date_from, date_to, headers):
    resp = request_with_retry(
        f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements", headers,
        params={"datetime_from": date_from, "datetime_to": date_to, "limit": 1000},  # correct param names
    )
    if resp is None:
        return pd.DataFrame()
    results = resp.json().get("results", [])
    rows = [
        {
            "sensor_id": sensor_id,
            "value": r.get("value"),
            "parameter": r.get("parameter", {}).get("name"),
            "datetime": r.get("period", {}).get("datetimeFrom", {}).get("utc"),
        }
        for r in results
    ]
    return pd.DataFrame(rows)


def collect_openaq(query_points, headers, lookback_days=LOOKBACK_DAYS, request_delay=REQUEST_DELAY):
    openaq_frames = []
    location_cache = {}

    for idx, row in query_points.iterrows():
        lat, lon = round(row["lat"], 2), round(row["lon"], 2)
        key = (lat, lon)

        if key not in location_cache:
            location_cache[key] = get_locations_near(lat, lon, headers)
            time.sleep(request_delay)

        locations = location_cache[key]
        if not locations:
            continue

        loc_id = locations[0]["id"]
        sensors_resp = request_with_retry(f"https://api.openaq.org/v3/locations/{loc_id}/sensors", headers)
        time.sleep(request_delay)
        if sensors_resp is None:
            continue
        sensors = sensors_resp.json().get("results", [])

        start = (row["ref_time"] - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end = row["ref_time"].strftime("%Y-%m-%d")

        for sensor in sensors:
            sensor_id = sensor["id"]
            df = pull_openaq_measurements(sensor_id, start, end, headers)
            if len(df) > 0:
                df["query_point_idx"] = idx
                df["point_type"] = row["point_type"]
                openaq_frames.append(df)
            time.sleep(request_delay)

        if idx % 20 == 0:
            print(f"  {idx}/{len(query_points)} processed, {len(openaq_frames)} successful pulls so far")

    return pd.concat(openaq_frames, ignore_index=True) if openaq_frames else pd.DataFrame()


def verify_date_ranges(openaq_df, seasons):
    """Mandatory sanity check: confirm no rows fall outside the expected season windows.
    This exact check caught a silent parameter-name bug during development."""
    if len(openaq_df) == 0:
        return
    openaq_df["datetime"] = pd.to_datetime(openaq_df["datetime"], utc=True)
    in_range_mask = pd.Series(False, index=openaq_df.index)
    for s in seasons:
        in_range_mask |= (
            openaq_df["datetime"] >= pd.Timestamp(s["start"]).tz_localize("UTC") - pd.Timedelta(days=10)
        ) & (openaq_df["datetime"] <= pd.Timestamp(s["end"]).tz_localize("UTC"))
    n_out = (~in_range_mask).sum()
    print(f"Rows outside expected season windows: {n_out} (should be 0 or near 0)")
    if n_out > 0:
        print("WARNING: unexpected out-of-range rows detected -- check API parameters before trusting this data.")


def main():
    parser = argparse.ArgumentParser(description="Collect ground-level air quality for all query points.")
    parser.add_argument("--in_dir", type=str, default="../data/raw")
    parser.add_argument("--out_dir", type=str, default="../data/raw")
    parser.add_argument("--lookback_days", type=int, default=LOOKBACK_DAYS)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAQ_API_KEY environment variable not set. "
            "Register for a free key at https://explore.openaq.org/register"
        )
    headers = {"X-API-Key": api_key}

    query_points = pd.read_csv(os.path.join(args.in_dir, "query_points.csv"), parse_dates=["ref_time"])

    seasons = [
        {"start": f"{y}-08-01", "end": f"{y}-10-31"} for y in sorted(query_points["season"].unique())
    ]

    openaq_df = collect_openaq(query_points, headers, lookback_days=args.lookback_days)
    verify_date_ranges(openaq_df, seasons)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "openaq_raw.csv")
    openaq_df.to_csv(out_path, index=False)
    n_covered = openaq_df["query_point_idx"].nunique() if len(openaq_df) > 0 else 0
    print(f"Total: {len(openaq_df)} rows, {n_covered}/{len(query_points)} points covered")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

"""
06_build_features.py

Builds leakage-safe, per-(query_point, horizon) aggregated features from
the raw weather/OpenAQ pulls, then constructs the five sensor-ablation
arms (weather_only, weather_matched, combined, openaq_only,
missingness_aware) used for evaluation, dropping sparse OpenAQ columns
(>50% missing) within each arm's relevant subset.

Usage:
    python 06_build_features.py --in_dir ../data/raw --out_dir ../data/processed
"""

import argparse
import os

import pandas as pd

HORIZONS_HOURS = [6, 12, 24, 48, 72]
FEATURE_WINDOW_HOURS = 48
SPARSE_COLUMN_THRESHOLD = 0.5


def extract_weather_features(qp_idx, cutoff, weather_df, window_hours=FEATURE_WINDOW_HOURS):
    """Aggregate weather readings strictly before `cutoff` -- the leakage-safe boundary."""
    window_start = cutoff - pd.Timedelta(hours=window_hours)
    subset = weather_df[
        (weather_df["query_point_idx"] == qp_idx)
        & (weather_df["time"] < cutoff)  # strict inequality: no leakage
        & (weather_df["time"] >= window_start)
    ]
    if len(subset) == 0:
        return None

    feats = {}
    for col in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]:
        feats[f"weather_{col}_mean"] = subset[col].mean()
        feats[f"weather_{col}_std"] = subset[col].std()
        feats[f"weather_{col}_max"] = subset[col].max()
        feats[f"weather_{col}_min"] = subset[col].min()
        feats[f"weather_{col}_last"] = subset.sort_values("time")[col].iloc[-1]
    feats["weather_n_obs"] = len(subset)
    return feats


def extract_openaq_features(qp_idx, cutoff, openaq_df, window_hours=FEATURE_WINDOW_HOURS):
    window_start = cutoff - pd.Timedelta(hours=window_hours)
    subset = openaq_df[
        (openaq_df["query_point_idx"] == qp_idx)
        & (openaq_df["datetime"] < cutoff)
        & (openaq_df["datetime"] >= window_start)
    ]
    if len(subset) == 0:
        return None

    feats = {}
    for param in subset["parameter"].unique():
        param_data = subset[subset["parameter"] == param]["value"]
        feats[f"openaq_{param}_mean"] = param_data.mean()
        feats[f"openaq_{param}_max"] = param_data.max()
        feats[f"openaq_{param}_last"] = param_data.iloc[-1]
    feats["openaq_n_obs"] = len(subset)
    return feats


def build_feature_table(query_points, weather_df, openaq_df, horizons=HORIZONS_HOURS):
    rows = []
    for idx, row in query_points.iterrows():
        for horizon in horizons:
            cutoff = row["ref_time"] - pd.Timedelta(hours=horizon)
            weather_feats = extract_weather_features(idx, cutoff, weather_df)
            openaq_feats = extract_openaq_features(idx, cutoff, openaq_df)

            record = {
                "query_point_idx": idx,
                "horizon_hours": horizon,
                "cutoff_time": cutoff,
                "label": 1 if row["point_type"] == "positive" else 0,
                "split": row["split"],
                "has_weather": weather_feats is not None,
                "has_openaq": openaq_feats is not None,
            }
            if weather_feats:
                record.update(weather_feats)
            if openaq_feats:
                record.update(openaq_feats)
            rows.append(record)

        if idx % 100 == 0:
            print(f"  {idx}/{len(query_points)} points processed")

    features_df = pd.DataFrame(rows)
    print(f"\nTotal (point, horizon) rows: {len(features_df)}")
    print(f"Weather availability: {features_df['has_weather'].mean():.1%}")
    print(f"OpenAQ availability: {features_df['has_openaq'].mean():.1%}")
    return features_df


def verify_no_leakage(query_points, weather_df, openaq_df, features_df, horizon=6):
    """Confirm zero rows at/after cutoff are ever included in extracted features."""
    sample = query_points[query_points["point_type"] == "positive"].iloc[0]
    sample_idx = sample.name
    cutoff = sample["ref_time"] - pd.Timedelta(hours=horizon)

    leaked = weather_df[(weather_df["query_point_idx"] == sample_idx) & (weather_df["time"] >= cutoff)]
    print(f"Weather leakage check -- rows at/after cutoff that MUST be excluded: {len(leaked)}")

    if len(openaq_df) > 0:
        oaq_idx = openaq_df["query_point_idx"].iloc[0]
        oaq_cutoff = query_points.loc[oaq_idx, "ref_time"] - pd.Timedelta(hours=horizon)
        oaq_leaked = openaq_df[(openaq_df["query_point_idx"] == oaq_idx) & (openaq_df["datetime"] >= oaq_cutoff)]
        print(f"OpenAQ leakage check -- rows at/after cutoff that MUST be excluded: {len(oaq_leaked)}")


def drop_sparse_columns(df, base_cols, threshold=SPARSE_COLUMN_THRESHOLD, condition_col=None):
    """
    Drop feature columns missing in more than `threshold` fraction of rows.
    If condition_col is given, compute missingness only within rows where
    that column is True (used for missingness_aware, where most rows
    legitimately lack OpenAQ coverage and that shouldn't count against a
    column's availability among rows that DO have coverage).
    """
    feature_cols = [c for c in df.columns if c not in base_cols]
    relevant_rows = df[df[condition_col] == True] if condition_col else df
    nan_frac = relevant_rows[feature_cols].isna().mean()
    keep_cols = nan_frac[nan_frac <= threshold].index.tolist()
    dropped_cols = nan_frac[nan_frac > threshold].index.tolist()
    if dropped_cols:
        print(f"  dropped {len(dropped_cols)} sparse columns")
    return df[list(base_cols) + keep_cols]


def build_arms(features_df):
    """
    Build the five sensor-ablation arms. weather_matched/combined/openaq_only
    share the identical OpenAQ-covered subset so sample size and class
    balance are held fixed across the paired comparison -- this isolates
    OpenAQ's marginal contribution from confounds introduced by its sparse,
    non-random coverage (see paper Section 3.6).
    """
    base_cols = {"query_point_idx", "horizon_hours", "cutoff_time", "label", "split"}
    weather_cols = [c for c in features_df.columns if c.startswith("weather_")]
    openaq_cols = [c for c in features_df.columns if c.startswith("openaq_")]

    arm_weather_only = features_df[list(base_cols) + weather_cols].copy()
    arm_openaq_only = features_df[features_df["has_openaq"] == True][list(base_cols) + openaq_cols].copy()
    arm_combined = features_df[
        (features_df["has_weather"] == True) & (features_df["has_openaq"] == True)
    ][list(base_cols) + weather_cols + openaq_cols].copy()
    arm_missingness_aware = features_df[
        list(base_cols) + ["has_weather", "has_openaq"] + weather_cols + openaq_cols
    ].copy()

    matched_indices = arm_combined[["query_point_idx", "horizon_hours"]]
    arm_weather_matched = arm_weather_only.merge(matched_indices, on=["query_point_idx", "horizon_hours"], how="inner")

    arms = {
        "weather_only": arm_weather_only,
        "weather_matched": arm_weather_matched,
        "combined": arm_combined,
        "openaq_only": arm_openaq_only,
        "missingness_aware": arm_missingness_aware,
    }

    cleaned_arms = {}
    for name, df in arms.items():
        if name == "missingness_aware":
            cleaned_df = drop_sparse_columns(df, base_cols, condition_col="has_openaq")
        else:
            cleaned_df = drop_sparse_columns(df, base_cols)
        cleaned_arms[name] = cleaned_df
        print(f"{name}: {len(cleaned_df)} rows, {len(cleaned_df.columns) - len(base_cols)} features")

    return cleaned_arms


def main():
    parser = argparse.ArgumentParser(description="Build leakage-safe features and sensor-ablation arms.")
    parser.add_argument("--in_dir", type=str, default="../data/raw")
    parser.add_argument("--out_dir", type=str, default="../data/processed")
    args = parser.parse_args()

    query_points = pd.read_csv(os.path.join(args.in_dir, "query_points.csv"))
    query_points["ref_time"] = pd.to_datetime(query_points["ref_time"], utc=True)

    weather_df = pd.read_csv(os.path.join(args.in_dir, "weather_raw.csv"))
    weather_df["time"] = pd.to_datetime(weather_df["time"], utc=True)

    openaq_path = os.path.join(args.in_dir, "openaq_raw.csv")
    if os.path.exists(openaq_path):
        openaq_df = pd.read_csv(openaq_path)
        openaq_df["datetime"] = pd.to_datetime(openaq_df["datetime"], utc=True)
    else:
        openaq_df = pd.DataFrame(columns=["query_point_idx", "datetime", "parameter", "value"])

    features_df = build_feature_table(query_points, weather_df, openaq_df)
    verify_no_leakage(query_points, weather_df, openaq_df, features_df)

    os.makedirs(args.out_dir, exist_ok=True)
    features_df.to_csv(os.path.join(args.out_dir, "features_all_horizons.csv"), index=False)

    cleaned_arms = build_arms(features_df)
    for name, df in cleaned_arms.items():
        df.to_csv(os.path.join(args.out_dir, f"arm_{name}.csv"), index=False)

    print(f"Saved features and {len(cleaned_arms)} arms to {args.out_dir}")


if __name__ == "__main__":
    main()

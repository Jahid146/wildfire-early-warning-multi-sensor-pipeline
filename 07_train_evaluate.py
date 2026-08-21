"""
07_train_evaluate.py

Trains and evaluates:
  (1) Logistic regression + XGBoost baselines across all five sensor-
      ablation arms and five lead-time horizons, with bootstrap PR-AUC
      confidence intervals and chance-level lift.
  (2) GRU-D on raw weather sequences (weather_only arm equivalent),
      using the corrected, leakage-safe sequence-construction logic.
  (3) A location-only vs. weather+location confound check, to verify
      predictive skill reflects genuine temporal signal rather than
      static geographic fire-proneness.

IMPORTANT -- read before adapting build_sequence() to a new project:
An earlier version of this sequence-construction code contained a subtle
bug: it floored the grid to the hour but not the raw timestamp filter,
which caused every negative example (constructed with exact-hour pseudo-
timestamps) to align perfectly with hourly weather data while every
positive example (using sub-hour-precision satellite timestamps) did not.
This produced a mask channel that perfectly predicted the label,
independent of any real signal, and inflated GRU-D's validation PR-AUC to
a suspicious 1.000 at every horizon. The fix -- flooring the cutoff
timestamp consistently in both the filter and the grid -- is applied
below. If you see near-perfect validation scores on an event-detection
task with differently-sourced positive/negative timestamps, check for
this exact failure mode before trusting the result.

Usage:
    python 07_train_evaluate.py --in_dir ../data/processed --raw_dir ../data/raw --out_dir ../data/results
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

HORIZONS_HOURS = [6, 12, 24, 48, 72]
BASE_COLS = {"query_point_idx", "horizon_hours", "cutoff_time", "label", "split"}
SEQ_LENGTH = 48
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]


# ---------------------------------------------------------------------------
# Shared evaluation utility
# ---------------------------------------------------------------------------

def bootstrap_pr_auc(y_true, y_scores, n_bootstrap=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_b, s_b = y_true[idx], y_scores[idx]
        if len(np.unique(y_b)) < 2:
            continue
        scores.append(average_precision_score(y_b, s_b))
    return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


# ---------------------------------------------------------------------------
# Part 1: baseline models (logistic regression, XGBoost) across all arms
# ---------------------------------------------------------------------------

def prepare_arm_horizon(df, horizon, feature_cols):
    subset = df[df["horizon_hours"] == horizon].copy()
    train = subset[subset["split"] == "train"]
    test = subset[subset["split"] == "test"]

    X_train_raw = train[feature_cols].values
    X_test_raw = test[feature_cols].values
    y_train = train["label"].values
    y_test = test["label"].values

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    return X_train_imp, X_test_imp, X_train_scaled, X_test_scaled, y_train, y_test


def train_baselines(arms, horizons=HORIZONS_HOURS):
    results = []
    for arm_name, df in arms.items():
        feature_cols = [c for c in df.columns if c not in BASE_COLS]
        print(f"\n=== Arm: {arm_name} ({len(feature_cols)} features) ===")

        for horizon in horizons:
            X_train_imp, X_test_imp, X_train_s, X_test_s, y_train, y_test = prepare_arm_horizon(df, horizon, feature_cols)

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                print(f"  horizon {horizon}h: skipped, single-class split")
                continue

            pos_rate_test = y_test.mean()

            lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
            lr.fit(X_train_s, y_train)
            lr_scores = lr.predict_proba(X_test_s)[:, 1]
            lr_mean, lr_lo, lr_hi = bootstrap_pr_auc(y_test, lr_scores)

            xgb = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
                eval_metric="logloss", random_state=42,
            )
            xgb.fit(X_train_imp, y_train)
            xgb_scores = xgb.predict_proba(X_test_imp)[:, 1]
            xgb_mean, xgb_lo, xgb_hi = bootstrap_pr_auc(y_test, xgb_scores)

            results.append({"arm": arm_name, "horizon_hours": horizon, "model": "logistic_regression",
                             "pr_auc": lr_mean, "ci_low": lr_lo, "ci_high": lr_hi,
                             "n_train": len(y_train), "n_test": len(y_test), "test_pos_rate": pos_rate_test})
            results.append({"arm": arm_name, "horizon_hours": horizon, "model": "xgboost",
                             "pr_auc": xgb_mean, "ci_low": xgb_lo, "ci_high": xgb_hi,
                             "n_train": len(y_train), "n_test": len(y_test), "test_pos_rate": pos_rate_test})

            print(f"  horizon {horizon}h: LR PR-AUC={lr_mean:.3f} [{lr_lo:.3f},{lr_hi:.3f}] | "
                  f"XGB PR-AUC={xgb_mean:.3f} [{xgb_lo:.3f},{xgb_hi:.3f}]")

    results_df = pd.DataFrame(results)
    results_df["chance_level"] = results_df["test_pos_rate"]
    results_df["lift_over_chance"] = results_df["pr_auc"] / results_df["chance_level"]
    return results_df


# ---------------------------------------------------------------------------
# Part 2: GRU-D on raw weather sequences (corrected, leakage-safe version)
# ---------------------------------------------------------------------------

def build_sequence(qp_idx, cutoff, weather_df, seq_hours=SEQ_LENGTH):
    """
    Corrected version: the cutoff is floored to the hour CONSISTENTLY in
    both the raw-data filter and the grid construction. Flooring only
    moves the boundary earlier in time, so leakage-safety is preserved;
    critically, it also puts positive and negative examples on the same
    timestamp reference frame, closing the mask-leak bug described in
    this file's module docstring.
    """
    cutoff_floor = cutoff.floor("h")
    window_start = cutoff_floor - pd.Timedelta(hours=seq_hours)

    subset = weather_df[
        (weather_df["query_point_idx"] == qp_idx)
        & (weather_df["time"] < cutoff_floor)
        & (weather_df["time"] >= window_start)
    ].sort_values("time")

    if len(subset) == 0:
        return None

    full_grid = pd.date_range(window_start, cutoff_floor, freq="h", inclusive="left")
    grid_df = pd.DataFrame({"time": full_grid})
    merged = grid_df.merge(subset[["time"] + WEATHER_VARS], on="time", how="left")

    values = merged[WEATHER_VARS].values
    mask = (~merged[WEATHER_VARS].isna()).values.astype(float)

    deltas = np.zeros_like(values, dtype=float)
    for v in range(values.shape[1]):
        last_obs = 0
        for t in range(len(values)):
            if mask[t, v] == 1:
                deltas[t, v] = 0
                last_obs = t
            else:
                deltas[t, v] = t - last_obs if t > 0 else 0

    return values, mask, deltas


def build_horizon_dataset(query_points, weather_df, horizon):
    X_list, M_list, D_list, y_list, split_list = [], [], [], [], []

    for idx, row in query_points.iterrows():
        cutoff = row["ref_time"] - pd.Timedelta(hours=horizon)
        result = build_sequence(idx, cutoff, weather_df)
        if result is None:
            continue
        values, mask, deltas = result
        values_filled = pd.DataFrame(values).ffill().bfill().fillna(0).values

        X_list.append(values_filled)
        M_list.append(mask)
        D_list.append(deltas)
        y_list.append(1 if row["point_type"] == "positive" else 0)
        split_list.append(row["split"])

    return (np.array(X_list), np.array(M_list), np.array(D_list), np.array(y_list), np.array(split_list))


def verify_no_mask_leak(query_points, weather_df, horizons=HORIZONS_HOURS):
    """
    Diagnostic check to run before trusting any GRU-D result: confirms
    the observation mask does not trivially predict the label. A
    ConstantInputWarning / undefined correlation is the expected, healthy
    outcome when weather coverage is complete and uniform across classes.
    """
    import scipy.stats as stats

    for horizon in horizons:
        X, M, D, y, split = build_horizon_dataset(query_points, weather_df, horizon)
        test_mask = split == "test"
        M_test, y_test = M[test_mask], y[test_mask]
        fully_missing = (M_test.sum(axis=(1, 2)) == 0)
        mask_sums = M_test.sum(axis=(1, 2))
        if len(np.unique(mask_sums)) < 2:
            print(f"horizon {horizon}h: fully_missing={fully_missing.sum()}/{len(M_test)}, "
                  f"mask_sums constant ({mask_sums[0]:.0f}) -- no mask-label correlation possible (healthy)")
        else:
            corr = stats.pointbiserialr(y_test, mask_sums)
            print(f"horizon {horizon}h: fully_missing={fully_missing.sum()}/{len(M_test)}, "
                  f"mask-label correlation={corr.statistic:.3f} (p={corr.pvalue:.3f})")
            if abs(corr.statistic) > 0.5:
                print("  WARNING: substantial mask-label correlation detected -- "
                      "investigate before trusting downstream GRU-D results.")


class GRUD(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.gamma_x = nn.Linear(input_size, input_size)
        self.gamma_h = nn.Linear(input_size, hidden_size)

        self.gru_cell = nn.GRUCell(input_size * 2, hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, X, M, D):
        batch_size, seq_len, _ = X.shape
        h = torch.zeros(batch_size, self.hidden_size, device=X.device)
        x_last_obs = X[:, 0, :].clone()

        for t in range(seq_len):
            x_t, m_t, d_t = X[:, t, :], M[:, t, :], D[:, t, :]

            gamma_x_t = torch.exp(-torch.relu(self.gamma_x(d_t)))
            x_decayed = gamma_x_t * x_last_obs + (1 - gamma_x_t) * x_t
            x_last_obs = torch.where(m_t.bool(), x_t, x_decayed)

            gamma_h_t = torch.exp(-torch.relu(self.gamma_h(d_t)))
            h = gamma_h_t * h

            gru_input = torch.cat([x_decayed, m_t], dim=1)
            h = self.gru_cell(gru_input, h)

        return self.classifier(h).squeeze(-1)


def train_grud_for_horizon(query_points, weather_df, horizon, epochs=30, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    X, M, D, y, split = build_horizon_dataset(query_points, weather_df, horizon)

    train_mask = split == "train"
    test_mask = split == "test"

    X_train = torch.tensor(X[train_mask], dtype=torch.float32).to(device)
    M_train = torch.tensor(M[train_mask], dtype=torch.float32).to(device)
    D_train = torch.tensor(D[train_mask], dtype=torch.float32).to(device)
    y_train = torch.tensor(y[train_mask], dtype=torch.float32).to(device)

    X_test = torch.tensor(X[test_mask], dtype=torch.float32).to(device)
    M_test = torch.tensor(M[test_mask], dtype=torch.float32).to(device)
    D_test = torch.tensor(D[test_mask], dtype=torch.float32).to(device)
    y_test_np = y[test_mask]

    model = GRUD(input_size=X.shape[2], hidden_size=32).to(device)
    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train, M_train, D_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"    epoch {epoch}: loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test, M_test, D_test)
        test_scores = torch.sigmoid(test_logits).cpu().numpy()

    pr_mean, pr_lo, pr_hi = bootstrap_pr_auc(y_test_np, test_scores)
    return pr_mean, pr_lo, pr_hi, len(y_train), len(y_test_np), y_test_np.mean()


def train_grud(query_points, weather_df, horizons=HORIZONS_HOURS):
    grud_results = []
    for horizon in horizons:
        print(f"\n=== GRU-D, horizon {horizon}h ===")
        pr_mean, pr_lo, pr_hi, n_train, n_test, pos_rate = train_grud_for_horizon(query_points, weather_df, horizon)
        grud_results.append({"arm": "grud_weather", "horizon_hours": horizon, "model": "grud",
                              "pr_auc": pr_mean, "ci_low": pr_lo, "ci_high": pr_hi,
                              "n_train": n_train, "n_test": n_test, "test_pos_rate": pos_rate,
                              "chance_level": pos_rate, "lift_over_chance": pr_mean / pos_rate})
        print(f"  PR-AUC={pr_mean:.3f} [{pr_lo:.3f},{pr_hi:.3f}]")
    return pd.DataFrame(grud_results)


# ---------------------------------------------------------------------------
# Part 3: location-only vs. weather+location confound check
# ---------------------------------------------------------------------------

def location_confound_check(weather_only_arm, query_points, horizons=HORIZONS_HOURS):
    rows = []
    for horizon in horizons:
        subset = weather_only_arm[weather_only_arm["horizon_hours"] == horizon]
        train = subset[subset["split"] == "train"]
        test = subset[subset["split"] == "test"]

        train_latlon = train.merge(query_points[["lat", "lon"]], left_on="query_point_idx", right_index=True)
        test_latlon = test.merge(query_points[["lat", "lon"]], left_on="query_point_idx", right_index=True)

        # location-only baseline
        lr_loc = LogisticRegression(class_weight="balanced", random_state=42)
        lr_loc.fit(train_latlon[["lat", "lon"]], train_latlon["label"])
        loc_scores = lr_loc.predict_proba(test_latlon[["lat", "lon"]])[:, 1]
        loc_pr, loc_lo, loc_hi = bootstrap_pr_auc(test_latlon["label"].values, loc_scores)

        # weather + location
        weather_cols = [c for c in weather_only_arm.columns if c.startswith("weather_")]
        feat_cols = weather_cols + ["lat", "lon"]
        X_train = train_latlon[feat_cols].fillna(train_latlon[feat_cols].median())
        X_test = test_latlon[feat_cols].fillna(train_latlon[feat_cols].median())

        lr_full = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        lr_full.fit(X_train, train_latlon["label"])
        full_scores = lr_full.predict_proba(X_test)[:, 1]
        full_pr, full_lo, full_hi = bootstrap_pr_auc(test_latlon["label"].values, full_scores)

        print(f"horizon {horizon}h -- location only: {loc_pr:.3f} [{loc_lo:.3f},{loc_hi:.3f}] | "
              f"weather+location: {full_pr:.3f} [{full_lo:.3f},{full_hi:.3f}]")

        rows.append({"horizon_hours": horizon, "location_only_pr_auc": loc_pr,
                      "location_only_ci_low": loc_lo, "location_only_ci_high": loc_hi,
                      "weather_location_pr_auc": full_pr,
                      "weather_location_ci_low": full_lo, "weather_location_ci_high": full_hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate baselines, GRU-D, and confound checks.")
    parser.add_argument("--in_dir", type=str, default="../data/processed", help="Directory with arm_*.csv files")
    parser.add_argument("--raw_dir", type=str, default="../data/raw", help="Directory with query_points.csv, weather_raw.csv")
    parser.add_argument("--out_dir", type=str, default="../data/results")
    parser.add_argument("--skip_grud", action="store_true", help="Skip GRU-D training (e.g., no GPU available)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    arm_names = ["weather_only", "weather_matched", "combined", "openaq_only", "missingness_aware"]
    arms = {name: pd.read_csv(os.path.join(args.in_dir, f"arm_{name}.csv")) for name in arm_names}

    print("=== Baseline models (logistic regression, XGBoost) ===")
    results_df = train_baselines(arms)
    results_df.to_csv(os.path.join(args.out_dir, "baseline_results.csv"), index=False)

    pivot = results_df.pivot_table(index=["arm", "horizon_hours"], columns="model", values="lift_over_chance")
    print("\n=== Lift-over-chance pivot ===")
    print(pivot)

    query_points = pd.read_csv(os.path.join(args.raw_dir, "query_points.csv"))
    query_points["ref_time"] = pd.to_datetime(query_points["ref_time"], utc=True)

    print("\n=== Location confound check ===")
    confound_df = location_confound_check(arms["weather_only"], query_points)
    confound_df.to_csv(os.path.join(args.out_dir, "location_confound_check.csv"), index=False)

    if not args.skip_grud:
        weather_df = pd.read_csv(os.path.join(args.raw_dir, "weather_raw.csv"))
        weather_df["time"] = pd.to_datetime(weather_df["time"], utc=True)

        print("\n=== GRU-D mask-leak diagnostic (run before trusting results) ===")
        verify_no_mask_leak(query_points, weather_df)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n=== Training GRU-D on device: {device} ===")
        grud_results_df = train_grud(query_points, weather_df)
        grud_results_df.to_csv(os.path.join(args.out_dir, "grud_results.csv"), index=False)

    print(f"\nAll results saved to {args.out_dir}")


if __name__ == "__main__":
    main()

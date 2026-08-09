"""
Offline ML Training for the Integrated FDC Framework
=====================================================
Journal Extension of ICAE2025 Paper (Applied Energy)

Reads the fault-free baseline CSV produced by Phase1_PoC.py,
engineers features, trains OCSVM / Isolation Forest / LOF,
calibrates thresholds, serialises all artifacts, and produces
diagnostic plots.

Outputs (all saved to MODEL_DIR):
    ocsvm.joblib          – trained One-Class SVM
    iforest.joblib        – trained Isolation Forest
    lof.joblib            – trained Local Outlier Factor (novelty=True)
    scaler.joblib         – fitted StandardScaler
    thresholds.joblib     – dict of per-model, per-percentile thresholds
    training_config.json  – human-readable record of all settings
    training_data_featured.csv – filtered+featured training set

Plots (all saved to PLOT_DIR):
    01_feature_distributions.png
    02_feature_correlation.png
    03_score_distributions.png
    04_scores_over_time.png
    05_feature_scatter_ocsvm.png
    06_cross_model_agreement.png

Author:  Nima Monghasemi
Date:    February 2026

Prerequisites
    conda activate FDC
    (scikit-learn 1.8.0, numpy 2.4.2, pandas 3.0.0, matplotlib)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import joblib


# ================================================================
#  CONFIGURATION — edit paths / parameters here
# ================================================================

# Project root defaults to this file's directory; override with FDC_PROJECT_DIR
# when the runs and models live outside the repo checkout.
PROJECT_DIR = os.environ.get(
    "FDC_PROJECT_DIR", os.path.dirname(os.path.abspath(__file__))
)
INPUT_CSV   = os.path.join(PROJECT_DIR, "output_poc", "api_sensor_log.csv")
MODEL_DIR   = os.path.join(PROJECT_DIR, "models")
PLOT_DIR    = os.path.join(PROJECT_DIR, "plots", "training")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Feature-engineering knobs
ROLLING_WINDOW         = 12      # 12 × 10 min = 2 h  (matches the Erl proxy)
OCCUPIED_SP_THRESHOLD  = 18.0    # °C — above this ⇒ occupied heating mode
MIN_FLOW_THRESHOLD     = 0.001   # kg/s — above this ⇒ heating is active

# Model-training knobs
CONTAMINATION  = 0.05   # target outlier fraction on training data
RANDOM_STATE   = 42
OCSVM_NU       = 0.05
OCSVM_KERNEL   = "rbf"
OCSVM_GAMMA    = "scale"
IF_ESTIMATORS  = 200
LOF_NEIGHBORS  = 20

# Feature set
FEATURE_COLS = [
    "temp_error",           # T_zone_core − T_setpoint
    "flow_ratio",           # mdot / peak_mdot
    "temp_error_2h_avg",    # rolling 2-h mean of temp_error
    "dT_zone_dt",           # ΔT_zone per timestep  (≈ per 10 min)
    "T_outdoor",            # outdoor drybulb (context)
    "delta_T_hw",           # T_hw_inlet − T_hw_outlet
]

# Percentile thresholds to store (for parametric sweep in the paper)
THRESHOLD_PERCENTILES = [1, 2, 3, 5, 7, 10]


# ================================================================
#  STEP 1 — Load the Phase-1 baseline CSV
# ================================================================

def load_baseline_csv(path: str) -> pd.DataFrame:
    """Read api_sensor_log.csv and build a proper datetime index."""
    print("=" * 65)
    print("STEP 1 : Loading baseline data")
    print("=" * 65)

    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows  ×  {len(df.columns)} columns")
    print(f"  Columns : {list(df.columns)}")

    # ---- build datetime from EnergyPlus time columns ----
    # month >= 10 → 2017  (Oct–Dec);  month <= 4 → 2018  (Jan–Apr)
    years = df["month"].apply(lambda m: 2017 if m >= 10 else 2018)

    date_str = (
        years.astype(str)
        + "-"
        + df["month"].astype(int).astype(str).str.zfill(2)
        + "-"
        + df["day"].astype(int).astype(str).str.zfill(2)
    )
    base_dates = pd.to_datetime(date_str, format="%Y-%m-%d")

    # hour may be 24 (end-of-day in E+) → timedelta handles it
    offsets = pd.to_timedelta(df["hour"].astype(int), unit="h") + pd.to_timedelta(
        df["minute"].astype(int), unit="m"
    )
    df["datetime"] = base_dates + offsets
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    n_bad = df["datetime"].isna().sum()
    if n_bad:
        print(f"  WARNING : {n_bad} rows with unparseable timestamps — dropped")
        df.dropna(subset=["datetime"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    print(f"  Date range : {df['datetime'].iloc[0]}  ->  {df['datetime'].iloc[-1]}")

    sensor_cols = [
        c for c in df.columns
        if c not in ("hour", "minute", "month", "day", "datetime")
    ]
    print(f"\n  Sensor summary statistics (all {len(df):,} timesteps) :")
    print(df[sensor_cols].describe().round(3).to_string())

    return df


# ================================================================
#  STEP 2 — Filter to occupied + actively-heating timesteps
# ================================================================

def filter_active_heating(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where the building is occupied AND heating."""
    print("\n" + "=" * 65)
    print("STEP 2 : Filtering for active-heating periods")
    print("=" * 65)

    mask_occ  = df["htg_sp_core"] >= OCCUPIED_SP_THRESHOLD
    mask_flow = df["hw_flow_core"] > MIN_FLOW_THRESHOLD
    mask      = mask_occ & mask_flow

    df_active = df.loc[mask].copy().reset_index(drop=True)

    print(f"  Total timesteps          : {len(df):>8,}")
    print(f"  Occupied  (SP >= {OCCUPIED_SP_THRESHOLD}°C)   : {mask_occ.sum():>8,}")
    print(f"  Flow active (> {MIN_FLOW_THRESHOLD} kg/s) : {mask_flow.sum():>8,}")
    print(f"  Both -> training pool     : {len(df_active):>8,}  "
          f"({100 * len(df_active) / len(df):.1f}%)")

    return df_active


# ================================================================
#  STEP 3 — Feature engineering
# ================================================================

def engineer_features(df: pd.DataFrame):
    """Compute ML features; return (df_with_features, feature_cols, peak_flow)."""
    print("\n" + "=" * 65)
    print("STEP 3 : Feature engineering")
    print("=" * 65)

    df = df.copy()

    # 1. Temperature error  (comfort deviation)
    df["temp_error"] = df["zone_temp_core"] - df["htg_sp_core"]

    # 2. Flow ratio  (normalised by observed baseline peak)
    peak_flow = df["hw_flow_core"].max()
    df["flow_ratio"] = df["hw_flow_core"] / peak_flow
    print(f"  Peak observed baseline flow : {peak_flow:.4f} kg/s")

    # 3. Rolling 2-h average of temperature error
    df["temp_error_2h_avg"] = (
        df["temp_error"]
        .rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW)
        .mean()
    )

    # 4. Rate of change of zone temperature (per 10-min step)
    df["dT_zone_dt"] = df["zone_temp_core"].diff()

    # 5. Outdoor temperature (rename for clarity)
    df["T_outdoor"] = df["oa_temp"]

    # 6. Heat-extraction ΔT across the baseboard
    df["delta_T_hw"] = df["hw_inlet_temp_core"] - df["hw_outlet_temp_core"]

    # 7. (optional) supply-return temperature ratio — kept but not in
    #    default FEATURE_COLS; add if needed
    df["supply_return_ratio"] = np.where(
        df["hw_supply_temp"] > 1.0,
        df["hw_outlet_temp_core"] / df["hw_supply_temp"],
        np.nan,
    )

    # Drop NaN rows produced by rolling() and diff()
    n_before = len(df)
    df.dropna(subset=FEATURE_COLS, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Rows dropped (rolling/diff NaN) : {n_before - len(df)}")
    print(f"  Final training-set size          : {len(df):,} timesteps")

    print(f"\n  Feature statistics :")
    print(df[FEATURE_COLS].describe().round(4).to_string())

    return df, FEATURE_COLS, peak_flow


# ================================================================
#  STEP 4 — Scaling
# ================================================================

def fit_scaler(df: pd.DataFrame, feature_cols: list):
    """Fit a StandardScaler and return (X_scaled, scaler)."""
    print("\n" + "=" * 65)
    print("STEP 4 : Feature scaling (StandardScaler)")
    print("=" * 65)

    scaler = StandardScaler()
    X = df[feature_cols].values.astype(np.float64)
    X_scaled = scaler.fit_transform(X)

    print(f"  {'Feature':<25s}  {'mean':>8s}  {'std':>8s}  "
          f"{'scaled_mean':>12s}  {'scaled_std':>11s}")
    print("  " + "-" * 70)
    for i, col in enumerate(feature_cols):
        print(f"  {col:<25s}  {scaler.mean_[i]:>8.4f}  {scaler.scale_[i]:>8.4f}  "
              f"{X_scaled[:, i].mean():>12.6f}  {X_scaled[:, i].std():>11.6f}")

    return X_scaled, scaler


# ================================================================
#  STEP 5 — Train the three anomaly-detection models
# ================================================================

def train_models(X: np.ndarray) -> dict:
    """Train OCSVM, Isolation Forest, LOF.  Return dict of fitted models."""
    print("\n" + "=" * 65)
    print("STEP 5 : Training anomaly-detection models")
    print("=" * 65)

    models = {}

    # ---- OCSVM ----
    print(f"\n  [1/3] One-Class SVM  (kernel={OCSVM_KERNEL}, "
          f"gamma={OCSVM_GAMMA}, nu={OCSVM_NU})")
    ocsvm = OneClassSVM(kernel=OCSVM_KERNEL, gamma=OCSVM_GAMMA, nu=OCSVM_NU)
    ocsvm.fit(X)
    preds = ocsvm.predict(X)
    frac = (preds == -1).mean()
    print(f"    Training outlier fraction : {frac:.4f}  ({frac * 100:.2f}%)")
    models["ocsvm"] = ocsvm

    # ---- Isolation Forest ----
    print(f"\n  [2/3] Isolation Forest  (n_estimators={IF_ESTIMATORS}, "
          f"contamination={CONTAMINATION})")
    iforest = IsolationForest(
        n_estimators=IF_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )
    iforest.fit(X)
    preds = iforest.predict(X)
    frac = (preds == -1).mean()
    print(f"    Training outlier fraction : {frac:.4f}  ({frac * 100:.2f}%)")
    models["iforest"] = iforest

    # ---- LOF  (novelty mode) ----
    print(f"\n  [3/3] Local Outlier Factor  (n_neighbors={LOF_NEIGHBORS}, "
          f"novelty=True, contamination={CONTAMINATION})")
    lof = LocalOutlierFactor(
        n_neighbors=LOF_NEIGHBORS,
        contamination=CONTAMINATION,
        novelty=True,
    )
    lof.fit(X)
    preds = lof.predict(X)
    frac = (preds == -1).mean()
    print(f"    Training outlier fraction : {frac:.4f}  ({frac * 100:.2f}%)")
    models["lof"] = lof

    return models


# ================================================================
#  STEP 6 — Extract anomaly scores & calibrate thresholds
# ================================================================

def calibrate_thresholds(models: dict, X: np.ndarray):
    """
    Compute continuous anomaly scores on the training set and store
    thresholds at multiple percentiles for the parametric sweep.
    
    Returns
    -------
    scores : dict   — {model_name: 1-D array}
    thresholds : dict — {model_name: {"default": ..., "p2": ..., ...}}
    """
    print("\n" + "=" * 65)
    print("STEP 6 : Threshold calibration")
    print("=" * 65)

    scores = {}
    thresholds = {}

    extractors = {
        "ocsvm":   lambda m, X: m.decision_function(X),   # +ve = normal
        "iforest": lambda m, X: m.score_samples(X),       # higher = more normal
        "lof":     lambda m, X: m.score_samples(X),       # higher = more normal
    }

    for name in ["ocsvm", "iforest", "lof"]:
        s = extractors[name](models[name], X)
        scores[name] = s

        # percentile-based thresholds
        t = {}
        for p in THRESHOLD_PERCENTILES:
            t[f"p{p}"] = float(np.percentile(s, p))

        # default = contamination-matched (≈ 5th percentile)
        t["default"] = float(np.percentile(s, CONTAMINATION * 100))
        thresholds[name] = t

        fpr_def = (s < t["default"]).mean()
        print(f"\n  {name.upper()}")
        print(f"    Score range : [{s.min():.4f},  {s.max():.4f}]   "
              f"mean={s.mean():.4f}")
        print(f"    Default threshold ({t['default']:.4f})  ->  "
              f"FPR = {fpr_def:.4f}")
        for p in THRESHOLD_PERCENTILES:
            print(f"    p{p:<3d} threshold ({t[f'p{p}']:.4f})  ->  "
                  f"FPR = {p / 100:.4f}")

    return scores, thresholds


# ================================================================
#  STEP 7 — Serialise everything
# ================================================================

def save_artifacts(
    models, scaler, thresholds, feature_cols, peak_flow, df_train
):
    """Persist all artifacts needed for in-the-loop deployment."""
    print("\n" + "=" * 65)
    print("STEP 7 : Serialising artifacts")
    print("=" * 65)

    for name, model in models.items():
        p = os.path.join(MODEL_DIR, f"{name}.joblib")
        joblib.dump(model, p)
        print(f"  Saved : {name}.joblib")

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    print("  Saved : scaler.joblib")

    joblib.dump(thresholds, os.path.join(MODEL_DIR, "thresholds.joblib"))
    print("  Saved : thresholds.joblib")

    # Human-readable config
    config = {
        "training_timestamp":        datetime.now().isoformat(),
        "input_csv":                 INPUT_CSV,
        "n_training_samples":        len(df_train),
        "feature_columns":           feature_cols,
        "peak_flow_kg_s":            float(peak_flow),
        "rolling_window_steps":      ROLLING_WINDOW,
        "rolling_window_minutes":    ROLLING_WINDOW * 10,
        "contamination":             CONTAMINATION,
        "ocsvm_nu":                  OCSVM_NU,
        "ocsvm_kernel":              OCSVM_KERNEL,
        "ocsvm_gamma":               OCSVM_GAMMA,
        "if_n_estimators":           IF_ESTIMATORS,
        "lof_n_neighbors":           LOF_NEIGHBORS,
        "occupied_sp_threshold_C":   OCCUPIED_SP_THRESHOLD,
        "min_flow_threshold_kg_s":   MIN_FLOW_THRESHOLD,
        "random_state":              RANDOM_STATE,
        "threshold_percentiles":     THRESHOLD_PERCENTILES,
        "thresholds":                thresholds,
        "scaler_mean":               scaler.mean_.tolist(),
        "scaler_scale":              scaler.scale_.tolist(),
    }
    cfg_path = os.path.join(MODEL_DIR, "training_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    print("  Saved : training_config.json")

    # Save the full featured training DataFrame
    csv_path = os.path.join(MODEL_DIR, "training_data_featured.csv")
    df_train.to_csv(csv_path, index=False)
    print(f"  Saved : training_data_featured.csv  ({len(df_train):,} rows)")

    print(f"\n  All artifacts -> {MODEL_DIR}")


# ================================================================
#  STEP 8 — Diagnostic plots
# ================================================================

def plot_diagnostics(df, feature_cols, scores, thresholds):
    """Generate six diagnostic figures."""
    print("\n" + "=" * 65)
    print("STEP 8 : Diagnostic plots")
    print("=" * 65)

    model_names  = ["ocsvm", "iforest", "lof"]
    model_labels = ["OCSVM", "Isolation Forest", "LOF"]
    palette      = ["#4CAF50", "#FF9800", "#9C27B0"]

    # ---- 01  Feature distributions ----
    n_feat = len(feature_cols)
    ncols  = 3
    nrows  = (n_feat + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        ax.hist(df[col], bins=80, color="#2196F3", alpha=0.75,
                edgecolor="white", linewidth=0.3)
        mu = df[col].mean()
        ax.axvline(mu, color="red", ls="--", lw=1,
                   label=f"μ = {mu:.3f}")
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    for j in range(n_feat, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions — Fault-Free Training Data",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "01_feature_distributions.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved : 01_feature_distributions.png")

    # ---- 02  Correlation matrix ----
    fig, ax = plt.subplots(figsize=(8, 6.5))
    corr = df[feature_cols].corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_feat))
    ax.set_yticks(range(n_feat))
    ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(feature_cols, fontsize=9)
    for r in range(n_feat):
        for c in range(n_feat):
            val = corr.iloc[r, c]
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(val) > 0.6 else "black")
    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.82)
    ax.set_title("Feature Correlation Matrix", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "02_feature_correlation.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved : 02_feature_correlation.png")

    # ---- 03  Score distributions with threshold lines ----
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for i, (name, label, clr) in enumerate(
        zip(model_names, model_labels, palette)
    ):
        ax = axes[i]
        s = scores[name]
        t = thresholds[name]

        ax.hist(s, bins=120, color=clr, alpha=0.7,
                edgecolor="white", linewidth=0.2, density=True)
        ax.axvline(t["default"], color="red",    ls="-",  lw=1.5,
                   label=f"default  (FPR ≈ {CONTAMINATION:.0%})")
        ax.axvline(t["p2"],      color="orange", ls="--", lw=1.2,
                   label="p2  (FPR = 2%)")
        ax.axvline(t["p10"],     color="blue",   ls=":",  lw=1.2,
                   label="p10 (FPR = 10%)")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Anomaly score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle("Anomaly-Score Distributions on Training Data",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "03_score_distributions.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved : 03_score_distributions.png")

    # ---- 04  Scores over time ----
    fig, axes = plt.subplots(3, 1, figsize=(17, 10), sharex=True)
    t_axis = df["datetime"].values[: len(scores["ocsvm"])]

    for i, (name, label, clr) in enumerate(
        zip(model_names, model_labels, palette)
    ):
        ax = axes[i]
        s  = scores[name]
        th = thresholds[name]["default"]

        ax.plot(t_axis, s, color=clr, alpha=0.35, linewidth=0.3)
        ax.axhline(th, color="red", ls="--", lw=1, label="default threshold")

        outlier_mask = s < th
        if outlier_mask.any():
            ax.scatter(
                t_axis[outlier_mask], s[outlier_mask],
                color="red", s=4, alpha=0.6, zorder=5,
                label=f"flagged ({outlier_mask.sum():,} pts)",
            )

        ax.set_ylabel(f"{label}\nscore", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.25)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].set_xlabel("Date")
    fig.suptitle("Anomaly Scores Over Time — Fault-Free Baseline",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "04_scores_over_time.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved : 04_scores_over_time.png")

    # ---- 05  Feature-pair scatter coloured by OCSVM score ----
    pairs = [
        ("temp_error",       "flow_ratio"),
        ("temp_error",       "temp_error_2h_avg"),
        ("flow_ratio",       "T_outdoor"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ocsvm_s = scores["ocsvm"]

    for i, (fx, fy) in enumerate(pairs):
        ax = axes[i]
        sc = ax.scatter(
            df[fx].values[: len(ocsvm_s)],
            df[fy].values[: len(ocsvm_s)],
            c=ocsvm_s, cmap="RdYlGn", s=1, alpha=0.5,
        )
        ax.set_xlabel(fx)
        ax.set_ylabel(fy)
        ax.set_title(f"{fx}  vs  {fy}", fontsize=10, fontweight="bold")
        fig.colorbar(sc, ax=ax, label="OCSVM score", shrink=0.82)

    fig.suptitle("Feature-Pair Scatter (coloured by OCSVM score)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "05_feature_scatter_ocsvm.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved : 05_feature_scatter_ocsvm.png")

    # ---- 06  Cross-model Venn-style agreement bar chart ----
    flags = {}
    for name in model_names:
        flags[name] = scores[name] < thresholds[name]["default"]

    n_total = len(ocsvm_s)
    only_oc = ( flags["ocsvm"] & ~flags["iforest"] & ~flags["lof"]).sum()
    only_if = (~flags["ocsvm"] &  flags["iforest"] & ~flags["lof"]).sum()
    only_lo = (~flags["ocsvm"] & ~flags["iforest"] &  flags["lof"]).sum()
    two_any = (
        (flags["ocsvm"].astype(int)
         + flags["iforest"].astype(int)
         + flags["lof"].astype(int)) == 2
    ).sum()
    all_three = (
        flags["ocsvm"] & flags["iforest"] & flags["lof"]
    ).sum()

    labels = ["OCSVM\nonly", "IF\nonly", "LOF\nonly", "Any 2", "All 3"]
    counts = [only_oc, only_if, only_lo, two_any, all_three]
    colors_bar = ["#4CAF50", "#FF9800", "#9C27B0", "#607D8B", "#F44336"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=colors_bar, edgecolor="white", lw=0.8)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(cnt), ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    ax.set_ylabel("Number of flagged timesteps")
    ax.set_title("Cross-Model Agreement on Training-Data Outliers",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "06_cross_model_agreement.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved : 06_cross_model_agreement.png")

    print(f"\n  All plots -> {PLOT_DIR}")


# ================================================================
#  STEP 9 — Validation summary (printed report)
# ================================================================

def print_validation_summary(scores, thresholds, X):
    """Console summary of FPR, cross-model agreement, and next steps."""
    print("\n" + "=" * 65)
    print("STEP 9 : Training-set validation summary")
    print("=" * 65)

    n = X.shape[0]

    print(f"\n  {'Model':<22s} {'Threshold':>10s} {'Outliers':>9s} "
          f"{'FPR':>7s}  {'Inliers':>9s}")
    print("  " + "-" * 62)
    for name, label in [("ocsvm", "OCSVM"),
                        ("iforest", "Isolation Forest"),
                        ("lof", "LOF")]:
        s   = scores[name]
        th  = thresholds[name]["default"]
        n_o = int((s < th).sum())
        fpr = n_o / n
        print(f"  {label:<22s} {th:>10.4f} {n_o:>9,d} {fpr:>7.4f}  "
              f"{n - n_o:>9,d}")

    # Cross-model
    f_oc = scores["ocsvm"]   < thresholds["ocsvm"]["default"]
    f_if = scores["iforest"] < thresholds["iforest"]["default"]
    f_lo = scores["lof"]     < thresholds["lof"]["default"]

    any_flag = (f_oc | f_if | f_lo).sum()
    ge2      = ((f_oc.astype(int) + f_if.astype(int) + f_lo.astype(int)) >= 2).sum()
    all3     = (f_oc & f_if & f_lo).sum()

    print(f"\n  Cross-model agreement (default thresholds) :")
    print(f"    Flagged by ANY model  : {any_flag:>7,d}  "
          f"({100 * any_flag / n:.2f}%)")
    print(f"    Flagged by >= 2 models : {ge2:>7,d}  "
          f"({100 * ge2 / n:.2f}%)")
    print(f"    Flagged by ALL 3      : {all3:>7,d}  "
          f"({100 * all3 / n:.2f}%)")

    print("\n  Deployment recommendation :")
    print(f"    Use the 'default' threshold (contamination = {CONTAMINATION})")
    print( "    as the starting operating point.  The parametric analysis")
    print( "    in the journal paper will sweep p1 -> p10 for sensitivity curves.")


# ================================================================
#  MAIN
# ================================================================

if __name__ == "__main__":

    t_start = datetime.now()
    print("=" * 65)
    print("   OFFLINE ML TRAINING — FDC Framework")
    print("   Journal Extension of ICAE2025 -> Applied Energy")
    print(f"   {t_start.strftime('%Y-%m-%d  %H:%M:%S')}")
    print("=" * 65)

    # 1. Load
    df_raw = load_baseline_csv(INPUT_CSV)

    # 2. Filter
    df_active = filter_active_heating(df_raw)

    # 3. Features
    df_feat, feat_cols, peak_flow = engineer_features(df_active)

    # 4. Scale
    X_scaled, scaler = fit_scaler(df_feat, feat_cols)

    # 5. Train
    models = train_models(X_scaled)

    # 6. Thresholds
    scores, thresholds = calibrate_thresholds(models, X_scaled)

    # 7. Save
    save_artifacts(models, scaler, thresholds, feat_cols, peak_flow, df_feat)

    # 8. Plots
    plot_diagnostics(df_feat, feat_cols, scores, thresholds)

    # 9. Summary
    print_validation_summary(scores, thresholds, X_scaled)

    # Final report
    elapsed = (datetime.now() - t_start).total_seconds()
    print("\n" + "=" * 65)
    print(f"   TRAINING COMPLETE   ({elapsed:.1f} s)")
    print("=" * 65)
    print(f"\n  Artifact inventory ({MODEL_DIR}) :")
    for fname in sorted(os.listdir(MODEL_DIR)):
        fpath = os.path.join(MODEL_DIR, fname)
        size  = os.path.getsize(fpath) / 1024
        print(f"    {fname:<40s}  {size:>8.1f} KB")

    print(f"\n  Plot inventory ({PLOT_DIR}) :")
    for fname in sorted(os.listdir(PLOT_DIR)):
        print(f"    {fname}")

    print("\n  > Next step : ML-in-the-loop integration")
    print("    Load serialised models inside the Python API callback")
    print("    to perform real-time anomaly detection at each zone timestep.")
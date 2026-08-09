# -*- coding: utf-8 -*-
"""
Phase 4b: Post-Hoc Threshold Sweep & Persistence Filter Analysis
=================================================================
Supply Curve Fault (System-Level Supply Temperature Bias)

Operates entirely on the already-logged fdc_runtime_log.csv files.
No re-simulation required.

Produces:
    1. Threshold sweep: metrics at each stored percentile (p1–p10)
    2. Persistence filter: requires N consecutive flags before declaring fault
    3. Combined sweep: threshold × persistence grid
    4. Sensitivity curves for the journal paper
    5. Updated fault window detail plot with best configuration
    6. Multi-zone impact analysis (system-level fault signature)

Author:  Nima Monghasemi
Date:    February 2026
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

import joblib

# ================================================================
#  CONFIGURATION
# ================================================================

MODEL_NAMES  = ["ocsvm", "iforest", "lof"]
MODEL_LABELS = {"ocsvm": "OCSVM", "iforest": "Isolation Forest", "lof": "LOF"}
MODEL_COLORS = {"ocsvm": "#4CAF50", "iforest": "#FF9800", "lof": "#9C27B0"}

PERSISTENCE_VALUES = [1, 2, 3, 4, 5, 6]
THRESHOLD_KEYS     = ["p1", "p2", "p3", "p5", "p7", "p10", "default"]

# Fault window: supply curve fault runs 24h (persistent sensor bias)
FAULT_WINDOW = {
    "start_month": 1, "end_month": 1,
    "start_day": 15,  "end_day": 16,
    "start_hour": 0,  "end_hour": 24
}

TIMESTEP_MINUTES = 10

# Extra zones for system-level fault analysis
EXTRA_ZONES = ["Back_Space", "Front_Retail"]

FAULT_TAG = "supplycurve"


# ================================================================
#  DATA LOADING
# ================================================================

def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """Load runtime log, build datetime, tag fault window and gating."""
    df = pd.read_csv(csv_path)

    # Build datetime
    years = df["month"].apply(lambda m: 2017 if m >= 10 else 2018)
    date_str = (
        years.astype(str) + "-"
        + df["month"].astype(int).astype(str).str.zfill(2) + "-"
        + df["day"].astype(int).astype(str).str.zfill(2)
    )
    base_dates = pd.to_datetime(date_str, format="%Y-%m-%d")
    offsets = (pd.to_timedelta(df["hour"].astype(int), unit="h")
               + pd.to_timedelta(df["minute"].astype(int), unit="m"))
    df["datetime"] = base_dates + offsets

    # Tag fault window
    # Supply curve fault: 24h window, so hour check is 0 <= hour < 24 (always true)
    fw = FAULT_WINDOW
    if fw["start_hour"] == 0 and fw["end_hour"] == 24:
        # Full-day fault: only check month/day
        df["in_fault_window"] = (
            (df["month"] >= fw["start_month"]) & (df["month"] <= fw["end_month"]) &
            (df["day"] >= fw["start_day"]) & (df["day"] <= fw["end_day"])
        ).astype(int)
    else:
        df["in_fault_window"] = (
            (df["month"] >= fw["start_month"]) & (df["month"] <= fw["end_month"]) &
            (df["day"] >= fw["start_day"]) & (df["day"] <= fw["end_day"]) &
            (df["hour"] >= fw["start_hour"]) & (df["hour"] < fw["end_hour"])
        ).astype(int)

    # Use in_fault column from runtime if available (more accurate for 24h faults)
    if "in_fault" in df.columns:
        df["in_fault_window"] = df["in_fault"].astype(int)

    # Gating mask
    df["gated"] = (
        (df["occupied"] == 1) &
        (df["heating_active"] == 1) &
        (df["feature_ready"] == 1)
    ).astype(int)

    return df


# ================================================================
#  RE-FLAGGING UTILITIES
# ================================================================

def reflag_at_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Return binary flags (1 = anomaly) for a given threshold."""
    return (scores < threshold).astype(int)


def apply_persistence_filter(flags: np.ndarray, n_persist: int,
                             session_ids: np.ndarray) -> tuple:
    """
    Require n_persist consecutive flags within the same daily session.
    
    Returns
    -------
    filtered_flags : retroactive marking for TP/FP/FN/TN counting
    realtime_flags : causal view for latency calculation
    """
    filtered = np.zeros_like(flags)
    realtime = np.zeros_like(flags)
    run_length = 0
    prev_session = -1

    for i in range(len(flags)):
        if session_ids[i] != prev_session:
            run_length = 0
        prev_session = session_ids[i]

        if flags[i] == 1:
            run_length += 1
        else:
            run_length = 0

        if run_length >= n_persist:
            filtered[max(0, i - n_persist + 1): i + 1] = 1
            realtime[i] = 1

    return filtered, realtime


def compute_session_ids(df: pd.DataFrame) -> np.ndarray:
    """Assign integer session IDs; new session at each gated 0→1 transition."""
    gated = df["gated"].values
    session = np.zeros(len(gated), dtype=int)
    current_id = 0
    prev_gated = 0

    for i in range(len(gated)):
        if gated[i] == 1 and prev_gated == 0:
            current_id += 1
        session[i] = current_id if gated[i] == 1 else 0
        prev_gated = gated[i]

    return session


# ================================================================
#  METRICS COMPUTATION
# ================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    fault_window_abs_start_idx: int = None,
                    gated_indices: np.ndarray = None,
                    realtime_flags: np.ndarray = None) -> dict:
    """Compute precision, recall, F1, FPR, and detection latency."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    latency_steps = None
    latency_minutes = None
    if fault_window_abs_start_idx is not None and gated_indices is not None:
        flags_for_latency = realtime_flags if realtime_flags is not None else y_pred
        fault_realtime_mask = (y_true == 1) & (flags_for_latency == 1)
        if fault_realtime_mask.any():
            first_tp_pos = np.where(fault_realtime_mask)[0][0]
            first_tp_global = gated_indices[first_tp_pos]
            latency_steps = int(first_tp_global - fault_window_abs_start_idx)
            latency_minutes = latency_steps * TIMESTEP_MINUTES

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
        "latency_steps": latency_steps, "latency_minutes": latency_minutes,
    }


# ================================================================
#  SWEEP ENGINE
# ================================================================

def run_sweep(df: pd.DataFrame, thresholds: dict,
              baseline_df: pd.DataFrame = None) -> pd.DataFrame:
    """Sweep threshold × persistence × model grid on faulted data."""
    session_ids = compute_session_ids(df)
    gated_mask = df["gated"].values == 1
    gated_indices = np.where(gated_mask)[0]
    y_true = df.loc[gated_mask, "in_fault_window"].values

    # Find absolute index of fault window start
    fw = FAULT_WINDOW
    fw_start_rows = df[
        (df["month"] == fw["start_month"]) &
        (df["day"] == fw["start_day"]) &
        (df["hour"] == fw["start_hour"])
    ]
    fault_abs_start = int(fw_start_rows.index[0]) if len(fw_start_rows) > 0 else None

    results = []

    for thresh_key in THRESHOLD_KEYS:
        model_raw_flags = {}

        for mname in MODEL_NAMES:
            scores_gated = df.loc[gated_mask, f"score_{mname}"].values
            thresh_val = thresholds[mname].get(thresh_key)
            if thresh_val is None:
                continue

            raw_flags = reflag_at_threshold(scores_gated, thresh_val)
            model_raw_flags[mname] = raw_flags

            for n_persist in PERSISTENCE_VALUES:
                if n_persist == 1:
                    filtered, realtime = raw_flags, raw_flags
                else:
                    filtered, realtime = apply_persistence_filter(
                        raw_flags, n_persist, session_ids[gated_mask]
                    )

                metrics = compute_metrics(
                    y_true, filtered, fault_abs_start, gated_indices,
                    realtime_flags=realtime
                )
                metrics.update({
                    "model": mname, "threshold_key": thresh_key,
                    "threshold_value": thresh_val, "persistence": n_persist
                })
                results.append(metrics)

        # Majority vote
        if len(model_raw_flags) == 3:
            vote_sum = sum(model_raw_flags.values())
            for n_persist in PERSISTENCE_VALUES:
                mv_raw = (vote_sum >= 2).astype(int)
                if n_persist == 1:
                    filtered, realtime = mv_raw, mv_raw
                else:
                    filtered, realtime = apply_persistence_filter(
                        mv_raw, n_persist, session_ids[gated_mask]
                    )

                metrics = compute_metrics(
                    y_true, filtered, fault_abs_start, gated_indices,
                    realtime_flags=realtime
                )
                metrics.update({
                    "model": "majority_vote", "threshold_key": thresh_key,
                    "threshold_value": "N/A", "persistence": n_persist
                })
                results.append(metrics)

    return pd.DataFrame(results)


def run_baseline_fpr_sweep(df_base: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Compute FPR on baseline (fault-free) data across the same sweep grid."""
    session_ids = compute_session_ids(df_base)
    gated_mask = df_base["gated"].values == 1
    n_gated = gated_mask.sum()
    results = []

    for thresh_key in THRESHOLD_KEYS:
        model_raw_flags = {}

        for mname in MODEL_NAMES:
            scores_gated = df_base.loc[gated_mask, f"score_{mname}"].values
            thresh_val = thresholds[mname].get(thresh_key)
            if thresh_val is None:
                continue

            raw_flags = reflag_at_threshold(scores_gated, thresh_val)
            model_raw_flags[mname] = raw_flags

            for n_persist in PERSISTENCE_VALUES:
                if n_persist == 1:
                    filtered = raw_flags
                else:
                    filtered, _ = apply_persistence_filter(
                        raw_flags, n_persist, session_ids[gated_mask]
                    )

                fpr = float(filtered.sum()) / n_gated if n_gated > 0 else 0.0
                results.append({
                    "model": mname, "threshold_key": thresh_key,
                    "persistence": n_persist, "baseline_fpr": fpr,
                    "baseline_flags": int(filtered.sum()),
                    "baseline_gated": n_gated,
                })

        # Majority vote
        if len(model_raw_flags) == 3:
            vote_sum = sum(model_raw_flags.values())
            for n_persist in PERSISTENCE_VALUES:
                mv_raw = (vote_sum >= 2).astype(int)
                if n_persist > 1:
                    filtered, _ = apply_persistence_filter(
                        mv_raw, n_persist, session_ids[gated_mask]
                    )
                else:
                    filtered = mv_raw

                fpr = float(filtered.sum()) / n_gated if n_gated > 0 else 0.0
                results.append({
                    "model": "majority_vote", "threshold_key": thresh_key,
                    "persistence": n_persist, "baseline_fpr": fpr,
                    "baseline_flags": int(filtered.sum()),
                    "baseline_gated": n_gated,
                })

    return pd.DataFrame(results)


# ================================================================
#  PLOTTING
# ================================================================

def plot_threshold_sensitivity(sweep_df: pd.DataFrame, plot_dir: Path):
    """F1, precision, recall vs. threshold at persistence = 1 and 3."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    thresh_order = ["p1", "p2", "p3", "p5", "p7", "p10", "default"]
    thresh_labels = ["1%", "2%", "3%", "5%", "7%", "10%", "5%(def)"]

    models_to_plot = MODEL_NAMES + ["majority_vote"]
    colors = {**MODEL_COLORS, "majority_vote": "#F44336"}
    labels = {**MODEL_LABELS, "majority_vote": "Majority Vote"}

    for ax_idx, n_persist in enumerate([1, 3]):
        for mname in models_to_plot:
            sub = sweep_df[
                (sweep_df["model"] == mname) &
                (sweep_df["persistence"] == n_persist) &
                (sweep_df["threshold_key"].isin(thresh_order))
            ].copy()
            sub["thresh_rank"] = sub["threshold_key"].map(
                {k: i for i, k in enumerate(thresh_order)}
            )
            sub = sub.sort_values("thresh_rank")
            x = range(len(sub))

            axes[ax_idx].plot(
                x, sub["f1"].values, "o-",
                color=colors[mname], label=labels[mname], lw=1.5, ms=5
            )
            axes[ax_idx + 2].plot(
                x, sub["precision"].values, "s--",
                color=colors[mname], label=labels[mname], lw=1.2, ms=4, alpha=0.85
            )
            axes[ax_idx + 2].plot(
                x, sub["recall"].values, "^:",
                color=colors[mname], lw=1.0, ms=4, alpha=0.6
            )

        axes[ax_idx].set_title(
            f"F1 Score  (persistence = {n_persist} step{'s' if n_persist > 1 else ''})",
            fontweight="bold", fontsize=11
        )
        axes[ax_idx].set_xticks(range(len(thresh_order)))
        axes[ax_idx].set_xticklabels(thresh_labels)
        axes[ax_idx].set_xlabel("Threshold percentile")
        axes[ax_idx].set_ylabel("F1")
        axes[ax_idx].set_ylim(0, 1.05)
        axes[ax_idx].legend(fontsize=8)
        axes[ax_idx].grid(True, alpha=0.3)

        axes[ax_idx + 2].set_title(
            f"Precision (solid) & Recall (dotted)  (persist = {n_persist})",
            fontweight="bold", fontsize=11
        )
        axes[ax_idx + 2].set_xticks(range(len(thresh_order)))
        axes[ax_idx + 2].set_xticklabels(thresh_labels)
        axes[ax_idx + 2].set_xlabel("Threshold percentile")
        axes[ax_idx + 2].set_ylabel("Score")
        axes[ax_idx + 2].set_ylim(0, 1.05)
        axes[ax_idx + 2].legend(fontsize=8)
        axes[ax_idx + 2].grid(True, alpha=0.3)

    fig.suptitle(
        "Threshold Sensitivity — Supply Curve Fault (−8K Bias)",
        fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    out = plot_dir / f"threshold_sensitivity_{FAULT_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_persistence_sensitivity(sweep_df: pd.DataFrame, plot_dir: Path):
    """F1, precision, recall vs. persistence at default threshold."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    models_to_plot = MODEL_NAMES + ["majority_vote"]
    colors = {**MODEL_COLORS, "majority_vote": "#F44336"}
    labels = {**MODEL_LABELS, "majority_vote": "Majority Vote"}

    for ax_idx, (metric, title) in enumerate(
        zip(["f1", "precision", "recall"], ["F1 Score", "Precision", "Recall"])
    ):
        ax = axes[ax_idx]
        for mname in models_to_plot:
            sub = sweep_df[
                (sweep_df["model"] == mname) &
                (sweep_df["threshold_key"] == "default")
            ].sort_values("persistence")

            ax.plot(
                sub["persistence"].values, sub[metric].values, "o-",
                color=colors[mname], label=labels[mname], lw=1.5, ms=6
            )

        ax.set_xlabel("Persistence filter (consecutive steps)")
        ax.set_ylabel(title)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(PERSISTENCE_VALUES)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Persistence Filter Sensitivity — Default Threshold — Supply Curve",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    out = plot_dir / f"persistence_sensitivity_{FAULT_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_fpr_vs_recall(sweep_df: pd.DataFrame, baseline_fpr_df: pd.DataFrame,
                       plot_dir: Path):
    """ROC-style: baseline FPR vs. recall across all configurations."""
    fig, ax = plt.subplots(figsize=(9, 7))

    models_to_plot = MODEL_NAMES + ["majority_vote"]
    colors = {**MODEL_COLORS, "majority_vote": "#F44336"}
    labels = {**MODEL_LABELS, "majority_vote": "Majority Vote"}
    markers = {"ocsvm": "o", "iforest": "s", "lof": "^", "majority_vote": "D"}

    for mname in models_to_plot:
        merged = sweep_df[sweep_df["model"] == mname].merge(
            baseline_fpr_df[baseline_fpr_df["model"] == mname][
                ["threshold_key", "persistence", "baseline_fpr"]
            ],
            on=["threshold_key", "persistence"], how="left"
        ).sort_values("baseline_fpr")

        ax.plot(
            merged["baseline_fpr"].values * 100,
            merged["recall"].values,
            marker=markers[mname], color=colors[mname],
            label=labels[mname], lw=1.2, ms=4, alpha=0.75
        )

    ax.set_xlabel("Baseline False Positive Rate [%]", fontsize=11)
    ax.set_ylabel("Recall (supply curve fault window)", fontsize=11)
    ax.set_title("Detection Trade-off: Baseline FPR vs. Fault Recall",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=-0.2)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    out = plot_dir / f"fpr_vs_recall_{FAULT_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_best_config_detail(df_fault: pd.DataFrame, df_base: pd.DataFrame,
                            thresholds: dict, best_cfg: dict, plot_dir: Path):
    """
    Fault window detail plot with multi-zone overlay for supply curve fault.
    7 panels: multi-zone temps, supply temp, multi-zone flows, temp error,
    and 3 model score panels with best-config flags.
    """
    session_ids = compute_session_ids(df_fault)
    gated_mask = df_fault["gated"].values == 1
    best_thresh_key = best_cfg["threshold_key"]
    best_persist = int(best_cfg["persistence"])

    # Recompute flags for best config
    for mname in MODEL_NAMES:
        scores = df_fault[f"score_{mname}"].values.copy()
        thresh_val = thresholds[mname][best_thresh_key]
        raw_flags = np.zeros(len(df_fault), dtype=int)
        valid_mask = ~np.isnan(scores)
        raw_flags[valid_mask] = (scores[valid_mask] < thresh_val).astype(int)

        gated_flags = raw_flags[gated_mask]
        if best_persist > 1:
            gated_flags, _ = apply_persistence_filter(
                gated_flags, best_persist, session_ids[gated_mask]
            )
        full_flags = np.zeros(len(df_fault), dtype=int)
        full_flags[gated_mask] = gated_flags
        df_fault[f"flag_best_{mname}"] = full_flags

    df_fault["flag_best_mv"] = (
        (df_fault["flag_best_ocsvm"]
         + df_fault["flag_best_iforest"]
         + df_fault["flag_best_lof"]) >= 2
    ).astype(int)

    # Zoom window
    zoom_start = pd.Timestamp("2018-01-14")
    zoom_end = pd.Timestamp("2018-01-18")
    df_z = df_fault[
        (df_fault["datetime"] >= zoom_start) & (df_fault["datetime"] <= zoom_end)
    ].copy()
    df_b = df_base[
        (df_base["datetime"] >= zoom_start) & (df_base["datetime"] <= zoom_end)
    ].copy()

    # Fault spans — 24h window for supply curve
    fault_spans = [
        (pd.Timestamp("2018-01-15 00:00"), pd.Timestamp("2018-01-17 00:00")),
    ]

    fig, axes = plt.subplots(7, 1, figsize=(16, 26), sharex=True)

    for ax in axes:
        for fs, fe in fault_spans:
            ax.axvspan(fs, fe, alpha=0.12, color="red")
        ax.grid(True, alpha=0.3)

    # --- Panel 0: Multi-zone temperatures ---
    ax = axes[0]
    ax.plot(df_z["datetime"], df_z["zone_temp"], lw=1.2, color="#E53935",
            label="Core_Retail (faulted)")
    if len(df_b) > 0:
        ax.plot(df_b["datetime"], df_b["zone_temp"], lw=0.8, ls="--",
                color="#E53935", alpha=0.5, label="Core_Retail (baseline)")
    ax.plot(df_z["datetime"], df_z["intended_sp"], lw=0.8, ls=":",
            color="black", label="Setpoint")

    zone_colors = {"Back_Space": "#1E88E5", "Front_Retail": "#43A047"}
    for zone_label in EXTRA_ZONES:
        col = f"{zone_label}_zone_temp"
        if col in df_z.columns:
            ax.plot(df_z["datetime"], df_z[col], lw=1.0,
                    color=zone_colors.get(zone_label, "gray"), label=zone_label)
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("Multi-Zone Air Temperatures", fontweight="bold")
    ax.legend(fontsize=8, ncol=3)

    # --- Panel 1: Supply temperature ---
    ax = axes[1]
    if "t_supply" in df_z.columns:
        ax.plot(df_z["datetime"], df_z["t_supply"], lw=1.2, color="#FF6F00",
                label="HW Supply Temp (faulted)")
    if "t_supply" in df_b.columns and len(df_b) > 0:
        ax.plot(df_b["datetime"], df_b["t_supply"], lw=0.8, ls="--",
                color="#FF6F00", alpha=0.5, label="HW Supply Temp (baseline)")
    ax.axhline(60.0, color="gray", ls=":", lw=0.8, label="Baseline (60°C)")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("HW Supply Temperature", fontweight="bold")
    ax.legend(fontsize=8)

    # --- Panel 2: Multi-zone flow rates ---
    ax = axes[2]
    ax.plot(df_z["datetime"], df_z["m_dot"], lw=1.0, color="#E53935",
            label="Core_Retail (faulted)")
    if len(df_b) > 0:
        ax.plot(df_b["datetime"], df_b["m_dot"], lw=0.8, ls="--",
                color="#E53935", alpha=0.5, label="Core_Retail (baseline)")
    for zone_label in EXTRA_ZONES:
        col = f"{zone_label}_m_dot"
        if col in df_z.columns:
            ax.plot(df_z["datetime"], df_z[col], lw=0.8,
                    color=zone_colors.get(zone_label, "gray"), label=zone_label)
    ax.set_ylabel("Mass flow [kg/s]")
    ax.set_title("HW Mass Flow Rates (Multi-Zone)", fontweight="bold")
    ax.legend(fontsize=8)

    # --- Panel 3: Temperature error ---
    ax = axes[3]
    ax.plot(df_z["datetime"], df_z["temp_error"], lw=0.8, color="#5E35B1",
            label="temp_error")
    if "temp_error_2h_avg" in df_z.columns:
        ax.plot(df_z["datetime"], df_z["temp_error_2h_avg"], lw=1.0,
                color="#00897B", label="2h rolling avg")
    ax.axhline(0, color="black", ls="-", lw=0.5)
    ax.set_ylabel("Error [°C]")
    ax.set_title("Temperature Error (Core_Retail)", fontweight="bold")
    ax.legend(fontsize=8)

    # --- Panels 4–6: Anomaly scores with best flags ---
    for i, mname in enumerate(MODEL_NAMES):
        ax = axes[4 + i]
        score_col = f"score_{mname}"
        flag_col = f"flag_best_{mname}"

        valid = df_z[df_z[score_col].notna()]
        ax.plot(valid["datetime"], valid[score_col],
                color=MODEL_COLORS[mname], lw=0.8, alpha=0.7)

        thresh_val = thresholds[mname][best_thresh_key]
        ax.axhline(thresh_val, color="gray", ls="--", lw=1.0, alpha=0.6,
                   label=f"threshold ({best_thresh_key})")

        flagged = df_z[df_z[flag_col] == 1]
        n_in_window = flagged["in_fault_window"].sum() if "in_fault_window" in flagged.columns else 0
        n_outside = len(flagged) - n_in_window
        ax.scatter(flagged["datetime"], flagged[score_col],
                   color="red", s=14, zorder=5, alpha=0.8,
                   label=f"Flagged (TP={n_in_window}, FP={n_outside})")

        ax.set_ylabel(f"{MODEL_LABELS[mname]}\nscore")
        ax.set_title(f"{MODEL_LABELS[mname]} — {best_thresh_key} threshold, "
                     f"persist={best_persist}", fontweight="bold")
        ax.legend(fontsize=8, loc="lower left")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    axes[-1].set_xlabel("Date / Time")

    fig.suptitle(
        f"Best Configuration Detail — Supply Curve Fault "
        f"(thresh={best_thresh_key}, persist={best_persist})",
        fontsize=14, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    out = plot_dir / f"fault_window_best_config_{FAULT_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ================================================================
#  BEST CONFIGURATION SELECTION
# ================================================================

def select_best_config(sweep_df: pd.DataFrame, min_recall: float = 0.70) -> dict:
    """Select best config: maximize F1 subject to recall >= min_recall."""
    candidates = sweep_df[sweep_df["recall"] >= min_recall].copy()

    if len(candidates) == 0:
        print(f"  WARNING: No config achieves recall >= {min_recall:.0%}. "
              f"Selecting highest recall instead.")
        candidates = sweep_df.copy()
        candidates = candidates.sort_values("recall", ascending=False)
        best = candidates.iloc[0]
    else:
        best = candidates.sort_values("f1", ascending=False).iloc[0]

    return best.to_dict()


# ================================================================
#  CONSOLE REPORT
# ================================================================

def print_report(sweep_df: pd.DataFrame, baseline_fpr_df: pd.DataFrame,
                 best_cfg: dict):
    """Print structured summary of sweep results."""
    print("\n" + "=" * 75)
    print("  SWEEP RESULTS SUMMARY — SUPPLY CURVE FAULT")
    print("=" * 75)

    # Original (default, persist=1)
    orig = sweep_df[
        (sweep_df["threshold_key"] == "default") &
        (sweep_df["persistence"] == 1)
    ]

    print(f"\n  {'Config':<32s} {'Model':<16s} {'Prec':>6s} {'Rec':>6s} "
          f"{'F1':>6s} {'FPR':>6s} {'Latency':>8s}")
    print("  " + "-" * 88)

    for _, row in orig.iterrows():
        lat = f"{row['latency_minutes']}m" if row['latency_minutes'] is not None else "N/A"
        print(f"  {'Original (default, p=1)':<32s} {row['model']:<16s} "
              f"{row['precision']:>6.3f} {row['recall']:>6.3f} "
              f"{row['f1']:>6.3f} {row['fpr']:>6.3f} {lat:>8s}")

    print()
    best_model = best_cfg["model"]
    best_tk = best_cfg["threshold_key"]
    best_p = int(best_cfg["persistence"])
    lat = f"{best_cfg['latency_minutes']}m" if best_cfg['latency_minutes'] is not None else "N/A"
    print(f"  >>> BEST CONFIG: model={best_model}, threshold={best_tk}, "
          f"persistence={best_p}")
    print(f"      Precision={best_cfg['precision']:.3f}  "
          f"Recall={best_cfg['recall']:.3f}  "
          f"F1={best_cfg['f1']:.3f}  "
          f"FPR={best_cfg['fpr']:.3f}  "
          f"Latency={lat}")

    print(f"\n  Conference paper rule-based proxy (supply curve):")
    print(f"      (Not available — supply curve fault is new to the journal extension)")

    # Baseline FPR for best config
    bf_row = baseline_fpr_df[
        (baseline_fpr_df["model"] == best_model) &
        (baseline_fpr_df["threshold_key"] == best_tk) &
        (baseline_fpr_df["persistence"] == best_p)
    ]
    if len(bf_row) > 0:
        print(f"  Baseline FPR for best config: {bf_row.iloc[0]['baseline_fpr']:.4%}")


# ================================================================
#  MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 4b: Post-hoc threshold & persistence sweep (Supply Curve)"
    )
    parser.add_argument("--faulted-log", required=True,
                        help="Path to faulted fdc_runtime_log.csv")
    parser.add_argument("--baseline-log", required=True,
                        help="Path to baseline fdc_runtime_log.csv")
    parser.add_argument("--models-dir", default="models",
                        help="Directory containing trained models and thresholds")
    parser.add_argument("--plot-dir",
                        default="plots/fault_analysis_supplycurve",
                        help="Output directory for plots and CSVs")
    parser.add_argument("--min-recall", type=float, default=0.70,
                        help="Minimum recall for best-config selection")
    args = parser.parse_args()

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)

    print("=" * 75)
    print("  PHASE 4b: POST-HOC THRESHOLD & PERSISTENCE SWEEP")
    print("  Supply Curve Fault (System-Level Temperature Bias)")
    print("=" * 75)

    # Load thresholds
    thresholds = joblib.load(models_dir / "thresholds.joblib")
    with open(models_dir / "training_config.json", encoding="utf-8") as f:
        training_cfg = json.load(f)

    # Load data
    print("\nLoading faulted log...")
    df_fault = load_and_prepare(args.faulted_log)
    print(f"  {len(df_fault):,} rows, "
          f"{df_fault['in_fault_window'].sum()} in fault window")

    print("Loading baseline log...")
    df_base = load_and_prepare(args.baseline_log)
    print(f"  {len(df_base):,} rows")

    # Run sweeps
    print("\nRunning threshold × persistence sweep on faulted data...")
    sweep_df = run_sweep(df_fault, thresholds, df_base)
    print(f"  {len(sweep_df)} configurations evaluated")

    print("Running baseline FPR sweep...")
    baseline_fpr_df = run_baseline_fpr_sweep(df_base, thresholds)

    # Select best
    best_cfg = select_best_config(sweep_df, min_recall=args.min_recall)

    # Report
    print_report(sweep_df, baseline_fpr_df, best_cfg)

    # Plots
    print("\n" + "=" * 75)
    print("  GENERATING PLOTS")
    print("=" * 75)
    plot_threshold_sensitivity(sweep_df, plot_dir)
    plot_persistence_sensitivity(sweep_df, plot_dir)
    plot_fpr_vs_recall(sweep_df, baseline_fpr_df, plot_dir)
    plot_best_config_detail(df_fault, df_base, thresholds, best_cfg, plot_dir)

    # Save results
    sweep_path = plot_dir / f"sweep_results_{FAULT_TAG}.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"\n  Full sweep results: {sweep_path}")

    baseline_fpr_path = plot_dir / f"baseline_fpr_sweep_{FAULT_TAG}.csv"
    baseline_fpr_df.to_csv(baseline_fpr_path, index=False)
    print(f"  Baseline FPR sweep: {baseline_fpr_path}")

    best_path = plot_dir / f"best_config_{FAULT_TAG}.json"
    best_serializable = {
        k: (int(v) if isinstance(v, (np.integer,)) else
            float(v) if isinstance(v, (np.floating,)) else v)
        for k, v in best_cfg.items()
    }
    with open(best_path, "w") as f:
        json.dump(best_serializable, f, indent=2)
    print(f"  Best config: {best_path}")

    print("\n" + "=" * 75)
    print("  SWEEP COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()

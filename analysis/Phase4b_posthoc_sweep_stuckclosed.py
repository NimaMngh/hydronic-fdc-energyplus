# -*- coding: utf-8 -*-
"""
Phase 4b: Post-Hoc Threshold Sweep & Persistence Filter Analysis
=================================================================
Operates entirely on the already-logged fdc_runtime_log.csv files.
No re-simulation required.

Produces:
    1. Threshold sweep: metrics at each stored percentile (p1–p10)
    2. Persistence filter: requires N consecutive flags before declaring fault
    3. Combined sweep: threshold × persistence grid
    4. Sensitivity curves for the journal paper
    5. Updated fault window detail plot with best configuration

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
from itertools import product

import joblib


# ================================================================
#  CONFIGURATION
# ================================================================

MODEL_NAMES  = ["ocsvm", "iforest", "lof"]
MODEL_LABELS = {"ocsvm": "OCSVM", "iforest": "Isolation Forest", "lof": "LOF"}
MODEL_COLORS = {"ocsvm": "#4CAF50", "iforest": "#FF9800", "lof": "#9C27B0"}

PERSISTENCE_VALUES = [1, 2, 3, 4, 5, 6]  # consecutive flagged steps required
THRESHOLD_KEYS     = ["p1", "p2", "p3", "p5", "p7", "p10", "default"]

# Fault window (must match the IDF / run config)
FAULT_WINDOW = {
    "start_month": 1, "end_month": 1,
    "start_day": 15,  "end_day": 16,
    "start_hour": 8,  "end_hour": 17
}

TIMESTEP_MINUTES = 10


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
    fw = FAULT_WINDOW
    df["in_fault_window"] = (
        (df["month"] >= fw["start_month"]) & (df["month"] <= fw["end_month"]) &
        (df["day"] >= fw["start_day"]) & (df["day"] <= fw["end_day"]) &
        (df["hour"] >= fw["start_hour"]) & (df["hour"] < fw["end_hour"])
    ).astype(int)

    # Gating mask: occupied, heating active, features ready
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
    Require n_persist consecutive flags within the same daily session
    before declaring a fault.  Resets the counter at session boundaries
    to prevent bleed across overnight gaps.

    Parameters
    ----------
    flags : 1-D array of 0/1
    n_persist : minimum consecutive flags required
    session_ids : integer array identifying each contiguous active session
                  (changes value at each daily startup)

    Returns
    -------
    filtered_flags : 1-D array of 0/1
        Retroactively marks all n_persist steps in the run as flagged.
        Use this for TP/FP/FN/TN counting so precision/recall are unaffected.
    realtime_flags : 1-D array of 0/1
        Causal view: a flag appears only at the timestep where the persistence
        criterion is *first met*, then stays on while the run continues.
        Use this for latency calculation — it reflects when an operator would
        actually know a fault is present in real time.
    """
    filtered = np.zeros_like(flags)
    realtime = np.zeros_like(flags)
    run_length = 0
    prev_session = -1

    for i in range(len(flags)):
        # Reset counter at session boundary
        if session_ids[i] != prev_session:
            run_length = 0
        prev_session = session_ids[i]

        if flags[i] == 1:
            run_length += 1
        else:
            run_length = 0

        if run_length >= n_persist:
            # Retroactive marking for TP/FP counting
            filtered[max(0, i - n_persist + 1): i + 1] = 1
            # Causal marking: flag only from the decision point onward
            realtime[i] = 1

    return filtered, realtime


def compute_session_ids(df: pd.DataFrame) -> np.ndarray:
    """
    Assign an integer session ID to each row.  A new session starts
    whenever the 'gated' column transitions from 0 to 1 (i.e., daily
    system startup after overnight setback).
    """
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
    """
    Compute precision, recall, F1, FPR, and detection latency.

    Parameters
    ----------
    y_true : ground-truth fault labels (1 = fault, 0 = normal)
    y_pred : predicted flags — used for TP/FP/FN/TN counting.
             For persistence-filtered configs this should be the *retroactive*
             filtered array so that precision/recall are unaffected.
    fault_window_abs_start_idx : absolute row index of fault-window onset (08:00)
    gated_indices : absolute row indices of gated timesteps in the full DataFrame
    realtime_flags : causal flag array returned by apply_persistence_filter.
        If provided, latency is measured from the first realtime flag inside the
        fault window — i.e. the earliest moment an operator would know a fault
        is present.  If None, latency falls back to using y_pred (legacy
        behaviour, which under-estimates latency for persist > 1).
    """
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Detection latency from absolute fault window start.
    # Use realtime_flags for a causally-correct measurement: the flag array
    # used here only lights up at the timestep where the persistence criterion
    # is first met, matching what an operator would observe in real time.
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
        "precision": precision, "recall": recall, "f1": f1,
        "fpr": fpr,
        "latency_steps": latency_steps,
        "latency_minutes": latency_minutes,
    }


# ================================================================
#  SWEEP ENGINE
# ================================================================

def run_sweep(df: pd.DataFrame, thresholds: dict,
              baseline_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Sweep across all combinations of:
      - model (ocsvm, iforest, lof, majority_vote)
      - threshold percentile (p1 ... p10, default)
      - persistence filter (1 ... 6 steps)

    Returns a DataFrame with one row per configuration.
    """
    session_ids = compute_session_ids(df)
    gated_mask = df["gated"].values == 1
    gated_indices = np.where(gated_mask)[0]

    y_true = df.loc[gated_mask, "in_fault_window"].values

    # Find the absolute index of the start of the fault window (e.g., 08:00 on Jan 15)
    # Note: Removed the (df["minute"] == 0) check to account for EnergyPlus logging delays
    fw_start_rows = df[
        (df["month"] == FAULT_WINDOW["start_month"]) &
        (df["day"] == FAULT_WINDOW["start_day"]) &
        (df["hour"] == FAULT_WINDOW["start_hour"])
    ]
    fault_abs_start = int(fw_start_rows.index[0]) if len(fw_start_rows) > 0 else None

    results = []

    for thresh_key in THRESHOLD_KEYS:
        # --- Individual models ---
        model_raw_flags = {}

        for mname in MODEL_NAMES:
            score_col = f"score_{mname}"
            scores_gated = df.loc[gated_mask, score_col].values

            thresh_val = thresholds[mname].get(thresh_key)
            if thresh_val is None:
                continue

            raw_flags = reflag_at_threshold(scores_gated, thresh_val)
            model_raw_flags[mname] = raw_flags

            for n_persist in PERSISTENCE_VALUES:
                if n_persist == 1:
                    filtered = raw_flags
                    realtime = raw_flags   # no persistence delay at n=1
                else:
                    filtered, realtime = apply_persistence_filter(
                        raw_flags, n_persist, session_ids[gated_mask]
                    )

                metrics = compute_metrics(
                    y_true, filtered, fault_abs_start, gated_indices,
                    realtime_flags=realtime
                )
                metrics["model"] = mname
                metrics["threshold_key"] = thresh_key
                metrics["threshold_value"] = thresh_val
                metrics["persistence"] = n_persist
                results.append(metrics)

        # --- Majority vote ---
        if len(model_raw_flags) == 3:
            vote_sum = sum(model_raw_flags.values())

            for n_persist in PERSISTENCE_VALUES:
                mv_raw = (vote_sum >= 2).astype(int)

                if n_persist == 1:
                    filtered = mv_raw
                    realtime = mv_raw
                else:
                    filtered, realtime = apply_persistence_filter(
                        mv_raw, n_persist, session_ids[gated_mask]
                    )

                metrics = compute_metrics(
                    y_true, filtered, fault_abs_start, gated_indices,
                    realtime_flags=realtime
                )
                metrics["model"] = "majority_vote"
                metrics["threshold_key"] = thresh_key
                metrics["threshold_value"] = "N/A"
                metrics["persistence"] = n_persist
                results.append(metrics)

    results_df = pd.DataFrame(results)
    return results_df


# ================================================================
#  BASELINE FPR SWEEP (for comparison)
# ================================================================

def run_baseline_fpr_sweep(df_base: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Compute FPR on baseline (fault-free) data across the same sweep grid."""
    session_ids = compute_session_ids(df_base)
    gated_mask = df_base["gated"].values == 1

    # For baseline, all gated rows are "normal" (in_fault_window = 0 everywhere)
    n_gated = gated_mask.sum()
    results = []

    for thresh_key in THRESHOLD_KEYS:
        model_raw_flags = {}

        for mname in MODEL_NAMES:
            score_col = f"score_{mname}"
            scores_gated = df_base.loc[gated_mask, score_col].values

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
                    "model": mname,
                    "threshold_key": thresh_key,
                    "persistence": n_persist,
                    "baseline_fpr": fpr,
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
                    "model": "majority_vote",
                    "threshold_key": thresh_key,
                    "persistence": n_persist,
                    "baseline_fpr": fpr,
                    "baseline_flags": int(filtered.sum()),
                    "baseline_gated": n_gated,
                })

    return pd.DataFrame(results)


# ================================================================
#  PLOTTING
# ================================================================

def plot_threshold_sensitivity(sweep_df: pd.DataFrame, plot_dir: Path):
    """
    For each model, plot F1, precision, recall vs. threshold percentile
    at persistence = 1 and persistence = 3.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    # Order thresholds from tight to loose
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

            # F1 plot (top row)
            axes[ax_idx].plot(
                x, sub["f1"].values, "o-",
                color=colors[mname], label=labels[mname], lw=1.5, ms=5
            )
            # Precision plot (bottom row)
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
        "Threshold Sensitivity — Stuck-Closed Fault Detection",
        fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    out = plot_dir / "threshold_sensitivity_stuckclosed.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_persistence_sensitivity(sweep_df: pd.DataFrame, plot_dir: Path):
    """
    For each model at the 'default' threshold, plot F1, precision, recall
    vs. persistence filter length.
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    models_to_plot = MODEL_NAMES + ["majority_vote"]
    colors = {**MODEL_COLORS, "majority_vote": "#F44336"}
    labels = {**MODEL_LABELS, "majority_vote": "Majority Vote"}

    metric_names = ["f1", "precision", "recall"]
    metric_titles = ["F1 Score", "Precision", "Recall"]

    for ax_idx, (metric, title) in enumerate(zip(metric_names, metric_titles)):
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
        "Persistence Filter Sensitivity — Default Threshold — Stuck-Closed",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    out = plot_dir / "persistence_sensitivity_stuckclosed.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_fpr_vs_recall(sweep_df: pd.DataFrame, baseline_fpr_df: pd.DataFrame,
                       plot_dir: Path):
    """
    ROC-style plot: baseline FPR (x-axis) vs. recall on fault window (y-axis)
    across threshold × persistence configurations.  Each model gets a curve.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    models_to_plot = MODEL_NAMES + ["majority_vote"]
    colors = {**MODEL_COLORS, "majority_vote": "#F44336"}
    labels = {**MODEL_LABELS, "majority_vote": "Majority Vote"}
    markers = {"ocsvm": "o", "iforest": "s", "lof": "^", "majority_vote": "D"}

    for mname in models_to_plot:
        # Merge sweep metrics with baseline FPR
        sw = sweep_df[sweep_df["model"] == mname].copy()
        bf = baseline_fpr_df[baseline_fpr_df["model"] == mname].copy()

        merged = sw.merge(
            bf[["model", "threshold_key", "persistence", "baseline_fpr"]],
            on=["model", "threshold_key", "persistence"],
            how="left"
        )

        # Sort by baseline_fpr for a clean curve
        merged = merged.sort_values("baseline_fpr")

        ax.plot(
            merged["baseline_fpr"].values * 100,
            merged["recall"].values,
            marker=markers[mname], color=colors[mname],
            label=labels[mname], lw=1.2, ms=4, alpha=0.75
        )

    ax.set_xlabel("Baseline False Positive Rate [%]", fontsize=11)
    ax.set_ylabel("Recall (stuck-closed fault window)", fontsize=11)
    ax.set_title("Detection Trade-off: Baseline FPR vs. Fault Recall",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=-0.2)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    out = plot_dir / "fpr_vs_recall_stuckclosed.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_best_config_detail(df_fault: pd.DataFrame, df_base: pd.DataFrame,
                            thresholds: dict, best_cfg: dict, plot_dir: Path):
    """
    Reproduce the fault window detail plot using the best configuration
    found by the sweep.
    """
    session_ids = compute_session_ids(df_fault)
    gated_mask = df_fault["gated"].values == 1

    # Recompute flags for the best configuration
    best_thresh_key = best_cfg["threshold_key"]
    best_persist = best_cfg["persistence"]

    for mname in MODEL_NAMES:
        scores = df_fault[f"score_{mname}"].values.copy()
        thresh_val = thresholds[mname][best_thresh_key]

        # Flag full array (with NaN scores getting flag=0)
        raw_flags = np.zeros(len(df_fault), dtype=int)
        valid_mask = ~np.isnan(scores)
        raw_flags[valid_mask] = (scores[valid_mask] < thresh_val).astype(int)

        # Apply persistence only on gated rows, then map back
        gated_flags = raw_flags[gated_mask]
        if best_persist > 1:
            gated_flags, _ = apply_persistence_filter(
                gated_flags, best_persist, session_ids[gated_mask]
            )
        # Write back
        full_flags = np.zeros(len(df_fault), dtype=int)
        full_flags[gated_mask] = gated_flags
        df_fault[f"flag_best_{mname}"] = full_flags

    # Majority vote on best flags
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

    fault_spans = [
        (pd.Timestamp("2018-01-15 08:00"), pd.Timestamp("2018-01-15 17:00")),
        (pd.Timestamp("2018-01-16 08:00"), pd.Timestamp("2018-01-16 17:00")),
    ]

    fig, axes = plt.subplots(5, 1, figsize=(16, 18), sharex=True)

    for ax in axes:
        for fs, fe in fault_spans:
            ax.axvspan(fs, fe, alpha=0.12, color="red")
        ax.grid(True, alpha=0.3)

    # Panel 1: Zone temperature
    axes[0].plot(df_z["datetime"], df_z["zone_temp"], "r-", lw=1.2, label="Faulted")
    axes[0].plot(df_b["datetime"], df_b["zone_temp"], "b--", lw=1.0, alpha=0.7, label="Baseline")
    axes[0].plot(df_z["datetime"], df_z["htg_sp"], "k:", lw=0.8, label="Setpoint")
    axes[0].set_ylabel("Temperature [°C]")
    axes[0].set_title("Zone Air Temperature", fontweight="bold")
    axes[0].legend(fontsize=9)

    # Panel 2: Flow rate
    axes[1].plot(df_z["datetime"], df_z["m_dot"], "r-", lw=1.2, label="Faulted")
    axes[1].plot(df_b["datetime"], df_b["m_dot"], "b--", lw=1.0, alpha=0.7, label="Baseline")
    axes[1].set_ylabel("Flow rate [kg/s]")
    axes[1].set_title("HW Mass Flow Rate", fontweight="bold")
    axes[1].legend(fontsize=9)

    # Panels 3–5: Anomaly scores with BEST flags
    for i, mname in enumerate(MODEL_NAMES):
        ax = axes[2 + i]
        score_col = f"score_{mname}"
        flag_col = f"flag_best_{mname}"

        valid = df_z[df_z[score_col].notna()]
        ax.plot(valid["datetime"], valid[score_col],
                color=MODEL_COLORS[mname], lw=0.8, alpha=0.7)

        # Threshold line
        thresh_val = thresholds[mname][best_thresh_key]
        ax.axhline(thresh_val, color="gray", ls="--", lw=1.0, alpha=0.6,
                   label=f"threshold ({best_thresh_key})")

        # Filtered flags
        flagged = df_z[df_z[flag_col] == 1]
        n_in_window = flagged["in_fault_window"].sum()
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
        f"Best Configuration Detail — Stuck-Closed "
        f"(thresh={best_thresh_key}, persist={best_persist})",
        fontsize=14, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    out = plot_dir / "fault_window_best_config_stuckclosed.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ================================================================
#  BEST CONFIGURATION SELECTION
# ================================================================

def select_best_config(sweep_df: pd.DataFrame, min_recall: float = 0.70) -> dict:
    """
    Select the best configuration by maximizing F1 subject to
    recall >= min_recall.  If no config meets the recall floor,
    relax to the highest-recall config.
    """
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
    """Print a structured summary of the sweep results."""

    print("\n" + "=" * 75)
    print("  SWEEP RESULTS SUMMARY — STUCK-CLOSED FAULT")
    print("=" * 75)

    # Show original (default, persist=1) vs best
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

    # Conference paper comparison
    print(f"\n  Conference paper rule-based proxy (stuck-closed):")
    print(f"      Precision=0.540  Recall=0.850  F1=0.660")

    improvement = best_cfg["f1"] - 0.66
    print(f"\n  F1 delta vs. conference proxy: {improvement:+.3f}")

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
        description="Phase 4b: Post-hoc threshold & persistence sweep"
    )
    parser.add_argument("--faulted-log", required=True,
                        help="Path to faulted fdc_runtime_log.csv")
    parser.add_argument("--baseline-log", required=True,
                        help="Path to baseline fdc_runtime_log.csv")
    parser.add_argument("--models-dir", default="models",
                        help="Directory containing trained models and thresholds")
    parser.add_argument("--plot-dir", default="plots/fault_analysis_stuckclosed",
                        help="Output directory for plots and CSVs")
    parser.add_argument("--min-recall", type=float, default=0.70,
                        help="Minimum recall for best-config selection")
    args = parser.parse_args()

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)

    print("=" * 75)
    print("  PHASE 4b: POST-HOC THRESHOLD & PERSISTENCE SWEEP")
    print("  Stuck-Closed Valve Fault")
    print("=" * 75)

    # Load thresholds
    thresholds = joblib.load(models_dir / "thresholds.joblib")
    with open(models_dir / "training_config.json") as f:
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

    # Save full sweep results
    sweep_path = plot_dir / "sweep_results_stuckclosed.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"\n  Full sweep results: {sweep_path}")

    baseline_fpr_path = plot_dir / "baseline_fpr_sweep_stuckclosed.csv"
    baseline_fpr_df.to_csv(baseline_fpr_path, index=False)
    print(f"  Baseline FPR sweep: {baseline_fpr_path}")

    best_path = plot_dir / "best_config_stuckclosed.json"
    # Convert numpy types for JSON serialization
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
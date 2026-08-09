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

import joblib

# ================================================================
#  CONFIGURATION
# ================================================================

MODEL_NAMES  = ["ocsvm", "iforest", "lof"]
MODEL_LABELS = {"ocsvm": "OCSVM", "iforest": "Isolation Forest", "lof": "LOF"}
MODEL_COLORS = {"ocsvm": "#4CAF50", "iforest": "#FF9800", "lof": "#9C27B0"}

PERSISTENCE_VALUES = [1, 2, 3, 4, 5, 6]
THRESHOLD_KEYS     = ["p1", "p2", "p3", "p5", "p7", "p10", "default"]

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
    df = pd.read_csv(csv_path)
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

    fw = FAULT_WINDOW
    df["in_fault_window"] = (
        (df["month"] >= fw["start_month"]) & (df["month"] <= fw["end_month"]) &
        (df["day"] >= fw["start_day"]) & (df["day"] <= fw["end_day"]) &
        (df["hour"] >= fw["start_hour"]) & (df["hour"] < fw["end_hour"])
    ).astype(int)

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
    return (scores < threshold).astype(int)

def apply_persistence_filter(flags: np.ndarray, n_persist: int, session_ids: np.ndarray) -> tuple:
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

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, fault_window_abs_start_idx: int = None,
                    gated_indices: np.ndarray = None, realtime_flags: np.ndarray = None) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    latency_steps, latency_minutes = None, None
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

def run_sweep(df: pd.DataFrame, thresholds: dict, baseline_df: pd.DataFrame = None) -> pd.DataFrame:
    session_ids = compute_session_ids(df)
    gated_mask = df["gated"].values == 1
    gated_indices = np.where(gated_mask)[0]
    y_true = df.loc[gated_mask, "in_fault_window"].values

    fw_start_rows = df[
        (df["month"] == FAULT_WINDOW["start_month"]) &
        (df["day"] == FAULT_WINDOW["start_day"]) &
        (df["hour"] == FAULT_WINDOW["start_hour"])
    ]
    fault_abs_start = int(fw_start_rows.index[0]) if len(fw_start_rows) > 0 else None

    results = []
    for thresh_key in THRESHOLD_KEYS:
        model_raw_flags = {}
        for mname in MODEL_NAMES:
            scores_gated = df.loc[gated_mask, f"score_{mname}"].values
            thresh_val = thresholds[mname].get(thresh_key)
            if thresh_val is None: continue

            raw_flags = reflag_at_threshold(scores_gated, thresh_val)
            model_raw_flags[mname] = raw_flags

            for n_persist in PERSISTENCE_VALUES:
                filtered, realtime = raw_flags, raw_flags
                if n_persist > 1:
                    filtered, realtime = apply_persistence_filter(raw_flags, n_persist, session_ids[gated_mask])
                
                metrics = compute_metrics(y_true, filtered, fault_abs_start, gated_indices, realtime_flags=realtime)
                metrics.update({"model": mname, "threshold_key": thresh_key, "threshold_value": thresh_val, "persistence": n_persist})
                results.append(metrics)

        if len(model_raw_flags) == 3:
            vote_sum = sum(model_raw_flags.values())
            for n_persist in PERSISTENCE_VALUES:
                mv_raw = (vote_sum >= 2).astype(int)
                filtered, realtime = mv_raw, mv_raw
                if n_persist > 1:
                    filtered, realtime = apply_persistence_filter(mv_raw, n_persist, session_ids[gated_mask])
                
                metrics = compute_metrics(y_true, filtered, fault_abs_start, gated_indices, realtime_flags=realtime)
                metrics.update({"model": "majority_vote", "threshold_key": thresh_key, "threshold_value": "N/A", "persistence": n_persist})
                results.append(metrics)
    return pd.DataFrame(results)

def run_baseline_fpr_sweep(df_base: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    session_ids = compute_session_ids(df_base)
    gated_mask = df_base["gated"].values == 1
    n_gated = gated_mask.sum()
    results = []

    for thresh_key in THRESHOLD_KEYS:
        model_raw_flags = {}
        for mname in MODEL_NAMES:
            scores_gated = df_base.loc[gated_mask, f"score_{mname}"].values
            thresh_val = thresholds[mname].get(thresh_key)
            if thresh_val is None: continue

            raw_flags = reflag_at_threshold(scores_gated, thresh_val)
            model_raw_flags[mname] = raw_flags

            for n_persist in PERSISTENCE_VALUES:
                filtered = raw_flags if n_persist == 1 else apply_persistence_filter(raw_flags, n_persist, session_ids[gated_mask])[0]
                fpr = float(filtered.sum()) / n_gated if n_gated > 0 else 0.0
                results.append({"model": mname, "threshold_key": thresh_key, "persistence": n_persist, "baseline_fpr": fpr, "baseline_flags": int(filtered.sum()), "baseline_gated": n_gated})

        if len(model_raw_flags) == 3:
            vote_sum = sum(model_raw_flags.values())
            for n_persist in PERSISTENCE_VALUES:
                mv_raw = (vote_sum >= 2).astype(int)
                filtered = mv_raw if n_persist == 1 else apply_persistence_filter(mv_raw, n_persist, session_ids[gated_mask])[0]
                fpr = float(filtered.sum()) / n_gated if n_gated > 0 else 0.0
                results.append({"model": "majority_vote", "threshold_key": thresh_key, "persistence": n_persist, "baseline_fpr": fpr, "baseline_flags": int(filtered.sum()), "baseline_gated": n_gated})
    return pd.DataFrame(results)

# ================================================================
#  PLOTTING
# ================================================================

def plot_threshold_sensitivity(sweep_df: pd.DataFrame, plot_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()
    thresh_order = ["p1", "p2", "p3", "p5", "p7", "p10", "default"]
    thresh_labels = ["1%", "2%", "3%", "5%", "7%", "10%", "5%(def)"]
    models_to_plot = MODEL_NAMES + ["majority_vote"]
    colors = {**MODEL_COLORS, "majority_vote": "#F44336"}
    labels = {**MODEL_LABELS, "majority_vote": "Majority Vote"}

    for ax_idx, n_persist in enumerate([1, 3]):
        for mname in models_to_plot:
            sub = sweep_df[(sweep_df["model"] == mname) & (sweep_df["persistence"] == n_persist) & (sweep_df["threshold_key"].isin(thresh_order))].copy()
            sub["thresh_rank"] = sub["threshold_key"].map({k: i for i, k in enumerate(thresh_order)})
            sub = sub.sort_values("thresh_rank")
            x = range(len(sub))

            axes[ax_idx].plot(x, sub["f1"].values, "o-", color=colors[mname], label=labels[mname], lw=1.5, ms=5)
            axes[ax_idx + 2].plot(x, sub["precision"].values, "s--", color=colors[mname], label=labels[mname], lw=1.2, ms=4, alpha=0.85)
            axes[ax_idx + 2].plot(x, sub["recall"].values, "^:", color=colors[mname], lw=1.0, ms=4, alpha=0.6)

        axes[ax_idx].set_title(f"F1 Score  (persistence = {n_persist})", fontweight="bold", fontsize=11)
        axes[ax_idx].set_xticks(range(len(thresh_order)))
        axes[ax_idx].set_xticklabels(thresh_labels)
        axes[ax_idx].set_ylabel("F1")
        axes[ax_idx].set_ylim(0, 1.05)
        axes[ax_idx].legend(fontsize=8)
        axes[ax_idx].grid(True, alpha=0.3)

        axes[ax_idx + 2].set_title(f"Precision (solid) & Recall (dotted)  (persist = {n_persist})", fontweight="bold", fontsize=11)
        axes[ax_idx + 2].set_xticks(range(len(thresh_order)))
        axes[ax_idx + 2].set_xticklabels(thresh_labels)
        axes[ax_idx + 2].set_ylabel("Score")
        axes[ax_idx + 2].set_ylim(0, 1.05)
        axes[ax_idx + 2].legend(fontsize=8)
        axes[ax_idx + 2].grid(True, alpha=0.3)

    fig.suptitle("Threshold Sensitivity — Stuck-Open Valve (+2°C Setpoint Bias)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = plot_dir / "threshold_sensitivity_stuckopen.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_persistence_sensitivity(sweep_df: pd.DataFrame, plot_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    models_to_plot = MODEL_NAMES + ["majority_vote"]
    colors = {**MODEL_COLORS, "majority_vote": "#F44336"}
    labels = {**MODEL_LABELS, "majority_vote": "Majority Vote"}

    for ax_idx, (metric, title) in enumerate(zip(["f1", "precision", "recall"], ["F1 Score", "Precision", "Recall"])):
        ax = axes[ax_idx]
        for mname in models_to_plot:
            sub = sweep_df[(sweep_df["model"] == mname) & (sweep_df["threshold_key"] == "default")].sort_values("persistence")
            ax.plot(sub["persistence"].values, sub[metric].values, "o-", color=colors[mname], label=labels[mname], lw=1.5, ms=6)

        ax.set_xlabel("Persistence filter (consecutive steps)")
        ax.set_ylabel(title)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(PERSISTENCE_VALUES)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Persistence Filter Sensitivity — Default Threshold — Stuck-Open", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = plot_dir / "persistence_sensitivity_stuckopen.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_fpr_vs_recall(sweep_df: pd.DataFrame, baseline_fpr_df: pd.DataFrame, plot_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 7))
    models_to_plot = MODEL_NAMES + ["majority_vote"]
    colors = {**MODEL_COLORS, "majority_vote": "#F44336"}
    labels = {**MODEL_LABELS, "majority_vote": "Majority Vote"}
    markers = {"ocsvm": "o", "iforest": "s", "lof": "^", "majority_vote": "D"}

    for mname in models_to_plot:
        merged = sweep_df[sweep_df["model"] == mname].merge(
            baseline_fpr_df[baseline_fpr_df["model"] == mname][["threshold_key", "persistence", "baseline_fpr"]],
            on=["threshold_key", "persistence"], how="left"
        ).sort_values("baseline_fpr")
        ax.plot(merged["baseline_fpr"].values * 100, merged["recall"].values, marker=markers[mname], color=colors[mname], label=labels[mname], lw=1.2, ms=4, alpha=0.75)

    ax.set_xlabel("Baseline False Positive Rate [%]", fontsize=11)
    ax.set_ylabel("Recall (stuck-open fault window)", fontsize=11)
    ax.set_title("Detection Trade-off: Baseline FPR vs. Fault Recall", fontweight="bold", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=-0.2)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    out = plot_dir / "fpr_vs_recall_stuckopen.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_best_config_detail(df_fault: pd.DataFrame, df_base: pd.DataFrame, thresholds: dict, best_cfg: dict, plot_dir: Path):
    session_ids = compute_session_ids(df_fault)
    gated_mask = df_fault["gated"].values == 1
    best_thresh_key, best_persist = best_cfg["threshold_key"], best_cfg["persistence"]

    for mname in MODEL_NAMES:
        scores = df_fault[f"score_{mname}"].values.copy()
        thresh_val = thresholds[mname][best_thresh_key]
        raw_flags = np.zeros(len(df_fault), dtype=int)
        valid_mask = ~np.isnan(scores)
        raw_flags[valid_mask] = (scores[valid_mask] < thresh_val).astype(int)

        gated_flags = raw_flags[gated_mask]
        if best_persist > 1:
            gated_flags, _ = apply_persistence_filter(gated_flags, best_persist, session_ids[gated_mask])
        
        full_flags = np.zeros(len(df_fault), dtype=int)
        full_flags[gated_mask] = gated_flags
        df_fault[f"flag_best_{mname}"] = full_flags

    df_fault["flag_best_mv"] = ((df_fault["flag_best_ocsvm"] + df_fault["flag_best_iforest"] + df_fault["flag_best_lof"]) >= 2).astype(int)

    zoom_start, zoom_end = pd.Timestamp("2018-01-14"), pd.Timestamp("2018-01-18")
    df_z = df_fault[(df_fault["datetime"] >= zoom_start) & (df_fault["datetime"] <= zoom_end)].copy()
    df_b = df_base[(df_base["datetime"] >= zoom_start) & (df_base["datetime"] <= zoom_end)].copy()

    fault_spans = [(pd.Timestamp("2018-01-15 08:00"), pd.Timestamp("2018-01-15 17:00")),
                   (pd.Timestamp("2018-01-16 08:00"), pd.Timestamp("2018-01-16 17:00"))]

    # Updated to 6 rows to accommodate the setpoint panel
    fig, axes = plt.subplots(6, 1, figsize=(16, 21), sharex=True)

    for ax in axes:
        for fs, fe in fault_spans:
            ax.axvspan(fs, fe, alpha=0.12, color="red")
        ax.grid(True, alpha=0.3)

    # Panel 1: Zone temperature
    axes[0].plot(df_z["datetime"], df_z["zone_temp"], "r-", lw=1.2, label="Faulted")
    axes[0].plot(df_b["datetime"], df_b["zone_temp"], "b--", lw=1.0, alpha=0.7, label="Baseline")
    axes[0].set_ylabel("Temperature [°C]")
    axes[0].set_title("Zone Air Temperature", fontweight="bold")
    axes[0].legend(fontsize=9)

    # Panel 2: Setpoint comparison (NEW)
    if 'intended_sp' in df_z.columns:
        axes[1].plot(df_z["datetime"], df_z["htg_sp"], "r-", lw=1.2, label="Actuated setpoint (faulted)")
        axes[1].plot(df_z["datetime"], df_z["intended_sp"], "b--", lw=1.0, alpha=0.7, label="Intended setpoint (schedule)")
        if 'intended_sp' in df_b.columns:
            axes[1].plot(df_b["datetime"], df_b["htg_sp"], "g:", lw=0.8, alpha=0.5, label="Baseline setpoint")
    axes[1].set_ylabel("Setpoint [°C]")
    axes[1].set_title("Heating Setpoint: Intended vs. Actuated", fontweight="bold")
    axes[1].legend(fontsize=9)

    # Panel 3: Flow rate
    axes[2].plot(df_z["datetime"], df_z["m_dot"], "r-", lw=1.2, label="Faulted")
    axes[2].plot(df_b["datetime"], df_b["m_dot"], "b--", lw=1.0, alpha=0.7, label="Baseline")
    axes[2].set_ylabel("Flow rate [kg/s]")
    axes[2].set_title("HW Mass Flow Rate", fontweight="bold")
    axes[2].legend(fontsize=9)

    # Panels 4–6: Anomaly scores with BEST flags
    for i, mname in enumerate(MODEL_NAMES):
        ax = axes[3 + i]
        score_col, flag_col = f"score_{mname}", f"flag_best_{mname}"
        valid = df_z[df_z[score_col].notna()]
        ax.plot(valid["datetime"], valid[score_col], color=MODEL_COLORS[mname], lw=0.8, alpha=0.7)

        thresh_val = thresholds[mname][best_thresh_key]
        ax.axhline(thresh_val, color="gray", ls="--", lw=1.0, alpha=0.6, label=f"threshold ({best_thresh_key})")

        flagged = df_z[df_z[flag_col] == 1]
        n_in_window = flagged["in_fault_window"].sum()
        ax.scatter(flagged["datetime"], flagged[score_col], color="red", s=14, zorder=5, alpha=0.8,
                   label=f"Flagged (TP={n_in_window}, FP={len(flagged) - n_in_window})")

        ax.set_ylabel(f"{MODEL_LABELS[mname]}\nscore")
        ax.set_title(f"{MODEL_LABELS[mname]} — {best_thresh_key} threshold, persist={best_persist}", fontweight="bold")
        ax.legend(fontsize=8, loc="lower left")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    axes[-1].set_xlabel("Date / Time")

    fig.suptitle(f"Best Configuration Detail — Stuck-Open (thresh={best_thresh_key}, persist={best_persist})", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = plot_dir / "fault_window_best_config_stuckopen.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ================================================================
#  BEST CONFIGURATION SELECTION & CONSOLE REPORT
# ================================================================

def select_best_config(sweep_df: pd.DataFrame, min_recall: float = 0.70) -> dict:
    candidates = sweep_df[sweep_df["recall"] >= min_recall].copy()
    if len(candidates) == 0:
        print(f"  WARNING: No config achieves recall >= {min_recall:.0%}. Selecting highest recall instead.")
        return sweep_df.sort_values("recall", ascending=False).iloc[0].to_dict()
    return candidates.sort_values("f1", ascending=False).iloc[0].to_dict()

def print_report(sweep_df: pd.DataFrame, baseline_fpr_df: pd.DataFrame, best_cfg: dict):
    print("\n" + "=" * 75)
    print("  SWEEP RESULTS SUMMARY — STUCK-OPEN FAULT")
    print("=" * 75)

    orig = sweep_df[(sweep_df["threshold_key"] == "default") & (sweep_df["persistence"] == 1)]
    print(f"\n  {'Config':<32s} {'Model':<16s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'FPR':>6s} {'Latency':>8s}")
    print("  " + "-" * 88)

    for _, row in orig.iterrows():
        lat = f"{row['latency_minutes']}m" if row['latency_minutes'] is not None else "N/A"
        print(f"  {'Original (default, p=1)':<32s} {row['model']:<16s} {row['precision']:>6.3f} {row['recall']:>6.3f} {row['f1']:>6.3f} {row['fpr']:>6.3f} {lat:>8s}")

    lat = f"{best_cfg['latency_minutes']}m" if best_cfg['latency_minutes'] is not None else "N/A"
    print(f"\n  >>> BEST CONFIG: model={best_cfg['model']}, threshold={best_cfg['threshold_key']}, persistence={int(best_cfg['persistence'])}")
    print(f"      Precision={best_cfg['precision']:.3f}  Recall={best_cfg['recall']:.3f}  F1={best_cfg['f1']:.3f}  FPR={best_cfg['fpr']:.3f}  Latency={lat}")

    # Reference values commented out until established
    print(f"\n  Conference paper rule-based proxy (stuck-open):")
    print(f"      (Not yet available for stuck-open. Proceed to baseline FPR check)")

    bf_row = baseline_fpr_df[(baseline_fpr_df["model"] == best_cfg["model"]) & (baseline_fpr_df["threshold_key"] == best_cfg["threshold_key"]) & (baseline_fpr_df["persistence"] == best_cfg["persistence"])]
    if len(bf_row) > 0:
        print(f"  Baseline FPR for best config: {bf_row.iloc[0]['baseline_fpr']:.4%}")

# ================================================================
#  MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 4b: Post-hoc threshold & persistence sweep (Stuck-Open)")
    parser.add_argument("--faulted-log", required=True, help="Path to faulted fdc_runtime_log.csv")
    parser.add_argument("--baseline-log", required=True, help="Path to baseline fdc_runtime_log.csv")
    parser.add_argument("--models-dir", default="models", help="Directory containing trained models and thresholds")
    parser.add_argument("--plot-dir", default="plots/fault_analysis_stuckopen", help="Output directory for plots and CSVs")
    parser.add_argument("--min-recall", type=float, default=0.70, help="Minimum recall for best-config selection")
    args = parser.parse_args()

    plot_dir, models_dir = Path(args.plot_dir), Path(args.models_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75 + "\n  PHASE 4b: POST-HOC SWEEP (Stuck-Open Valve)\n" + "=" * 75)

    thresholds = joblib.load(models_dir / "thresholds.joblib")
    df_fault, df_base = load_and_prepare(args.faulted_log), load_and_prepare(args.baseline_log)

    print("\nRunning sweeps...")
    sweep_df = run_sweep(df_fault, thresholds, df_base)
    baseline_fpr_df = run_baseline_fpr_sweep(df_base, thresholds)
    best_cfg = select_best_config(sweep_df, min_recall=args.min_recall)

    print_report(sweep_df, baseline_fpr_df, best_cfg)

    print("\n" + "=" * 75 + "\n  GENERATING PLOTS\n" + "=" * 75)
    plot_threshold_sensitivity(sweep_df, plot_dir)
    plot_persistence_sensitivity(sweep_df, plot_dir)
    plot_fpr_vs_recall(sweep_df, baseline_fpr_df, plot_dir)
    plot_best_config_detail(df_fault, df_base, thresholds, best_cfg, plot_dir)

    sweep_df.to_csv(plot_dir / "sweep_results_stuckopen.csv", index=False)
    baseline_fpr_df.to_csv(plot_dir / "baseline_fpr_sweep_stuckopen.csv", index=False)
    
    with open(plot_dir / "best_config_stuckopen.json", "w") as f:
        json.dump({k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v) for k, v in best_cfg.items()}, f, indent=2)

    print("\n" + "=" * 75 + "\n  SWEEP COMPLETE\n" + "=" * 75)

if __name__ == "__main__":
    main()
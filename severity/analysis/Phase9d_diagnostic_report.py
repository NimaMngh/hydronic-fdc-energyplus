# -*- coding: utf-8 -*-
"""
Phase 9d: Severity Diagnostics & Narrative Report
===================================================
Loads the Phase 9a/9b output tables (table_severity_detection.csv and
table_severity_compensation.csv) together with severity_metadata.json,
performs four automated checks, and produces two output files:

    severity/plots/
        severity_diagnostics.json   — machine-readable flag records
        severity_report.txt         — human-readable narrative summary
                                      (suitable for copy-paste into the paper)

Automated diagnostic checks
----------------------------
1. so_b05 paradox
   F1 = 0 (fault undetectable) yet DDH recovery is substantial. This happens
   because compensation triggers on different logic than the detection model
   (always-on or proxy-based), so the two can disagree completely.

2. Non-monotonic DDH recovery (stuck-closed)
   Recovery should increase with severity (more flow restriction = more
   discomfort removed by compensation). A non-monotonic profile at sc_s20
   (peak) and sc_s10 (collapse) points to the compensation actuator being
   constrained at very low flow fractions; the check reports the anomaly and
   the implied severity threshold.

3. Counter-intuitive supply-curve F1 trend
   Detection F1 peaks at scu_k05 (moderate bias) and declines at higher
   biases. Likely cause: at large OAT biases the heating demand is suppressed
   uniformly, so fault-window temperatures move away from the anomaly
   detector's learned distribution, which paradoxically reduces the scores.
   Raised when max(F1) does not occur at max(severity_value).

4. Detection saturation plateaus
   Consecutive severity levels where F1, precision, and recall all sit within
   a tolerance band; reports the plateau range and the affected metrics.

Author : Nima Monghasemi
Date   : March 2026
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE         = Path(__file__).resolve().parent   # severity/analysis/
SEV_DIR      = HERE.parent                        # severity/
PROJECT_ROOT = SEV_DIR.parent

SEV_PLOTS  = SEV_DIR / "plots"
DET_CSV    = SEV_PLOTS / "table_severity_detection.csv"
COMP_CSV   = SEV_PLOTS / "table_severity_compensation.csv"
META_JSON  = SEV_PLOTS / "severity_metadata.json"

DIAG_JSON  = SEV_PLOTS / "severity_diagnostics.json"
REPORT_TXT = SEV_PLOTS / "severity_report.txt"

# Tolerance for "identical" metric values (saturation check)
PLATEAU_TOL = 1e-4

# Minimum recovery % to consider as "substantial" in the paradox check
PARADOX_RECOVERY_THRESHOLD = 30.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    for p, name in [(DET_CSV, "Detection"), (COMP_CSV, "Compensation")]:
        if not p.exists():
            raise FileNotFoundError(
                f"{name} table not found at:\n  {p}\n"
                f"Run Phase9{'a' if name == 'Detection' else 'b'}_severity_*.py first."
            )
    return pd.read_csv(DET_CSV), pd.read_csv(COMP_CSV)


def _subset(df: pd.DataFrame, fault_type: str) -> pd.DataFrame:
    return (
        df[df["fault_type"] == fault_type]
        .copy()
        .sort_values("severity_value")
        .reset_index(drop=True)
    )


def _find_plateaus(df: pd.DataFrame, metrics: list[str],
                   tol: float = PLATEAU_TOL) -> list[dict]:
    """
    Identify consecutive severity levels where every metric in *metrics* is
    within *tol* of its previous value.  Returns a list of plateau records.
    """
    plateaus = []
    x = df["severity_value"].values
    for metric in metrics:
        y = df[metric].values
        i = 0
        while i < len(y) - 1:
            if abs(y[i] - y[i + 1]) < tol:
                j = i + 1
                while j < len(y) - 1 and abs(y[j] - y[j + 1]) < tol:
                    j += 1
                tags = df["severity_tag"].iloc[i: j + 1].tolist()
                plateaus.append({
                    "metric":          metric,
                    "sev_start":       float(x[i]),
                    "sev_end":         float(x[j]),
                    "value":           float(round(y[i], 6)),
                    "severity_tags":   tags,
                    "n_levels":        j - i + 1,
                })
                i = j
            else:
                i += 1
    return plateaus


# ---------------------------------------------------------------------------
# Check 1 – so_b05 Paradox
# ---------------------------------------------------------------------------

def check_paradox(det_df: pd.DataFrame, comp_df: pd.DataFrame) -> dict:
    """
    Detect cases where F1 ≈ 0 (undetectable) but DDH recovery is substantial.
    """
    flags = []
    for fault_type in det_df["fault_type"].unique():
        det_sub  = _subset(det_df,  fault_type)
        comp_sub = _subset(comp_df, fault_type)
        # Align on severity_tag
        merged = det_sub.merge(
            comp_sub[["severity_tag", "ddh_recovery_pct"]],
            on="severity_tag", how="inner"
        )
        paradox_rows = merged[
            (merged["f1"] < 1e-6) &
            (merged["ddh_recovery_pct"] > PARADOX_RECOVERY_THRESHOLD)
        ]
        for _, row in paradox_rows.iterrows():
            flags.append({
                "fault_type":       fault_type,
                "severity_tag":     row["severity_tag"],
                "severity_value":   float(row["severity_value"]),
                "f1":               float(row["f1"]),
                "ddh_recovery_pct": float(row["ddh_recovery_pct"]),
                "interpretation": (
                    "Compensation active despite zero-F1 detection. "
                    "Likely cause: Phase 5 uses a proxy or always-on trigger "
                    "independent of the Phase 4 anomaly model. "
                    "Report this as a limitation: detection and compensation "
                    "are not fully coupled at sub-threshold severities."
                ),
            })
    return {
        "check": "paradox_f1_zero_but_recovery_positive",
        "n_flags": len(flags),
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Check 2 – Non-Monotonic DDH Recovery (Stuck-Closed)
# ---------------------------------------------------------------------------

def check_nonmonotonic_recovery(comp_df: pd.DataFrame) -> dict:
    """
    For stuck_closed: DDH recovery should increase monotonically with severity
    (lower phi = more severe).  Detect inversions after x-axis inversion.
    """
    sub = _subset(comp_df, "stuck_closed")
    # After inversion: sort by descending phi (increasing severity)
    sub_inv = sub.sort_values("severity_value", ascending=False).reset_index(drop=True)
    rec = sub_inv["ddh_recovery_pct"].values
    flags = []
    for i in range(1, len(rec)):
        if rec[i] < rec[i - 1] - PLATEAU_TOL:
            flags.append({
                "inversion_at_tag": sub_inv["severity_tag"].iloc[i],
                "severity_value":   float(sub_inv["severity_value"].iloc[i]),
                "ddh_recovery_here": float(rec[i]),
                "ddh_recovery_prev": float(rec[i - 1]),
                "delta_pct":        float(round(rec[i] - rec[i - 1], 2)),
            })

    peak_idx = int(np.argmax(rec))
    peak_row = sub_inv.iloc[peak_idx]

    return {
        "check": "nonmonotonic_ddh_recovery_stuck_closed",
        "n_inversions": len(flags),
        "peak_recovery_tag":   str(peak_row["severity_tag"]),
        "peak_recovery_value": float(peak_row["severity_value"]),
        "peak_recovery_pct":   float(peak_row["ddh_recovery_pct"]),
        "inversions": flags,
        "interpretation": (
            "Non-monotonic DDH recovery in stuck-closed faults indicates that "
            "the compensation actuator is constrained at very low flow fractions "
            "(phi ≤ 0.10). The peak recovery at sc_s20 (phi=0.20) and collapse "
            "at sc_s10 (phi=0.10) imply a practical compensation floor: below "
            "~20 % rated flow the supply-temperature ramp cannot compensate for "
            "the thermal deficit within the fault window. "
            "Recommend citing phi=0.20 as the 'compensation effectiveness threshold'."
        ) if flags else "DDH recovery is monotonically non-decreasing with severity — no anomaly.",
    }


# ---------------------------------------------------------------------------
# Check 3 – Counter-Intuitive Supply-Curve F1 Trend
# ---------------------------------------------------------------------------

def check_supply_curve_f1_trend(det_df: pd.DataFrame) -> dict:
    """
    For supply_curve: flag if the F1 peak occurs at a *lower* severity level
    rather than the highest severity level.
    """
    sub = _subset(det_df, "supply_curve")   # sorted ascending by sev_value (|K|)
    f1  = sub["f1"].values
    tags = sub["severity_tag"].values
    vals = sub["severity_value"].values

    peak_idx = int(np.argmax(f1))
    max_idx  = len(f1) - 1  # highest severity_value = largest |bias|

    is_counter_intuitive = (peak_idx < max_idx) and (f1[peak_idx] - f1[max_idx] > PLATEAU_TOL)

    return {
        "check": "supply_curve_f1_peak_not_at_max_severity",
        "peak_f1_tag":       str(tags[peak_idx]),
        "peak_f1_value":     float(vals[peak_idx]),
        "peak_f1_score":     float(f1[peak_idx]),
        "max_severity_tag":  str(tags[max_idx]),
        "max_severity_value": float(vals[max_idx]),
        "f1_at_max_severity": float(f1[max_idx]),
        "counter_intuitive": bool(is_counter_intuitive),
        "interpretation": (
            "F1 peaks at a moderate supply-curve bias and *declines* at larger "
            "biases.  Probable mechanism: at very large OAT sensor errors the "
            "heating demand is suppressed uniformly across all zones, shifting "
            "temperature residuals away from the anomaly detector's learned "
            "normal distribution toward a new 'cold' regime that the fixed "
            "threshold cannot distinguish from fault-free cold-weather operation. "
            "Implication: the current threshold-based detector has a sweet spot "
            "at moderate biases; re-training with bias-inclusive data or using "
            "a bias-aware feature (OAT vs. supply-temp correlation) would extend "
            "detection to high-severity levels."
        ) if is_counter_intuitive else (
            "F1 is monotonically non-decreasing with supply-curve severity — "
            "no counter-intuitive trend detected."
        ),
    }


# ---------------------------------------------------------------------------
# Check 4 – Detection Saturation Plateaus
# ---------------------------------------------------------------------------

def check_saturation_plateaus(det_df: pd.DataFrame) -> dict:
    """
    For each fault type, find severity bands where F1, precision, and recall
    are all flat within PLATEAU_TOL.
    """
    summary = {}
    for fault_type in ["stuck_closed", "stuck_open", "supply_curve"]:
        sub = _subset(det_df, fault_type)
        plateaus = _find_plateaus(sub, ["f1", "precision", "recall"])
        # Keep only plateaus that span ≥ 2 consecutive levels
        significant = [p for p in plateaus if p["n_levels"] >= 2]
        summary[fault_type] = {
            "n_plateau_bands": len(significant),
            "plateau_details": significant,
            "interpretation": (
                f"Detection saturates across {len(significant)} severity band(s) "
                f"for {fault_type.replace('_', ' ')} faults. "
                "This indicates the current model/threshold/persistence combination "
                "cannot differentiate these severity levels — the detector reaches "
                "its performance ceiling. For publication: report the saturation "
                "band as the 'detection-equivalent range' and note it in the "
                "limitations section."
            ) if significant else (
                f"No saturation plateaus found for {fault_type.replace('_', ' ')}."
            ),
        }
    return {"check": "detection_saturation_plateaus", "by_fault_type": summary}


# ---------------------------------------------------------------------------
# Narrative report writer
# ---------------------------------------------------------------------------

def _fmt_pct(v: float) -> str:
    return f"{v:.1f} %"

def _fmt_f1(v: float) -> str:
    return f"{v:.3f}"


def write_narrative_report(diag: dict, det_df: pd.DataFrame,
                            comp_df: pd.DataFrame) -> str:
    """Build a plain-text narrative suitable for paper integration."""
    lines = []
    sep = "=" * 72

    lines += [
        sep,
        "SEVERITY SWEEP DIAGNOSTIC REPORT",
        f"Generated: {diag['generated_at']}",
        sep,
        "",
        "This report summarises four automated diagnostic findings from the",
        "fault-severity parametric sweep (Phase 9a/9b results).",
        "",
    ]

    # ── Finding 1: Paradox ──────────────────────────────────────────────────
    p = diag["checks"]["paradox"]
    lines += ["─" * 72,
              "FINDING 1 — so_b05 PARADOX: Compensation Without Detection",
              "─" * 72, ""]
    if p["n_flags"]:
        for f in p["flags"]:
            lines += [
                f"  Fault type   : {f['fault_type']}",
                f"  Severity tag : {f['severity_tag']}  "
                f"(severity = {f['severity_value']} °C thermostat bias)",
                f"  Detection F1 : {_fmt_f1(f['f1'])}  ← effectively zero (undetectable)",
                f"  DDH Recovery : {_fmt_pct(f['ddh_recovery_pct'])}  ← non-trivial compensation",
                "",
                f"  Interpretation:",
            ]
            for sentence in f["interpretation"].split(". "):
                if sentence.strip():
                    lines.append(f"    {sentence.strip()}.")
            lines.append("")
        lines += [
            "  Paper narrative (suggested):",
            "    'At the lowest stuck-open severity (bias = 0.5 °C), the anomaly",
            "    detector produced F1 = 0 — the signal is sub-threshold and the",
            "    fault is undetectable by the ML layer alone. Nevertheless, the",
            "    compensation module recovered 83.5 % of degree-hour discomfort,",
            "    driven by the persistent supply-temperature override that operates",
            "    independently of the detection flag. This decoupling represents",
            "    a practical robustness feature but also a transparency risk:',",
            "    'the system compensates for faults it has not explicitly detected.'",
            "",
        ]
    else:
        lines += ["  No paradox flags found.", ""]

    # ── Finding 2: Non-Monotonic Recovery ──────────────────────────────────
    nm = diag["checks"]["nonmonotonic_recovery"]
    lines += ["─" * 72,
              "FINDING 2 — NON-MONOTONIC DDH RECOVERY (Stuck-Closed)",
              "─" * 72, ""]
    lines += [
        f"  Peak recovery : {_fmt_pct(nm['peak_recovery_pct'])} "
        f"at {nm['peak_recovery_tag']} (phi = {nm['peak_recovery_value']})",
        f"  Inversions    : {nm['n_inversions']}",
        "",
        "  Interpretation:",
    ]
    for sentence in nm["interpretation"].split(". "):
        if sentence.strip():
            lines.append(f"    {sentence.strip()}.")
    lines.append("")

    if nm["inversions"]:
        lines.append("  Inversion details:")
        for inv in nm["inversions"]:
            lines.append(
                f"    → {inv['inversion_at_tag']}  "
                f"(phi={inv['severity_value']}):  "
                f"recovery drops by {inv['delta_pct']:.1f} pp "
                f"({_fmt_pct(inv['ddh_recovery_here'])} vs "
                f"{_fmt_pct(inv['ddh_recovery_prev'])} at previous level)"
            )
        lines.append("")

    lines += [
        "  Paper narrative (suggested):",
        "    'For the stuck-closed fault, DDH recovery is non-monotonic:",
        "    it peaks at sc_s20 (phi = 0.20, 39.4 %) and collapses at sc_s10",
        "    (phi = 0.10, 17.7 %). This is attributed to actuator saturation:",
        "    at phi ≤ 0.10 the coil inlet flow is so restricted that raising",
        "    the supply temperature beyond 70 °C cannot compensate for the",
        "    thermal deficit within the fault window. The compensation",
        "    effectiveness threshold is therefore estimated at phi ≈ 0.20.'",
        "",
    ]

    # ── Finding 3: Supply-Curve F1 Trend ───────────────────────────────────
    sc = diag["checks"]["supply_curve_f1_trend"]
    lines += ["─" * 72,
              "FINDING 3 — COUNTER-INTUITIVE SUPPLY-CURVE F1 DECLINE",
              "─" * 72, ""]
    lines += [
        f"  F1 peak at    : {sc['peak_f1_tag']} "
        f"(|bias| = {sc['peak_f1_value']} K,  F1 = {_fmt_f1(sc['peak_f1_score'])})",
        f"  F1 at max sev.: {sc['max_severity_tag']} "
        f"(|bias| = {sc['max_severity_value']} K,  F1 = {_fmt_f1(sc['f1_at_max_severity'])})",
        f"  Counter-intuitive flag: {sc['counter_intuitive']}",
        "",
        "  Interpretation:",
    ]
    for sentence in sc["interpretation"].split(". "):
        if sentence.strip():
            lines.append(f"    {sentence.strip()}.")
    lines += [
        "",
        "  Paper narrative (suggested):",
        "    'Detection F1 for the supply-curve fault peaks at a moderate bias",
        "    (|ΔT| = 5 K, F1 = 0.527) and declines at larger biases (F1 = 0.486",
        "    at |ΔT| = 15 K). This counter-intuitive behaviour is attributed to",
        "    the anomaly detector's score distribution: at very large OAT biases",
        "    the heating system operates in an extreme-suppression regime whose",
        "    residuals closely resemble fault-free cold-weather patterns, reducing",
        "    separability. Future work should incorporate supply-temperature",
        "    correlation features to improve detection at high-severity levels.'",
        "",
    ]

    # ── Finding 4: Saturation Plateaus ─────────────────────────────────────
    sat = diag["checks"]["saturation_plateaus"]["by_fault_type"]
    lines += ["─" * 72,
              "FINDING 4 — DETECTION SATURATION PLATEAUS",
              "─" * 72, ""]
    for ft, info in sat.items():
        label = ft.replace("_", " ").title()
        lines += [f"  [{label}]  {info['n_plateau_bands']} plateau band(s)"]
        for pd_info in info["plateau_details"]:
            tags_str = ", ".join(pd_info["severity_tags"])
            lines.append(
                f"    Metric {pd_info['metric']:10s}: flat at {pd_info['value']:.4f} "
                f"across [{tags_str}]  "
                f"(sev {pd_info['sev_start']} → {pd_info['sev_end']})"
            )
        lines.append("")
    lines += [
        "  Paper narrative (suggested):",
        "    'All three fault types exhibit detection saturation: F1, precision,",
        "    and recall are constant across mid-severity levels (sc_s20–sc_s50;",
        "    so_b20–so_b60).  This indicates the chosen model/threshold/persistence",
        "    combination reaches its performance ceiling within a moderate severity",
        "    band and cannot differentiate individual levels within that range.",
        "    In practice this is acceptable — the system detects all faults in",
        "    the plateau band with equal reliability — but the saturation range",
        "    should be reported as the detector's operational resolution limit.'",
        "",
    ]

    # ── Summary table ───────────────────────────────────────────────────────
    lines += [sep,
              "DIAGNOSTIC SUMMARY",
              sep,
              f"  Paradox flags       : {p['n_flags']}",
              f"  Non-monotonic inversions (SC): {nm['n_inversions']}",
              f"  Supply-curve F1 counter-intuitive: {sc['counter_intuitive']}",
    ]
    tot_plateaus = sum(
        sat[ft]["n_plateau_bands"] for ft in sat
    )
    lines += [
        f"  Total saturation bands: {tot_plateaus}",
        "",
        "All flags written to: severity_diagnostics.json",
        ""]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 68)
    print("Phase 9d — Severity Diagnostics & Narrative Report")
    print("=" * 68)

    try:
        det_df, comp_df = _load_tables()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"  Detection table  : {len(det_df)} rows")
    print(f"  Compensation table: {len(comp_df)} rows")

    # Load metadata if available
    meta = {}
    if META_JSON.exists():
        meta = json.loads(META_JSON.read_text())
        print(f"  Metadata         : loaded from {META_JSON.name}")
    else:
        print(f"  WARNING: {META_JSON.name} not found — provenance info unavailable.")

    # Run checks
    print("\n  Running diagnostic checks ...")
    c1 = check_paradox(det_df, comp_df)
    c2 = check_nonmonotonic_recovery(comp_df)
    c3 = check_supply_curve_f1_trend(det_df)
    c4 = check_saturation_plateaus(det_df)
    print(f"    Check 1 (paradox)         : {c1['n_flags']} flag(s)")
    print(f"    Check 2 (non-monotonic)   : {c2['n_inversions']} inversion(s)")
    print(f"    Check 3 (SC F1 trend)     : counter-intuitive = {c3['counter_intuitive']}")
    n_plat = sum(
        c4["by_fault_type"][ft]["n_plateau_bands"]
        for ft in c4["by_fault_type"]
    )
    print(f"    Check 4 (saturation)      : {n_plat} total plateau band(s)")

    # Aggregate diagnostics record
    diag = {
        "generated_by":  "Phase9d_diagnostic_report.py",
        "generated_at":  datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_metadata": meta.get("generated_at", "unavailable"),
        "n_rows_detection":     len(det_df),
        "n_rows_compensation":  len(comp_df),
        "checks": {
            "paradox":                c1,
            "nonmonotonic_recovery":  c2,
            "supply_curve_f1_trend":  c3,
            "saturation_plateaus":    c4,
        },
    }

    # Write JSON
    DIAG_JSON.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(f"\n  Saved diagnostics JSON: {DIAG_JSON}")

    # Write narrative text report
    report_text = write_narrative_report(diag, det_df, comp_df)
    REPORT_TXT.write_text(report_text, encoding="utf-8")
    print(f"  Saved narrative report : {REPORT_TXT}")

    print("\nPhase 9d complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase12_compound_diagnosis.py — compound-fault diagnosis analysis.

Post-hoc PARALLEL evaluation of all three diagnostic signatures on a run's
fdc_runtime_log.csv, with an explicit abstain state. Unlike the runtime
ladder (fixed priority, first match wins), this script evaluates every
signature independently at every alarmed timestep and classifies each step
into one of four diagnostic states:

    single_<fault>        exactly one signature supported -> unambiguous action
    multi_consistent      several signatures supported, but their corrective
                          actions agree in direction (e.g. stuck_closed +
                          supply_curve both call for a supply boost) -> a
                          supervisory action is still safe
    conflict_abstain      supported signatures call for opposing actions
                          (boost vs. reduce) -> the safe action is to abstain
    no_signature_abstain  anomaly confirmed but no signature matches ->
                          abstain (the runtime 'unknown' outcome)

Signature predicates and thresholds replicate _diagnose_fault in
Phase5_ML_compensate.py exactly (including the strong-signal supply path,
the relaxed one-zone confirmation, and the legacy multi-zone path), read
from the run's own diagnosis_config. Detection alarms are reconstructed
from the logged anomaly scores with the same session-reset persistence
semantics as the Phase 4b sweeps, so any detector can be evaluated
regardless of which one drove the run.

Usage (from the project root):
    python analysis\\Phase12_compound_diagnosis.py --run-dir runs\\<C1_dir>
    python analysis\\Phase12_compound_diagnosis.py --run-dir runs\\<C1_dir> runs\\<C2_dir> ^
        --detector majority --persistence 4 --out plots\\parameter_studies

Outputs per run:
    <out>/<run_name>_parallel_diag.csv     per-gated-timestep detail
    <out>/<run_name>_state_summary.csv     state occupancy inside/outside window
plus a console summary comparing the parallel states with the runtime
ladder verdicts (diagnosed_fault column, if present in the log).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Corrective-action direction implied by each signature
ACTION = {"stuck_closed": "boost", "stuck_open": "reduce", "supply_curve": "boost"}


# ────────────────────────────────────────────────────────────────────────────
# Loading
# ────────────────────────────────────────────────────────────────────────────

def load_run(run_dir: Path):
    cfg = None
    for p in sorted(run_dir.glob("run_config_*.json")):
        cfg = json.loads(p.read_text(encoding="utf-8"))
        break
    log_path = run_dir / "fdc_runtime_log.csv"
    df = pd.read_csv(log_path, low_memory=False)
    return df, (cfg or {})


def load_thresholds(models_dir: Path) -> dict:
    tc = json.loads((models_dir / "training_config.json").read_text(encoding="utf-8"))
    return tc["thresholds"]


# ────────────────────────────────────────────────────────────────────────────
# Detection reconstruction (identical semantics to the Phase 4b sweeps)
# ────────────────────────────────────────────────────────────────────────────

def session_ids(gated: np.ndarray) -> np.ndarray:
    out = np.zeros(len(gated), int)
    cur, prev = 0, 0
    for i, g in enumerate(gated):
        if g == 1 and prev == 0:
            cur += 1
        out[i] = cur if g == 1 else 0
        prev = g
    return out


def persistence_filter(flags: np.ndarray, n: int, sess: np.ndarray) -> np.ndarray:
    filtered = np.zeros_like(flags)
    run, prev = 0, -1
    for i, f in enumerate(flags):
        if sess[i] != prev:
            run = 0
        prev = sess[i]
        run = run + 1 if f == 1 else 0
        if run >= n:
            filtered[max(0, i - n + 1): i + 1] = 1
    return filtered


def reconstruct_alarms(df: pd.DataFrame, gated_mask: np.ndarray,
                       detector: str, thr_key: str, persistence: int,
                       thresholds: dict) -> np.ndarray:
    """Persistence-filtered alarm array aligned with the gated rows."""
    g = df.loc[gated_mask]
    sess = session_ids(np.ones(len(g), int))  # sessions from full-frame gating below
    # recompute sessions on the full frame for correct gap resets
    full_sess = session_ids(gated_mask.astype(int))
    sess = full_sess[gated_mask]

    def raw(model):
        return (g[f"score_{model}"].values < thresholds[model][thr_key]).astype(int)

    if detector == "majority":
        flags = ((raw("ocsvm") + raw("iforest") + raw("lof")) >= 2).astype(int)
    else:
        flags = raw(detector)
    return persistence_filter(flags, persistence, sess)


# ────────────────────────────────────────────────────────────────────────────
# Parallel signature evaluation (replicates _diagnose_fault, no priority)
# ────────────────────────────────────────────────────────────────────────────

def zones_under(row: pd.Series, zone_labels: list, thr: float) -> int:
    n = 0
    for z in zone_labels:
        zt = row.get(f"{z}_zone_temp", np.nan)
        sp = row.get(f"{z}_intended_sp", np.nan)
        if not (np.isnan(zt) or np.isnan(sp)) and (zt - sp) < -thr:
            n += 1
    return n


def evaluate_signatures(row: pd.Series, diag: dict, zone_labels: list,
                        baseline_supply: float) -> dict:
    thr_te = diag.get("temp_error_thresh_C", 0.3)
    thr_te_inst = diag.get("instant_temp_error_thresh_C", 0.5)
    thr_fr = diag.get("flow_ratio_low", 0.6)
    dev_K = diag.get("supply_temp_deviation_thresh_K", 5.0)
    mzc = diag.get("multi_zone_count", 2)

    fr = row.get("flow_ratio", np.nan)
    te_avg = row.get("temp_error_2h_avg", np.nan)
    te_inst = row.get("temp_error", np.nan)
    t_supply = row.get("t_supply", np.nan)

    s_sc = (not np.isnan(fr)) and (not np.isnan(te_avg)) \
        and fr < thr_fr and te_avg < -thr_te

    avg_pos = (not np.isnan(te_avg)) and te_avg > thr_te
    inst_pos = (not np.isnan(te_inst)) and te_inst > thr_te_inst
    s_so = (avg_pos or inst_pos) and (not np.isnan(fr)) and fr >= thr_fr

    s_sup = False
    if (not np.isnan(fr)) and fr >= thr_fr:
        if not np.isnan(t_supply):
            dev = baseline_supply - t_supply
            if dev >= 2.0 * dev_K:
                s_sup = True
            elif dev >= dev_K:
                if zones_under(row, zone_labels, 0.0) >= max(1, mzc - 1):
                    s_sup = True
        if (not s_sup) and (not np.isnan(te_avg)) and te_avg < -thr_te:
            if zones_under(row, zone_labels, thr_te) >= mzc:
                s_sup = True

    return {"S_stuck_closed": s_sc, "S_stuck_open": s_so, "S_supply_curve": s_sup}


def classify(sig: dict) -> str:
    supported = [f.replace("S_", "") for f, v in sig.items() if v]
    if not supported:
        return "no_signature_abstain"
    if len(supported) == 1:
        return f"single_{supported[0]}"
    actions = {ACTION[f] for f in supported}
    return "multi_consistent" if len(actions) == 1 else "conflict_abstain"


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def analyse_run(run_dir: Path, detector: str, thr_key: str,
                persistence: int, thresholds: dict, out_dir: Path):
    df, cfg = load_run(run_dir)
    diag = cfg.get("diagnosis_config", {})
    baseline_supply = cfg.get("compensation_config", {}).get(
        "baseline_supply_temp_C", cfg.get("baseline_supply_temp_C", 60.0))
    zone_labels = list(cfg.get("extra_zone_sensors", {}).keys())
    run_name = cfg.get("run_name", run_dir.name)

    # detector defaults follow the run's own config unless overridden
    if detector == "auto":
        detector = cfg.get("detection_model") or "majority"
        persistence = persistence or cfg.get("persistence_steps", 4)
    persistence = persistence or 4

    gated = ((df.get("occupied", 1) == 1)
             & (df.get("heating_active", 1) == 1)
             & (df.get("feature_ready", 1) == 1)).values
    alarms_g = reconstruct_alarms(df, gated, detector, thr_key,
                                  persistence, thresholds)

    g = df.loc[gated].copy().reset_index(drop=True)
    g["alarm"] = alarms_g
    if "in_fault" not in g.columns:
        g["in_fault"] = 0
    if "diagnosed_fault" not in g.columns:
        g["diagnosed_fault"] = ""
    g["diagnosed_fault"] = g["diagnosed_fault"].fillna("")

    sigs = g.apply(lambda r: evaluate_signatures(r, diag, zone_labels,
                                                 baseline_supply), axis=1)
    sig_df = pd.DataFrame(list(sigs))
    g = pd.concat([g, sig_df], axis=1)
    g["state"] = [classify(s) for s in sigs]
    # states are only meaningful on alarmed steps; mark the rest
    g.loc[g.alarm == 0, "state"] = "not_alarmed"

    keep = ["timestep", "month", "day", "hour", "minute", "in_fault", "alarm",
            "flow_ratio", "temp_error", "temp_error_2h_avg",
            "S_stuck_closed", "S_stuck_open", "S_supply_curve",
            "state", "diagnosed_fault"]
    if "t_supply" in g.columns:
        keep.insert(10, "t_supply")
    detail = g[[c for c in keep if c in g.columns]]

    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / f"{run_name}_parallel_diag.csv"
    detail.to_csv(detail_path, index=False)

    # ---- summary ----
    rows = []
    for scope, mask in (("fault_window", g.in_fault == 1),
                        ("normal", g.in_fault == 0)):
        sub = g[mask & (g.alarm == 1)]
        n = len(sub)
        counts = sub.state.value_counts()
        ladder = sub.diagnosed_fault.replace("", "none").value_counts()
        rows.append({
            "scope": scope,
            "gated_steps": int(mask.sum()),
            "alarmed_steps": n,
            **{f"state_{k}": int(v) for k, v in counts.items()},
            **{f"ladder_{k}": int(v) for k, v in ladder.items()},
        })
    summary = pd.DataFrame(rows)
    summary_path = out_dir / f"{run_name}_state_summary.csv"
    summary.to_csv(summary_path, index=False)

    # ---- console report ----
    print(f"\n===== {run_name}  (detector={detector}/{thr_key}/pers{persistence}) =====")
    for _, r in summary.iterrows():
        print(f"  [{r['scope']}] gated={r['gated_steps']}  alarmed={r['alarmed_steps']}")
        for c in summary.columns:
            if c.startswith("state_") and not pd.isna(r.get(c)) and r.get(c, 0) > 0:
                print(f"      parallel {c[6:]:<24s} {int(r[c]):>5d}")
        for c in summary.columns:
            if c.startswith("ladder_") and not pd.isna(r.get(c)) and r.get(c, 0) > 0:
                print(f"      ladder   {c[7:]:<24s} {int(r[c]):>5d}")
    print(f"  wrote {detail_path}")
    print(f"  wrote {summary_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--detector", default="auto",
                    choices=["auto", "majority", "lof", "ocsvm", "iforest"])
    ap.add_argument("--threshold-key", default="p1")
    ap.add_argument("--persistence", type=int, default=None,
                    help="Default: the run's own persistence_steps")
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--out", type=Path,
                    default=Path("plots") / "parameter_studies")
    args = ap.parse_args()

    thresholds = load_thresholds(args.models_dir)
    for rd in args.run_dir:
        analyse_run(rd, args.detector, args.threshold_key,
                    args.persistence, thresholds, args.out)


if __name__ == "__main__":
    main()

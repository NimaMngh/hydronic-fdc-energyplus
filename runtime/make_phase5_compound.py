#!/usr/bin/env python3
"""
make_phase5_compound.py — compound-fault runner.

Derives Phase5_compound.py from Phase5_ML_compensate.py by applying a small
set of exact-match edits. The original runtime file is left untouched, so the
reference results stay reproducible from the unmodified Phase 5 runner.

The derived runner adds two config-driven capabilities:

1. "diagnosis_only": true
   Runs the full detection + signature-ladder diagnosis state machine and
   logs the ladder verdicts (diagnosed_fault column, console
   "DIAGNOSIS (log-only)" lines) without engaging compensation or writing
   any compensation actuator. Needed for the compound-fault runs, since in
   the stock runtime the diagnosis block is gated on compensation_enabled.

2. "fault_type": "compound_stuckopen_supplycurve"
   Injects the stuck-open thermostat bias (+fault_bias_C) and the
   supply-setpoint bias (supply_curve_bias_K) simultaneously during the
   same fault window. Internally the run behaves as a supply_curve run
   (actuator pathways, t_supply logging, extra-zone sensors) with the
   stuck-open injection added on top.

Each edit is matched exactly once. If Phase 5 has drifted from the baseline
these patterns were written against, the script aborts rather than emit a
silently wrong file.

Usage (from the project root):
    python runtime\\make_phase5_compound.py
    python runtime\\make_phase5_compound.py --src runtime\\Phase5_ML_compensate.py
"""
import argparse
import sys
from pathlib import Path

EDITS = [
    # ------------------------------------------------------------------
    # E1: __init__ — compound-mode + diagnosis-only flags
    # ------------------------------------------------------------------
    (
        """        self.config = config
        self.models = models""",
        """        self.config = config

        # ---- Compound-fault mode + log-only diagnosis ----
        _raw_ft = config.get('fault_type', '')
        self.compound_mode = (_raw_ft == 'compound_stuckopen_supplycurve')
        if self.compound_mode:
            # Base behaviour follows the supply_curve pathways (actuator
            # resolution, injection calling point, t_supply logging, extra
            # zone sensors); the stuck-open thermostat bias is injected
            # additionally in fault_injection_callback.
            config['fault_type'] = 'supply_curve'
            print('[compound] stuck_open + supply_curve compound mode ACTIVE')
        self.diagnosis_only = config.get('diagnosis_only', False)
        if self.diagnosis_only:
            print('[compound] diagnosis-only mode: ladder verdicts are '
                  'logged, no compensation is applied')

        self.models = models""",
    ),
    # ------------------------------------------------------------------
    # E2: setup_handles — resolve stuck-open actuators in compound mode
    # ------------------------------------------------------------------
    (
        """        if fault_type == 'stuck_open':
            self.fault_actuator = api.exchange.get_actuator_handle(
                state, \"Zone Temperature Control\", \"Heating Setpoint\",
                self.config['zone_name']
            )""",
        """        if fault_type == 'stuck_open' or self.compound_mode:
            self.fault_actuator = api.exchange.get_actuator_handle(
                state, \"Zone Temperature Control\", \"Heating Setpoint\",
                self.config['zone_name']
            )""",
    ),
    # ------------------------------------------------------------------
    # E3: fault_injection_callback — stuck-open branch fires in compound
    # ------------------------------------------------------------------
    (
        """        # ---- STUCK-OPEN: zone-level thermostat bias ----
        if fault_type == 'stuck_open':""",
        """        # ---- STUCK-OPEN: zone-level thermostat bias ----
        if fault_type == 'stuck_open' or self.compound_mode:""",
    ),
    # ------------------------------------------------------------------
    # E4: fault_injection_callback — supply_curve branch decoupled from
    #     the stuck-open branch (elif -> if) so both run in compound mode
    # ------------------------------------------------------------------
    (
        """        # ---- SUPPLY CURVE: system-level supply temperature bias ----
        elif fault_type == 'supply_curve':""",
        """        # ---- SUPPLY CURVE: system-level supply temperature bias ----
        if fault_type == 'supply_curve':""",
    ),
    # ------------------------------------------------------------------
    # E5: diagnosis gate — also runs in diagnosis-only mode
    # ------------------------------------------------------------------
    (
        """        if (self.comp_enabled and not self.comp_active
                and self.detect_streak >= self.persistence_steps):""",
        """        if ((self.comp_enabled or self.diagnosis_only)
                and not self.comp_active
                and self.detect_streak >= self.persistence_steps):""",
    ),
    # ------------------------------------------------------------------
    # E6: valid-diagnosis branch — log-only path before the comp path
    # ------------------------------------------------------------------
    (
        """            else:
                # Valid diagnosis -> activate compensation
                self.comp_active = True
                self.diagnosed_fault = candidate_fault
                self.unknown_streak = 0
                self.n_compensation_activations += 1""",
        """            elif self.diagnosis_only:
                # Log-only mode: record the ladder verdict
                # without engaging compensation or touching actuators.
                self.diagnosed_fault = candidate_fault
                self.unknown_streak = 0
                self.n_compensation_activations += 1
                if in_fault == 0:
                    self.n_false_comp_activations += 1
                self.detect_streak = 0  # re-arm: re-diagnose each window
                print(f\"  [{self.timestep_idx}] DIAGNOSIS (log-only)  \"
                      f\"verdict={candidate_fault}  \"
                      f\"te_inst={temp_error:+.3f}  \"
                      f\"te_2h={temp_error_2h_avg:+.3f}  \"
                      f\"fr={flow_ratio:.3f}\")
            else:
                # Valid diagnosis -> activate compensation
                self.comp_active = True
                self.diagnosed_fault = candidate_fault
                self.unknown_streak = 0
                self.n_compensation_activations += 1""",
    ),
    # ------------------------------------------------------------------
    # E7: clear the log-only verdict after the release window
    # ------------------------------------------------------------------
    (
        """        # Deactivation: sustained non-detection releases compensation""",
        """        # Log-only mode: clear the recorded verdict after the
        # release window so the log reflects distinct diagnosis episodes.
        if (self.diagnosis_only and self.diagnosed_fault is not None
                and self.clear_streak >= self.release_steps):
            self.diagnosed_fault = None

        # Deactivation: sustained non-detection releases compensation""",
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=None,
                    help="Path to Phase5_ML_compensate.py")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path (default: Phase5_compound.py next to src)")
    args = ap.parse_args()

    src = args.src
    if src is None:
        for cand in (Path("runtime") / "Phase5_ML_compensate.py",
                     Path("Phase5_ML_compensate.py")):
            if cand.exists():
                src = cand
                break
    if src is None or not src.exists():
        sys.exit("ERROR: Phase5_ML_compensate.py not found; pass --src.")

    out = args.out or src.with_name("Phase5_compound.py")
    text = src.read_text(encoding="utf-8")

    for i, (old, new) in enumerate(EDITS, 1):
        n = text.count(old)
        if n != 1:
            sys.exit(f"ERROR: edit E{i} matched {n} times (expected 1). "
                     f"Phase 5 has drifted from the patch baseline near:\n"
                     f"  {old.splitlines()[0][:70]!r}\n"
                     f"Nothing was written; re-check the edit pattern.")
        text = text.replace(old, new)

    banner = ("# " + "=" * 68 + "\n"
              "# GENERATED FILE - produced by make_phase5_compound.py from\n"
              "# Phase5_ML_compensate.py. Adds diagnosis_only mode and the\n"
              "# compound fault type 'compound_stuckopen_supplycurve'.\n"
              "# Do NOT edit this file; re-run the generator after changing\n"
              "# Phase 5.\n"
              "# " + "=" * 68 + "\n")
    out.write_text(banner + text, encoding="utf-8")
    print(f"wrote {out}  ({len(EDITS)} edits applied, all exact-match)")


if __name__ == "__main__":
    main()

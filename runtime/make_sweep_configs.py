#!/usr/bin/env python3
"""
make_sweep_configs.py — compensation-parameter sweeps.

Generates run-config JSONs by cloning the three existing comp configs and
changing ONLY the compensation parameters: the supply-temperature target for
the target sweep, the ramp gain for the gain sweep. No IDF edits are needed
for either, since target and alpha live entirely in the run config.

Usage (from the runtime/ directory that contains the base configs):
    python make_sweep_configs.py                      # target sweep
    python make_sweep_configs.py --sweep gain         # gain sweep, current targets
    python make_sweep_configs.py --sweep gain --targets 70 50 65
        # gain sweep once the target sweep has fixed the final targets
        # (order: stuck_closed stuck_open supply_curve)

Then run each config exactly as usual, e.g.:
    python Phase5_ML_compensate.py sweep_configs/<name>.json

Author : Nima Monghasemi
Date   : August 2026
"""
import argparse, copy, json
from pathlib import Path

BASE = {
    "stuck_closed": "run_config_comp_stuckclosed.json",
    "stuck_open":   "run_config_comp_stuckopen.json",
    "supply_curve": "run_config_comp_supplycurve.json",
}

# Supply-temperature targets per fault, at the reference severities.
# stuck_closed@70, stuck_open@50 and supply_curve@70 already have runs, so
# they are skipped by default to save compute.
TARGET_SWEEP = {
    "stuck_closed": [60.0, 65.0, 70.0],
    "stuck_open":   [50.0, 55.0, 60.0],
    "supply_curve": [60.0, 65.0, 70.0],
}
EXISTING_TARGETS = {("stuck_closed", 70.0), ("stuck_open", 50.0), ("supply_curve", 70.0)}

GAIN_SWEEP_ALPHAS = [0.1, 0.4]   # 0.2 is the reference gain


def load(fname):
    with open(fname, encoding="utf-8") as f:
        return json.load(f)


def emit(cfg, outdir, name):
    cfg = copy.deepcopy(cfg)
    cfg["run_name"] = name
    out = outdir / f"run_config_{name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=["target", "gain"], default="target")
    ap.add_argument("--targets", nargs=3, type=float, default=None,
                    metavar=("SC", "SO", "SUP"),
                    help="Final targets for the gain sweep: stuck_closed stuck_open supply_curve")
    ap.add_argument("--include-existing", action="store_true",
                    help="Target sweep: also emit configs for already-run reference targets")
    args = ap.parse_args()

    outdir = Path("sweep_configs")
    outdir.mkdir(exist_ok=True)

    if args.sweep == "target":
        for fault, targets in TARGET_SWEEP.items():
            base = load(BASE[fault])
            for t in targets:
                if (fault, t) in EXISTING_TARGETS and not args.include_existing:
                    print(f"skip {fault} target {t:g} (reference run exists)")
                    continue
                cfg = copy.deepcopy(base)
                cfg["compensation_config"]["targets"][fault] = t
                emit(cfg, outdir, f"{fault.replace('_','')}_comp_t{int(t)}")
    else:
        finals = dict(zip(["stuck_closed", "stuck_open", "supply_curve"],
                          args.targets)) if args.targets else None
        for fault in BASE:
            base = load(BASE[fault])
            for a in GAIN_SWEEP_ALPHAS:
                cfg = copy.deepcopy(base)
                if finals:
                    cfg["compensation_config"]["targets"][fault] = finals[fault]
                cfg["compensation_config"]["ramp_factor"] = a
                tgt = cfg["compensation_config"]["targets"][fault]
                emit(cfg, outdir,
                     f"{fault.replace('_','')}_comp_t{int(tgt)}_a{str(a).replace('.','')}")

    print("\nReminder: run each config with the unmodified Phase5 runtime; "
          "no IDF changes are needed for either sweep.")


if __name__ == "__main__":
    main()

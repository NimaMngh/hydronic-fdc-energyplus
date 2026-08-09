# Parameter studies

Run configurations and figure scripts for three studies that sit alongside the
main fault cases: a sweep over the compensation target, a sweep over the ramp
gain, and two compound-fault stress tests.

The two sweeps need no IDF changes. Both the target and the gain live entirely
in the run config, so the same three fault IDFs used for the reference runs are
reused unchanged.

## Contents

| Path | Purpose |
|---|---|
| `configs/target_sweep/` | Six configs sweeping the supply-temperature target. The reference targets (70 / 50 / 70 °C) already have runs and are not repeated here. |
| `configs/gain_sweep/` | Ramp gain at alpha = 0.1 and 0.4. Alpha = 0.2 is the reference used by the base runs. |
| `configs/compound_faults/` | Two simultaneous-fault configs: stuck-closed with supply-setpoint bias, and stuck-open with supply-setpoint bias. |
| `figures/` | Standalone figure scripts: literature positioning map, diagnosis-ladder diagram, recovery spectrum and target sensitivity. |
| `monthly_fpr_baseline.csv` | Month-by-month gated false-positive rate on the shared fault-free baseline. |

Two scripts referenced below live outside this folder:
`runtime/make_sweep_configs.py` regenerates the configs, and
`analysis/Phase10_sweep_pareto.py` computes the KPI table and the Pareto plots.

## Compensation-target sweep

1. Copy the `configs/target_sweep/*.json` files into `runtime/`, next to the
   existing configs and IDFs.
2. Run each one as an ordinary compensated run:

```bash
python runtime/Phase5_ML_compensate.py run_config_stuckopen_comp_t55.json
```

   Six runs in total: stuck-closed at 60 and 65 °C, stuck-open at 55 and
   60 °C, supply-setpoint at 60 and 65 °C. The stuck-closed t60 case is
   degenerate — the target equals the 60 °C baseline — so it costs little and
   bounds the Pareto front from below.

3. Analyse from the project root, which must contain `runs/` and
   `table_ablation_kpis.csv`:

```bash
python analysis/Phase10_sweep_pareto.py
```

   This writes `sweep_results/sweep_kpis.csv` plus one Pareto plot per fault
   type — recovery % against energy-vs-baseline %, coloured by the
   return-temperature shift J_R, with the non-dominated front marked.

4. Pick the final target per fault in the order comfort, then energy, then J_R.

The window construction and KPI arithmetic in the Pareto script match
`analysis/Phase6_cross_fault_synthesis.py`: feeding the baseline log through
it reproduces the baseline row of `table_ablation_kpis.csv` to full
floating-point precision.

## Ramp-gain sweep

If the target sweep keeps the 70 / 50 / 70 targets, the supplied `a01` and
`a04` configs are ready to run as they are. If the targets change, regenerate
them first:

```bash
python runtime/make_sweep_configs.py --sweep gain --targets <SC> <SO> <SUP>
```

Then run the six configs and re-run the Pareto analysis. Alpha values are
annotated automatically on the plots.

## Compound faults

The stuck-closed case combines the EMS flow restriction with the
supply-setpoint bias and reuses the existing machinery:
`RetailStandalone_stuckclosed_detect.idf` with `fault_type = "supply_curve"`
injects both at once.

The stuck-open case needs two simultaneous Python injections and the
parallel-signature diagnosis logging, which the stock Phase 5 runner does not
provide. Generate the extended runner first:

```bash
python runtime/make_phase5_compound.py
```

That derives `runtime/Phase5_compound.py` from the unmodified Phase 5 runner,
adding a `diagnosis_only` mode and the `compound_stuckopen_supplycurve` fault
type. Analyse the results with `analysis/Phase12_compound_diagnosis.py`, which
evaluates all three signatures in parallel and reports an explicit abstain
state where they conflict.

## Monthly false-positive rates

`monthly_fpr_baseline.csv` holds the per-month gated FPR on the shared
baseline log. The majority-vote columns reproduce the published sweep exactly
(24 flags out of 13,289 gated steps). The LOF p1/persistence-6 column shows
the winter concentration, peaking at 3.7 % in December and 8.1 % in January.

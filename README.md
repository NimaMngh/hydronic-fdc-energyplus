# Integrated fault detection and supervisory compensation for hydronic radiator faults in district-heated buildings

Code, models and simulation inputs for the study of the same name, submitted
to *Results in Engineering*.

Most fault detection and diagnostics work stops at identifying that something
is wrong. This study closes the loop: three unsupervised detectors run inside
the EnergyPlus solver through the Python API, a signature ladder decides which
fault they are seeing, and a supervisory controller adjusts the
district-heating supply setpoint in response — all within the same simulation,
so the corrected behaviour feeds back into the physics rather than being
estimated afterwards.

The detectors (one-class SVM, isolation forest, local outlier factor) are
trained only on a fault-free heating season, so no faulty data is needed to
commission them. Across 48 simulations the framework recovers a
fault-dependent share of the discomfort a fault causes, which is the study's
main point: control-type faults can be compensated automatically, while
physically constrained faults hit a hydraulic ceiling and are better routed to
maintenance.

## Results

Supervisory recovery of fault-induced discomfort degree-hours, with the change
in delivered hydronic energy:

| Fault | Recovery | Energy vs. fault-free baseline | Note |
|---|---|---|---|
| Stuck-open valve (over-delivery) | 68.8 % | +23.1 % | 15 % less energy than the uncompensated fault |
| Supply-setpoint bias | 57.2 % | -0.8 % | compensation raises the return-water temperature |
| Stuck-closed valve | 38.6 % | -7.1 % | hydraulic ceiling — flow, not supply temperature, is the constraint |

Best detection configuration per fault, against a shared fault-free baseline
of 13,289 gated timesteps:

| Fault | Detector | F1 | Recall | Latency |
|---|---|---|---|---|
| Stuck-closed valve | majority vote, p1, persistence 3 | 0.90 | 1.00 | 20 min |
| Stuck-open valve | majority vote, p1, persistence 4 | 0.86 | 0.92 | 100 min |
| Supply-setpoint bias | LOF, p1, persistence 6 | 0.52 | 0.98 | 550 min |

The supply-setpoint bias is the hard case. Recall is high but precision is
not, because a slow system-wide drift resembles ordinary operation to a
detector trained on zone-level features. That is also why its diagnosis path
reads the supply temperature directly instead of inferring the fault from zone
behaviour alone.

## Model

The building is the ASHRAE 90.1-2022 RetailStandalone prototype from the DOE
commercial reference set. Its packaged single-zone air conditioners have been
removed along with all mechanical cooling; each zone is now heated by a
hydronic baseboard fed from a plant loop with a district-heating source, and
keeps an energy-recovery ventilator so ventilation stays independent of
heating. Weather is Västerås, Sweden. Runs cover the October–April heating
season on a 10-minute timestep, which leaves 13,289 gated timesteps in the
fault-free baseline.

`Core_Retail` is the instrumented zone. The other zones are logged as well,
because the diagnosis ladder uses simultaneous under-delivery across zones to
separate a system-level supply fault from a local valve fault.

## Faults

| Fault | Injected by | Signature used to diagnose it |
|---|---|---|
| Stuck-closed valve | Erl EMS program in the IDF, capping coil mass flow | low flow ratio with a negative temperature error |
| Stuck-open valve | Python API, biasing the zone thermostat | positive temperature error at normal flow |
| Supply-setpoint bias | Python API, biasing the plant supply setpoint | supply temperature below baseline, normal flow, several zones under-delivering |

The severity sweep varies each one over five levels: flow fractions from 0.10
to 0.70 for the stuck-closed valve, thermostat bias from 0.5 to 6 K, and
supply-setpoint bias from 2 to 15 K.

## Requirements

EnergyPlus 25.1.0, installed separately. Its Python API (`pyenergyplus`) ships
with the installation and cannot be obtained from conda or PyPI, so the
runtime scripts add the EnergyPlus directory to `sys.path` before importing it.
Adjust that path if EnergyPlus lives somewhere other than the default.

Everything else comes from the environment file:

```bash
conda env create -f environment.yml
conda activate FDC
```

scikit-learn is pinned deliberately. The detectors in `models/` are joblib
pickles and will complain, or quietly misbehave, if they are loaded under a
different minor version.

## Layout

| Path | Contents |
|---|---|
| `Offline_ML_training.py` | Trains the three detectors on the fault-free log and calibrates percentile thresholds |
| `runtime/` | In-the-loop runners and their run configs — Phase 4 detects only, Phase 5 adds diagnosis and compensation |
| `analysis/` | Post-hoc threshold and persistence sweeps, cross-fault synthesis, manuscript figures |
| `severity/` | Severity-sweep IDFs, configs and analysis |
| `parameter_studies/` | Compensation-target and ramp-gain sweeps, and the compound-fault stress tests |
| `models/` | Trained detectors, scaler, thresholds and the training record |
| `results/` | Derived KPI tables, in CSV and LaTeX form |
| `*.idf` | The baseline model and one variant per fault |

Simulation output is not committed. Each run writes a roughly 7 MB
`fdc_runtime_log.csv` into `runs/`, which `.gitignore` excludes; the analysis
scripts read those logs, and the derived tables in `results/` are what the
repository carries instead.

The article's figures are not included either. The scripts that draw them are
here in `analysis/` and `parameter_studies/figures/` so every figure can be
regenerated from the tables in `results/` and the run logs.

## Running it

Train the detectors, then run a fault case:

```bash
python Offline_ML_training.py
python runtime/Phase4_ML_in_loop_detect.py runtime/run_config_detect_stuckclosed.json
python runtime/Phase5_ML_compensate.py runtime/run_config_comp_stuckclosed.json
```

Phase 4 logs anomaly scores without acting on them, which is what the
threshold sweeps in `analysis/` consume. Phase 5 additionally diagnoses and
compensates: it ramps the supply setpoint from the 60 °C baseline towards a
per-fault target of 70, 50 and 70 °C, at a gain of 0.2 per timestep.

Compensation writes to the same `HW-Loop-Temp-Schedule` actuator that the
supply-setpoint fault uses. EnergyPlus permits one owner per actuator, so an
IDF used with Phase 5 must not declare an `EnergyManagementSystem:Actuator`
for that schedule, and its Erl compensation program must be disabled. The
fault-injection EMS objects stay, since the stuck-closed fault is injected
from the IDF.

The severity IDFs are generated rather than edited by hand:

```bash
python severity/idfs/patch_idf_flow_fraction.py --batch --project-root . --output-dir severity/idfs
```

## Provenance and licence

The prototype building model was developed by Pacific Northwest National
Laboratory for the US Department of Energy; the original citations are kept in
the comment header of each IDF file. Modifications to the HVAC system, the EMS
fault-injection logic, and all Python code are covered by the MIT licence in
`LICENSE`.

## Citation

Monghasemi, N., Vouros, S., Kyprianidis, K., Vadiee, A. *Integrated fault
detection and supervisory compensation for hydronic radiator faults in
district-heated buildings.* Submitted to Results in Engineering.

The software itself is archived on Zenodo. Use the concept DOI
`10.5281/zenodo.21862959`, which always resolves to the latest version.
`CITATION.cff` carries both citations in machine-readable form.

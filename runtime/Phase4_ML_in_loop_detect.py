# -*- coding: utf-8 -*-
"""
Phase 4: ML-in-the-loop Detection + Logging (No Compensation)
Runs EnergyPlus Library mode, loads trained artifacts, computes features online,
and logs sensors / features / scores / flags to CSV.

Supports three fault types:
  - stuck_open:     Zone-level thermostat bias via Python API
  - stuck_closed:   Zone-level flow restriction via Erl EMS (IDF-based)
  - supply_curve:   System-level supply temperature bias via Python API

Author:  Nima Monghasemi
Date:    February 2026
"""

import sys
import os
import json
import shutil
import argparse
from datetime import datetime
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, r"C:\EnergyPlusV25-1-0")
from pyenergyplus.api import EnergyPlusAPI


class MLRuntimeLogger:
    def __init__(self, config: dict, models: dict, scaler, thresholds: dict, training_cfg: dict):
        self.config = config
        self.models = models
        self.scaler = scaler
        self.thresholds = thresholds
        self.training_cfg = training_cfg

        # Feature engineering parameters
        self.feature_cols = training_cfg['feature_columns']
        self.roll_window = training_cfg['rolling_window_steps']
        self.peak_flow = training_cfg['peak_flow_kg_s']
        self.occupied_threshold = training_cfg['occupied_sp_threshold_C']
        self.min_flow = training_cfg['min_flow_threshold_kg_s']

        # State buffers
        self.temp_error_buffer = deque(maxlen=self.roll_window)
        self.prev_zone_temp = None
        self.was_active_last_step = False
        self.timestep_idx = 0
        self.log_rows = []

        # API Handles
        self.handles = {}
        self.handles_ready = False
        self.fault_actuator = -1       # stuck_open: heating setpoint
        self.cooling_actuator = -1     # stuck_open: cooling guard
        self.supply_temp_actuator = -1 # supply_curve: node temp SP
        self.sched_handle = -1
        self.sched_handle_no_opt = -1

        # Extra zone handles for system-level fault logging
        self.extra_zone_handles = {}

    def setup_handles(self, state) -> bool:
        """Resolve variable and actuator handles once API data is ready."""
        if self.handles_ready:
            return True
        if not api.exchange.api_data_fully_ready(state):
            return False

        vm = self.config['var_mapping']
        all_ok = True

        # --- Primary zone sensors ---
        for key, (var_type, var_name) in vm.items():
            h = api.exchange.get_variable_handle(state, var_type, var_name)
            if h == -1:
                print(f"ERROR: Handle failed for {key}: '{var_type}' '{var_name}'")
                all_ok = False
            self.handles[key] = h

        # --- Fault-type-specific actuators ---
        fault_type = self.config.get('fault_type', '')

        if fault_type == 'stuck_open':
            # Zone thermostat override
            self.fault_actuator = api.exchange.get_actuator_handle(
                state, "Zone Temperature Control", "Heating Setpoint",
                self.config['zone_name']
            )
            if self.fault_actuator == -1:
                print(f"ERROR: Actuator failed for Heating Setpoint on {self.config['zone_name']}")
                all_ok = False

            self.cooling_actuator = api.exchange.get_actuator_handle(
                state, "Zone Temperature Control", "Cooling Setpoint",
                self.config['zone_name']
            )
            if self.cooling_actuator == -1:
                print(f"WARNING: Cooling setpoint actuator not found. "
                      f"Stuck-open bias may be limited by cooling setpoint conflict.")

        elif fault_type == 'supply_curve':
            # Check if config forces the schedule override path
            force_sched = self.config.get('force_schedule_override', False)
            
            if not force_sched:
                # Try the node setpoint actuator first
                supply_node = self.config.get('supply_node_name', 'HW Supply Outlet Node')
                self.supply_temp_actuator = api.exchange.get_actuator_handle(
                    state,
                    "System Node Setpoint",
                    "Temperature Setpoint",
                    supply_node
                )
                if self.supply_temp_actuator != -1:
                    print(f"  Supply node actuator resolved: System Node Setpoint / "
                          f"Temperature Setpoint / {supply_node}")
                    print(f"  WARNING: This actuator may not be respected by the plant solver.")
                    print(f"  If supply temp stays at baseline, re-run with force_schedule_override=true")
                    self.config['_supply_fault_via_schedule'] = False
                else:
                    print(f"  Node actuator failed. Falling back to schedule override.")
                    force_sched = True  # Fall through to schedule path
            
            if force_sched:
                self.supply_temp_actuator = api.exchange.get_actuator_handle(
                    state,
                    "Schedule:Compact",
                    "Schedule Value",
                    "HW-Loop-Temp-Schedule"
                )
                if self.supply_temp_actuator == -1:
                    print(f"ERROR: Schedule actuator failed for HW-Loop-Temp-Schedule!")
                    print(f"  Make sure the EMS actuator declaration for this schedule")
                    print(f"  has been REMOVED from the IDF (actuator ownership rule).")
                    all_ok = False
                else:
                    print(f"  Schedule override actuator resolved: Schedule:Compact / "
                          f"Schedule Value / HW-Loop-Temp-Schedule")
                    self.config['_supply_fault_via_schedule'] = True


        # --- Extra zone sensors (for multi-zone logging) ---
        extra_zones = self.config.get('extra_zone_sensors', {})
        for zone_label, zone_vm in extra_zones.items():
            self.extra_zone_handles[zone_label] = {}
            for key, (var_type, var_name) in zone_vm.items():
                h = api.exchange.get_variable_handle(state, var_type, var_name)
                if h == -1:
                    print(f"WARNING: Extra zone handle failed: {zone_label}/{key}: "
                          f"'{var_type}' '{var_name}'")
                    # Non-fatal: we log NaN for missing extra-zone sensors
                self.extra_zone_handles[zone_label][key] = h

        # --- Intended schedule handle (for temp_error computation) ---
        self.sched_handle = api.exchange.get_variable_handle(
            state, "Schedule Value", "HTGSETP_SCH_YES_OPTIMUM"
        )
        if self.sched_handle == -1:
            print("ERROR: Variable handle failed for Schedule Value HTGSETP_SCH_YES_OPTIMUM")
            all_ok = False
            
        self.sched_handle_no_opt = api.exchange.get_variable_handle(
            state, "Schedule Value", "HTGSETP_SCH_NO_OPTIMUM"
        )
        if self.sched_handle_no_opt == -1:
            print("WARNING: Variable handle failed for Schedule Value HTGSETP_SCH_NO_OPTIMUM")

        if not all_ok:
            api.runtime.stop_simulation(state)
            return False

        print(f"[{self.timestep_idx}] All handles resolved. Fault type: '{fault_type}'")
        n_extra = sum(len(zh) for zh in self.extra_zone_handles.values())
        print(f"  Extra zone sensors: {n_extra} across {len(self.extra_zone_handles)} zones")
        self.handles_ready = True
        return True

    def fault_injection_callback(self, state):
        """Runs BEFORE the heat balance init to inject faults directly into the solver."""
        if api.exchange.warmup_flag(state):
            return
        if not self.setup_handles(state):
            return

        month = api.exchange.month(state)
        day = api.exchange.day_of_month(state)
        hour = api.exchange.hour(state)

        # Check if we are inside the fault window
        fw = self.config.get('fault_window', {})
        in_fault = False
        if fw:
            if (fw['start_month'] <= month <= fw['end_month'] and
                fw['start_day'] <= day <= fw['end_day'] and
                fw['start_hour'] <= hour < fw['end_hour']):
                in_fault = True

        fault_type = self.config.get('fault_type', '')

        # ================================================================
        #  STUCK-OPEN: zone-level thermostat bias
        # ================================================================
        if fault_type == 'stuck_open':
            if in_fault:
                base_sp = api.exchange.get_variable_value(state, self.sched_handle)
                bias = self.config.get('fault_bias_C', 2.0)
                biased_sp = base_sp + bias
                api.exchange.set_actuator_value(state, self.fault_actuator, biased_sp)
                if self.cooling_actuator != -1:
                    cooling_guard = biased_sp + 0.5
                    api.exchange.set_actuator_value(state, self.cooling_actuator, cooling_guard)
            else:
                api.exchange.reset_actuator(state, self.fault_actuator)
                if self.cooling_actuator != -1:
                    api.exchange.reset_actuator(state, self.cooling_actuator)

        # ================================================================
        #  SUPPLY CURVE FAULT: system-level supply temperature bias
        # ================================================================
        elif fault_type == 'supply_curve':
            if in_fault:
                baseline_supply = self.config.get('baseline_supply_temp_C', 60.0)
                bias = self.config.get('supply_curve_bias_K', -8.0)
                faulty_supply = baseline_supply + bias

                api.exchange.set_actuator_value(
                    state, self.supply_temp_actuator, faulty_supply
                )
            else:
                api.exchange.reset_actuator(state, self.supply_temp_actuator)

        # ================================================================
        #  STUCK-CLOSED: handled by Erl EMS in the IDF — no Python action
        # ================================================================
        # (fault_type == 'stuck_closed' → nothing to do here)

    def logging_callback(self, state):
        """Runs AFTER Zone Reporting to capture sensors and run ML inference."""
        if api.exchange.warmup_flag(state):
            return
        if not self.setup_handles(state):
            return

        self.timestep_idx += 1

        # Read raw sensors (primary zone)
        s = {key: api.exchange.get_variable_value(state, h)
             for key, h in self.handles.items()}
        intended_sp = api.exchange.get_variable_value(state, self.sched_handle)
        
        intended_sp_no_opt = api.exchange.get_variable_value(state, self.sched_handle_no_opt)

        month, day, hour, minute = (
            api.exchange.month(state), api.exchange.day_of_month(state),
            api.exchange.hour(state), api.exchange.minutes(state)
        )

        # ---- Determine fault-active flag for logging ----
        fw = self.config.get('fault_window', {})
        in_fault = 0
        if fw:
            if (fw['start_month'] <= month <= fw['end_month'] and
                fw['start_day'] <= day <= fw['end_day'] and
                fw['start_hour'] <= hour < fw['end_hour']):
                in_fault = 1

        # ---- Feature Engineering (primary zone) ----
        zone_temp = s['zone_temp']
        m_dot = s['m_dot']

        temp_error = zone_temp - intended_sp
        flow_ratio = (m_dot / self.peak_flow) if self.peak_flow > 0 else 0.0
        delta_T_hw = s['t_inlet'] - s['t_outlet']

        occupied = 1 if (intended_sp >= self.occupied_threshold) else 0
        heating_active = 1 if (m_dot > self.min_flow) else 0

        # Buffer flush on daily startup
        is_active_now = (occupied == 1 and heating_active == 1)
        if is_active_now and not self.was_active_last_step:
            self.temp_error_buffer.clear()
            self.prev_zone_temp = None

        self.was_active_last_step = is_active_now

        if is_active_now:
            self.temp_error_buffer.append(temp_error)
            dT_zone_dt = (zone_temp - self.prev_zone_temp
                          if self.prev_zone_temp is not None else np.nan)
            self.prev_zone_temp = zone_temp
        else:
            dT_zone_dt = np.nan
            self.prev_zone_temp = None

        temp_error_2h_avg = (np.mean(self.temp_error_buffer)
                             if len(self.temp_error_buffer) >= self.roll_window
                             else np.nan)

        feature_ready = 1 if (len(self.temp_error_buffer) >= self.roll_window
                              and self.prev_zone_temp is not None
                              and not np.isnan(dT_zone_dt)) else 0

        # ---- ML Inference ----
        scores = {'ocsvm': np.nan, 'iforest': np.nan, 'lof': np.nan}
        flags = {'ocsvm': 0, 'iforest': 0, 'lof': 0}

        if feature_ready and is_active_now:
            X_raw = np.array([[temp_error, flow_ratio, temp_error_2h_avg,
                               dT_zone_dt, s['t_outdoor'], delta_T_hw]])
            X_scaled = self.scaler.transform(X_raw)

            scores['ocsvm'] = float(self.models['ocsvm'].decision_function(X_scaled)[0])
            scores['iforest'] = float(self.models['iforest'].score_samples(X_scaled)[0])
            scores['lof'] = float(self.models['lof'].score_samples(X_scaled)[0])

            thresh_key = self.config.get('threshold_key', 'default')
            for mname in ['ocsvm', 'iforest', 'lof']:
                if scores[mname] < self.thresholds[mname][thresh_key]:
                    flags[mname] = 1

        # ---- Build log row ----
        row = {
            'timestep': self.timestep_idx,
            'month': month, 'day': day, 'hour': hour, 'minute': minute,
            'zone_temp': zone_temp, 'htg_sp': s['htg_sp'],
            'intended_sp': intended_sp, 'm_dot': m_dot,
            't_inlet': s['t_inlet'], 't_outlet': s['t_outlet'],
            't_outdoor': s['t_outdoor'],
            'in_fault': in_fault,
            'temp_error': temp_error, 'flow_ratio': flow_ratio,
            'temp_error_2h_avg': temp_error_2h_avg,
            'dT_zone_dt': dT_zone_dt, 'delta_T_hw': delta_T_hw,
            'occupied': occupied, 'heating_active': heating_active,
            'feature_ready': feature_ready,
            'score_ocsvm': scores['ocsvm'], 'score_iforest': scores['iforest'],
            'score_lof': scores['lof'],
            'flag_ocsvm': flags['ocsvm'], 'flag_iforest': flags['iforest'],
            'flag_lof': flags['lof']
        }

        # ---- Supply temperature (always log if handle exists) ----
        if 't_supply' in self.handles:
            row['t_supply'] = s.get('t_supply', np.nan)

        # ---- Extra zone sensors (multi-zone logging) ----
        for zone_label, zone_handles in self.extra_zone_handles.items():
            for key, h in zone_handles.items():
                col_name = f"{zone_label}_{key}"
                if h != -1:
                    row[col_name] = api.exchange.get_variable_value(state, h)
                else:
                    row[col_name] = np.nan
            
            # Extra zones follow the non-optimum schedule, not the primary one
            row[f"{zone_label}_intended_sp"] = intended_sp_no_opt

        self.log_rows.append(row)

    def save(self, run_dir: str):
        df = pd.DataFrame(self.log_rows)
        csv_path = Path(run_dir) / "fdc_runtime_log.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nRuntime log saved: {csv_path}")
        return df


def validate_results(df: pd.DataFrame, training_cfg: dict, config: dict):
    """Post-run validation with fault-type-aware diagnostics."""
    print("\n" + "=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)
    print(f"1. Total timesteps logged: {len(df):,}")
    if len(df) == 0:
        return

    roll_window = training_cfg['rolling_window_steps']
    early = df.head(roll_window + 2)
    print(f"2. Warm-up check: feature_ready == 0 count in first "
          f"{roll_window + 2} rows: {(early['feature_ready'] == 0).sum()} "
          f"(expected ~{roll_window})")

    gated = df[(df['occupied'] == 1) & (df['heating_active'] == 1)
               & (df['feature_ready'] == 1)]
    print(f"\n3. Gated population: {len(gated):,} rows")
    if len(gated) == 0:
        print("   WARNING: No gated rows found!")
        return

    print(f"\n4. Flagged rates under '{config.get('threshold_key', 'default')}' threshold:")
    for m in ['ocsvm', 'iforest', 'lof']:
        print(f"   {m:12s}: {gated[f'flag_{m}'].mean():>6.2%}  "
              f"(n={gated[f'flag_{m}'].sum()})")

    # ---- Fault-window-specific diagnostics ----
    if 'in_fault' in df.columns:
        fault_rows = gated[gated['in_fault'] == 1]
        normal_rows = gated[gated['in_fault'] == 0]

        print(f"\n5. Fault window analysis:")
        print(f"   Fault-active gated rows:  {len(fault_rows):,}")
        print(f"   Normal gated rows:        {len(normal_rows):,}")

        if len(fault_rows) > 0:
            print(f"\n   During fault window (primary zone):")
            print(f"     Avg zone temp:    {fault_rows['zone_temp'].mean():.2f} °C")
            print(f"     Avg intended SP:  {fault_rows['intended_sp'].mean():.2f} °C")
            print(f"     Avg temp_error:   {fault_rows['temp_error'].mean():.3f} °C")
            print(f"     Avg flow rate:    {fault_rows['m_dot'].mean():.4f} kg/s")
            if 't_supply' in fault_rows.columns:
                print(f"     Avg supply temp:  {fault_rows['t_supply'].mean():.2f} °C")
            print(f"     Avg delta_T_hw:   {fault_rows['delta_T_hw'].mean():.2f} °C")

            print(f"\n   Detection rates during fault window:")
            for m in ['ocsvm', 'iforest', 'lof']:
                if len(fault_rows) > 0:
                    recall = fault_rows[f'flag_{m}'].mean()
                    print(f"     {m:12s}: {recall:>6.2%}  "
                          f"({fault_rows[f'flag_{m}'].sum()}/{len(fault_rows)})")

            # Extra zone impact summary
            extra_zones = config.get('extra_zone_sensors', {})
            if extra_zones:
                print(f"\n   Multi-zone impact during fault window:")
                for zone_label in extra_zones:
                    zt_col = f"{zone_label}_zone_temp"
                    sp_col = f"{zone_label}_intended_sp"
                    md_col = f"{zone_label}_m_dot"
                    if zt_col in fault_rows.columns:
                        z_temp = fault_rows[zt_col].mean()
                        z_sp = fault_rows[sp_col].mean() if sp_col in fault_rows.columns else np.nan
                        z_flow = fault_rows[md_col].mean() if md_col in fault_rows.columns else np.nan
                        z_err = z_temp - z_sp if not np.isnan(z_sp) else np.nan
                        print(f"     {zone_label:20s}:  Tavg={z_temp:.2f}°C  "
                              f"SP={z_sp:.2f}°C  err={z_err:+.2f}°C  "
                              f"mdot={z_flow:.4f} kg/s")

        if len(normal_rows) > 0:
            print(f"\n   False positive rates (normal periods):")
            for m in ['ocsvm', 'iforest', 'lof']:
                fpr = normal_rows[f'flag_{m}'].mean()
                print(f"     {m:12s}: {fpr:>6.2%}  "
                      f"({normal_rows[f'flag_{m}'].sum()}/{len(normal_rows)})")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: ML Detection in EnergyPlus loop"
    )
    parser.add_argument("config", help="Path to run config json")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:         
        config = json.load(f)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = Path(config['output_root']) / f"{timestamp}_{config['run_name']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / Path(args.config).name)

    mdir = Path(config['models_dir'])
    with open(mdir / "training_config.json", encoding='utf-8') as f:  
            training_cfg = json.load(f)
    models = {
        'ocsvm': joblib.load(mdir / "ocsvm.joblib"),
        'iforest': joblib.load(mdir / "iforest.joblib"),
        'lof': joblib.load(mdir / "lof.joblib")
    }
    scaler = joblib.load(mdir / "scaler.joblib")
    thresholds = joblib.load(mdir / "thresholds.joblib")

    global api
    api = EnergyPlusAPI()
    state = api.state_manager.new_state()

    # Request primary zone variables
    for key, (vtype, vname) in config['var_mapping'].items():
        api.exchange.request_variable(state, vtype, vname)
        
    api.exchange.request_variable(state, "Schedule Value", "HTGSETP_SCH_YES_OPTIMUM")
    api.exchange.request_variable(state, "Schedule Value", "HTGSETP_SCH_NO_OPTIMUM")

    # Request extra zone variables
    for zone_label, zone_vm in config.get('extra_zone_sensors', {}).items():
        for key, (vtype, vname) in zone_vm.items():
            api.exchange.request_variable(state, vtype, vname)

    logger = MLRuntimeLogger(config, models, scaler, thresholds, training_cfg)

    # Fault injection: fires once per zone timestep, before heat balance init
    api.runtime.callback_begin_zone_timestep_before_init_heat_balance(
        state, logger.fault_injection_callback
    )
    # Logging: fires after zone reporting
    api.runtime.callback_end_zone_timestep_after_zone_reporting(
        state, logger.logging_callback
    )

    exit_code = api.runtime.run_energyplus(state, [
        "-w", str(config['epw_path']),
        "-d", str(run_dir),
        str(config['idf_path'])
    ])

    print(f"\nEnergyPlus exit code: {exit_code}")
    df = logger.save(run_dir)
    validate_results(df, training_cfg, config)


if __name__ == "__main__":
    main()
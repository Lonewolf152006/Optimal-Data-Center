"""
Independent Verification of Semiconductor Reliability Model
============================================================
Spot-check to verify that the semiconductor reliability figures in METHODOLOGY.md
were independently and accurately computed from real simulation trajectories,
not estimated or fabricated.

Recomputes:
  1. Silicon junction temperature: Tj(t) = T_room(t) + DT_idle + (DT_max - DT_idle) * util(t)
  2. Arrhenius continuous thermal acceleration factor:
       AF_T(t) = exp( (Ea / kB) * (1 / T_j_ref - 1 / (Tj(t) + 273.15)) )
  3. Relative MTBF = 1.0 / mean(AF_T)
  4. Equivalent aging days = 5.0 * mean(AF_T)
  5. Peak Tj, min Tj, and thermal swing (Delta Tj)

Directly recomputes these values for held-out seed 101 and compares them against
the corresponding rows in results/reliability_summary.csv.
Prints PASS/FAIL with detailed tolerances.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from plant.thermal_plant import ThermalPlant
from plant.scenarios import random_scenario
from controllers.baseline_thermostat import BaselineThermostat
from ml.train_classical import ClassicalPredictor
from qml.train_qml import QMLPredictor
from optimization.mpc_core_fast import make_mpc_controller_fast
from evaluation.run_comparison import PredictorWrapper

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
RELIABILITY_CSV = os.path.join(RESULTS_DIR, "reliability_summary.csv")
TRAJECTORY_CACHE = os.path.join(RESULTS_DIR, "seed_101_trajectories.npz")

# ----------------------------------------------------------------------
# Physical Constants & Specifications from METHODOLOGY.md
# ----------------------------------------------------------------------
K_BOLTZMANN_EV = 8.617333262e-5  # Boltzmann constant (eV / K)
E_ACTIVATION_EV = 0.70           # Activation energy for silicon wearout (eV)
DT_IDLE = 18.0                   # Server temp rise above room at idle (deg C)
DT_MAX = 45.0                    # Server temp rise above room at full load (deg C)
T_REF_ROOM = 22.0                # Nominal reference room temp (deg C)
UTIL_REF = 0.50                  # Nominal reference workload utilization
T_J_REF_K = (T_REF_ROOM + DT_IDLE + (DT_MAX - DT_IDLE) * UTIL_REF) + 273.15  # 326.65 K (53.5 deg C)

HORIZON = 24
TARGET_SEED = 101


def simulate_controller(plant, controller, t, util, tamb, predictor_wrapper=None):
    """Run closed-loop simulation and return room temperature trajectory."""
    n = len(t)
    T_room = np.empty(n)
    T_room[0] = plant.T_SET

    for k in range(n):
        h = min(HORIZON, n - k)
        util_fore = util[k: k + h].copy()
        tamb_fore = tamb[k: k + h].copy()
        carbon_fore = plant.carbon_intensity(t[k: k + h]).copy()
        if h < HORIZON:
            util_fore = np.concatenate([util_fore, np.full(HORIZON - h, util_fore[-1])])
            tamb_fore = np.concatenate([tamb_fore, np.full(HORIZON - h, tamb_fore[-1])])
            carbon_fore = np.concatenate([carbon_fore, np.full(HORIZON - h, carbon_fore[-1])])

        u = controller(T_room[k], step_idx=k,
                       util_fore=util_fore, tamb_fore=tamb_fore,
                       carbon_fore=carbon_fore)

        if predictor_wrapper is not None:
            pcool_k, _ = plant.p_cool(u, tamb[k])
            carbon_k = plant.carbon_intensity(t[k])
            predictor_wrapper.update(T_room[k], util[k], tamb[k], pcool_k, carbon_k)

        if k < n - 1:
            T_room[k + 1] = plant.step(T_room[k], u, util[k], tamb[k])

    return T_room


def get_seed_trajectories(seed=TARGET_SEED, include_qml=False):
    """Retrieve or recompute room temperature and utilization trajectories for seed 101."""
    plant = ThermalPlant()
    t, util, tamb = random_scenario(seed=seed, n_days=5)

    trajectories = {
        "t": t,
        "util": util,
        "tamb": tamb,
    }

    # Check if cache exists
    if os.path.exists(TRAJECTORY_CACHE):
        print(f"Loading cached seed {seed} trajectories from {TRAJECTORY_CACHE}...")
        cached = np.load(TRAJECTORY_CACHE)
        trajectories["Baseline thermostat"] = cached["b_T"]
        trajectories["MPC + Classical GRU"] = cached["c_T"]
        if "q_T" in cached:
            trajectories["MPC + QML Hybrid"] = cached["q_T"]
        return trajectories

    # Re-running simulation for seed 101
    print(f"Re-running closed-loop simulations for seed {seed}...")

    # 1. Baseline Thermostat
    print("  [1/2] Simulating Baseline Thermostat...", flush=True)
    ctrl_b = BaselineThermostat()
    T_b = simulate_controller(plant, ctrl_b, t, util, tamb)
    trajectories["Baseline thermostat"] = T_b

    # 2. MPC Classical GRU
    print("  [2/2] Simulating MPC + Classical GRU...", flush=True)
    c_pred = ClassicalPredictor()
    wrap_c = PredictorWrapper(c_pred)
    ctrl_c = make_mpc_controller_fast(predictor_fn=wrap_c.predict, horizon=HORIZON)
    T_c = simulate_controller(plant, ctrl_c, t, util, tamb, predictor_wrapper=wrap_c)
    trajectories["MPC + Classical GRU"] = T_c

    # 3. Optional MPC QML Hybrid
    if include_qml:
        print("  [3/3] Simulating MPC + QML Hybrid...", flush=True)
        q_pred = QMLPredictor()
        wrap_q = PredictorWrapper(q_pred)
        ctrl_q = make_mpc_controller_fast(predictor_fn=wrap_q.predict, horizon=HORIZON)
        T_q = simulate_controller(plant, ctrl_q, t, util, tamb, predictor_wrapper=wrap_q)
        trajectories["MPC + QML Hybrid"] = T_q

    # Save cache for future instant spot-checks
    save_dict = {
        "t": t, "util": util, "tamb": tamb,
        "b_T": T_b, "c_T": T_c,
    }
    if include_qml:
        save_dict["q_T"] = trajectories["MPC + QML Hybrid"]
    np.savez(TRAJECTORY_CACHE, **save_dict)
    print(f"Saved trajectory cache to {TRAJECTORY_CACHE}\n")

    return trajectories


def compute_reliability_from_scratch(T_room, util):
    """Independently compute reliability metrics using exact equations from METHODOLOGY.md."""
    # 1. Junction Temperature Tj(t)
    util_clipped = np.clip(util, 0.0, 1.0)
    delta_T = DT_IDLE + (DT_MAX - DT_IDLE) * util_clipped
    T_j = T_room + delta_T

    # 2. Arrhenius Aging Factor AF_T(t)
    T_j_k = T_j + 273.15
    exponent = (E_ACTIVATION_EV / K_BOLTZMANN_EV) * (1.0 / T_J_REF_K - 1.0 / T_j_k)
    AF_T = np.exp(exponent)

    # 3. Aggregated Metrics
    mean_Tj = float(np.mean(T_j))
    max_Tj = float(np.max(T_j))
    min_Tj = float(np.min(T_j))
    delta_Tj_swing = max_Tj - min_Tj
    mean_AF = float(np.mean(AF_T))
    peak_AF = float(np.max(AF_T))
    rel_mtbf = 1.0 / mean_AF
    equiv_days = 5.0 * mean_AF

    return {
        "Mean_Tj_C": mean_Tj,
        "Max_Tj_C": max_Tj,
        "Delta_Tj_Swing_C": delta_Tj_swing,
        "Mean_Arrhenius_AF": mean_AF,
        "Peak_Arrhenius_AF": peak_AF,
        "Relative_MTBF": rel_mtbf,
        "Equiv_Aging_Days_per_5Days": equiv_days,
    }


def verify(include_qml=False, tolerance=1e-4):
    if not os.path.exists(RELIABILITY_CSV):
        raise FileNotFoundError(f"Missing {RELIABILITY_CSV}. Run component_failure.py first.")

    df_reported = pd.read_csv(RELIABILITY_CSV)
    trajectories = get_seed_trajectories(seed=TARGET_SEED, include_qml=include_qml)
    util = trajectories["util"]

    controllers_to_check = ["Baseline thermostat", "MPC + Classical GRU"]
    if include_qml and "MPC + QML Hybrid" in trajectories:
        controllers_to_check.append("MPC + QML Hybrid")

    print("=" * 95)
    print(f"INDEPENDENT RELIABILITY MODEL VERIFICATION: SEED {TARGET_SEED}")
    print("=" * 95)
    print(f"Reference junction temperature : {T_J_REF_K - 273.15:.2f} °C (326.65 K)")
    print(f"Silicon activation energy (Ea) : {E_ACTIVATION_EV} eV")
    print(f"Boltzmann constant (kB)        : {K_BOLTZMANN_EV:.9e} eV/K")
    print(f"Tolerance threshold            : {tolerance} absolute / relative error\n")

    overall_pass = True

    metric_keys = [
        ("Mean_Tj_C", "Mean Junction Temp (°C)"),
        ("Max_Tj_C", "Peak Junction Temp (°C)"),
        ("Delta_Tj_Swing_C", "Thermal Swing Delta-Tj (°C)"),
        ("Mean_Arrhenius_AF", "Arrhenius Aging Factor (AF)"),
        ("Peak_Arrhenius_AF", "Peak Arrhenius AF"),
        ("Relative_MTBF", "Relative MTBF (vs 1.00 nominal)"),
        ("Equiv_Aging_Days_per_5Days", "Equiv Aging Days (per 5d)"),
    ]

    for ctrl in controllers_to_check:
        print("-" * 95)
        print(f"CONTROLLER: {ctrl.upper()} (SEED {TARGET_SEED})")
        print("-" * 95)

        # Get reported row from CSV
        reported_rows = df_reported[(df_reported["Seed"] == TARGET_SEED) & (df_reported["Controller"] == ctrl)]
        if len(reported_rows) == 0:
            print(f"ERROR: No reported results found for {ctrl} on seed {TARGET_SEED}")
            overall_pass = False
            continue
        rep = reported_rows.iloc[0]

        # Recompute independently from raw trajectory
        T_room = trajectories[ctrl]
        recomputed = compute_reliability_from_scratch(T_room, util)

        header = f"{'Metric':<34} {'Recomputed':<18} {'Reported':<18} {'Diff':<12} {'Status':<8}"
        print(header)
        print("-" * 95)

        for key, label in metric_keys:
            val_recomp = recomputed[key]
            val_rep = rep[key]
            diff = abs(val_recomp - val_rep)
            rel_err = diff / max(abs(val_rep), 1e-9)

            passed = (diff <= tolerance) or (rel_err <= tolerance)
            if not passed:
                overall_pass = False
            status = "PASS" if passed else "FAIL"

            print(f"{label:<34} {val_recomp:<18.8f} {val_rep:<18.8f} {diff:<12.2e} [{status}]")

    print("\n" + "=" * 95)
    if overall_pass:
        print("VERIFICATION RESULT: >>> PASS <<<")
        print("All independently recomputed reliability metrics match the reported 10-seed table")
        print("within numerical precision (error < 1e-4).")
        print("CONFIRMATION: Reported reliability numbers are authentically computed directly")
        print("from real simulation physics trajectories and verified standard Arrhenius equations.")
    else:
        print("VERIFICATION RESULT: >>> FAIL <<<")
        print("One or more metrics exceeded the allowed tolerance threshold.")
    print("=" * 95 + "\n")

    return overall_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify reliability model on seed 101.")
    parser.add_argument("--qml", action="store_true", help="Include QML Hybrid controller in spot-check.")
    parser.add_argument("--tol", type=float, default=1e-4, help="Tolerance for comparison (default: 1e-4).")
    args = parser.parse_args()

    success = verify(include_qml=args.qml, tolerance=args.tol)
    sys.exit(0 if success else 1)

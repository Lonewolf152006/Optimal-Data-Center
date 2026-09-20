"""
Multi-Rack Spatial Workload Placement Closed-Loop Evaluation
============================================================
Evaluates Thermal-Unaware (Uniform) vs. Thermal-Aware Optimal Workload Placement
across 10 held-out seeds (101-110) in a 4-rack data center with heat recirculation.

Quantifies:
  1. Hot-spot elimination and rack-to-rack thermal gradient reduction.
  2. Critical task thermal SLA violation rate (T_j > 65°C or T_inlet > 24°C).
  3. Server hardware reliability (Arrhenius AF and relative MTBF) for critical loads.
  4. Facility cooling electrical energy (kWh) and carbon footprint (kg CO2).
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from plant.spatial_thermal_plant import SpatialThermalPlant
from plant.scenarios import random_scenario, carbon_intensity
from ml.train_classical import ClassicalPredictor
from optimization.mpc_core_fast import make_mpc_controller_fast
from evaluation.run_comparison import PredictorWrapper
from controllers.spatial_workload_dispatcher import (
    WorkloadClassifier,
    ThermalUnawareDispatcher,
    ThermalAwareOptimalDispatcher,
)

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
SEEDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
HORIZON = 24


def run_spatial_simulation(condition_name, t, util, tamb, seed, save_trajectories=False):
    """Run a single 5-day multi-rack spatial simulation under the given workload policy."""
    n = len(t)
    carbon = carbon_intensity(t)
    plant = SpatialThermalPlant()
    plant.reset()

    # MPC controller using classical GRU predictor
    predictor = ClassicalPredictor()
    wrapper = PredictorWrapper(predictor)
    controller = make_mpc_controller_fast(predictor_fn=wrapper.predict, horizon=HORIZON)

    classifier = WorkloadClassifier(crit_fraction=0.35)
    if condition_name == "Thermal-Unaware (Uniform)":
        dispatcher = ThermalUnawareDispatcher(n_racks=plant.N_RACKS)
    else:
        dispatcher = ThermalAwareOptimalDispatcher(n_racks=plant.N_RACKS)

    # History arrays
    T_inlet_hist = np.empty((n, plant.N_RACKS))
    T_rack_hist = np.empty((n, plant.N_RACKS))
    T_junc_hist = np.empty((n, plant.N_RACKS))
    AF_hist = np.empty((n, plant.N_RACKS))
    w_crit_hist = np.empty((n, plant.N_RACKS))
    w_tot_hist = np.empty((n, plant.N_RACKS))
    u_hist = np.empty(n)
    pcool_hist = np.empty(n)

    current_Tin = np.full(plant.N_RACKS, plant.T_SET)

    for k in range(n):
        # 1. Aggregate room state for MPC
        T_mean = float(np.mean(plant.T_rack))

        # 2. MPC cooling decision
        h = min(HORIZON, n - k)
        util_fore = util[k: k + h].copy()
        tamb_fore = tamb[k: k + h].copy()
        carbon_fore = carbon[k: k + h].copy()
        if h < HORIZON:
            util_fore = np.concatenate([util_fore, np.full(HORIZON - h, util_fore[-1])])
            tamb_fore = np.concatenate([tamb_fore, np.full(HORIZON - h, tamb_fore[-1])])
            carbon_fore = np.concatenate([carbon_fore, np.full(HORIZON - h, carbon_fore[-1])])

        u = controller(T_mean, step_idx=k,
                       util_fore=util_fore, tamb_fore=tamb_fore,
                       carbon_fore=carbon_fore)
        u_hist[k] = u

        # 3. Workload dispatch
        w_crit, w_batch = classifier.split_demand(util[k], n_racks=plant.N_RACKS)
        if condition_name == "Thermal-Unaware (Uniform)":
            dispatch_res = dispatcher.dispatch(w_crit, w_batch)
        else:
            dispatch_res = dispatcher.dispatch(w_crit, w_batch, T_inlet_current=current_Tin)

        w_tot = dispatch_res["w_total"]
        w_crit_hist[k, :] = dispatch_res["w_crit"]
        w_tot_hist[k, :] = w_tot

        # 4. Advance 4-rack plant
        step_res = plant.step(u=u, w_vector=w_tot, tamb=tamb[k])
        current_Tin = step_res["T_inlet"]

        T_inlet_hist[k, :] = step_res["T_inlet"]
        T_rack_hist[k, :] = step_res["T_rack"]
        T_junc_hist[k, :] = step_res["T_junction"]
        AF_hist[k, :] = step_res["Arrhenius_AF"]
        pcool_hist[k] = step_res["P_cool"]

        # 5. Update MPC predictor
        carbon_k = carbon_intensity(t[k])
        wrapper.update(T_mean, util[k], tamb[k], step_res["P_cool"], carbon_k)

    # Compute aggregate metrics
    dt = plant.DT_HR
    energy_kwh = float(np.sum(pcool_hist * dt))
    carbon_kg = float(np.sum(pcool_hist * carbon * dt) / 1000.0)

    # Thermal metrics
    peak_inlet_temp = float(np.max(T_inlet_hist))
    peak_junc_temp = float(np.max(T_junc_hist))
    mean_thermal_gradient = float(np.mean(np.max(T_inlet_hist, axis=1) - np.min(T_inlet_hist, axis=1)))

    # Hot-spot duration: minutes where any rack inlet exceeds 24.0°C
    hotspot_steps = np.sum(np.any(T_inlet_hist > plant.T_INLET_CRIT_MAX, axis=1))
    hotspot_minutes = float(hotspot_steps * (plant.DT_HR * 60))

    # Critical task thermal violations:
    # A critical task is violated if run on a rack where T_junc > 65°C or T_inlet > 24°C
    crit_violated_weights = np.sum(
        w_crit_hist * ((T_junc_hist > plant.T_JUNCTION_CRIT_MAX) | (T_inlet_hist > plant.T_INLET_CRIT_MAX))
    )
    crit_total_weights = np.sum(w_crit_hist)
    crit_violation_pct = float(100.0 * crit_violated_weights / max(1e-6, crit_total_weights))

    # Critical task hardware MTBF
    crit_weighted_af = np.sum(w_crit_hist * AF_hist) / max(1e-6, crit_total_weights)
    crit_mtbf = float(1.0 / crit_weighted_af)

    # Overall rack MTBF
    mean_overall_af = float(np.mean(AF_hist))
    overall_mtbf = float(1.0 / mean_overall_af)

    # Per-rack max inlet temps
    per_rack_max_inlet = [float(np.max(T_inlet_hist[:, i])) for i in range(plant.N_RACKS)]

    out = {
        "Seed": seed,
        "Condition": condition_name,
        "Energy_kWh": energy_kwh,
        "Carbon_kg": carbon_kg,
        "Peak_Inlet_Temp": peak_inlet_temp,
        "Peak_Junction_Temp": peak_junc_temp,
        "Mean_Thermal_Gradient": mean_thermal_gradient,
        "Hotspot_Minutes": hotspot_minutes,
        "Crit_Violation_Pct": crit_violation_pct,
        "Crit_MTBF": crit_mtbf,
        "Overall_MTBF": overall_mtbf,
        "Rack1_Max_Inlet": per_rack_max_inlet[0],
        "Rack2_Max_Inlet": per_rack_max_inlet[1],
        "Rack3_Max_Inlet": per_rack_max_inlet[2],
        "Rack4_Max_Inlet": per_rack_max_inlet[3],
    }

    trajectories = None
    if save_trajectories:
        trajectories = {
            "t": t,
            "T_inlet": T_inlet_hist,
            "T_junc": T_junc_hist,
            "w_crit": w_crit_hist,
            "w_tot": w_tot_hist,
            "u": u_hist,
            "pcool": pcool_hist,
        }

    return out, trajectories


def evaluate_seed(seed, is_first_seed=False):
    """Run both conditions for a single seed."""
    t, util, tamb = random_scenario(seed=seed, n_days=5)

    res_unaware, traj_unaware = run_spatial_simulation(
        "Thermal-Unaware (Uniform)", t, util, tamb, seed, save_trajectories=is_first_seed
    )
    res_optimal, traj_optimal = run_spatial_simulation(
        "Thermal-Aware (Optimal)", t, util, tamb, seed, save_trajectories=is_first_seed
    )

    return [res_unaware, res_optimal], (traj_unaware, traj_optimal) if is_first_seed else None


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 90)
    print("MULTI-RACK SPATIAL WORKLOAD PLACEMENT EVALUATION (10 Seeds x 2 Policies)")
    print("=" * 90)
    print("Evaluating 4-Rack Data Center with Heat Recirculation & Critical Workloads...")
    print(f"Seeds: {SEEDS}\n")

    t0 = time.time()
    all_rows = []
    saved_trajs = None

    # First run seed 101 to capture detailed trajectories for plotting
    print("Simulating Seed 101 (capturing detailed trajectories)...", flush=True)
    rows_101, saved_trajs = evaluate_seed(101, is_first_seed=True)
    all_rows.extend(rows_101)

    # Save trajectories for fig6
    if saved_trajs is not None:
        traj_unaware, traj_optimal = saved_trajs
        np.savez(
            os.path.join(RESULTS_DIR, "spatial_trajectories.npz"),
            t=traj_unaware["t"],
            unaware_inlet=traj_unaware["T_inlet"],
            unaware_junc=traj_unaware["T_junc"],
            unaware_wcrit=traj_unaware["w_crit"],
            unaware_wtot=traj_unaware["w_tot"],
            optimal_inlet=traj_optimal["T_inlet"],
            optimal_junc=traj_optimal["T_junc"],
            optimal_wcrit=traj_optimal["w_crit"],
            optimal_wtot=traj_optimal["w_tot"],
        )
        print("Saved detailed trajectories to results/spatial_trajectories.npz", flush=True)

    # Run remaining 9 seeds in parallel
    remaining_seeds = SEEDS[1:]
    print(f"Evaluating remaining {len(remaining_seeds)} seeds in parallel...", flush=True)

    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_seed = {executor.submit(evaluate_seed, s, False): s for s in remaining_seeds}
        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                rows, _ = future.result()
                all_rows.extend(rows)
                print(f"  [Seed {seed}] Completed both policies.", flush=True)
            except Exception as e:
                print(f"  [Seed {seed}] ERROR: {e}", flush=True)

    elapsed = time.time() - t0
    print(f"\nAll 10 seeds evaluated in {elapsed:.1f} s ({elapsed / 60:.2f} min)\n", flush=True)

    # Save detailed CSV
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(RESULTS_DIR, "spatial_workload_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved per-seed results to {csv_path}\n", flush=True)

    # Summary table
    print("=" * 95)
    print(f"{'Metric':<36} {'Thermal-Unaware (Uniform)':<28} {'Thermal-Aware (Optimal)':<26} {'Improvement'}")
    print("=" * 95)

    summary_data = []

    metrics_def = [
        ("Peak Rack Inlet Temp (°C)", "Peak_Inlet_Temp", "{:.2f} ± {:.2f}", "{:+.2f} °C", True),
        ("Peak CPU Junction Temp (°C)", "Peak_Junction_Temp", "{:.2f} ± {:.2f}", "{:+.2f} °C", True),
        ("Mean Rack Thermal Gradient (°C)", "Mean_Thermal_Gradient", "{:.2f} ± {:.2f}", "{:+.2f} °C", True),
        ("Hot-Spot Duration >24°C (min)", "Hotspot_Minutes", "{:.1f} ± {:.1f}", "{:+.1f} min", True),
        ("Critical Task Violation Rate (%)", "Crit_Violation_Pct", "{:.2f}% ± {:.2f}%", "{:+.2f} pp", True),
        ("Critical Workload MTBF (relative)", "Crit_MTBF", "{:.3f} ± {:.3f}", "{:+.3f}x", False),
        ("Overall Facility MTBF (relative)", "Overall_MTBF", "{:.3f} ± {:.3f}", "{:+.3f}x", False),
        ("Electrical Energy (kWh)", "Energy_kWh", "{:.1f} ± {:.1f}", "{:+.1f} kWh", True),
        ("Carbon Footprint (kg CO2)", "Carbon_kg", "{:.1f} ± {:.1f}", "{:+.1f} kg", True),
    ]

    for label, col, fmt, diff_fmt, lower_is_better in metrics_def:
        unaware_vals = df[df["Condition"] == "Thermal-Unaware (Uniform)"][col]
        optimal_vals = df[df["Condition"] == "Thermal-Aware (Optimal)"][col]

        m_un = unaware_vals.mean()
        s_un = unaware_vals.std()
        m_opt = optimal_vals.mean()
        s_opt = optimal_vals.std()

        diff = m_opt - m_un
        diff_str = diff_fmt.format(diff)

        str_un = fmt.format(m_un, s_un)
        str_opt = fmt.format(m_opt, s_opt)

        print(f"{label:<36} {str_un:<28} {str_opt:<26} {diff_str}")

        summary_data.append({
            "Metric": label,
            "Thermal_Unaware_Mean": m_un,
            "Thermal_Unaware_Std": s_un,
            "Thermal_Aware_Mean": m_opt,
            "Thermal_Aware_Std": s_opt,
            "Delta": diff,
        })

    print("=" * 95)

    sum_df = pd.DataFrame(summary_data)
    sum_csv_path = os.path.join(RESULTS_DIR, "spatial_workload_summary.csv")
    sum_df.to_csv(sum_csv_path, index=False)
    print(f"Saved summary statistics to {sum_csv_path}\n", flush=True)


if __name__ == "__main__":
    main()

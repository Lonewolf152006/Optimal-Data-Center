"""
Compute MAE Broken Out by Horizon Step Position Across 10 Seeds
==============================================================
Evaluates how closed-loop prediction error compounds across horizon
steps h = 1, 2, ..., 24 (5 min to 120 min ahead) for both Classical GRU
and Hybrid QML predictors across all 10 held-out scenario seeds.

Outputs:
  - results/mae_by_horizon_step.csv
  - results/fig4_mae_by_horizon_step.png
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from plant.thermal_plant import ThermalPlant
from plant.scenarios import random_scenario
from ml.train_classical import ClassicalPredictor
from qml.train_qml import QMLPredictor
from optimization.mpc_core_fast import make_mpc_controller_fast
from evaluation.run_comparison import PredictorWrapper

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
SEEDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
HORIZON = 24


def run_sim_with_horizon_errors(plant, controller, t, util, tamb, predictor_wrapper):
    """Run closed-loop simulation and record per-horizon-step errors."""
    n = len(t)
    T = np.empty(n)
    T[0] = plant.T_SET
    predictions = []

    for k in range(n):
        h = min(HORIZON, n - k)
        util_fore = util[k: k + h].copy()
        tamb_fore = tamb[k: k + h].copy()
        carbon_fore = plant.carbon_intensity(t[k: k + h]).copy()
        if h < HORIZON:
            util_fore = np.concatenate([util_fore, np.full(HORIZON - h, util_fore[-1])])
            tamb_fore = np.concatenate([tamb_fore, np.full(HORIZON - h, tamb_fore[-1])])
            carbon_fore = np.concatenate([carbon_fore, np.full(HORIZON - h, carbon_fore[-1])])

        u = controller(T[k], step_idx=k,
                       util_fore=util_fore, tamb_fore=tamb_fore,
                       carbon_fore=carbon_fore)

        if predictor_wrapper.last_prediction is not None:
            predictions.append((k, predictor_wrapper.last_prediction))

        pcool_k, _ = plant.p_cool(u, tamb[k])
        carbon_k = plant.carbon_intensity(t[k])
        predictor_wrapper.update(T[k], util[k], tamb[k], pcool_k, carbon_k)

        if k < n - 1:
            T[k + 1] = plant.step(T[k], u, util[k], tamb[k])

    # Compute MAE per horizon step h in {0..HORIZON-1}
    step_errors = [[] for _ in range(HORIZON)]
    for step_k, p_traj in predictions:
        for h in range(HORIZON):
            target_idx = step_k + 1 + h
            if target_idx < n:
                err = abs(p_traj[h] - T[target_idx])
                step_errors[h].append(err)

    mae_per_step = np.array([np.mean(step_errors[h]) for h in range(HORIZON)])
    return mae_per_step


def evaluate_seed(seed):
    plant = ThermalPlant()
    t, util, tamb = random_scenario(seed=seed, n_days=5)

    # 1. Classical GRU
    c_pred = ClassicalPredictor()
    wrap_c = PredictorWrapper(c_pred)
    ctrl_c = make_mpc_controller_fast(predictor_fn=wrap_c.predict, horizon=HORIZON)
    mae_c = run_sim_with_horizon_errors(plant, ctrl_c, t, util, tamb, wrap_c)

    # 2. Hybrid QML
    q_pred = QMLPredictor()
    wrap_q = PredictorWrapper(q_pred)
    ctrl_q = make_mpc_controller_fast(predictor_fn=wrap_q.predict, horizon=HORIZON)
    mae_q = run_sim_with_horizon_errors(plant, ctrl_q, t, util, tamb, wrap_q)

    return seed, mae_c, mae_q


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 80)
    print(f"COMPUTING MAE BROKEN OUT BY HORIZON STEP (10 SEEDS, HORIZON = {HORIZON})")
    print("=" * 80)

    all_mae_c = []
    all_mae_q = []

    t0 = time.time()
    max_workers = 4
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_seed, seed): seed for seed in SEEDS}
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                s, mc, mq = fut.result()
                all_mae_c.append(mc)
                all_mae_q.append(mq)
                print(f"  [DONE] Seed {seed} processed successfully", flush=True)
            except Exception as e:
                print(f"  [ERROR] Seed {seed} failed: {e}", flush=True)

    elapsed = time.time() - t0
    print(f"\nAll 10 seeds evaluated in {elapsed:.1f} s ({elapsed/60:.2f} min)\n", flush=True)

    all_mae_c = np.array(all_mae_c)  # (10, 24)
    all_mae_q = np.array(all_mae_q)  # (10, 24)

    mean_mae_c = np.mean(all_mae_c, axis=0)
    std_mae_c = np.std(all_mae_c, axis=0)
    mean_mae_q = np.mean(all_mae_q, axis=0)
    std_mae_q = np.std(all_mae_q, axis=0)

    steps = np.arange(1, HORIZON + 1)
    minutes = steps * 5

    df = pd.DataFrame({
        "Horizon_Step": steps,
        "Time_Ahead_Min": minutes,
        "Classical_GRU_MAE": mean_mae_c,
        "Classical_GRU_Std": std_mae_c,
        "Hybrid_QML_MAE": mean_mae_q,
        "Hybrid_QML_Std": std_mae_q,
        "Difference_QML_minus_GRU": mean_mae_q - mean_mae_c,
    })

    csv_path = os.path.join(RESULTS_DIR, "mae_by_horizon_step.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved horizon-step MAE breakdown to {csv_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("HORIZON-STEP MAE BREAKDOWN (Averaged across 10 Seeds)")
    print("=" * 80)
    print(f"{'Step':<6} {'Time Ahead':<12} {'Classical GRU MAE':<22} {'Hybrid QML MAE':<22} {'Diff (QML - GRU)':<18}")
    print("-" * 80)
    for i in range(HORIZON):
        h_step = steps[i]
        t_min = f"{minutes[i]} min"
        c_str = f"{mean_mae_c[i]:.4f} ± {std_mae_c[i]:.4f} °C"
        q_str = f"{mean_mae_q[i]:.4f} ± {std_mae_q[i]:.4f} °C"
        d_val = f"{mean_mae_q[i] - mean_mae_c[i]:+.4f} °C"
        print(f"{h_step:<6} {t_min:<12} {c_str:<22} {q_str:<22} {d_val:<18}")
    print("=" * 80 + "\n")

    # Plot line chart
    plt.figure(figsize=(10, 5.5), dpi=140)
    plt.plot(steps, mean_mae_c, "o-", color="#1f77b4", linewidth=2.2, label="MPC + Classical GRU", markersize=5)
    plt.fill_between(steps, mean_mae_c - std_mae_c, mean_mae_c + std_mae_c, color="#1f77b4", alpha=0.18)

    plt.plot(steps, mean_mae_q, "s-", color="#d62728", linewidth=2.2, label="MPC + QML Hybrid", markersize=5)
    plt.fill_between(steps, mean_mae_q - std_mae_q, mean_mae_q + std_mae_q, color="#d62728", alpha=0.18)

    plt.title("Closed-Loop Prediction MAE by Horizon Step Position (10 Seeds)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Horizon Step Position (Step 1 = 5 min, Step 24 = 120 min ahead)", fontsize=10)
    plt.ylabel("Closed-Loop Prediction MAE (°C)", fontsize=10)
    plt.xticks(steps, [f"{s}\n({s*5}m)" if s % 2 != 0 else f"{s}" for s in steps], fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(fontsize=10, loc="upper left", framealpha=0.9)
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, "fig4_mae_by_horizon_step.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved line chart to {plot_path}")


if __name__ == "__main__":
    main()

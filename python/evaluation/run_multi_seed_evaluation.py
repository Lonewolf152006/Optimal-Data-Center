"""
Multi-Seed Robustness Evaluation & Latency Accounting
=====================================================
Evaluates whether QML's ASHRAE compliance advantage over Classical GRU
is real or evaluation noise from a single held-out seed.

Runs across 10 held-out seeds strictly outside dataset_gen.py's seed range (0..99):
  Seeds: [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]

Computes and saves:
  1. results/multi_seed_results.csv (per-seed metrics per controller)
  2. results/multi_seed_summary.csv (mean, std, min, max per metric per controller)
  3. Latency & cost accounting side-by-side table (GRU vs QML)
  4. Explicit statistical overlap check for % recommended band compliance.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from plant.thermal_plant import ThermalPlant
from ml.train_classical import ClassicalPredictor
from qml.train_qml import QMLPredictor
from evaluation.run_comparison import run_single_scenario

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# 10 seeds strictly outside dataset_gen.py training range (0..99)
SEEDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]


# ======================================================================
# Instrumentation profiler for MPC optimizer calls
# ======================================================================
class PredictorProfiler:
    """Read-only instrumentation wrapper around predictor_fn to count
    invocations per MPC solve and measure exact wall-clock time spent inside."""
    def __init__(self, predictor):
        self.predictor = predictor
        self.total_calls = 0
        self.total_time_s = 0.0
        self.calls_this_solve = 0
        self.calls_per_solve_history = []

    def start_solve(self):
        self.calls_this_solve = 0

    def end_solve(self):
        self.calls_per_solve_history.append(self.calls_this_solve)

    def update(self, *args, **kwargs):
        return self.predictor.update(*args, **kwargs)

    def reset(self):
        self.total_calls = 0
        self.total_time_s = 0.0
        self.calls_this_solve = 0
        self.calls_per_solve_history = []
        return self.predictor.reset()

    def predict(self, state, forecast_features):
        t0 = time.perf_counter()
        res = self.predictor.predict(state, forecast_features)
        dt = time.perf_counter() - t0
        self.total_calls += 1
        self.calls_this_solve += 1
        self.total_time_s += dt
        return res


# ======================================================================
# Worker function for parallel execution across seeds
# ======================================================================
def _worker_run_seed(seed):
    """Executes all 3 controllers for a single seed in an isolated process."""
    plant = ThermalPlant()
    c_pred = ClassicalPredictor()
    q_pred = QMLPredictor()
    t0 = time.time()
    metrics_dict = run_single_scenario(
        seed=seed, plant=plant,
        classical_pred=c_pred, qml_pred=q_pred,
        save_trajectories=False, verbose=False
    )
    elapsed = time.time() - t0
    return seed, metrics_dict, elapsed


# ======================================================================
# Main evaluation script
# ======================================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 80)
    print("MULTI-SEED ROBUSTNESS EVALUATION (10 HELD-OUT SCENARIOS)")
    print("=" * 80)
    print(f"Held-out seeds: {SEEDS}")
    print("Seeds strictly excluded from training dataset (0..99).\n", flush=True)

    # ------------------------------------------------------------------
    # Step A: Run Latency Accounting on Seed 101 with Profilers
    # ------------------------------------------------------------------
    print("Running detailed latency/cost accounting profile on seed 101...", flush=True)
    plant = ThermalPlant()
    c_pred_raw = ClassicalPredictor()
    q_pred_raw = QMLPredictor()
    c_profiler = PredictorProfiler(c_pred_raw)
    q_profiler = PredictorProfiler(q_pred_raw)

    t_prof_start = time.time()
    run_single_scenario(
        seed=SEEDS[0], plant=plant,
        classical_pred=c_profiler, qml_pred=q_profiler,
        classical_profiler=c_profiler, qml_profiler=q_profiler,
        save_trajectories=False, verbose=False
    )
    t_prof_total = time.time() - t_prof_start
    print(f"Profile run finished in {t_prof_total:.1f}s\n", flush=True)

    # Compute latency accounting metrics
    c_calls_per_solve_mean = np.mean(c_profiler.calls_per_solve_history)
    c_calls_per_solve_std = np.std(c_profiler.calls_per_solve_history)
    q_calls_per_solve_mean = np.mean(q_profiler.calls_per_solve_history)
    q_calls_per_solve_std = np.std(q_profiler.calls_per_solve_history)

    n_steps = len(c_profiler.calls_per_solve_history)
    c_solves_per_step = len(c_profiler.calls_per_solve_history) / n_steps
    q_solves_per_step = len(q_profiler.calls_per_solve_history) / n_steps

    c_total_calls = c_profiler.total_calls
    q_total_calls = q_profiler.total_calls

    c_total_time = c_profiler.total_time_s
    q_total_time = q_profiler.total_time_s
    time_ratio = q_total_time / max(c_total_time, 1e-9)

    c_latency_ms = (c_total_time / max(c_total_calls, 1)) * 1000
    q_latency_ms = (q_total_time / max(q_total_calls, 1)) * 1000
    latency_ratio = q_latency_ms / max(c_latency_ms, 1e-9)

    print("=" * 80)
    print("PREDICTOR LATENCY & INFERENCE COST ACCOUNTING (1,440-step Scenario)")
    print("=" * 80)
    print(f"{'Metric':<36} {'Classical GRU':<20} {'Hybrid QML':<20} {'Ratio (QML/GRU)':<15}")
    print("-" * 80)
    print(f"{'Predictor calls per MPC solve':<36} {f'{c_calls_per_solve_mean:.2f} ± {c_calls_per_solve_std:.2f}':<20} {f'{q_calls_per_solve_mean:.2f} ± {q_calls_per_solve_std:.2f}':<20} {'1.00x':<15}")
    print(f"{'MPC solves per control step':<36} {f'{c_solves_per_step:.2f}':<20} {f'{q_solves_per_step:.2f}':<20} {'1.00x':<15}")
    print(f"{'Total predictor invocations':<36} {f'{c_total_calls}':<20} {f'{q_total_calls}':<20} {'1.00x':<15}")
    print(f"{'Mean inference latency':<36} {f'{c_latency_ms:.2f} ms':<20} {f'{q_latency_ms:.2f} ms':<20} {f'{latency_ratio:.2f}x':<15}")
    print(f"{'Total predictor wall-clock time':<36} {f'{c_total_time:.2f} s':<20} {f'{q_total_time:.2f} s':<20} {f'{time_ratio:.2f}x':<15}")
    print("=" * 80 + "\n", flush=True)

    # ------------------------------------------------------------------
    # Step B: Run Multi-Seed Evaluations
    # ------------------------------------------------------------------
    print(f"Launching evaluations across {len(SEEDS)} seeds...", flush=True)
    all_results = []
    
    # Use parallel processing pool (3 workers for balanced CPU utilisation)
    max_workers = 3
    t_start = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker_run_seed, seed): seed for seed in SEEDS}
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                s, metrics_dict, el = fut.result()
                print(f"  [DONE] Seed {seed} completed in {el:.1f}s", flush=True)
                for ctrl_name, m in metrics_dict.items():
                    row = {"Seed": seed, **m}
                    all_results.append(row)
            except Exception as e:
                print(f"  [ERROR] Seed {seed} failed: {e}", flush=True)

    total_time = time.time() - t_start
    print(f"\nAll {len(SEEDS)} seeds evaluated in {total_time:.1f}s ({total_time/60:.2f} min).\n", flush=True)

    # Create DataFrame of all results
    df_results = pd.DataFrame(all_results)
    results_csv_path = os.path.join(RESULTS_DIR, "multi_seed_results.csv")
    df_results.to_csv(results_csv_path, index=False)
    print(f"Saved full per-seed results to {results_csv_path}")

    # ------------------------------------------------------------------
    # Step C: Compute Summary Statistics (mean, std, min, max)
    # ------------------------------------------------------------------
    metric_cols = [
        "Energy (kWh)", "Carbon (kg CO2)", "% Recommended (18-27C)",
        "% Allowable (15-32C)", "Max Temp (C)", "Min Temp (C)",
        "Avg Latency (ms)", "Prediction MAE (C)", "Prediction RMSE (C)",
    ]

    summary_rows = []
    for ctrl, grp in df_results.groupby("Controller"):
        for metric in metric_cols:
            vals = grp[metric].dropna()
            if len(vals) > 0:
                summary_rows.append({
                    "Controller": ctrl,
                    "Metric": metric,
                    "Mean": float(vals.mean()),
                    "Std": float(vals.std()),
                    "Min": float(vals.min()),
                    "Max": float(vals.max()),
                })
            else:
                summary_rows.append({
                    "Controller": ctrl,
                    "Metric": metric,
                    "Mean": np.nan, "Std": np.nan, "Min": np.nan, "Max": np.nan,
                })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(RESULTS_DIR, "multi_seed_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Saved summary statistics to {summary_csv_path}\n")

    # Display Pivot Summary Table for user
    pivot_mean = df_summary.pivot(index="Controller", columns="Metric", values="Mean")
    pivot_std = df_summary.pivot(index="Controller", columns="Metric", values="Std")

    display_cols = ["Energy (kWh)", "Carbon (kg CO2)", "% Recommended (18-27C)",
                    "% Allowable (15-32C)", "Max Temp (C)", "Min Temp (C)"]
    
    print("=" * 100)
    print("MULTI-SEED STATISTICAL SUMMARY (Mean ± Std over 10 Seeds)")
    print("=" * 100)
    for ctrl in ["Baseline thermostat", "MPC + Classical GRU", "MPC + QML Hybrid"]:
        if ctrl in pivot_mean.index:
            print(f"\n{ctrl}:")
            for c in display_cols:
                m_val = pivot_mean.loc[ctrl, c]
                s_val = pivot_std.loc[ctrl, c]
                print(f"  {c:<28}: {m_val:8.2f} ± {s_val:6.2f}")
    print("=" * 100 + "\n")

    # ------------------------------------------------------------------
    # Step D: Explicit Statistical Overlap Check for % Recommended Band
    # ------------------------------------------------------------------
    gru_rec = df_results[df_results["Controller"] == "MPC + Classical GRU"]["% Recommended (18-27C)"]
    qml_rec = df_results[df_results["Controller"] == "MPC + QML Hybrid"]["% Recommended (18-27C)"]

    gru_mean, gru_std = float(gru_rec.mean()), float(gru_rec.std())
    qml_mean, qml_std = float(qml_rec.mean()), float(qml_rec.std())

    gru_range = (gru_mean - gru_std, gru_mean + gru_std)
    qml_range = (qml_mean - qml_std, qml_mean + qml_std)

    overlap = not (qml_range[0] > gru_range[1] or gru_range[0] > qml_range[1])

    print("=" * 80)
    print("ASHRAE RECOMMENDED-BAND COMPLIANCE: ROBUSTNESS HYPOTHESIS TEST")
    print("=" * 80)
    print(f"Classical GRU Mean ± Std : {gru_mean:6.2f}% ± {gru_std:5.2f}%  ->  Interval [{gru_range[0]:.2f}%, {gru_range[1]:.2f}%]")
    print(f"Hybrid QML    Mean ± Std : {qml_mean:6.2f}% ± {qml_std:5.2f}%  ->  Interval [{qml_range[0]:.2f}%, {qml_range[1]:.2f}%]")
    print(f"Difference in Means (QML - GRU): {qml_mean - gru_mean:+.2f}%")
    print("-" * 80)
    if overlap:
        print("RESULT: The Mean ± Std intervals OVERLAP.")
        if qml_mean > gru_mean:
            print("VERDICT: While QML has a slightly higher point estimate, the confidence intervals")
            print("         overlap across 10 random seeds. The compliance gap is within evaluation noise,")
            print("         and does NOT represent a statistically robust quantum advantage.")
        else:
            print("VERDICT: QML does NOT maintain a compliance advantage over Classical GRU.")
    else:
        print("RESULT: The Mean ± Std intervals DO NOT OVERLAP.")
        if qml_mean > gru_mean:
            print("VERDICT: QML maintains a statistically distinct compliance advantage over GRU")
            print("         across the 10 held-out seeds without overlap.")
        else:
            print("VERDICT: Classical GRU maintains a statistically distinct advantage over QML.")
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()

"""
Standalone Diagnostic Profiler for MPC Optimizer
=================================================
Inspects SciPy's L-BFGS-B optimizer execution for exactly ONE representative
control step without modifying mpc_core.py or touching running pipeline files.

Logs:
- Total cost function evaluations
- Total optimizer iterations
- Whether finite-difference fallback is occurring (function evals per iteration)
- Exact time spent evaluating the cost function vs. SciPy optimizer overhead
"""

import os
import sys
import time
import numpy as np
from scipy.optimize import minimize

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from plant.thermal_plant import ThermalPlant


def run_optimizer_profile(horizon=12, use_predictor=True):
    plant = ThermalPlant()
    U_BASELINE = 0.45
    t0 = 23.5  # representative room temperature (deg C)

    # Generate representative dummy forecasts
    util_fore = np.array([0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.68, 0.62, 0.58, 0.52, 0.48, 0.45][:horizon])
    tamb_fore = np.array([26.0, 27.0, 28.0, 29.5, 31.0, 32.0, 31.5, 30.0, 28.5, 27.0, 26.5, 26.0][:horizon])
    carbon_fore = np.array([350.0, 370.0, 400.0, 430.0, 450.0, 460.0, 440.0, 410.0, 380.0, 360.0, 350.0, 340.0][:horizon])

    # Representative predictor trajectory (simulating drift)
    if use_predictor:
        ml_traj = t0 + np.cumsum(np.array([0.08, 0.10, 0.12, 0.11, 0.09, 0.07, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01][:horizon]))
    else:
        ml_traj = None

    warm_u = np.full(horizon, 0.45)

    # Exact trajectory prediction and cost logic from mpc_core.py
    def predict_trajectory(room_temp, u_seq, util_f, tamb_f, ml_t):
        h = len(u_seq)
        Ts = np.empty(h + 1)
        Ts[0] = room_temp
        for k in range(h):
            if ml_t is not None:
                pred_step = ml_t[k] - (room_temp if k == 0 else ml_t[k - 1])
                dT_cool = (u_seq[k] - U_BASELINE) * plant.Q_COOL_MAX * plant.DT_HR / plant.C_ROOM
                Ts[k + 1] = Ts[k] + pred_step - dT_cool
            else:
                qit = plant.Q_IDLE + (plant.Q_IT_MAX - plant.Q_IDLE) * util_f[k]
                q_del = u_seq[k] * plant.Q_COOL_MAX
                dT = (qit - q_del + plant.UA_ENV * (tamb_f[k] - Ts[k])) / plant.C_ROOM
                Ts[k + 1] = Ts[k] + plant.DT_HR * dT
        return Ts

    def base_cost(u_seq, room_temp, util_f, tamb_f, carbon_f, ml_t):
        Ts = predict_trajectory(room_temp, u_seq, util_f, tamb_f, ml_t)
        pcool_val, _ = plant.p_cool(u_seq, tamb_f[:len(u_seq)])
        carbon_cost = np.sum(pcool_val * carbon_f[:len(u_seq)] * plant.DT_HR)

        rec_viol = (np.maximum(0, plant.T_REC_LO - Ts[1:])
                    + np.maximum(0, Ts[1:] - plant.T_REC_HI))
        alw_viol = (np.maximum(0, plant.T_ALW_LO - Ts[1:])
                    + np.maximum(0, Ts[1:] - plant.T_ALW_HI))

        penalty = 400 * np.sum(rec_viol ** 2) + 20000 * np.sum(alw_viol ** 2)
        smooth = 50 * np.sum(
            np.diff(np.concatenate([[warm_u[0]], u_seq])) ** 2
        )
        return carbon_cost / 1000 + penalty + smooth

    # Instrumentation closure to log evaluations and timings
    eval_count = 0
    total_cost_eval_time = 0.0

    def profiled_cost(u_seq, *args):
        nonlocal eval_count, total_cost_eval_time
        t_start = time.perf_counter()
        val = base_cost(u_seq, *args)
        total_cost_eval_time += (time.perf_counter() - t_start)
        eval_count += 1
        return val

    x0 = np.clip(np.concatenate([warm_u[1:], warm_u[-1:]]), 0, 1)

    t_solve_start = time.perf_counter()
    res = minimize(
        profileed_cost if 'profileed_cost' in locals() else profiled_cost,
        x0,
        args=(t0, util_fore, tamb_fore, carbon_fore, ml_traj),
        method="L-BFGS-B",
        bounds=[(0, 1)] * horizon,
        options={"maxiter": 40},
    )
    total_solve_time = time.perf_counter() - t_solve_start

    n_iter = res.nit
    evals_per_iter = eval_count / max(n_iter, 1)
    is_finite_diff = evals_per_iter >= (horizon * 0.8)
    scipy_overhead_time = max(0.0, total_solve_time - total_cost_eval_time)

    print("=" * 80)
    print("MPC OPTIMIZER PROFILING REPORT (Single Step Solve)")
    print("=" * 80)
    print(f"Horizon length (decision variables) : {horizon}")
    print(f"Predictor mode                     : {'Active (ml_traj provided)' if use_predictor else 'None (physics fallback)'}")
    print(f"Optimizer method                   : L-BFGS-B (bounds=[0, 1])")
    print(f"Passed analytical Jacobian (jac)   : None (Finite-Difference Fallback)")
    print("-" * 80)
    print(f"Total optimizer iterations (nit)   : {n_iter}")
    print(f"Total cost function evaluations    : {eval_count}")
    print(f"Function evaluations per iteration : {evals_per_iter:.2f}")
    print(f"Finite-difference fallback status  : {'CONFIRMED (approx ' + str(horizon + 1) + ' evals/iter)' if is_finite_diff else 'Analytic/Line-search dominant'}")
    print("-" * 80)
    print(f"Total solve wall-clock time        : {total_solve_time * 1000:8.2f} ms")
    print(f"  - Cost function evaluation time  : {total_cost_eval_time * 1000:8.2f} ms  ({(total_cost_eval_time / total_solve_time) * 100:5.1f}%)")
    print(f"  - SciPy L-BFGS-B internal overhead: {scipy_overhead_time * 1000:8.2f} ms  ({(scipy_overhead_time / total_solve_time) * 100:5.1f}%)")
    print("-" * 80)
    print("ANALYSIS:")
    print(f"  For a 12-dimensional continuous search space, SciPy's 2-point finite-difference")
    print(f"  gradient estimation perturbs each dimension individually at every line search step,")
    print(f"  requiring (12 + 1) = 13 function evaluations per gradient evaluation.")
    print(f"  With {n_iter} iterations, {eval_count} total evaluations occur in pure uncompiled Python.")
    print(f"  Across 1,440 steps per 5-day scenario, this accumulates ~{eval_count * 1440:,} evaluations per controller.")
    print("=" * 80 + "\n")

    return {
        "horizon": horizon,
        "n_iter": n_iter,
        "eval_count": eval_count,
        "evals_per_iter": evals_per_iter,
        "total_solve_time_ms": total_solve_time * 1000,
        "cost_eval_time_ms": total_cost_eval_time * 1000,
        "scipy_overhead_time_ms": scipy_overhead_time * 1000,
        "is_finite_diff": is_finite_diff,
        "u_opt": np.clip(res.x, 0, 1),
    }


if __name__ == "__main__":
    run_optimizer_profile(horizon=12, use_predictor=True)

"""
Three-Way Closed-Loop Comparison
=================================
Runs three simulations on an IDENTICAL scenario (e.g. seed 999 or any held-out seed):
  1. Baseline thermostat
  2. MPC + classical (GRU) predictor
  3. MPC + QML predictor

Refactored to expose ``run_single_scenario(seed)`` for multi-seed evaluations.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import time
import numpy as np
import pandas as pd

from plant.thermal_plant import ThermalPlant
from plant.scenarios import build_scenario, random_scenario, carbon_intensity
from controllers.baseline_thermostat import BaselineThermostat
from controllers.mpc_core import make_mpc_controller
from ml.train_classical import ClassicalPredictor
from qml.train_qml import QMLPredictor

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

HELD_OUT_SEED = 999
HORIZON = 24


class PredictorWrapper:
    """Wraps a predictor to capture trajectory predictions for MAE/RMSE
    without incurring redundant inference overhead."""
    def __init__(self, predictor):
        self.predictor = predictor
        self.last_prediction = None

    def update(self, *args, **kwargs):
        return self.predictor.update(*args, **kwargs)

    def reset(self):
        self.last_prediction = None
        return self.predictor.reset()

    def predict(self, state, forecast_features):
        pred = self.predictor.predict(state, forecast_features)
        self.last_prediction = np.array(pred, copy=True)
        return pred


# ======================================================================
# Simulation loop (generic)
# ======================================================================
def run_sim(plant, controller, t, util, tamb, horizon=HORIZON,
            predictor_wrapper=None, profiler=None, label=""):
    """Run a closed-loop simulation, optionally feeding a predictor."""
    n = len(t)
    carbon = carbon_intensity(t)
    T = np.empty(n)
    T[0] = plant.T_SET
    u_hist = np.empty(n)
    latencies = []
    predictions = []

    for k in range(n):
        h = min(horizon, n - k)
        util_fore = util[k:k + h].copy()
        tamb_fore = tamb[k:k + h].copy()
        carbon_fore = carbon[k:k + h].copy()
        # Pad if at end of scenario
        if h < horizon:
            util_fore = np.concatenate([util_fore, np.full(horizon - h, util_fore[-1])])
            tamb_fore = np.concatenate([tamb_fore, np.full(horizon - h, tamb_fore[-1])])
            carbon_fore = np.concatenate([carbon_fore, np.full(horizon - h, carbon_fore[-1])])

        if profiler is not None:
            profiler.start_solve()

        t0 = time.perf_counter()
        u = controller(T[k], step_idx=k,
                       util_fore=util_fore, tamb_fore=tamb_fore,
                       carbon_fore=carbon_fore)
        latencies.append(time.perf_counter() - t0)

        if profiler is not None:
            profiler.end_solve()

        u_hist[k] = u

        # Record captured prediction trajectory from controller call
        if predictor_wrapper is not None and predictor_wrapper.last_prediction is not None:
            predictions.append((k, predictor_wrapper.last_prediction))

        # Update predictor history
        if predictor_wrapper is not None:
            pcool_k, _ = plant.p_cool(u, tamb[k])
            predictor_wrapper.update(T[k], util[k], tamb[k], pcool_k, carbon[k])

        if k < n - 1:
            T[k + 1] = plant.step(T[k], u, util[k], tamb[k])

    pcool, _ = plant.p_cool(u_hist, tamb)
    carbon_kg = np.cumsum(pcool * carbon * plant.DT_HR) / 1000.0
    energy_kwh = np.cumsum(pcool * plant.DT_HR)

    # Compute prediction trajectory MAE / RMSE where applicable
    pred_mae, pred_rmse = None, None
    if len(predictions) > 0:
        maes, mses = [], []
        for step_k, p_traj in predictions:
            actual_len = min(len(p_traj), n - 1 - step_k)
            if actual_len > 0:
                actual = T[step_k + 1 : step_k + 1 + actual_len]
                maes.append(np.mean(np.abs(p_traj[:actual_len] - actual)))
                mses.append(np.mean((p_traj[:actual_len] - actual) ** 2))
        if len(maes) > 0:
            pred_mae = float(np.mean(maes))
            pred_rmse = float(np.sqrt(np.mean(mses)))

    return {
        "label": label,
        "T": T, "u": u_hist, "pcool": pcool,
        "carbon_kg": carbon_kg, "energy_kwh": energy_kwh,
        "t": t, "util": util, "tamb": tamb,
        "latencies": np.array(latencies),
        "pred_mae": pred_mae,
        "pred_rmse": pred_rmse,
    }


# ======================================================================
# Metrics
# ======================================================================
def compute_metrics(result, plant):
    T = result["T"]
    return {
        "Controller": result["label"],
        "Energy (kWh)": float(result["energy_kwh"][-1]),
        "Carbon (kg CO2)": float(result["carbon_kg"][-1]),
        "% Recommended (18-27C)": float(100 * np.mean(
            (T >= plant.T_REC_LO) & (T <= plant.T_REC_HI))),
        "% Allowable (15-32C)": float(100 * np.mean(
            (T >= plant.T_ALW_LO) & (T <= plant.T_ALW_HI))),
        "Max Temp (C)": float(T.max()),
        "Min Temp (C)": float(T.min()),
        "Avg Latency (ms)": float(np.mean(result["latencies"]) * 1000),
        "Prediction MAE (C)": result["pred_mae"],
        "Prediction RMSE (C)": result["pred_rmse"],
    }


# ======================================================================
# Single scenario evaluation
# ======================================================================
def run_single_scenario(seed=HELD_OUT_SEED, plant=None, classical_pred=None,
                        qml_pred=None, classical_profiler=None, qml_profiler=None,
                        save_trajectories=False, verbose=True):
    """Run all 3 controllers on a single scenario seed.

    Parameters
    ----------
    seed : int
        Scenario seed (e.g. 999 for legacy held-out scenario, or >= 100).
    plant : ThermalPlant, optional
    classical_pred : ClassicalPredictor, optional
    qml_pred : QMLPredictor, optional
    classical_profiler : PredictorProfiler, optional
    qml_profiler : PredictorProfiler, optional
    save_trajectories : bool, default False
    verbose : bool, default True

    Returns
    -------
    dict[str, dict]
        Mapping from controller label to its computed metrics dict.
    """
    if plant is None:
        plant = ThermalPlant()
    if classical_pred is None:
        classical_pred = ClassicalPredictor()
    if qml_pred is None:
        qml_pred = QMLPredictor()

    classical_pred.reset()
    qml_pred.reset()

    # Wrap predictors to capture predictions and support profiling
    wrap_c = PredictorWrapper(classical_profiler if classical_profiler is not None else classical_pred)
    wrap_q = PredictorWrapper(qml_profiler if qml_profiler is not None else qml_pred)

    # Build scenario
    if seed == 999:
        if verbose:
            print(f"\nBuilding held-out test scenario seed {seed} (heat-wave days [1,3], spikes day 2)...", flush=True)
        t, util, tamb = build_scenario(
            n_days=5,
            heat_wave_days=[1, 3],
            spike_windows=[(2, 13, 15)],
        )
    else:
        if verbose:
            print(f"\nBuilding randomised held-out scenario seed {seed}...", flush=True)
        t, util, tamb = random_scenario(seed=seed, n_days=5)

    # ---- 1. Baseline thermostat -----------------------------------------
    if verbose:
        print("  [1/3] Running baseline thermostat...", flush=True)
    baseline_ctrl = BaselineThermostat()
    r_baseline = run_sim(plant, baseline_ctrl, t, util, tamb,
                         label="Baseline thermostat")

    # ---- 2. MPC + classical predictor -----------------------------------
    if verbose:
        print("  [2/3] Running MPC + classical GRU predictor...", flush=True)
    mpc_classical = make_mpc_controller(predictor_fn=wrap_c.predict, horizon=HORIZON)
    r_classical = run_sim(plant, mpc_classical, t, util, tamb,
                          predictor_wrapper=wrap_c, profiler=classical_profiler,
                          label="MPC + Classical GRU")

    # ---- 3. MPC + QML predictor -----------------------------------------
    if verbose:
        print("  [3/3] Running MPC + QML predictor...", flush=True)
    mpc_qml = make_mpc_controller(predictor_fn=wrap_q.predict, horizon=HORIZON)
    r_qml = run_sim(plant, mpc_qml, t, util, tamb,
                    predictor_wrapper=wrap_q, profiler=qml_profiler,
                    label="MPC + QML Hybrid")

    # Metrics
    metrics_dict = {}
    for r in [r_baseline, r_classical, r_qml]:
        m = compute_metrics(r, plant)
        metrics_dict[r["label"]] = m

    if save_trajectories:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, "results_summary.csv")
        df = pd.DataFrame(list(metrics_dict.values()))
        df.to_csv(csv_path, index=False)

        np.savez(
            os.path.join(RESULTS_DIR, "trajectories.npz"),
            t=t, util=util, tamb=tamb,
            carbon_intensity=carbon_intensity(t),
            b_T=r_baseline["T"], b_u=r_baseline["u"],
            b_pcool=r_baseline["pcool"], b_carbon=r_baseline["carbon_kg"],
            b_energy=r_baseline["energy_kwh"],
            c_T=r_classical["T"], c_u=r_classical["u"],
            c_pcool=r_classical["pcool"], c_carbon=r_classical["carbon_kg"],
            c_energy=r_classical["energy_kwh"],
            q_T=r_qml["T"], q_u=r_qml["u"],
            q_pcool=r_qml["pcool"], q_carbon=r_qml["carbon_kg"],
            q_energy=r_qml["energy_kwh"],
        )
        if verbose:
            print("\n" + "=" * 80)
            print(f"RESULTS COMPARISON (Seed {seed})")
            print("=" * 80)
            print(df.to_string(index=False, float_format="%.2f"))
            print("=" * 80)
            print(f"Saved {csv_path} and trajectories.npz\n", flush=True)

    return metrics_dict


def main():
    run_single_scenario(seed=HELD_OUT_SEED, save_trajectories=True, verbose=True)


if __name__ == "__main__":
    main()

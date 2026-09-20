"""
Causal Bias Intervention Test
==============================
Tests whether the measured near-term prediction bias difference between
Classical GRU and Hybrid QML CAUSES the ASHRAE compliance gap (87.81% vs
99.99%), or whether it is merely a correlated observation.

Six experimental conditions are run across the same 10 held-out seeds
(101-110), using the fast analytical-gradient MPC controller:

  1. GRU (Control)               -- Unmodified Classical GRU predictor
  2. QML (Control)               -- Unmodified Hybrid QML predictor
  3. GRU + QML Near Bias (1-4)   -- GRU output + (QML-GRU) bias at steps 1-4
  4. GRU + QML Full Bias (1-24)  -- GRU output + (QML-GRU) bias at all 24 steps
  5. QML - Own Near Bias (1-4)   -- QML output - QML's bias at steps 1-4
  6. QML - Own Full Bias (1-24)  -- QML output - QML's bias at all 24 steps

If bias is causal:
  - Condition 3 compliance should rise toward 99.99%
  - Condition 5 compliance should fall toward 87.81%

Outputs:
  - results/causal_bias_test_results.csv
  - Console verdict with plain-language interpretation
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from plant.thermal_plant import ThermalPlant
from plant.scenarios import random_scenario, carbon_intensity
from ml.train_classical import ClassicalPredictor
from qml.train_qml import QMLPredictor
from optimization.mpc_core_fast import make_mpc_controller_fast
from evaluation.run_comparison import PredictorWrapper
from diagnostics.bias_shifted_predictors import (
    make_gru_plus_qml_near_bias,
    make_gru_plus_qml_full_bias,
    make_qml_minus_qml_near_bias,
    make_qml_minus_qml_full_bias,
)

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
SEEDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
HORIZON = 24


def run_closed_loop(plant, controller, t, util, tamb, predictor_wrapper):
    """Run a single closed-loop MPC simulation and return key metrics."""
    n = len(t)
    carbon = carbon_intensity(t)
    T = np.empty(n)
    T[0] = plant.T_SET
    u_hist = np.empty(n)

    for k in range(n):
        h = min(HORIZON, n - k)
        util_fore = util[k: k + h].copy()
        tamb_fore = tamb[k: k + h].copy()
        carbon_fore = carbon[k: k + h].copy()
        if h < HORIZON:
            util_fore = np.concatenate([util_fore, np.full(HORIZON - h, util_fore[-1])])
            tamb_fore = np.concatenate([tamb_fore, np.full(HORIZON - h, tamb_fore[-1])])
            carbon_fore = np.concatenate([carbon_fore, np.full(HORIZON - h, carbon_fore[-1])])

        u = controller(T[k], step_idx=k,
                       util_fore=util_fore, tamb_fore=tamb_fore,
                       carbon_fore=carbon_fore)
        u_hist[k] = u

        # Update predictor history
        pcool_k, _ = plant.p_cool(u, tamb[k])
        carbon_k = carbon_intensity(t[k])
        predictor_wrapper.update(T[k], util[k], tamb[k], pcool_k, carbon_k)

        if k < n - 1:
            T[k + 1] = plant.step(T[k], u, util[k], tamb[k])

    # Compute metrics
    pcool, _ = plant.p_cool(u_hist, tamb)
    energy_kwh = float(np.sum(pcool * plant.DT_HR))
    carbon_kg = float(np.sum(pcool * carbon * plant.DT_HR) / 1000.0)
    pct_rec = float(100 * np.mean((T >= plant.T_REC_LO) & (T <= plant.T_REC_HI)))
    pct_alw = float(100 * np.mean((T >= plant.T_ALW_LO) & (T <= plant.T_ALW_HI)))
    max_temp = float(T.max())
    min_temp = float(T.min())

    return {
        "Energy_kWh": energy_kwh,
        "Carbon_kg": carbon_kg,
        "Pct_Recommended": pct_rec,
        "Pct_Allowable": pct_alw,
        "Max_Temp": max_temp,
        "Min_Temp": min_temp,
    }


def evaluate_seed(seed):
    """Run all 6 experimental conditions for a single seed."""
    plant = ThermalPlant()
    t, util, tamb = random_scenario(seed=seed, n_days=5)

    results = {}

    # --- 1. GRU (Control) ---
    gru_pred = ClassicalPredictor()
    wrap = PredictorWrapper(gru_pred)
    ctrl = make_mpc_controller_fast(predictor_fn=wrap.predict, horizon=HORIZON)
    results["GRU (Control)"] = run_closed_loop(plant, ctrl, t, util, tamb, wrap)

    # --- 2. QML (Control) ---
    qml_pred = QMLPredictor()
    wrap = PredictorWrapper(qml_pred)
    ctrl = make_mpc_controller_fast(predictor_fn=wrap.predict, horizon=HORIZON)
    results["QML (Control)"] = run_closed_loop(plant, ctrl, t, util, tamb, wrap)

    # --- 3. GRU + QML Near Bias (Steps 1-4) ---
    gru_pred_3 = ClassicalPredictor()
    shifted_3 = make_gru_plus_qml_near_bias(gru_pred_3)
    wrap = PredictorWrapper(shifted_3)
    ctrl = make_mpc_controller_fast(predictor_fn=wrap.predict, horizon=HORIZON)
    results["GRU + QML Near Bias (1-4)"] = run_closed_loop(plant, ctrl, t, util, tamb, wrap)

    # --- 4. GRU + QML Full Bias (Steps 1-24) ---
    gru_pred_4 = ClassicalPredictor()
    shifted_4 = make_gru_plus_qml_full_bias(gru_pred_4)
    wrap = PredictorWrapper(shifted_4)
    ctrl = make_mpc_controller_fast(predictor_fn=wrap.predict, horizon=HORIZON)
    results["GRU + QML Full Bias (1-24)"] = run_closed_loop(plant, ctrl, t, util, tamb, wrap)

    # --- 5. QML - Own Near Bias (Steps 1-4) ---
    qml_pred_5 = QMLPredictor()
    shifted_5 = make_qml_minus_qml_near_bias(qml_pred_5)
    wrap = PredictorWrapper(shifted_5)
    ctrl = make_mpc_controller_fast(predictor_fn=wrap.predict, horizon=HORIZON)
    results["QML - Own Near Bias (1-4)"] = run_closed_loop(plant, ctrl, t, util, tamb, wrap)

    # --- 6. QML - Own Full Bias (Steps 1-24) ---
    qml_pred_6 = QMLPredictor()
    shifted_6 = make_qml_minus_qml_full_bias(qml_pred_6)
    wrap = PredictorWrapper(shifted_6)
    ctrl = make_mpc_controller_fast(predictor_fn=wrap.predict, horizon=HORIZON)
    results["QML - Own Full Bias (1-24)"] = run_closed_loop(plant, ctrl, t, util, tamb, wrap)

    return seed, results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 90)
    print("CAUSAL BIAS INTERVENTION TEST (6 Conditions x 10 Seeds)")
    print("=" * 90)
    print("Testing whether near-term prediction bias difference CAUSES")
    print("the ASHRAE compliance gap, or is merely correlated.\n")
    print("Conditions:")
    print("  1. GRU (Control)               -- Unmodified baseline")
    print("  2. QML (Control)               -- Unmodified baseline")
    print("  3. GRU + QML Near Bias (1-4)   -- GRU + (QML-GRU) offset at steps 1-4")
    print("  4. GRU + QML Full Bias (1-24)  -- GRU + (QML-GRU) offset at all steps")
    print("  5. QML - Own Near Bias (1-4)   -- QML - QML's bias at steps 1-4")
    print("  6. QML - Own Full Bias (1-24)  -- QML - QML's bias at all steps")
    print(f"\nSeeds: {SEEDS}\n", flush=True)

    all_rows = []
    t0 = time.time()

    # Run with 4 parallel workers (each seed runs all 6 conditions)
    max_workers = 4
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_seed, seed): seed for seed in SEEDS}
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                s, results_dict = fut.result()
                for cond_name, metrics in results_dict.items():
                    row = {"Seed": s, "Condition": cond_name, **metrics}
                    all_rows.append(row)
                print(f"  [DONE] Seed {seed} -- all 6 conditions complete", flush=True)
            except Exception as e:
                print(f"  [ERROR] Seed {seed} failed: {e}", flush=True)
                import traceback
                traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\nAll 10 seeds x 6 conditions evaluated in {elapsed:.1f} s ({elapsed / 60:.2f} min)\n",
          flush=True)

    # Build DataFrame
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(RESULTS_DIR, "causal_bias_test_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved per-seed results to {csv_path}\n", flush=True)

    # ------------------------------------------------------------------
    # Summary Table
    # ------------------------------------------------------------------
    CONDITIONS_ORDER = [
        "GRU (Control)",
        "QML (Control)",
        "GRU + QML Near Bias (1-4)",
        "GRU + QML Full Bias (1-24)",
        "QML - Own Near Bias (1-4)",
        "QML - Own Full Bias (1-24)",
    ]

    print("=" * 90)
    print("SIX-WAY COMPARISON TABLE (Mean +/- Std across 10 Seeds)")
    print("=" * 90)
    header = (f"{'Condition':<32} {'% Recommended':<18} {'Max Temp (C)':<16} "
              f"{'Energy (kWh)':<18} {'Carbon (kg)':<16}")
    print(header)
    print("-" * 90)

    summary_rows = []
    for cond in CONDITIONS_ORDER:
        grp = df[df["Condition"] == cond]
        if len(grp) == 0:
            continue
        rec_mean = grp["Pct_Recommended"].mean()
        rec_std = grp["Pct_Recommended"].std()
        max_t_mean = grp["Max_Temp"].mean()
        max_t_std = grp["Max_Temp"].std()
        e_mean = grp["Energy_kWh"].mean()
        e_std = grp["Energy_kWh"].std()
        c_mean = grp["Carbon_kg"].mean()
        c_std = grp["Carbon_kg"].std()

        rec_str = f"{rec_mean:.2f} +/- {rec_std:.2f}%"
        max_t_str = f"{max_t_mean:.2f} +/- {max_t_std:.2f}"
        e_str = f"{e_mean:.0f} +/- {e_std:.0f}"
        c_str = f"{c_mean:.0f} +/- {c_std:.0f}"

        print(f"{cond:<32} {rec_str:<18} {max_t_str:<16} {e_str:<18} {c_str:<16}")
        summary_rows.append({
            "Condition": cond,
            "Recommended_Mean": rec_mean,
            "Recommended_Std": rec_std,
            "MaxTemp_Mean": max_t_mean,
            "MaxTemp_Std": max_t_std,
            "Energy_Mean": e_mean,
            "Energy_Std": e_std,
            "Carbon_Mean": c_mean,
            "Carbon_Std": c_std,
        })

    print("=" * 90 + "\n")

    # Save summary
    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(RESULTS_DIR, "causal_bias_test_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved summary to {summary_csv}\n", flush=True)

    # ------------------------------------------------------------------
    # Automatic Verdict
    # ------------------------------------------------------------------
    def get_stats(condition):
        grp = df[df["Condition"] == condition]
        return {
            "rec_mean": grp["Pct_Recommended"].mean(),
            "rec_std": grp["Pct_Recommended"].std(),
            "max_t_mean": grp["Max_Temp"].mean(),
        }

    gru_ctrl = get_stats("GRU (Control)")
    qml_ctrl = get_stats("QML (Control)")
    gru_near = get_stats("GRU + QML Near Bias (1-4)")
    gru_full = get_stats("GRU + QML Full Bias (1-24)")
    qml_near = get_stats("QML - Own Near Bias (1-4)")
    qml_full = get_stats("QML - Own Full Bias (1-24)")

    gru_base = gru_ctrl["rec_mean"]
    qml_base = qml_ctrl["rec_mean"]
    gap = qml_base - gru_base  # ~12.18 pp

    print("=" * 90)
    print("CAUSAL VERDICT")
    print("=" * 90)
    print(f"Reference GRU compliance:  {gru_base:.2f}%")
    print(f"Reference QML compliance:  {qml_base:.2f}%")
    print(f"Original compliance gap:   {gap:+.2f} pp (QML - GRU)\n")

    # Intervention effects
    gru_near_lift = gru_near["rec_mean"] - gru_base
    gru_full_lift = gru_full["rec_mean"] - gru_base
    qml_near_drop = qml_near["rec_mean"] - qml_base
    qml_full_drop = qml_full["rec_mean"] - qml_base

    print(f"GRU + Near Bias compliance: {gru_near['rec_mean']:.2f}% (delta: {gru_near_lift:+.2f} pp from GRU control)")
    print(f"GRU + Full Bias compliance: {gru_full['rec_mean']:.2f}% (delta: {gru_full_lift:+.2f} pp from GRU control)")
    print(f"QML - Near Bias compliance: {qml_near['rec_mean']:.2f}% (delta: {qml_near_drop:+.2f} pp from QML control)")
    print(f"QML - Full Bias compliance: {qml_full['rec_mean']:.2f}% (delta: {qml_full_drop:+.2f} pp from QML control)")
    print("")

    # Thresholds for "meaningful" movement
    # We consider closing >= 30% of the gap as meaningful causal evidence
    meaningful_frac = 0.30
    meaningful_pp = meaningful_frac * gap  # ~3.65 pp

    print(f"Threshold for 'meaningful' causal evidence: closing >= {meaningful_frac*100:.0f}% of the "
          f"{gap:.1f} pp gap = {meaningful_pp:.1f} pp\n")
    print("-" * 90)

    # Test 1: Does grafting QML's near-term bias onto GRU boost compliance?
    near_closes_gap = gru_near_lift / gap if gap > 0 else 0
    full_closes_gap = gru_full_lift / gap if gap > 0 else 0

    print("TEST 1: Does grafting QML's bias onto GRU boost compliance toward 99.99%?")
    if gru_near_lift >= meaningful_pp:
        print(f"  POSITIVE: GRU + Near Bias closes {near_closes_gap*100:.1f}% of the gap ({gru_near_lift:+.2f} pp).")
        print(f"  Near-term bias (steps 1-4) is a REAL CAUSAL DRIVER of the compliance advantage.")
    else:
        print(f"  NEGATIVE: GRU + Near Bias only closes {near_closes_gap*100:.1f}% of the gap ({gru_near_lift:+.2f} pp).")
        print(f"  Near-term bias alone does NOT explain the compliance advantage.")

    print("")
    print("TEST 2: Is the effect distributed across the full horizon, or concentrated in steps 1-4?")
    if gru_full_lift >= meaningful_pp:
        full_vs_near = gru_full_lift - gru_near_lift
        if full_vs_near > meaningful_pp * 0.3:
            print(f"  Full-horizon bias adds +{full_vs_near:.2f} pp beyond near-term only.")
            print(f"  The causal effect is DISTRIBUTED across the horizon, not just steps 1-4.")
        else:
            print(f"  Full-horizon bias adds only +{full_vs_near:.2f} pp beyond near-term.")
            print(f"  The causal effect is CONCENTRATED in near-term steps 1-4.")
    else:
        print(f"  NEGATIVE: Even full-horizon bias only closes {full_closes_gap*100:.1f}% of the gap.")

    print("")
    print("TEST 3: Does removing QML's own near-term bias drop compliance toward GRU's level?")
    if qml_near_drop <= -meaningful_pp:
        near_erodes_gap = abs(qml_near_drop) / gap if gap > 0 else 0
        print(f"  CONFIRMING: QML - Near Bias drops {near_erodes_gap*100:.1f}% of the gap ({qml_near_drop:+.2f} pp).")
        print(f"  Removing QML's near-term positive bias degrades compliance -- confirming causal role.")
    else:
        print(f"  NEGATIVE: QML - Near Bias only drops {qml_near_drop:+.2f} pp from QML control.")
        print(f"  QML's compliance is NOT driven by its near-term bias.")

    print("")
    print("TEST 4: Full bias removal from QML")
    if qml_full_drop <= -meaningful_pp:
        full_erodes_gap = abs(qml_full_drop) / gap if gap > 0 else 0
        print(f"  QML - Full Bias drops {full_erodes_gap*100:.1f}% of the gap ({qml_full_drop:+.2f} pp).")
    else:
        print(f"  QML - Full Bias only drops {qml_full_drop:+.2f} pp from QML control.")

    print("")
    print("-" * 90)
    print("OVERALL CAUSAL CONCLUSION:")
    print("-" * 90)

    # Determine overall verdict
    gru_side_causal = (gru_near_lift >= meaningful_pp) or (gru_full_lift >= meaningful_pp)
    qml_side_causal = (qml_near_drop <= -meaningful_pp) or (qml_full_drop <= -meaningful_pp)

    if gru_side_causal and qml_side_causal:
        print("  STRONG CAUSAL EVIDENCE: Both directions confirm that prediction bias")
        print("  differences between GRU and QML ARE a causal driver of the ASHRAE compliance gap.")
        if gru_near_lift >= meaningful_pp and qml_near_drop <= -meaningful_pp:
            print("  The effect is concentrated in NEAR-TERM steps 1-4 (5-20 min ahead).")
        elif gru_full_lift >= meaningful_pp and qml_full_drop <= -meaningful_pp:
            print("  The effect requires the FULL HORIZON bias profile, not just near-term steps.")
    elif gru_side_causal or qml_side_causal:
        print("  PARTIAL CAUSAL EVIDENCE: One direction supports the bias hypothesis,")
        print("  but the other does not confirm. The bias mechanism is a contributing factor")
        print("  but likely interacts with other aspects of the prediction trajectory")
        print("  (e.g., trajectory shape, dynamics, or optimizer sensitivity).")
    else:
        print("  NO CAUSAL EVIDENCE: Neither adding QML's bias to GRU nor removing QML's")
        print("  own bias meaningfully moves compliance. The prediction bias difference is")
        print("  NOT the causal mechanism behind the compliance gap.")
        print("  The compliance gap likely stems from other differences in how the two")
        print("  models' full predicted trajectories interact with the MPC optimizer --")
        print("  such as trajectory shape, curvature, step-to-step dynamics, or how")
        print("  the optimizer's cost landscape responds to the complete prediction profile.")

    print("=" * 90 + "\n", flush=True)


if __name__ == "__main__":
    main()

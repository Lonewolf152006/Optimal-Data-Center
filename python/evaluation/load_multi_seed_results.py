"""
Multi-Seed Results Loader and Hypothesis Testing
================================================
Post-processing utility to load and display multi-seed evaluation results
once the background pipeline run finishes.

Checks:
1. File existence (results/multi_seed_results.csv, results/multi_seed_summary.csv)
2. Seed completion count (reports count if < 10)
3. Clean summary table: Mean ± Std for Energy, Carbon, % ASHRAE Recommended Band
4. Hypothesis test on % ASHRAE Recommended compliance: interval overlap check
"""

import os
import sys
import numpy as np
import pandas as pd

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "multi_seed_results.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "multi_seed_summary.csv")


def load_and_report():
    if not os.path.exists(RESULTS_CSV) or not os.path.exists(SUMMARY_CSV):
        print("not ready yet")
        return

    df_results = pd.read_csv(RESULTS_CSV)
    df_summary = pd.read_csv(SUMMARY_CSV)

    unique_seeds = df_results["Seed"].unique()
    num_seeds = len(unique_seeds)

    if num_seeds < 10:
        print(f"Results partially available ({num_seeds}/10 seeds present: {list(unique_seeds)}). Not all 10 seeds ready yet.")
        return

    print("=" * 88)
    print(f"MULTI-SEED EVALUATION RESULTS ({num_seeds} HELD-OUT SCENARIO SEEDS)")
    print("=" * 88)
    print(f"Seeds evaluated: {sorted(list(unique_seeds))}\n")

    # Table of Mean +/- Std across completed seeds
    controllers = ["Baseline thermostat", "MPC + Classical GRU", "MPC + QML Hybrid"]
    metrics_to_show = [
        ("Energy (kWh)", "Cooling Energy (kWh)"),
        ("Carbon (kg CO2)", "Carbon Footprint (kg CO2)"),
        ("% Recommended (18-27C)", "% ASHRAE Recommended"),
        ("% Allowable (15-32C)", "% ASHRAE Allowable"),
    ]

    print(f"{'Controller':<26} {'Cooling Energy (kWh)':<22} {'Carbon (kg CO2)':<20} {'% Recommended':<18}")
    print("-" * 88)

    for ctrl in controllers:
        c_res = df_results[df_results["Controller"] == ctrl]
        if len(c_res) == 0:
            continue
        e_mean, e_std = c_res["Energy (kWh)"].mean(), c_res["Energy (kWh)"].std()
        c_mean, c_std = c_res["Carbon (kg CO2)"].mean(), c_res["Carbon (kg CO2)"].std()
        r_mean, r_std = c_res["% Recommended (18-27C)"].mean(), c_res["% Recommended (18-27C)"].std()

        print(f"{ctrl:<26} {e_mean:8.1f} ± {e_std:5.1f} kWh     {c_mean:7.1f} ± {c_std:4.1f} kg     {r_mean:6.2f} ± {r_std:4.2f} %")
    print("=" * 88 + "\n")

    # Statistical hypothesis test on % Recommended compliance
    gru_data = df_results[df_results["Controller"] == "MPC + Classical GRU"]["% Recommended (18-27C)"]
    qml_data = df_results[df_results["Controller"] == "MPC + QML Hybrid"]["% Recommended (18-27C)"]

    gru_mean, gru_std = gru_data.mean(), gru_data.std()
    qml_mean, qml_std = qml_data.mean(), qml_data.std()

    gru_interval = (gru_mean - gru_std, gru_mean + gru_std)
    qml_interval = (qml_mean - qml_std, qml_mean + qml_std)

    overlap = not (qml_interval[0] > gru_interval[1] or gru_interval[0] > qml_interval[1])

    print("=" * 88)
    print("HYPOTHESIS TEST: ASHRAE RECOMMENDED-BAND COMPLIANCE (ROBUSTNESS CHECK)")
    print("=" * 88)
    print(f"Classical GRU Compliance Mean ± Std : {gru_mean:6.2f}% ± {gru_std:4.2f}%  ->  Range: [{gru_interval[0]:.2f}%, {gru_interval[1]:.2f}%]")
    print(f"Hybrid QML    Compliance Mean ± Std : {qml_mean:6.2f}% ± {qml_std:4.2f}%  ->  Range: [{qml_interval[0]:.2f}%, {qml_interval[1]:.2f}%]")
    print(f"Mean Difference (QML - GRU)         : {qml_mean - gru_mean:+.2f} percentage points")
    print("-" * 88)
    if overlap:
        print("EMPIRICAL FINDING: The Mean ± Std compliance ranges OVERLAP across the 10 seeds.")
        print("  - The single-seed compliance advantage observed on seed 999 does NOT represent")
        print("    a statistically distinct quantum advantage.")
        print("  - The observed difference is within the cross-scenario variation noise.")
    else:
        print("EMPIRICAL FINDING: The Mean ± Std compliance ranges DO NOT OVERLAP.")
        winner = "Hybrid QML" if qml_mean > gru_mean else "Classical GRU"
        print(f"  - {winner} maintains a statistically distinct advantage across held-out seeds.")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    load_and_report()

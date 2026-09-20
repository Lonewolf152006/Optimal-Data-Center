"""
Dataset Generator
==================
Runs 100 randomised scenarios using the baseline thermostat on the
thermal plant and logs every timestep to ``data/thermal_dataset.csv``.

Columns:
  time, run_id, utilization, ambient_temp, room_temp, cooling_power,
  carbon_intensity
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from plant.thermal_plant import ThermalPlant
from plant.scenarios import random_scenario, carbon_intensity
from controllers.baseline_thermostat import BaselineThermostat

# Paths (relative to repo root)
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DATA_DIR = os.path.join(REPO_ROOT, "data")
CSV_PATH = os.path.join(DATA_DIR, "thermal_dataset.csv")

N_RUNS = 100
N_DAYS = 5


def generate_dataset():
    plant = ThermalPlant()
    os.makedirs(DATA_DIR, exist_ok=True)
    all_rows = []

    for run_id in range(N_RUNS):
        t, util, tamb = random_scenario(seed=run_id, n_days=N_DAYS)
        carbon = carbon_intensity(t)
        n = len(t)

        controller = BaselineThermostat()
        T_room = np.empty(n)
        T_room[0] = plant.T_SET
        u_hist = np.empty(n)

        for k in range(n):
            u = controller(T_room[k])
            u_hist[k] = u
            if k < n - 1:
                T_room[k + 1] = plant.step(T_room[k], u, util[k], tamb[k])

        pcool, _ = plant.p_cool(u_hist, tamb)

        for k in range(n):
            all_rows.append({
                "time": t[k],
                "run_id": run_id,
                "utilization": util[k],
                "ambient_temp": tamb[k],
                "room_temp": T_room[k],
                "cooling_power": pcool[k],
                "carbon_intensity": carbon[k],
            })

        if (run_id + 1) % 10 == 0:
            print(f"  completed run {run_id + 1}/{N_RUNS}")

    df = pd.DataFrame(all_rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nDataset saved: {CSV_PATH}")
    print(f"  {len(df)} rows, {df['run_id'].nunique()} runs")
    return df


if __name__ == "__main__":
    print("Generating dataset (100 runs × 5 days each)...\n")
    generate_dataset()

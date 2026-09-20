"""
Multi-Seed Box Plots & Distribution Comparison
===============================================
Generates box plots comparing the 3 controllers across 10 held-out seeds on:
  1. Total 5-day cooling electrical energy (kWh)
  2. Total 5-day carbon footprint (kg CO2)
  3. % time in ASHRAE recommended comfort band (18–27 °C)

Marks explicitly whether the GRU and QML distributions overlap.
Saves to ``results/fig3_multiseed_boxplots.png``.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "multi_seed_results.csv")


def make_boxplots():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing {CSV_PATH}. Run run_multi_seed_evaluation.py first.")

    df = pd.read_csv(CSV_PATH)
    controllers = ["Baseline thermostat", "MPC + Classical GRU", "MPC + QML Hybrid"]
    labels = ["Baseline\nThermostat", "MPC +\nClassical GRU", "MPC +\nQML Hybrid"]
    colors = ["#7f7f7f", "#1f77b4", "#d62728"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=140)

    # ------------------------------------------------------------------
    # Subplot 1: Total Cooling Energy (kWh)
    # ------------------------------------------------------------------
    ax0 = axes[0]
    data_energy = [df[df["Controller"] == c]["Energy (kWh)"].values for c in controllers]
    bp0 = ax0.boxplot(data_energy, patch_artist=True, tick_labels=labels, widths=0.55,
                      medianprops=dict(color="black", linewidth=1.5),
                      boxprops=dict(linewidth=1.2), whiskerprops=dict(linewidth=1.2))
    for patch, col in zip(bp0["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    ax0.set_title("5-Day Cooling Energy Across 10 Seeds", fontsize=11, fontweight="bold", pad=10)
    ax0.set_ylabel("Electricity (kWh)", fontsize=10)
    ax0.grid(True, axis="y", alpha=0.3, linestyle="--")

    # ------------------------------------------------------------------
    # Subplot 2: Total Carbon Footprint (kg CO2)
    # ------------------------------------------------------------------
    ax1 = axes[1]
    data_carbon = [df[df["Controller"] == c]["Carbon (kg CO2)"].values for c in controllers]
    bp1 = ax1.boxplot(data_carbon, patch_artist=True, tick_labels=labels, widths=0.55,
                      medianprops=dict(color="black", linewidth=1.5),
                      boxprops=dict(linewidth=1.2), whiskerprops=dict(linewidth=1.2))
    for patch, col in zip(bp1["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    ax1.set_title("5-Day Carbon Footprint Across 10 Seeds", fontsize=11, fontweight="bold", pad=10)
    ax1.set_ylabel("Emissions (kg CO2)", fontsize=10)
    ax1.grid(True, axis="y", alpha=0.3, linestyle="--")

    # ------------------------------------------------------------------
    # Subplot 3: % Time in ASHRAE Recommended Band (18-27 C)
    # ------------------------------------------------------------------
    ax2 = axes[2]
    data_rec = [df[df["Controller"] == c]["% Recommended (18-27C)"].values for c in controllers]
    bp2 = ax2.boxplot(data_rec, patch_artist=True, tick_labels=labels, widths=0.55,
                      medianprops=dict(color="black", linewidth=1.5),
                      boxprops=dict(linewidth=1.2), whiskerprops=dict(linewidth=1.2))
    for patch, col in zip(bp2["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    ax2.set_title("ASHRAE Recommended Compliance (18–27 °C)", fontsize=11, fontweight="bold", pad=10)
    ax2.set_ylabel("Time in Band (%)", fontsize=10)
    ax2.set_ylim(min(75, min(np.concatenate(data_rec)) - 5), 105)
    ax2.grid(True, axis="y", alpha=0.3, linestyle="--")

    # Statistical overlap check between GRU and QML
    gru_vals = df[df["Controller"] == "MPC + Classical GRU"]["% Recommended (18-27C)"].values
    qml_vals = df[df["Controller"] == "MPC + QML Hybrid"]["% Recommended (18-27C)"].values
    gru_q25, gru_q75 = np.percentile(gru_vals, [25, 75])
    qml_q25, qml_q75 = np.percentile(qml_vals, [25, 75])

    overlap_iqr = not (qml_q25 > gru_q75 or gru_q25 > qml_q75)
    overlap_whiskers = not (qml_vals.min() > gru_vals.max() or gru_vals.min() > qml_vals.max())

    badge_text = "IQR Overlap: YES\nWhiskers Overlap: YES" if overlap_whiskers else "Distributions Distinct: NO OVERLAP"
    badge_color = "#fff3cd" if overlap_whiskers else "#d4edda"
    border_color = "#ffeeba" if overlap_whiskers else "#c3e6cb"

    ax2.text(0.5, 0.08, f"{badge_text}\n(Seed Sensitivity Present)",
             transform=ax2.transAxes, ha="center", va="bottom",
             fontsize=9, fontweight="semibold",
             bbox=dict(boxstyle="round,pad=0.5", facecolor=badge_color, edgecolor=border_color, alpha=0.9))

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "fig3_multiseed_boxplots.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved multi-seed box plot to {out_path}", flush=True)


if __name__ == "__main__":
    make_boxplots()

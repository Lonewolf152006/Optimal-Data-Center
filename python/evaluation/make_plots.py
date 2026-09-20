"""
Comparison Plots — Three Controllers
======================================
Overlays all THREE controllers (baseline thermostat, MPC + classical,
MPC + QML) on:
  1. Temperature trajectories with ASHRAE bands
  2. Cooling electrical power
  3. Cumulative carbon footprint
  4. Summary bar chart (energy / carbon / time-in-band)

Reads ``results/trajectories.npz`` produced by ``run_comparison.py``.
Saves figures to ``results/``.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
NPZ_PATH = os.path.join(RESULTS_DIR, "trajectories.npz")

T_REC_LO, T_REC_HI = 18.0, 27.0
T_ALW_LO, T_ALW_HI = 15.0, 32.0

COLORS = {
    "baseline": "#888780",
    "classical": "#185FA5",
    "qml": "#D4442A",
}
LABELS = {
    "baseline": "Baseline thermostat",
    "classical": "MPC + Classical GRU",
    "qml": "MPC + QML Hybrid",
}


def main():
    d = np.load(NPZ_PATH)
    t = d["t"]

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})

    # ==================================================================
    # Figure 1: time-series (temperature / cooling power / carbon)
    # ==================================================================
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # --- Temperature ---
    ax = axes[0]
    ax.axhspan(T_REC_LO, T_REC_HI, color="green", alpha=0.08, label="ASHRAE recommended")
    ax.axhline(T_ALW_HI, color="gray", ls=":", lw=1)
    ax.axhline(T_ALW_LO, color="gray", ls=":", lw=1)
    ax.plot(t, d["b_T"], color=COLORS["baseline"], lw=1.2, label=LABELS["baseline"])
    ax.plot(t, d["c_T"], color=COLORS["classical"], lw=1.4, label=LABELS["classical"])
    ax.plot(t, d["q_T"], color=COLORS["qml"], lw=1.4, ls="--", label=LABELS["qml"])
    ylim = ax.get_ylim()
    ax.set_ylim(min(ylim[0], T_ALW_LO - 1), max(ylim[1], T_ALW_HI + 1))
    ax.set_ylabel("Room air temp (°C)")
    ax.set_title("Room temperature — green = ASHRAE recommended (18–27 °C), dotted = allowable limits")
    ax.legend(loc="upper left", ncol=2, fontsize=9)

    # --- Cooling power ---
    ax = axes[1]
    ax.plot(t, d["b_pcool"], color=COLORS["baseline"], lw=1.0, label=LABELS["baseline"])
    ax.plot(t, d["c_pcool"], color=COLORS["classical"], lw=1.2, label=LABELS["classical"])
    ax.plot(t, d["q_pcool"], color=COLORS["qml"], lw=1.2, ls="--", label=LABELS["qml"])
    ax.set_ylabel("Cooling power (kW)")
    ax.set_title("Cooling electrical power (fan + chiller)")

    # --- Cumulative carbon ---
    ax = axes[2]
    ax.plot(t, d["b_carbon"], color=COLORS["baseline"], lw=1.4, label=LABELS["baseline"])
    ax.plot(t, d["c_carbon"], color=COLORS["classical"], lw=1.6, label=LABELS["classical"])
    ax.plot(t, d["q_carbon"], color=COLORS["qml"], lw=1.6, ls="--", label=LABELS["qml"])
    ax.set_ylabel("Cumulative CO₂ (kg)")
    ax.set_xlabel("Simulation time (hours)")
    ax.set_title("Cumulative carbon footprint")

    for a in axes:
        for dnum in range(1, 5):
            a.axvline(dnum * 24, color="black", lw=0.5, alpha=0.3)

    plt.tight_layout()
    fig1_path = os.path.join(RESULTS_DIR, "fig1_timeseries.png")
    plt.savefig(fig1_path, dpi=140)
    print(f"Saved {fig1_path}")

    # ==================================================================
    # Figure 2: summary bar chart (3 controllers)
    # ==================================================================
    labels_bar = ["Baseline\nthermostat", "MPC +\nClassical GRU", "MPC +\nQML Hybrid"]
    colors_bar = [COLORS["baseline"], COLORS["classical"], COLORS["qml"]]

    energies = [d["b_energy"][-1], d["c_energy"][-1], d["q_energy"][-1]]
    carbons = [d["b_carbon"][-1], d["c_carbon"][-1], d["q_carbon"][-1]]
    pct_rec = [
        100 * np.mean((d["b_T"] >= T_REC_LO) & (d["b_T"] <= T_REC_HI)),
        100 * np.mean((d["c_T"] >= T_REC_LO) & (d["c_T"] <= T_REC_HI)),
        100 * np.mean((d["q_T"] >= T_REC_LO) & (d["q_T"] <= T_REC_HI)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    def bar(ax, vals, title, ylabel, fmt="{:.0f}"):
        bars = ax.bar(labels_bar, vals, color=colors_bar, width=0.55)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v,
                    fmt.format(v), ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, max(vals) * 1.18)

    bar(axes[0], energies, "5-day cooling energy", "kWh")
    bar(axes[1], carbons, "5-day carbon footprint", "kg CO2")
    bar(axes[2], pct_rec, "Time in ASHRAE\nrecommended band", "%", fmt="{:.1f}")

    plt.tight_layout()
    fig2_path = os.path.join(RESULTS_DIR, "fig2_summary.png")
    plt.savefig(fig2_path, dpi=140)
    print(f"Saved {fig2_path}")


if __name__ == "__main__":
    main()

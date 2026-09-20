"""
Generate Figure 6: Multi-Rack Spatial Workload Placement & Hot-Spot Elimination
================================================================================
Visualizes:
  - Panel A: 4-Rack inlet temperature time series (Uniform vs. Optimal placement).
  - Panel B: Spatial workload distribution across racks (Critical vs. Batch).
  - Panel C: Hot-spot duration and critical task SLA violation comparison.
  - Panel D: Semiconductor reliability (Arrhenius MTBF) across server racks.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def make_spatial_plots():
    traj_path = os.path.join(RESULTS_DIR, "spatial_trajectories.npz")
    summary_path = os.path.join(RESULTS_DIR, "spatial_workload_summary.csv")
    results_path = os.path.join(RESULTS_DIR, "spatial_workload_results.csv")

    if not os.path.exists(traj_path) or not os.path.exists(results_path):
        print("Missing trajectory or results files. Run run_spatial_workload_evaluation.py first.")
        return

    trajs = np.load(traj_path)
    df_results = pd.read_csv(results_path)

    t_hr = trajs["t"]
    unaware_inlet = trajs["unaware_inlet"]
    optimal_inlet = trajs["optimal_inlet"]
    unaware_wcrit = trajs["unaware_wcrit"]
    optimal_wcrit = trajs["optimal_wcrit"]
    unaware_wtot = trajs["unaware_wtot"]
    optimal_wtot = trajs["optimal_wtot"]

    # Style settings
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.family"] = "sans-serif"
    fig = plt.figure(figsize=(16, 11), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.24)

    rack_colors = ["#2563EB", "#059669", "#D97706", "#DC2626"]  # Blue, Green, Amber, Red

    # -------------------------------------------------------------------------
    # Panel A: Rack Inlet Temperatures (Zoom-in Days 2-4: Hours 48-96)
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    mask = (t_hr >= 48) & (t_hr <= 96)
    t_zoom = t_hr[mask]

    # Baseline (Uniform) as dashed lines, Optimal as solid lines
    for i in range(4):
        ax_a.plot(t_zoom, unaware_inlet[mask, i], color=rack_colors[i], linestyle="--", alpha=0.65,
                  label=f"Rack {i+1} (Uniform)" if i in [0, 3] else None)
        ax_a.plot(t_zoom, optimal_inlet[mask, i], color=rack_colors[i], linestyle="-", linewidth=2.0,
                  label=f"Rack {i+1} (Thermal-Aware)")

    ax_a.axhline(24.0, color="#B91C1C", linestyle=":", linewidth=1.8, label="Critical Target (24°C)")
    ax_a.axhline(27.0, color="#4B5563", linestyle="-.", linewidth=1.5, label="ASHRAE Limit (27°C)")

    ax_a.set_title("Panel A: 4-Rack Inlet Temperatures (Uniform vs. Optimal)", fontsize=12, fontweight="bold", pad=8)
    ax_a.set_xlabel("Scenario Time (Hours)", fontsize=10)
    ax_a.set_ylabel("Rack Inlet Temperature (°C)", fontsize=10)
    ax_a.set_xlim(48, 96)
    ax_a.set_ylim(16.5, 28.5)
    ax_a.grid(True, alpha=0.3, linestyle="--")
    ax_a.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)

    # Highlight hot-spot zone
    ax_a.fill_between(t_zoom, 24.0, 28.5, color="#FEE2E2", alpha=0.35, zorder=0)
    ax_a.text(50, 27.5, "Severe Recirculation Hot Spot (Rack 4)", color="#991B1B", fontsize=8.5, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel B: Mean Workload Allocation per Rack (Critical vs. Batch)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    racks = ["Rack 1\n(Closest CRAC)", "Rack 2\n(Mid-Left)", "Rack 3\n(Mid-Right)", "Rack 4\n(Dead-End)"]
    x = np.arange(4)
    width = 0.35

    # Uniform mean utilization
    un_crit_mean = np.mean(unaware_wcrit, axis=0)
    un_batch_mean = np.mean(unaware_wtot - unaware_wcrit, axis=0)

    # Optimal mean utilization
    opt_crit_mean = np.mean(optimal_wcrit, axis=0)
    opt_batch_mean = np.mean(optimal_wtot - optimal_wcrit, axis=0)

    # Uniform bars
    p1 = ax_b.bar(x - width/2, un_crit_mean, width, label="Uniform: Critical", color="#3B82F6", alpha=0.85, edgecolor="#1E3A8A")
    p2 = ax_b.bar(x - width/2, un_batch_mean, width, bottom=un_crit_mean, label="Uniform: Batch", color="#93C5FD", alpha=0.85, edgecolor="#1E3A8A")

    # Optimal bars
    p3 = ax_b.bar(x + width/2, opt_crit_mean, width, label="Optimal: Critical", color="#10B981", alpha=0.9, edgecolor="#064E3B")
    p4 = ax_b.bar(x + width/2, opt_batch_mean, width, bottom=opt_crit_mean, label="Optimal: Batch", color="#A7F3D0", alpha=0.9, edgecolor="#064E3B")

    ax_b.set_title("Panel B: Spatial Workload Allocation across Racks", fontsize=12, fontweight="bold", pad=8)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(racks, fontsize=9.5)
    ax_b.set_ylabel("Mean Rack Workload Utilization", fontsize=10)
    ax_b.set_ylim(0, 1.05)
    ax_b.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax_b.legend(loc="upper right", fontsize=8.5, ncol=2, framealpha=0.9)

    # Annotation arrow
    ax_b.annotate("Critical tasks shifted to\ncoolest supply racks",
                  xy=(0.175, 0.75), xytext=(0.4, 0.90),
                  arrowprops=dict(arrowstyle="->", color="#065F46", lw=1.5),
                  fontsize=8.5, fontweight="bold", color="#065F46")

    # -------------------------------------------------------------------------
    # Panel C: Hot-Spot Duration & Critical Task SLA Violations (10 Seeds)
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])

    un_res = df_results[df_results["Condition"] == "Thermal-Unaware (Uniform)"]
    opt_res = df_results[df_results["Condition"] == "Thermal-Aware (Optimal)"]

    categories = ["Hot-Spot Duration\n>24°C (Hours)", "Critical SLA\nViolation Rate (%)"]
    un_vals = [un_res["Hotspot_Minutes"].mean() / 60.0, un_res["Crit_Violation_Pct"].mean()]
    un_errs = [un_res["Hotspot_Minutes"].std() / 60.0, un_res["Crit_Violation_Pct"].std()]

    opt_vals = [opt_res["Hotspot_Minutes"].mean() / 60.0, opt_res["Crit_Violation_Pct"].mean()]
    opt_errs = [opt_res["Hotspot_Minutes"].std() / 60.0, opt_res["Crit_Violation_Pct"].std()]

    x_c = np.arange(2)
    b1 = ax_c.bar(x_c - width/2, un_vals, width, yerr=un_errs, capsize=5,
                  color="#EF4444", alpha=0.85, edgecolor="#7F1D1D", label="Thermal-Unaware (Uniform)")
    b2 = ax_c.bar(x_c + width/2, opt_vals, width, yerr=opt_errs, capsize=5,
                  color="#10B981", alpha=0.85, edgecolor="#064E3B", label="Thermal-Aware (Optimal)")

    ax_c.set_title("Panel C: Hot-Spot & SLA Violation Comparison (10 Seeds)", fontsize=12, fontweight="bold", pad=8)
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax_c.set_ylabel("Duration (Hours) / Rate (%)", fontsize=10)
    ax_c.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax_c.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Value labels on bars
    for bar in b1:
        h = bar.get_height()
        ax_c.text(bar.get_x() + bar.get_width()/2., h + 1.2, f"{h:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#991B1B")
    for bar in b2:
        h = bar.get_height()
        ax_c.text(bar.get_x() + bar.get_width()/2., h + 1.2, f"{h:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#065F46")

    # -------------------------------------------------------------------------
    # Panel D: Relative MTBF / Hardware Reliability (Arrhenius Model)
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])

    mtbf_labels = ["Critical Tasks", "Rack 1\n(Cool)", "Rack 2\n(Mid)", "Rack 3\n(Mid)", "Rack 4\n(Hot Spot)", "Facility\nOverall"]
    x_d = np.arange(len(mtbf_labels))
    w_d = 0.35

    # Compute per-rack relative MTBF for uniform and optimal from saved results
    un_crit_mtbf = un_res["Crit_MTBF"].mean()
    opt_crit_mtbf = opt_res["Crit_MTBF"].mean()
    un_ov_mtbf = un_res["Overall_MTBF"].mean()
    opt_ov_mtbf = opt_res["Overall_MTBF"].mean()

    # Per-rack approximations from trajectories
    un_af_racks = np.mean(trajs["unaware_junc"], axis=0)  # surrogate for relative comparison
    # We can use accurate Arrhenius calculation
    T_j_ref_k = 53.5 + 273.15
    E_a = 0.70
    k_B = 8.6173e-5

    un_af = np.mean(np.exp((E_a / k_B) * (1.0 / T_j_ref_k - 1.0 / (trajs["unaware_junc"] + 273.15))), axis=0)
    opt_af = np.mean(np.exp((E_a / k_B) * (1.0 / T_j_ref_k - 1.0 / (trajs["optimal_junc"] + 273.15))), axis=0)

    un_mtbfs = [un_crit_mtbf, 1.0/un_af[0], 1.0/un_af[1], 1.0/un_af[2], 1.0/un_af[3], un_ov_mtbf]
    opt_mtbfs = [opt_crit_mtbf, 1.0/opt_af[0], 1.0/opt_af[1], 1.0/opt_af[2], 1.0/opt_af[3], opt_ov_mtbf]

    b_m1 = ax_d.bar(x_d - w_d/2, un_mtbfs, w_d, color="#64748B", alpha=0.85, edgecolor="#1E293B", label="Uniform Placement")
    b_m2 = ax_d.bar(x_d + w_d/2, opt_mtbfs, w_d, color="#8B5CF6", alpha=0.85, edgecolor="#4C1D95", label="Optimal Placement")

    ax_d.axhline(1.0, color="#374151", linestyle="--", linewidth=1.2, label="Nominal Reference (1.0×)")
    ax_d.set_title("Panel D: Relative Server Hardware Lifespan (Arrhenius MTBF)", fontsize=12, fontweight="bold", pad=8)
    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(mtbf_labels, fontsize=9)
    ax_d.set_ylabel("Relative MTBF (vs. Nominal 1.00×)", fontsize=10)
    ax_d.set_ylim(0, 1.25)
    ax_d.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax_d.legend(loc="upper right", fontsize=8.5, ncol=3, framealpha=0.9)

    # Highlight Rack 4 and Critical gains
    crit_gain = (opt_crit_mtbf - un_crit_mtbf) / un_crit_mtbf * 100.0
    r4_gain = (opt_mtbfs[4] - un_mtbfs[4]) / un_mtbfs[4] * 100.0
    ax_d.text(0, opt_crit_mtbf + 0.05, f"+{crit_gain:.1f}%", ha="center", color="#6D28D9", fontsize=8.5, fontweight="bold")
    ax_d.text(4, opt_mtbfs[4] + 0.05, f"+{r4_gain:.1f}%", ha="center", color="#6D28D9", fontsize=8.5, fontweight="bold")

    plt.suptitle("MathWorks Project #196: Multi-Rack Thermal Modeling & Optimal Critical Workload Placement",
                 fontsize=14, fontweight="bold", y=0.98)

    save_fig_path = os.path.join(RESULTS_DIR, "fig6_spatial_workload_placement.png")
    plt.savefig(save_fig_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Generated Figure 6: {save_fig_path}")


if __name__ == "__main__":
    make_spatial_plots()

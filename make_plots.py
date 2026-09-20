import os
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
sim_path = os.path.join(script_dir, "results", "sim_results.npz")
if not os.path.exists(sim_path):
    sim_path = os.path.join(script_dir, "sim_results.npz")
d = np.load(sim_path)
t = d["t"]
hour = t % 24
day = (t // 24).astype(int)
heat_wave = (day == 2) | (day == 4)
spike = ((day == 3) | (day == 4)) & (hour >= 13) & (hour < 15)

T_REC_LO, T_REC_HI = 18.0, 27.0
T_ALW_LO, T_ALW_HI = 15.0, 32.0

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})

# ---------------------------------------------------------------
# Figure 1: temperature / cooling power / cumulative carbon
# ---------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

def shade_events(ax):
    ax.fill_between(t, *ax.get_ylim(), where=heat_wave, color="orange", alpha=0.08, zorder=0, label="_nolegend_")
    ax.fill_between(t, *ax.get_ylim(), where=spike, color="red", alpha=0.10, zorder=0, label="_nolegend_")

ax = axes[0]
ax.axhspan(T_REC_LO, T_REC_HI, color="green", alpha=0.08)
ax.axhline(T_ALW_HI, color="gray", ls=":", lw=1)
ax.axhline(T_ALW_LO, color="gray", ls=":", lw=1)
ax.plot(t, d["b_T"], label="Baseline thermostat", color="#888780", lw=1.3)
ax.plot(t, d["m_T"], label="Predictive MPC", color="#185FA5", lw=1.5)
ylim = ax.get_ylim()
ax.set_ylim(min(ylim[0], T_ALW_LO-1), max(ylim[1], T_ALW_HI+1))
shade_events(ax)
ax.set_ylabel("Room air temp (°C)")
ax.set_title("Room temperature — green band = ASHRAE recommended (18-27°C), dotted = allowable limits\norange = heat-wave day, red = load-spike window")
ax.legend(loc="upper left", ncol=2, fontsize=9)

ax = axes[1]
ax.plot(t, d["b_pcool"], label="Baseline thermostat", color="#888780", lw=1.1)
ax.plot(t, d["m_pcool"], label="Predictive MPC", color="#185FA5", lw=1.3)
shade_events(ax)
ax.set_ylabel("Cooling power (kW)")
ax.set_title("Cooling electrical power (fan + chiller)")

ax = axes[2]
ax.plot(t, d["b_carbon"], label="Baseline thermostat", color="#888780", lw=1.6)
ax.plot(t, d["m_carbon"], label="Predictive MPC", color="#185FA5", lw=1.8)
shade_events(ax)
ax.set_ylabel("Cumulative CO2 (kg)")
ax.set_xlabel("Simulation time (hours)")
ax.set_title("Cumulative carbon footprint — gap between lines = carbon saved")

for a in axes:
    for dnum in range(1, 5):
        a.axvline(dnum*24, color="black", lw=0.5, alpha=0.3)

plt.tight_layout()
fig1_path = os.path.join(script_dir, "fig1_timeseries.png")
plt.savefig(fig1_path, dpi=140)
print(f"saved {fig1_path}")

# ---------------------------------------------------------------
# Figure 2: summary bar chart
# ---------------------------------------------------------------
b_energy, m_energy = d["b_energy"][-1], d["m_energy"][-1]
b_carbon, m_carbon = d["b_carbon"][-1], d["m_carbon"][-1]
b_pct_rec = 100*np.mean((d["b_T"]>=T_REC_LO)&(d["b_T"]<=T_REC_HI))
m_pct_rec = 100*np.mean((d["m_T"]>=T_REC_LO)&(d["m_T"]<=T_REC_HI))

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
labels = ["Baseline\nthermostat", "Predictive\nMPC"]
colors = ["#888780", "#185FA5"]

def bar(ax, vals, title, ylabel, fmt="{:.0f}"):
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(vals)*1.18)

bar(axes[0], [b_energy, m_energy], "5-day cooling energy", "kWh")
bar(axes[1], [b_carbon, m_carbon], "5-day carbon footprint", "kg CO2")
bar(axes[2], [b_pct_rec, m_pct_rec], "Time in ASHRAE\nrecommended band", "%", fmt="{:.1f}")

plt.tight_layout()
fig2_path = os.path.join(script_dir, "fig2_summary.png")
plt.savefig(fig2_path, dpi=140)
print(f"saved {fig2_path}")

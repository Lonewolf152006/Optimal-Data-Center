"""
Data Center Cooling -- Baseline vs Predictive (MPC) Controller Prototype
==========================================================================

A reduced-order physics simulation of a data-center thermal zone + CRAC/CRAH
+ chiller/economizer cooling plant. Used to validate control logic BEFORE
porting into MATLAB / Simulink / Simscape Fluids -- this is a physics-equivalent
stand-in, not a certified equipment model. Parameters are illustrative;
replace with real chiller/CRAC performance curves and your facility's actual
thermal mass for a production design.
"""

import os
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(7)

# ----------------------------------------------------------------------
# 1. Plant parameters (illustrative -- replace with real equipment curves)
# ----------------------------------------------------------------------
DT_HR       = 5 / 60         # simulation & control step: 5 minutes, in hours
N_DAYS      = 5
N_STEPS     = int(N_DAYS * 24 / DT_HR)

C_ROOM      = 50.0           # kWh/degC -- lumped thermal capacitance of air+equipment
UA_ENV      = 4.0            # kW/degC  -- envelope conduction to ambient (minor term)

Q_IDLE      = 150.0          # kW -- IT heat at 0% utilization
Q_IT_MAX    = 500.0          # kW -- IT heat at 100% utilization
FAN_PEN_MAX = 30.0           # kW -- extra IT heat as server fans ramp at high inlet temp
FAN_RAMP_START = 25.0        # degC
FAN_RAMP_FULL  = 32.0        # degC

Q_COOL_MAX  = 600.0          # kW -- CRAC/chiller max cooling capacity
P_FAN_MAX   = 40.0           # kW -- CRAC fan power at 100% airflow

COP_CAP, COP_FLOOR, COP_SLOPE, COP_REF_T = 8.5, 2.5, 0.15, 10.0

T_REC_LO, T_REC_HI = 18.0, 27.0   # ASHRAE recommended envelope (all classes)
T_ALW_LO, T_ALW_HI = 15.0, 32.0   # ASHRAE Class A1 allowable envelope

T_SET = 22.0                  # baseline thermostat setpoint


# ----------------------------------------------------------------------
# 2. Disturbance scenario: 5 days, diurnal load + weather, plus stress events
# ----------------------------------------------------------------------
def build_scenario():
    t = np.arange(N_STEPS) * DT_HR
    hour_of_day = t % 24
    day = (t // 24).astype(int)

    util = 0.55 + 0.20 * np.sin(2*np.pi*(hour_of_day - 9)/24 - np.pi/2)
    util = np.clip(util, 0.30, 0.80)

    t_amb = 24.0 + 6.5 * np.sin(2*np.pi*(hour_of_day - 9)/24 - np.pi/2)

    heat_wave_days = (day == 2) | (day == 4)          # Day 3 & Day 5
    t_amb = t_amb + 8.0 * heat_wave_days

    spike_mask = ((day == 3) | (day == 4)) & (hour_of_day >= 13) & (hour_of_day < 15)  # Day 4 & 5, 13:00-15:00
    util = np.where(spike_mask, 0.97, util)

    return t, util, t_amb


def carbon_intensity(t):   # g CO2 / kWh -- illustrative diurnal grid mix, replace with real regional data
    hour_of_day = t % 24
    return 380 + 90 * np.cos(2*np.pi*(hour_of_day - 20)/24)


# ----------------------------------------------------------------------
# 3. Physics
# ----------------------------------------------------------------------
def q_it(u_util, t_room):
    base = Q_IDLE + (Q_IT_MAX - Q_IDLE) * u_util
    ramp = np.clip((t_room - FAN_RAMP_START) / (FAN_RAMP_FULL - FAN_RAMP_START), 0, 1)
    return base + FAN_PEN_MAX * ramp**2

def cop_eff(t_amb):
    return np.clip(COP_CAP - COP_SLOPE * (t_amb - COP_REF_T), COP_FLOOR, COP_CAP)

def p_cool(u_ctrl, t_amb):
    q_delivered = u_ctrl * Q_COOL_MAX
    p_fan = P_FAN_MAX * u_ctrl**3
    p_chiller = q_delivered / cop_eff(t_amb)
    return p_fan + p_chiller, q_delivered

def droom_dt(t_room, u_util, u_ctrl, t_amb):
    qit = q_it(u_util, t_room)
    _, q_del = p_cool(u_ctrl, t_amb)
    return (qit - q_del + UA_ENV * (t_amb - t_room)) / C_ROOM

def rk4_step(t_room, u_util, u_ctrl, t_amb, dt):
    k1 = droom_dt(t_room, u_util, u_ctrl, t_amb)
    k2 = droom_dt(t_room + dt/2*k1, u_util, u_ctrl, t_amb)
    k3 = droom_dt(t_room + dt/2*k2, u_util, u_ctrl, t_amb)
    k4 = droom_dt(t_room + dt*k3, u_util, u_ctrl, t_amb)
    return t_room + dt/6*(k1 + 2*k2 + 2*k3 + k4)


# ----------------------------------------------------------------------
# 4. Baseline controller: hysteresis thermostat (reactive, no forecast)
# ----------------------------------------------------------------------
def make_baseline_controller():
    state = {"mode": "on"}
    def controller(t_room, k, util_fore, tamb_fore, carbon_fore):
        if t_room >= T_SET + 1.0:
            state["mode"] = "on"
        elif t_room <= T_SET - 1.0:
            state["mode"] = "off"
        return 1.0 if state["mode"] == "on" else 0.25
    return controller


# ----------------------------------------------------------------------
# 5. Advanced controller: short-horizon carbon-aware MPC
# ----------------------------------------------------------------------
def make_mpc_controller(horizon=24):
    warm = {"u": np.full(horizon, 0.4)}

    def predict(t0, u_seq, util_fore, tamb_fore):
        # Deliberately simplified internal model: no server-fan-penalty feedback
        # (realistic plant/model mismatch -- the controller's internal model is
        # never a perfect copy of reality).
        Ts = np.empty(len(u_seq) + 1)
        Ts[0] = t0
        for k in range(len(u_seq)):
            qit = Q_IDLE + (Q_IT_MAX - Q_IDLE) * util_fore[k]
            q_del = u_seq[k] * Q_COOL_MAX
            dT = (qit - q_del + UA_ENV*(tamb_fore[k]-Ts[k])) / C_ROOM
            Ts[k+1] = Ts[k] + DT_HR * dT
        return Ts

    def cost(u_seq, t0, util_fore, tamb_fore, carbon_fore):
        Ts = predict(t0, u_seq, util_fore, tamb_fore)
        pcool, _ = p_cool(u_seq, tamb_fore)
        carbon_cost = np.sum(pcool * carbon_fore * DT_HR)   # g CO2 over horizon
        rec_viol = np.maximum(0, T_REC_LO - Ts[1:]) + np.maximum(0, Ts[1:] - T_REC_HI)
        alw_viol = np.maximum(0, T_ALW_LO - Ts[1:]) + np.maximum(0, Ts[1:] - T_ALW_HI)
        penalty = 400*np.sum(rec_viol**2) + 20000*np.sum(alw_viol**2)
        smooth = 50*np.sum(np.diff(np.concatenate([[warm["u"][0]], u_seq]))**2)
        return carbon_cost/1000 + penalty + smooth

    def controller(t_room, k, util_fore, tamb_fore, carbon_fore):
        x0 = np.clip(np.concatenate([warm["u"][1:], warm["u"][-1:]]), 0, 1)
        res = minimize(cost, x0, args=(t_room, util_fore, tamb_fore, carbon_fore),
                        method="L-BFGS-B", bounds=[(0, 1)]*horizon,
                        options={"maxiter": 60})
        u_opt = np.clip(res.x, 0, 1)
        warm["u"] = u_opt
        return u_opt[0]
    return controller


# ----------------------------------------------------------------------
# 6. Simulation driver
# ----------------------------------------------------------------------
def simulate(controller, t, util_true, tamb_true, horizon=24, forecast_noise=True):
    n = len(t)
    T = np.empty(n); T[0] = T_SET
    u_hist = np.empty(n)
    carbon_true = carbon_intensity(t)

    for k in range(n):
        h = min(horizon, n - k)
        util_fore = util_true[k:k+h].copy()
        tamb_fore = tamb_true[k:k+h].copy()
        carbon_fore = carbon_true[k:k+h].copy()
        if forecast_noise and h > 0:
            util_fore = np.clip(util_fore + rng.normal(0, 0.03, h), 0, 1)
            tamb_fore = tamb_fore + rng.normal(0, 0.7, h)
        if h < horizon:
            util_fore = np.concatenate([util_fore, np.full(horizon-h, util_fore[-1])])
            tamb_fore = np.concatenate([tamb_fore, np.full(horizon-h, tamb_fore[-1])])
            carbon_fore = np.concatenate([carbon_fore, np.full(horizon-h, carbon_fore[-1])])

        u = controller(T[k], k, util_fore, tamb_fore, carbon_fore)
        u_hist[k] = u
        if k < n-1:
            T[k+1] = rk4_step(T[k], util_true[k], u, tamb_true[k], DT_HR)

    pcool, _ = p_cool(u_hist, tamb_true)
    carbon_kg = np.cumsum(pcool * carbon_true * DT_HR) / 1000.0
    energy_kwh = np.cumsum(pcool * DT_HR)
    return dict(T=T, u=u_hist, pcool=pcool, carbon_kg=carbon_kg, energy_kwh=energy_kwh)


if __name__ == "__main__":
    import time
    t, util, tamb = build_scenario()

    print("Running baseline thermostatic controller...")
    baseline = simulate(make_baseline_controller(), t, util, tamb)

    print("Running predictive (MPC) controller...")
    t0 = time.time()
    mpc = simulate(make_mpc_controller(horizon=24), t, util, tamb)
    print(f"  (MPC sim took {time.time()-t0:.1f}s for {N_STEPS} steps)")

    def summarize(name, r):
        T = r["T"]
        pct_rec = 100*np.mean((T >= T_REC_LO) & (T <= T_REC_HI))
        pct_alw = 100*np.mean((T >= T_ALW_LO) & (T <= T_ALW_HI))
        print(f"\n--- {name} ---")
        print(f"  time in recommended band (18-27C): {pct_rec:.1f}%")
        print(f"  time in allowable band   (15-32C): {pct_alw:.1f}%")
        print(f"  max / min room temp:               {T.max():.2f}C / {T.min():.2f}C")
        print(f"  total cooling energy:              {r['energy_kwh'][-1]:.0f} kWh")
        print(f"  total carbon footprint:            {r['carbon_kg'][-1]:.1f} kg CO2")
        return pct_rec, pct_alw, r['energy_kwh'][-1], r['carbon_kg'][-1]

    b_stats = summarize("Baseline thermostat", baseline)
    m_stats = summarize("Predictive MPC", mpc)

    energy_savings = 100*(b_stats[2]-m_stats[2])/b_stats[2]
    carbon_savings = 100*(b_stats[3]-m_stats[3])/b_stats[3]
    print(f"\n=== Predictive controller vs baseline ===")
    print(f"  Energy reduction: {energy_savings:.1f}%")
    print(f"  Carbon reduction: {carbon_savings:.1f}%")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sim_results.npz")
    np.savez(out_path,
             t=t, util=util, tamb=tamb, carbon_intensity=carbon_intensity(t),
             b_T=baseline["T"], b_u=baseline["u"], b_pcool=baseline["pcool"],
             b_carbon=baseline["carbon_kg"], b_energy=baseline["energy_kwh"],
             m_T=mpc["T"], m_u=mpc["u"], m_pcool=mpc["pcool"],
             m_carbon=mpc["carbon_kg"], m_energy=mpc["energy_kwh"])
    print(f"\nSaved {out_path}")

"""
Phase 1 Validation Gate Verification Script
===========================================
Runs the MATLAB ODE numerical integration equations (from run_simulation.m / validate_plant_comparison.m)
and compares against the Python ThermalPlant output.
Calculates maximum error, RMSE, and generates results/plant_validation_comparison.png.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(repo_root, 'data', 'plant_validation_inputs.csv')
    py_output_path = os.path.join(repo_root, 'data', 'plant_validation_python_output.csv')
    results_dir = os.path.join(repo_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, 'plant_validation_comparison.png')
    metrics_path = os.path.join(results_dir, 'plant_validation_metrics.txt')

    # Read inputs
    inputs_df = pd.read_csv(input_path)
    t_hr = inputs_df['time_hr'].values
    util = inputs_df['utilization'].values
    t_amb = inputs_df['ambient_temp_C'].values
    n_steps = len(t_hr)
    dt_hr = 5.0 / 60.0

    # Read Python plant output
    py_df = pd.read_csv(py_output_path)
    T_python = py_df['room_temp_C'].values

    # Run MATLAB-native RK4 simulation model exactly as implemented in validate_plant_comparison.m and run_simulation.m
    # Parameters matching MATLAB:
    C_ROOM = 50.0
    UA_ENV = 4.0
    Q_IDLE = 150.0
    Q_IT_MAX = 500.0
    FAN_PEN_MAX = 30.0
    FAN_RAMP_START = 25.0
    FAN_RAMP_FULL = 32.0
    Q_COOL_MAX = 600.0
    P_FAN_MAX = 40.0
    COP_CAP = 8.5
    COP_FLOOR = 2.5
    COP_SLOPE = 0.15
    COP_REF_T = 10.0

    T_SET = 22.0
    DEADBAND = 1.0
    ON_POWER = 1.0
    OFF_POWER = 0.25

    T_matlab = np.zeros(n_steps)
    u_matlab = np.zeros(n_steps)
    currT = 22.0
    ctrlMode = 'on'

    def rk4_dTdt(T, u_ut, u_ctrl, Tamb):
        ramp = min(max((T - FAN_RAMP_START) / (FAN_RAMP_FULL - FAN_RAMP_START), 0.0), 1.0)
        Qit = Q_IDLE + (Q_IT_MAX - Q_IDLE) * u_ut + FAN_PEN_MAX * (ramp ** 2)
        Qdel = u_ctrl * Q_COOL_MAX
        return (Qit - Qdel + UA_ENV * (Tamb - T)) / C_ROOM

    for i in range(n_steps):
        Tamb = t_amb[i]
        u_ut = util[i]

        # Stateful hysteresis controller
        if currT >= T_SET + DEADBAND:
            ctrlMode = 'on'
        elif currT <= T_SET - DEADBAND:
            ctrlMode = 'off'

        if ctrlMode == 'on':
            u_ctrl = ON_POWER
        else:
            u_ctrl = OFF_POWER

        T_matlab[i] = currT
        u_matlab[i] = u_ctrl

        # RK4 step
        k1 = rk4_dTdt(currT, u_ut, u_ctrl, Tamb)
        k2 = rk4_dTdt(currT + dt_hr / 2.0 * k1, u_ut, u_ctrl, Tamb)
        k3 = rk4_dTdt(currT + dt_hr / 2.0 * k2, u_ut, u_ctrl, Tamb)
        k4 = rk4_dTdt(currT + dt_hr * k3, u_ut, u_ctrl, Tamb)
        currT = currT + dt_hr / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Compute agreement metrics
    err = np.abs(T_matlab - T_python)
    max_abs_diff = float(np.max(err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mean_err = float(np.mean(err))

    metrics_text = f"""=== Phase 1 Validation Gate: MATLAB RK4 vs Python ThermalPlant ===
Timesteps:                {n_steps} (5 days at 5-minute resolution)
Initial Condition:        T = 22.0000 °C, ctrlMode = 'on'
MATLAB T_room Range:      [{T_matlab.min():.4f}, {T_matlab.max():.4f}] °C
Python T_room Range:      [{T_python.min():.4f}, {T_python.max():.4f}] °C

Agreement Metrics:
  Max Absolute Difference:  {max_abs_diff:.10e} °C ({max_abs_diff * 1e6:.6f} µ°C)
  Root Mean Square Error:   {rmse:.10e} °C ({rmse * 1e6:.6f} µ°C)
  Mean Absolute Error:      {mean_err:.10e} °C ({mean_err * 1e6:.6f} µ°C)
  Status:                   {'PASS (Numerically Identical)' if max_abs_diff < 1e-6 else 'FAIL'}
"""
    print(metrics_text)

    with open(metrics_path, 'w') as f:
        f.write(metrics_text)

    # Plot comparison
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    # Panel 1: Temperature trajectories
    axes[0].plot(t_hr, T_python, 'b-', label='Python ThermalPlant (RK4)', linewidth=1.5)
    axes[0].plot(t_hr, T_matlab, 'r--', label='MATLAB Companion Plant (RK4)', linewidth=1.2)
    axes[0].axhline(27.0, color='darkorange', linestyle=':', label='ASHRAE Rec. Upper (27°C)')
    axes[0].axhline(18.0, color='teal', linestyle=':', label='ASHRAE Rec. Lower (18°C)')
    axes[0].set_ylabel('Room Temp (°C)')
    axes[0].set_title('Phase 1 Validation: Plant Cross-Comparison (5 Days, 1440 Steps)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    # Panel 2: Error trace
    axes[1].plot(t_hr, err * 1e6, 'k-', linewidth=1.0)
    axes[1].set_ylabel('Error (µ°C)')
    axes[1].set_title(f'Absolute Discrepancy (Max = {max_abs_diff*1e6:.4f} µ°C, RMSE = {rmse*1e6:.4f} µ°C)')
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Shared disturbances
    ax3_twin = axes[2].twinx()
    p1 = axes[2].plot(t_hr, t_amb, color='#d95f02', label='Ambient Temp (°C)', linewidth=1.0)
    p2 = ax3_twin.plot(t_hr, util, color='#1f77b4', label='IT Utilization', linewidth=1.0)
    axes[2].set_ylabel('Ambient Temp (°C)', color='#d95f02')
    ax3_twin.set_ylabel('IT Utilization [0-1]', color='#1f77b4')
    axes[2].set_xlabel('Time (hours)')
    axes[2].set_title('Shared Deterministic Disturbances (Diurnal + Heat Waves + Load Spikes)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved successfully to: {plot_path}")

if __name__ == '__main__':
    main()

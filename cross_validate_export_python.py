"""
Cross-Validation: Export Python plant reference trajectory
===========================================================
Generates a deterministic 5-day simulation using the Python ThermalPlant
with the baseline thermostat and fixed (non-random) scenario inputs.
Exports to data/crossval_python_reference.csv for comparison with the
MATLAB/Simulink companion model.

Run:
    python cross_validate_export_python.py

Output:
    data/crossval_python_reference.csv
"""

import sys
import os
import numpy as np
import csv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from python.plant.thermal_plant import ThermalPlant
from python.plant.scenarios import build_scenario
from python.controllers.baseline_thermostat import BaselineThermostat


def main():
    # --- Deterministic scenario (no randomness) ---
    # Use the default scenario: 5 days, heat-wave on days 2 & 4,
    # load spikes on days 3 & 4 at 13:00-15:00
    n_days = 5
    t, utilization, ambient_temp = build_scenario(
        n_days=n_days,
        heat_wave_days=[2, 4],
        spike_windows=[(3, 13, 15), (4, 13, 15)],
    )

    plant = ThermalPlant()
    ctrl = BaselineThermostat(setpoint=22.0, deadband=1.0,
                               on_power=1.0, off_power=0.25)

    n_steps = len(t)
    T_room = np.zeros(n_steps)
    u_ctrl_log = np.zeros(n_steps)
    p_cool_log = np.zeros(n_steps)
    q_del_log = np.zeros(n_steps)

    room_temp = 22.0  # initial condition

    for i in range(n_steps):
        # Controller decision
        u = ctrl(room_temp)
        u_ctrl_log[i] = u

        # Record state BEFORE the step (consistent with MATLAB logging)
        T_room[i] = room_temp

        # Cooling power
        p_elec, q_del = plant.p_cool(u, ambient_temp[i])
        p_cool_log[i] = float(p_elec)
        q_del_log[i] = float(q_del)

        # Advance plant (RK4)
        room_temp = plant.step(room_temp, u, utilization[i], ambient_temp[i])

    # Export to CSV
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'crossval_python_reference.csv')

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_hr', 'utilization', 'ambient_temp_C',
                          'room_temp_C', 'u_ctrl', 'p_cool_kW', 'q_delivered_kW'])
        for i in range(n_steps):
            writer.writerow([
                f'{t[i]:.6f}',
                f'{utilization[i]:.6f}',
                f'{ambient_temp[i]:.6f}',
                f'{T_room[i]:.6f}',
                f'{u_ctrl_log[i]:.6f}',
                f'{p_cool_log[i]:.6f}',
                f'{q_del_log[i]:.6f}',
            ])

    print(f'Exported {n_steps} timesteps to {out_path}')
    print(f'  Duration: {n_days} days ({t[-1]:.1f} hours)')
    print(f'  T_room range: [{T_room.min():.2f}, {T_room.max():.2f}] °C')
    print(f'  Mean T_room: {T_room.mean():.2f} °C')

    return out_path


if __name__ == '__main__':
    main()

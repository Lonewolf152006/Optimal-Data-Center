"""
Python Plant Validation Runner
================================
Called by validate_plant_comparison.m via subprocess.
Reads deterministic inputs from a CSV, runs ThermalPlant + BaselineThermostat,
writes output room temperatures to another CSV.

Usage:
    python run_python_plant_validation.py <input.csv> <output.csv>
"""

import sys
import os
import csv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from python.plant.thermal_plant import ThermalPlant
from python.controllers.baseline_thermostat import BaselineThermostat


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input.csv> <output.csv>",
              file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Read inputs
    time_hr = []
    utilization = []
    ambient_temp = []

    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_hr.append(float(row['time_hr']))
            utilization.append(float(row['utilization']))
            ambient_temp.append(float(row['ambient_temp_C']))

    n_steps = len(time_hr)

    # Run plant with baseline thermostat
    plant = ThermalPlant()
    ctrl = BaselineThermostat(setpoint=22.0, deadband=1.0,
                               on_power=1.0, off_power=0.25)

    room_temp = 22.0  # Initial condition matching MATLAB
    T_room = []
    u_ctrl_log = []

    for i in range(n_steps):
        # Controller decision
        u = ctrl(room_temp)
        u_ctrl_log.append(u)

        # Record state BEFORE the step
        T_room.append(room_temp)

        # Advance plant (RK4)
        room_temp = plant.step(room_temp, u, utilization[i], ambient_temp[i])

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_hr', 'room_temp_C', 'u_ctrl'])
        for i in range(n_steps):
            writer.writerow([
                f'{time_hr[i]:.6f}',
                f'{T_room[i]:.10f}',
                f'{u_ctrl_log[i]:.6f}',
            ])

    print(f'Exported {n_steps} timesteps to {output_path}')
    print(f'  T_room range: [{min(T_room):.4f}, {max(T_room):.4f}] C')
    print(f'  Mean T_room:  {sum(T_room)/len(T_room):.4f} C')


if __name__ == '__main__':
    main()

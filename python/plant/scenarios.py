"""
Scenario Generation for Data Center Thermal Simulations
========================================================
Refactored from legacy simulate.py ``build_scenario()`` and
``carbon_intensity()``.  All profile equations are preserved exactly;
the interface is now parameterised so callers can vary heat-wave days,
spike windows, and use a fixed seed for reproducibility.
"""

import numpy as np

# Must match the plant timestep
DT_HR = 5 / 60  # 5-minute steps in hours


def build_scenario(n_days=5, heat_wave_days=None, spike_windows=None):
    """Generate disturbance profiles (utilization & ambient temperature).

    Parameters
    ----------
    n_days : int
        Duration of the scenario in days.
    heat_wave_days : list[int] or None
        0-indexed day numbers that receive +8 °C ambient offset.
        Default ``[2, 4]`` matches the legacy scenario.
    spike_windows : list[tuple(int, float, float)] or None
        Each entry ``(day, hour_start, hour_end)`` sets utilization to 0.97.
        Default ``[(3, 13, 15), (4, 13, 15)]`` matches legacy.

    Returns
    -------
    t : ndarray          Simulation time (hours).
    utilization : ndarray  IT utilization fraction [0, 1].
    ambient_temp : ndarray Outdoor temperature (°C).
    """
    n_steps = int(n_days * 24 / DT_HR)
    t = np.arange(n_steps) * DT_HR
    hour_of_day = t % 24
    day = (t // 24).astype(int)

    # --- base utilization (diurnal) ---
    util = 0.55 + 0.20 * np.sin(2 * np.pi * (hour_of_day - 9) / 24 - np.pi / 2)
    util = np.clip(util, 0.30, 0.80)

    # --- base ambient temperature (diurnal) ---
    t_amb = 24.0 + 6.5 * np.sin(2 * np.pi * (hour_of_day - 9) / 24 - np.pi / 2)

    # --- heat-wave overlay ---
    if heat_wave_days is None:
        heat_wave_days = [2, 4]
    hw_mask = np.zeros(n_steps, dtype=bool)
    for d in heat_wave_days:
        hw_mask |= (day == d)
    t_amb = t_amb + 8.0 * hw_mask

    # --- load-spike overlay ---
    if spike_windows is None:
        spike_windows = [(3, 13, 15), (4, 13, 15)]
    spike_mask = np.zeros(n_steps, dtype=bool)
    for (d, h_start, h_end) in spike_windows:
        spike_mask |= ((day == d) & (hour_of_day >= h_start) & (hour_of_day < h_end))
    util = np.where(spike_mask, 0.97, util)

    return t, util, t_amb


def carbon_intensity(t):
    """Grid carbon intensity (g CO₂ / kWh) — illustrative diurnal curve.

    Identical to ``ThermalPlant.carbon_intensity``; provided here for
    convenience so scenario code does not need to import the plant.
    """
    hour_of_day = t % 24
    return 380 + 90 * np.cos(2 * np.pi * (hour_of_day - 20) / 24)


def random_scenario(seed, n_days=5):
    """Build a scenario with randomised heat-wave and spike days.

    Uses ``seed`` to deterministically pick which days are heat-wave /
    spike days, giving reproducible variety across runs.

    Returns
    -------
    t, utilization, ambient_temp : ndarrays  (same as ``build_scenario``).
    """
    rng = np.random.default_rng(seed)
    n_hw = rng.integers(0, n_days)
    hw_days = sorted(rng.choice(n_days, size=n_hw, replace=False).tolist())
    n_sp = rng.integers(0, min(3, n_days))
    sp_days = sorted(rng.choice(n_days, size=n_sp, replace=False).tolist())
    spike_windows = [(d, 13, 15) for d in sp_days]
    return build_scenario(n_days=n_days, heat_wave_days=hw_days,
                          spike_windows=spike_windows)

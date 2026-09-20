"""
Baseline Hysteresis Thermostat Controller
==========================================
Refactored from legacy ``make_baseline_controller()``.
Logic is *unchanged*: cool at full power when room temp exceeds the
setpoint by ``deadband``, drop to minimum power when it falls below,
and hold the previous mode otherwise.
"""


class BaselineThermostat:
    """Simple bang-bang thermostat with hysteresis deadband.

    Default parameters reproduce the legacy controller exactly:
    setpoint 22 °C, ±1 °C deadband, on-power 1.0, off-power 0.25.
    """

    def __init__(self, setpoint=22.0, deadband=1.0,
                 on_power=1.0, off_power=0.25):
        self.setpoint = setpoint
        self.deadband = deadband
        self.on_power = on_power
        self.off_power = off_power
        self.mode = "on"

    def __call__(self, room_temp, step_idx=None,
                 util_fore=None, tamb_fore=None, carbon_fore=None):
        """Return cooling control signal in [0, 1].

        Extra keyword arguments (forecasts) are accepted for interface
        compatibility with the MPC controller but are ignored.
        """
        if room_temp >= self.setpoint + self.deadband:
            self.mode = "on"
        elif room_temp <= self.setpoint - self.deadband:
            self.mode = "off"
        return self.on_power if self.mode == "on" else self.off_power

    def reset(self):
        """Reset internal state for a fresh simulation run."""
        self.mode = "on"

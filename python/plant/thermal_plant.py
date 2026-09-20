"""
Data Center Thermal Plant — Lumped-Parameter Model
====================================================
Refactored from legacy simulate.py.  Every equation, constant, and
parameter value is preserved exactly from the original.  The monolithic
script is now a reusable class with a clean ``step()`` interface so that
*any* controller can drive it.
"""

import numpy as np


class ThermalPlant:
    """Lumped-parameter data-center thermal zone + CRAC/chiller model.

    All constants are class attributes so they can be inspected or overridden
    without modifying module-level globals.
    """

    # ---- simulation timing ------------------------------------------------
    DT_HR = 5 / 60  # simulation & control step: 5 minutes, in hours

    # ---- thermal zone -----------------------------------------------------
    C_ROOM = 50.0       # kWh/°C  lumped thermal capacitance (air + equipment)
    UA_ENV = 4.0         # kW/°C   envelope conduction to ambient

    # ---- IT heat load -----------------------------------------------------
    Q_IDLE = 150.0       # kW  IT heat at 0 % utilization
    Q_IT_MAX = 500.0     # kW  IT heat at 100 % utilization
    FAN_PEN_MAX = 30.0   # kW  extra IT heat as server fans ramp up
    FAN_RAMP_START = 25.0  # °C  onset of fan ramp
    FAN_RAMP_FULL = 32.0   # °C  fan ramp saturated

    # ---- cooling plant ----------------------------------------------------
    Q_COOL_MAX = 600.0   # kW  CRAC/chiller max cooling capacity
    P_FAN_MAX = 40.0     # kW  CRAC fan power at 100 % airflow

    # ---- chiller COP curve ------------------------------------------------
    COP_CAP = 8.5
    COP_FLOOR = 2.5
    COP_SLOPE = 0.15
    COP_REF_T = 10.0

    # ---- ASHRAE temperature envelopes -------------------------------------
    T_REC_LO, T_REC_HI = 18.0, 27.0   # recommended (all classes)
    T_ALW_LO, T_ALW_HI = 15.0, 32.0   # Class A1 allowable

    # ---- default setpoint -------------------------------------------------
    T_SET = 22.0

    # ------------------------------------------------------------------
    # Physics helpers (match legacy exactly)
    # ------------------------------------------------------------------
    def q_it(self, utilization, room_temp):
        """IT heat load including server-fan penalty (kW)."""
        base = self.Q_IDLE + (self.Q_IT_MAX - self.Q_IDLE) * utilization
        ramp = np.clip(
            (room_temp - self.FAN_RAMP_START)
            / (self.FAN_RAMP_FULL - self.FAN_RAMP_START),
            0, 1,
        )
        return base + self.FAN_PEN_MAX * ramp ** 2

    def cop_eff(self, ambient_temp):
        """Effective COP (dimensionless), temperature-dependent."""
        return np.clip(
            self.COP_CAP - self.COP_SLOPE * (ambient_temp - self.COP_REF_T),
            self.COP_FLOOR, self.COP_CAP,
        )

    def p_cool(self, u_ctrl, ambient_temp):
        """Cooling electrical power (kW) and heat removed (kW).

        Returns ``(p_electrical, q_delivered)``; both may be arrays.
        """
        q_delivered = u_ctrl * self.Q_COOL_MAX
        p_fan = self.P_FAN_MAX * u_ctrl ** 3
        p_chiller = q_delivered / self.cop_eff(ambient_temp)
        return p_fan + p_chiller, q_delivered

    def droom_dt(self, room_temp, utilization, u_ctrl, ambient_temp):
        """Rate of change of room temperature (°C / hr)."""
        qit = self.q_it(utilization, room_temp)
        _, q_del = self.p_cool(u_ctrl, ambient_temp)
        return (qit - q_del + self.UA_ENV * (ambient_temp - room_temp)) / self.C_ROOM

    def rk4_step(self, room_temp, utilization, u_ctrl, ambient_temp):
        """Single RK4 integration step over ``DT_HR``."""
        dt = self.DT_HR
        k1 = self.droom_dt(room_temp, utilization, u_ctrl, ambient_temp)
        k2 = self.droom_dt(room_temp + dt / 2 * k1, utilization, u_ctrl, ambient_temp)
        k3 = self.droom_dt(room_temp + dt / 2 * k2, utilization, u_ctrl, ambient_temp)
        k4 = self.droom_dt(room_temp + dt * k3, utilization, u_ctrl, ambient_temp)
        return room_temp + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # ------------------------------------------------------------------
    # Public step interface
    # ------------------------------------------------------------------
    def step(self, room_temp, u_ctrl, utilization, ambient_temp):
        """Advance the plant by one timestep.

        Parameters
        ----------
        room_temp : float     Current room air temperature (°C).
        u_ctrl : float        Cooling control signal in [0, 1].
        utilization : float   IT utilization fraction in [0, 1].
        ambient_temp : float  Outside ambient temperature (°C).

        Returns
        -------
        float  Room temperature at the next timestep (°C).
        """
        return self.rk4_step(room_temp, utilization, u_ctrl, ambient_temp)

    # ------------------------------------------------------------------
    # Carbon intensity (grid-level)
    # ------------------------------------------------------------------
    @staticmethod
    def carbon_intensity(t):
        """Grid carbon intensity (g CO₂ / kWh) — illustrative diurnal curve."""
        hour_of_day = t % 24
        return 380 + 90 * np.cos(2 * np.pi * (hour_of_day - 20) / 24)

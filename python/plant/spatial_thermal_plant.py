"""
Multi-Rack Spatial Data Center Thermal Plant Model
===================================================
Models a 4-rack enterprise server room arranged in a Hot Aisle–Cold Aisle
containment architecture with localized heat recirculation cross-interference.

Implements the thermodynamic framework established by Tang et al. (IEEE TPDS)
and Mousavi et al. (IEEE ETFA), fully coupled with our validated non-linear
chiller COP, variable ambient conditions, and Arrhenius silicon reliability.
"""

import numpy as np


class SpatialThermalPlant:
    """4-Rack Hot Aisle–Cold Aisle data center with cross-rack heat recirculation.

    Attributes:
        N_RACKS: Number of discrete server racks (4).
        D_MATRIX: 4x4 heat recirculation matrix D_ij representing the fraction
                  of exhaust heat from Rack j entrained into the intake of Rack i.
        C_RACK: Thermal capacitance per rack (kWh/°C).
        UA_RACK: Thermal conduction to room ambient per rack (kW/°C).
    """

    N_RACKS = 4
    DT_HR = 5 / 60  # 5-minute control step in hours

    # ---- Thermal Capacitance and Airflow per Rack -------------------------
    # Total room C_ROOM = 50.0 kWh/°C partitioned across 4 server racks
    C_RACK = 12.5  # kWh/°C per rack (4 * 12.5 = 50.0 kWh/°C total)
    UA_RACK = 1.0  # kW/°C envelope conduction per rack (4 * 1.0 = 4.0 kW/°C)

    # Air mass flow rate * specific heat per rack: m_dot_i * c_p (kW/°C)
    MC_AIR_RACK = 18.0  # kW/°C nominal airflow heat capacity rate per rack

    # ---- IT Heat Load per Rack --------------------------------------------
    # 4 racks * 125 kW max = 500 kW total room capacity (matching ThermalPlant)
    Q_RACK_IDLE = 37.5  # kW idle heat per rack (150 kW total across 4 racks)
    Q_RACK_MAX = 125.0  # kW peak heat per rack (500 kW total across 4 racks)

    FAN_PEN_MAX_RACK = 7.5  # kW fan penalty per rack (30 kW total)
    FAN_RAMP_START = 25.0  # °C onset of server internal fan ramp
    FAN_RAMP_FULL = 32.0  # °C saturated server internal fan ramp

    # ---- Cooling Plant ----------------------------------------------------
    Q_COOL_MAX = 600.0  # kW CRAC/chiller max cooling capacity
    P_FAN_MAX = 40.0  # kW CRAC blower fan power at 100% airflow

    # Chiller COP curve
    COP_CAP = 8.5
    COP_FLOOR = 2.5
    COP_SLOPE = 0.15
    COP_REF_T = 10.0

    # ---- ASHRAE Envelopes -------------------------------------------------
    T_REC_LO, T_REC_HI = 18.0, 27.0  # Recommended rack intake envelope (°C)
    T_ALW_LO, T_ALW_HI = 15.0, 32.0  # Allowable envelope (°C)
    T_SET = 22.0  # Nominal setpoint (°C)

    # Critical task thermal SLA thresholds
    T_INLET_CRIT_MAX = 24.0  # °C: Recommended upper limit for critical tasks
    T_JUNCTION_CRIT_MAX = 65.0  # °C: Upper limit on CPU junction temperature

    # ---- Silicon Junction Model Constants ---------------------------------
    DELTA_T_IDLE = 18.0  # °C junction temp rise above inlet at 0% util
    DELTA_T_MAX = 45.0  # °C junction temp rise above inlet at 100% util
    T_J_REF = 53.5  # °C nominal reference junction temperature (326.65 K)
    E_A = 0.70  # eV activation energy for silicon failure mechanisms
    K_B = 8.6173e-5  # eV/K Boltzmann constant

    def __init__(self):
        # 4x4 Heat Recirculation Matrix (Tang et al. data center topology)
        # Rack 1: Closest to CRAC discharge -> minimal recirculation
        # Rack 2: Mid-aisle left -> low recirculation
        # Rack 3: Mid-aisle right -> moderate recirculation
        # Rack 4: Farthest from CRAC (dead-end corner) -> high recirculation
        self.D_MATRIX = np.array([
            [0.02, 0.01, 0.00, 0.00],  # Rack 1 intake
            [0.03, 0.05, 0.02, 0.01],  # Rack 2 intake
            [0.02, 0.04, 0.08, 0.05],  # Rack 3 intake
            [0.03, 0.06, 0.11, 0.18],  # Rack 4 intake (severe recirculation)
        ], dtype=np.float64)

        # State vector: rack internal temperatures (°C)
        self.T_rack = np.full(self.N_RACKS, self.T_SET, dtype=np.float64)

    def reset(self, initial_temp=None):
        """Reset all rack temperatures."""
        init = self.T_SET if initial_temp is None else initial_temp
        self.T_rack = np.full(self.N_RACKS, init, dtype=np.float64)
        return self.T_rack.copy()

    def cop_eff(self, tamb):
        """Effective chiller COP as a function of ambient temperature."""
        cop = self.COP_CAP - self.COP_SLOPE * (tamb - self.COP_REF_T)
        return np.clip(cop, self.COP_FLOOR, self.COP_CAP)

    def p_cool(self, u, tamb):
        """Total cooling electrical power (kW) = fan power + chiller compressor power."""
        u_arr = np.asarray(u, dtype=np.float64)
        tamb_arr = np.asarray(tamb, dtype=np.float64)
        p_fan = self.P_FAN_MAX * (u_arr ** 3)
        p_chiller = (u_arr * self.Q_COOL_MAX) / self.cop_eff(tamb_arr)
        return p_fan + p_chiller, p_fan

    def q_rack_heat(self, w_vector, T_rack_vector):
        """Compute electrical heat dissipation for each rack including fan ramp."""
        w = np.clip(np.asarray(w_vector, dtype=np.float64), 0.0, 1.0)
        t = np.asarray(T_rack_vector, dtype=np.float64)

        base_heat = self.Q_RACK_IDLE + (self.Q_RACK_MAX - self.Q_RACK_IDLE) * w
        fan_ramp = np.clip(
            (t - self.FAN_RAMP_START) / (self.FAN_RAMP_FULL - self.FAN_RAMP_START),
            0.0, 1.0
        )
        return base_heat + self.FAN_PEN_MAX_RACK * fan_ramp

    def solve_inlet_temperatures(self, T_supply, T_out):
        """Compute rack inlet temperatures based on recirculation matrix D.

        T_in,i = T_supply + sum_j D_ij * (T_out,j - T_supply)
        """
        delta_out = np.maximum(0.0, T_out - T_supply)
        recirc_rise = self.D_MATRIX @ delta_out
        return T_supply + recirc_rise

    def step(self, u, w_vector, tamb):
        """Execute one 5-minute integration step with 4 discrete racks.

        Args:
            u: Cooling effort [0, 1] applied to CRAC unit.
            w_vector: 4-element array of rack workload utilizations [0, 1].
            tamb: Ambient temperature (°C).

        Returns:
            dict containing:
                T_rack: updated rack internal temperatures (°C)
                T_inlet: rack inlet/intake temperatures (°C)
                T_outlet: rack exhaust temperatures (°C)
                T_junction: CPU junction temperatures (°C)
                T_supply: CRAC cold-air supply temperature (°C)
                P_cool: electrical cooling power (kW)
                Q_it_tot: total IT thermal load (kW)
        """
        u = float(np.clip(u, 0.0, 1.0))
        w = np.clip(np.asarray(w_vector, dtype=np.float64), 0.0, 1.0)
        dt = self.DT_HR

        # 1. Cooling supply air temperature from CRAC
        # Total air heat capacity rate across 4 racks
        mc_tot = self.N_RACKS * self.MC_AIR_RACK
        T_mean = float(np.mean(self.T_rack))
        q_cool = u * self.Q_COOL_MAX
        T_supply = T_mean - (q_cool / mc_tot)
        # Physics lower limit on chilled air supply (e.g. 12°C to prevent coil freeze)
        T_supply = max(12.0, min(T_supply, T_mean))

        # 2. Iterative thermal equilibrium for rack intake & exhaust
        # Rack exhaust air temperature: T_out,i = T_rack,i + Q_rack,i / (2 * mc_rack)
        q_rack = self.q_rack_heat(w, self.T_rack)
        T_out = self.T_rack + (q_rack / (2.0 * self.MC_AIR_RACK))
        T_inlet = self.solve_inlet_temperatures(T_supply, T_out)

        # 3. RK4 integration of rack thermal masses
        # dT_rack,i / dt = (1 / C_RACK) * [Q_rack,i - mc_rack * (T_rack,i - T_inlet,i) - UA_RACK * (T_rack,i - tamb)]
        def deriv(t_state):
            q = self.q_rack_heat(w, t_state)
            dTdt = (
                q
                - self.MC_AIR_RACK * (t_state - T_inlet)
                - self.UA_RACK * (t_state - tamb)
            ) / self.C_RACK
            return dTdt

        k1 = deriv(self.T_rack)
        k2 = deriv(self.T_rack + 0.5 * dt * k1)
        k3 = deriv(self.T_rack + 0.5 * dt * k2)
        k4 = deriv(self.T_rack + dt * k3)
        self.T_rack = self.T_rack + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # 4. Silicon CPU junction temperatures
        T_junction = T_inlet + self.DELTA_T_IDLE + (self.DELTA_T_MAX - self.DELTA_T_IDLE) * w

        # 5. Arrhenius degradation factors
        T_j_kelvin = T_junction + 273.15
        T_ref_kelvin = self.T_J_REF + 273.15
        arrhenius_af = np.exp((self.E_A / self.K_B) * (1.0 / T_ref_kelvin - 1.0 / T_j_kelvin))

        # 6. Electrical cooling power
        pcool, _ = self.p_cool(u, tamb)

        return {
            "T_rack": self.T_rack.copy(),
            "T_inlet": T_inlet.copy(),
            "T_outlet": T_out.copy(),
            "T_junction": T_junction.copy(),
            "Arrhenius_AF": arrhenius_af.copy(),
            "T_supply": float(T_supply),
            "P_cool": float(pcool),
            "Q_it_tot": float(np.sum(q_rack)),
        }

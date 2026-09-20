"""
Carbon-Aware Receding-Horizon MPC Controller
==============================================
Refactored from legacy ``make_mpc_controller()``.

The controller now accepts an optional **predictor function** that
replaces / augments the simplified internal physics model.  When
``predictor_fn`` is ``None`` the legacy simplified physics model is
used directly (deliberate plant/model mismatch — no server-fan penalty).

When a predictor is provided it is called **once per control step** to
produce a temperature correction that is added on top of the physics
prediction.  The optimiser still runs against the physics-based forward
model so that varying ``u_seq`` changes the predicted trajectory and
the ASHRAE + carbon cost responds properly.

Cost function preserved exactly from legacy:
  carbon_cost / 1000
  + 400 * Σ(recommended-band violation²)
  + 20000 * Σ(allowable-band violation²)
  + 50 * Σ(Δu²)   (smoothness)
"""

import numpy as np
from scipy.optimize import minimize

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from plant.thermal_plant import ThermalPlant


def make_mpc_controller(predictor_fn=None, horizon=24):
    """Return a callable MPC controller.

    Parameters
    ----------
    predictor_fn : callable or None
        ``predict(current_state, forecast_features) -> np.ndarray``
        where *current_state* is the current room temperature (float)
        and *forecast_features* is ``(horizon, 3)`` array of
        ``[utilization, ambient_temp, carbon_intensity]``.
        Returns predicted room temperatures of length ``horizon``.
    horizon : int
        Prediction / control horizon (number of steps).

    Returns
    -------
    controller : callable
        ``controller(room_temp, step_idx, util_fore, tamb_fore, carbon_fore)``
    """
    plant = ThermalPlant()
    warm = {"u": np.full(horizon, 0.4)}

    # Nominal baseline cooling control for predictor deviation
    U_BASELINE = 0.45

    def predict_trajectory(t0, u_seq, util_fore, tamb_fore, ml_traj):
        h = len(u_seq)
        Ts = np.empty(h + 1)
        Ts[0] = t0
        for k in range(h):
            if ml_traj is not None:
                # Thermal drift from predictor
                pred_step = ml_traj[k] - (t0 if k == 0 else ml_traj[k - 1])
                # Cooling effect relative to nominal baseline cooling
                dT_cool = (u_seq[k] - U_BASELINE) * plant.Q_COOL_MAX * plant.DT_HR / plant.C_ROOM
                Ts[k + 1] = Ts[k] + pred_step - dT_cool
            else:
                # Fallback to internal simplified physics model
                qit = plant.Q_IDLE + (plant.Q_IT_MAX - plant.Q_IDLE) * util_fore[k]
                q_del = u_seq[k] * plant.Q_COOL_MAX
                dT = (qit - q_del + plant.UA_ENV * (tamb_fore[k] - Ts[k])) / plant.C_ROOM
                Ts[k + 1] = Ts[k] + plant.DT_HR * dT
        return Ts

    # ------------------------------------------------------------------
    # Cost function (preserved exactly from legacy)
    # ------------------------------------------------------------------
    def cost(u_seq, t0, util_fore, tamb_fore, carbon_fore, ml_traj):
        Ts = predict_trajectory(t0, u_seq, util_fore, tamb_fore, ml_traj)
        pcool_val, _ = plant.p_cool(u_seq, tamb_fore[:len(u_seq)])
        carbon_cost = np.sum(pcool_val * carbon_fore[:len(u_seq)] * plant.DT_HR)

        rec_viol = (np.maximum(0, plant.T_REC_LO - Ts[1:])
                    + np.maximum(0, Ts[1:] - plant.T_REC_HI))
        alw_viol = (np.maximum(0, plant.T_ALW_LO - Ts[1:])
                    + np.maximum(0, Ts[1:] - plant.T_ALW_HI))

        penalty = 400 * np.sum(rec_viol ** 2) + 20000 * np.sum(alw_viol ** 2)
        smooth = 50 * np.sum(
            np.diff(np.concatenate([[warm["u"][0]], u_seq])) ** 2
        )
        return carbon_cost / 1000 + penalty + smooth

    # ------------------------------------------------------------------
    # Controller callable
    # ------------------------------------------------------------------
    def controller(room_temp, step_idx=None,
                   util_fore=None, tamb_fore=None, carbon_fore=None):
        h = horizon

        # --- evaluate predictor (once, before optimisation) ------------
        ml_traj = None
        if predictor_fn is not None:
            forecast_features = np.column_stack([
                util_fore[:h], tamb_fore[:h], carbon_fore[:h]
            ])
            ml_traj = predictor_fn(room_temp, forecast_features)

        # --- optimise control sequence ---------------------------------
        x0 = np.clip(
            np.concatenate([warm["u"][1:], warm["u"][-1:]]), 0, 1
        )
        res = minimize(
            cost, x0,
            args=(room_temp, util_fore[:h], tamb_fore[:h],
                  carbon_fore[:h], ml_traj),
            method="L-BFGS-B",
            bounds=[(0, 1)] * h,
            options={"maxiter": 40},
        )
        u_opt = np.clip(res.x, 0, 1)
        warm["u"] = u_opt
        return u_opt[0]

    return controller

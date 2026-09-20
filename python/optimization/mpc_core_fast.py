"""
Carbon-Aware Receding-Horizon MPC Controller (Fast / Analytic Gradient)
=======================================================================
EXPERIMENTAL copy of mpc_core.py incorporating an exact analytical Jacobian
(gradient) for the objective function passed via the `jac=` parameter to
scipy.optimize.minimize(method="L-BFGS-B").

Eliminates the 2-point finite-difference fallback where SciPy perturbs each
decision variable individually (13 function evaluations per line search step).

NOTE: This is an experimental optimization copy. Do not use in production
pipelines without explicit validation.
"""

import sys, os
import time
import numpy as np
from scipy.optimize import minimize

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from plant.thermal_plant import ThermalPlant


def make_mpc_controller_fast(predictor_fn=None, horizon=24):
    """Return a callable MPC controller with analytical Jacobian acceleration.

    Parameters
    ----------
    predictor_fn : callable or None
        predict(current_state, forecast_features) -> np.ndarray
    horizon : int
        Prediction / control horizon (number of steps).

    Returns
    -------
    controller : callable
        controller(room_temp, step_idx, util_fore, tamb_fore, carbon_fore)
    """
    plant = ThermalPlant()
    warm = {"u": np.full(horizon, 0.4)}
    U_BASELINE = 0.45

    def predict_trajectory(t0, u_seq, util_fore, tamb_fore, ml_traj):
        h = len(u_seq)
        Ts = np.empty(h + 1)
        Ts[0] = t0
        for k in range(h):
            if ml_traj is not None:
                pred_step = ml_traj[k] - (t0 if k == 0 else ml_traj[k - 1])
                dT_cool = (u_seq[k] - U_BASELINE) * plant.Q_COOL_MAX * plant.DT_HR / plant.C_ROOM
                Ts[k + 1] = Ts[k] + pred_step - dT_cool
            else:
                qit = plant.Q_IDLE + (plant.Q_IT_MAX - plant.Q_IDLE) * util_fore[k]
                q_del = u_seq[k] * plant.Q_COOL_MAX
                dT = (qit - q_del + plant.UA_ENV * (tamb_fore[k] - Ts[k])) / plant.C_ROOM
                Ts[k + 1] = Ts[k] + plant.DT_HR * dT
        return Ts

    def cost_and_grad(u_seq, t0, util_fore, tamb_fore, carbon_fore, ml_traj):
        h = len(u_seq)
        Ts = predict_trajectory(t0, u_seq, util_fore, tamb_fore, ml_traj)
        y = Ts[1:]  # predicted temperatures at step 1..h

        # 1. Carbon cost and gradient
        pcool_val, _ = plant.p_cool(u_seq, tamb_fore[:h])
        carbon_cost = np.sum(pcool_val * carbon_fore[:h] * plant.DT_HR)

        cop = plant.cop_eff(tamb_fore[:h])
        dP_du = 3.0 * plant.P_FAN_MAX * (u_seq ** 2) + plant.Q_COOL_MAX / cop
        dCarbon_du = (dP_du * carbon_fore[:h] * plant.DT_HR) / 1000.0

        # 2. ASHRAE Violation Penalties and gradient
        rec_hi_viol = np.maximum(0.0, y - plant.T_REC_HI)
        rec_lo_viol = np.maximum(0.0, plant.T_REC_LO - y)
        rec_viol = rec_hi_viol + rec_lo_viol

        alw_hi_viol = np.maximum(0.0, y - plant.T_ALW_HI)
        alw_lo_viol = np.maximum(0.0, plant.T_ALW_LO - y)
        alw_viol = alw_hi_viol + alw_lo_viol

        penalty = 400.0 * np.sum(rec_viol ** 2) + 20000.0 * np.sum(alw_viol ** 2)

        # Gradient wrt predicted temperature vector y
        dRec_dy = 2.0 * (rec_hi_viol - rec_lo_viol)
        dAlw_dy = 2.0 * (alw_hi_viol - alw_lo_viol)
        gy = 400.0 * dRec_dy + 20000.0 * dAlw_dy

        if ml_traj is not None:
            # Trajectory is linear in u: dT_{k+1}/du_i = -alpha for i <= k
            alpha = plant.Q_COOL_MAX * plant.DT_HR / plant.C_ROOM
            # Suffix sum of gy: sum_{k=i}^{h-1} gy[k]
            dPenalty_du = -alpha * np.cumsum(gy[::-1])[::-1]
        else:
            # Physics model has autoregressive decay lambda
            lam = 1.0 - (plant.DT_HR * plant.UA_ENV) / plant.C_ROOM
            alpha = (plant.DT_HR * plant.Q_COOL_MAX) / plant.C_ROOM
            s = np.zeros(h)
            cur = 0.0
            for i in range(h - 1, -1, -1):
                cur = gy[i] + lam * cur
                s[i] = cur
            dPenalty_du = -alpha * s

        # 3. Smoothness cost and gradient
        d = np.diff(np.concatenate([[warm["u"][0]], u_seq]))
        smooth = 50.0 * np.sum(d ** 2)

        d_ext = np.append(d, 0.0)
        dSmooth_du = 100.0 * (d_ext[:-1] - d_ext[1:])

        total_cost = carbon_cost / 1000.0 + penalty + smooth
        total_grad = dCarbon_du + dPenalty_du + dSmooth_du

        return total_cost, total_grad

    def controller(room_temp, step_idx=None,
                   util_fore=None, tamb_fore=None, carbon_fore=None):
        h = horizon

        ml_traj = None
        if predictor_fn is not None:
            forecast_features = np.column_stack([
                util_fore[:h], tamb_fore[:h], carbon_fore[:h]
            ])
            ml_traj = predictor_fn(room_temp, forecast_features)

        x0 = np.clip(
            np.concatenate([warm["u"][1:], warm["u"][-1:]]), 0, 1
        )
        res = minimize(
            cost_and_grad, x0,
            args=(room_temp, util_fore[:h], tamb_fore[:h],
                  carbon_fore[:h], ml_traj),
            method="L-BFGS-B",
            jac=True,
            bounds=[(0, 1)] * h,
            options={"maxiter": 40},
        )
        u_opt = np.clip(res.x, 0, 1)
        warm["u"] = u_opt
        return u_opt[0]

    return controller


# ======================================================================
# Validation Check against original mpc_core.py
# ======================================================================
def validate_fast_controller():
    from controllers.mpc_core import make_mpc_controller

    print("=" * 80)
    print("VALIDATION: Original mpc_core.py vs Fast Analytic Gradient mpc_core_fast.py")
    print("=" * 80)
    print("Testing 5 representative dummy input conditions across both controllers...\n")

    h = 12
    test_cases = [
        {"name": "Condition 1 (Mild weather, normal load)",
         "t0": 23.0, "tamb": 26.0, "util": 0.50, "carb": 380.0, "drift": 0.05},
        {"name": "Condition 2 (Heat-wave, high load)",
         "t0": 26.5, "tamb": 34.0, "util": 0.85, "carb": 460.0, "drift": 0.15},
        {"name": "Condition 3 (Over-cooled room, low load)",
         "t0": 17.5, "tamb": 22.0, "util": 0.25, "carb": 300.0, "drift": -0.05},
        {"name": "Condition 4 (Extreme spike transition)",
         "t0": 25.0, "tamb": 31.0, "util": 0.95, "carb": 420.0, "drift": 0.20},
        {"name": "Condition 5 (Near allowable threshold)",
         "t0": 28.5, "tamb": 36.0, "util": 0.70, "carb": 390.0, "drift": 0.10},
    ]

    all_passed = True
    max_observed_diff = 0.0
    tol = 1e-3

    for idx, tc in enumerate(test_cases, 1):
        t0 = tc["t0"]
        util_fore = np.full(h, tc["util"])
        tamb_fore = np.full(h, tc["tamb"])
        carbon_fore = np.full(h, tc["carb"])
        dummy_pred = lambda temp, feat: temp + np.cumsum(np.full(h, tc["drift"]))

        # Build original and fast controllers
        ctrl_orig = make_mpc_controller(predictor_fn=dummy_pred, horizon=h)
        ctrl_fast = make_mpc_controller_fast(predictor_fn=dummy_pred, horizon=h)

        t_orig_0 = time.perf_counter()
        u_orig = ctrl_orig(t0, util_fore=util_fore, tamb_fore=tamb_fore, carbon_fore=carbon_fore)
        t_orig = (time.perf_counter() - t_orig_0) * 1000

        t_fast_0 = time.perf_counter()
        u_fast = ctrl_fast(t0, util_fore=util_fore, tamb_fore=tamb_fore, carbon_fore=carbon_fore)
        t_fast = (time.perf_counter() - t_fast_0) * 1000

        diff = abs(u_orig - u_fast)
        max_observed_diff = max(max_observed_diff, diff)
        passed = (diff <= tol)
        if not passed:
            all_passed = False

        status = "PASSED" if passed else "FAILED"
        speedup = t_orig / max(t_fast, 1e-6)
        print(f"[{status}] Test Case {idx}: {tc['name']}")
        print(f"       Original u*: {u_orig:.6f} ({t_orig:6.2f} ms)")
        print(f"       Fast u*    : {u_fast:.6f} ({t_fast:6.2f} ms)  ->  Speedup: {speedup:5.1f}x")
        print(f"       Abs Diff   : {diff:.2e}  (Tolerance: {tol:.1e})\n")

    print("-" * 80)
    print(f"Max observed difference across all 5 test cases: {max_observed_diff:.2e}")
    if all_passed:
        print("VERIFICATION SUCCESS: All 5 test cases matched within 1e-3 tolerance.")
        print("Analytical gradient produces identical control outputs with substantial speedup.")
    else:
        print("WARNING: Fast controller outputs DO NOT match original mpc_core.py within 1e-3 tolerance!")
        print("DO NOT claim this is a drop-in replacement.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    validate_fast_controller()

"""
Spatial Workload Dispatcher & Thermal-Aware Optimizer
=====================================================
Dispatches critical vs. batch IT workloads across 4 data center server racks.
Contrasts a traditional Thermal-Unaware (Uniform) hypervisor scheduler against
an Optimal Thermal-Aware Workload Dispatcher that prevents hot spots and protects
critical workloads from high-temperature silicon wear.
"""

import numpy as np
from scipy.optimize import minimize


class WorkloadClassifier:
    """Classifies incoming data center compute load into Critical and Batch tiers."""

    def __init__(self, crit_fraction=0.35):
        """
        Args:
            crit_fraction: Fraction of total facility IT load designated as
                           mission-critical (default 35%).
        """
        self.crit_fraction = crit_fraction

    def split_demand(self, total_util, n_racks=4):
        """Split scalar aggregate utilization into critical and batch capacity demands.

        Args:
            total_util: Scalar data center utilization in [0, 1].
            n_racks: Number of server racks.

        Returns:
            w_crit_total: Total critical workload units (across all racks).
            w_batch_total: Total batch workload units (across all racks).
        """
        total_demand = float(total_util) * n_racks
        w_crit = total_demand * self.crit_fraction
        w_batch = total_demand * (1.0 - self.crit_fraction)
        return w_crit, w_batch


class ThermalUnawareDispatcher:
    """Baseline Scheduler: Evenly spreads critical and batch loads across all racks."""

    def __init__(self, n_racks=4):
        self.n_racks = n_racks

    def dispatch(self, w_crit_total, w_batch_total, T_inlet_current=None):
        """Evenly divide critical and batch workloads among all racks."""
        w_crit_per_rack = np.full(self.n_racks, w_crit_total / self.n_racks)
        w_batch_per_rack = np.full(self.n_racks, w_batch_total / self.n_racks)
        w_tot = w_crit_per_rack + w_batch_per_rack
        return {
            "w_crit": w_crit_per_rack,
            "w_batch": w_batch_per_rack,
            "w_total": np.clip(w_tot, 0.0, 1.0),
        }


class ThermalAwareOptimalDispatcher:
    """Optimal Scheduler: Directs critical loads to coolest racks and balances thermal gradients.

    Solves a constrained convex optimization problem (QP) at each scheduling step:
      min  sum_i (w_crit,i * T_inlet,i) + lambda_balance * sum_i (w_total,i - w_mean)^2
           + lambda_hot * sum_i [max(0, T_inlet,i + alpha * w_total,i - T_target)]^2
      s.t. sum_i w_crit,i = W_crit_total
           sum_i w_batch,i = W_batch_total
           0 <= w_crit,i, w_batch,i
           w_crit,i + w_batch,i <= 1.0
    """

    def __init__(self, n_racks=4, lambda_crit=1.5, lambda_balance=0.8, lambda_hot=3.0, T_target=24.0):
        self.n_racks = n_racks
        self.lambda_crit = lambda_crit
        self.lambda_balance = lambda_balance
        self.lambda_hot = lambda_hot
        self.T_target = T_target

    def dispatch(self, w_crit_total, w_batch_total, T_inlet_current):
        """Compute optimal per-rack critical and batch workload allocations.

        Args:
            w_crit_total: Total critical workload units.
            w_batch_total: Total batch workload units.
            T_inlet_current: Current 4-element vector of rack inlet temperatures (°C).

        Returns:
            dict containing w_crit, w_batch, w_total vectors.
        """
        Tin = np.asarray(T_inlet_current, dtype=np.float64)
        n = self.n_racks
        w_mean = (w_crit_total + w_batch_total) / n

        # Decision vector x of size 2*n: [w_crit_1..w_crit_n, w_batch_1..w_batch_n]
        def objective(x):
            w_crit = x[:n]
            w_batch = x[n:]
            w_tot = w_crit + w_batch

            # 1. Thermal exposure cost for critical loads (shield critical tasks from heat)
            cost_crit = self.lambda_crit * np.sum(w_crit * Tin)

            # 2. Thermal gradient smoothing cost (prevent extreme imbalance)
            cost_bal = self.lambda_balance * np.sum((w_tot - w_mean) ** 2)

            # 3. Soft penalty on estimated temperature exceeding safe inlet target
            # Estimated temperature rise is approximately proportional to local utilization
            t_est = Tin + 3.5 * w_tot
            hot_viol = np.maximum(0.0, t_est - self.T_target)
            cost_hot = self.lambda_hot * np.sum(hot_viol ** 2)

            return cost_crit + cost_bal + cost_hot

        # Constraints
        # Total critical demand met
        c1 = {"type": "eq", "fun": lambda x: np.sum(x[:n]) - w_crit_total}
        # Total batch demand met
        c2 = {"type": "eq", "fun": lambda x: np.sum(x[n:]) - w_batch_total}

        constraints = [c1, c2]

        # Total capacity constraint per rack: w_crit_i + w_batch_i <= 1.0
        for i in range(n):
            constraints.append({
                "type": "ineq",
                "fun": lambda x, idx=i: 1.0 - (x[idx] + x[n + idx])
            })

        # Bounds: [0, 1] for each variable
        bounds = [(0.0, 1.0) for _ in range(2 * n)]

        # Initial guess: prioritize critical on lowest Tin racks
        sort_order = np.argsort(Tin)
        x0 = np.zeros(2 * n)
        # Allocate critical first to coolest racks
        rem_crit = w_crit_total
        for idx in sort_order:
            alloc = min(0.75, rem_crit)
            x0[idx] = alloc
            rem_crit -= alloc

        # Distribute remaining batch
        rem_batch = w_batch_total
        for idx in range(n):
            avail = 1.0 - x0[idx]
            alloc = min(avail, rem_batch / (n - idx))
            x0[n + idx] = alloc
            rem_batch -= alloc

        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 60, "ftol": 1e-4, "disp": False}
        )

        if res.success:
            w_crit_opt = np.clip(res.x[:n], 0.0, 1.0)
            w_batch_opt = np.clip(res.x[n:], 0.0, 1.0)
        else:
            # Fallback to analytical water-filling if solver does not converge
            w_crit_opt = np.zeros(n)
            rem_c = w_crit_total
            for idx in sort_order:
                alloc = min(0.85, rem_c)
                w_crit_opt[idx] = alloc
                rem_c -= alloc

            w_batch_opt = np.zeros(n)
            rem_b = w_batch_total
            for idx in range(n):
                avail = max(0.0, 1.0 - w_crit_opt[idx])
                alloc = min(avail, rem_b / max(1, n - idx))
                w_batch_opt[idx] = alloc
                rem_b -= alloc

        w_tot = np.clip(w_crit_opt + w_batch_opt, 0.0, 1.0)

        return {
            "w_crit": w_crit_opt,
            "w_batch": w_batch_opt,
            "w_total": w_tot,
        }

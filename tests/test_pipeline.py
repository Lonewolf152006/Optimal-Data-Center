"""
Automated Test Suite for Optimal Data Center Cooling Pipeline
==============================================================
Compatible with both pytest and python's standard unittest runner.

Tests:
  1. Thermal Plant lumped-parameter thermodynamics and baseline thermostat
  2. Classical GRU and Hybrid QML PennyLane neural surrogates
  3. Fast analytical-gradient receding-horizon MPC controller
  4. Arrhenius continuous thermal reliability model and time-above-ceiling math
  5. 4-Rack spatial data center recirculation model and convex QP dispatcher
"""

import os
import sys
import unittest
import numpy as np

# Add repository paths
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from plant.thermal_plant import ThermalPlant
from controllers.baseline_thermostat import BaselineThermostat
from ml.train_classical import ClassicalPredictor
from qml.train_qml import QMLPredictor
from optimization.mpc_core_fast import make_mpc_controller_fast
from plant.spatial_thermal_plant import SpatialThermalPlant
from controllers.spatial_workload_dispatcher import WorkloadClassifier, ThermalAwareOptimalDispatcher
from diagnostics.verify_reliability_model import (
    compute_reliability_from_scratch,
    K_BOLTZMANN_EV,
    E_ACTIVATION_EV,
)


class TestThermalPlant(unittest.TestCase):
    def setUp(self):
        self.plant = ThermalPlant()

    def test_thermal_plant_rk4(self):
        """Verify 4th-order Runge-Kutta simulation step produces physically valid temperatures."""
        T0 = 22.0
        u = 0.5
        util = 0.6
        tamb = 28.0
        T1 = self.plant.step(T0, u, util, tamb)
        self.assertIsInstance(T1, float)
        self.assertTrue(15.0 < T1 < 35.0, f"Temperature {T1} out of physical bounds")

    def test_cooling_power_and_cop(self):
        """Verify non-linear cubic fan power and temperature-dependent COP curve."""
        # Check COP bounds
        cop_cold = self.plant.cop_eff(10.0)
        cop_hot = self.plant.cop_eff(35.0)
        self.assertAlmostEqual(cop_cold, 8.5)
        self.assertTrue(2.5 <= cop_hot <= 8.5)

        # Check cubic fan scaling
        p1, q1 = self.plant.p_cool(0.2, 25.0)
        p2, q2 = self.plant.p_cool(0.8, 25.0)
        self.assertGreater(p2, p1)
        self.assertGreater(q2, q1)

    def test_baseline_thermostat(self):
        """Verify thermostatic hysteresis controller around 22 C setpoint."""
        ctrl = BaselineThermostat()
        # High temperature -> full cooling
        u_hot = ctrl(25.0)
        self.assertGreater(u_hot, 0.4)
        # Low temperature -> lower cooling
        u_cold = ctrl(20.0)
        self.assertLess(u_cold, u_hot)


class TestNeuralSurrogates(unittest.TestCase):
    def test_classical_gru(self):
        """Verify Classical GRU model loads checkpoint and outputs 24 predictions."""
        predictor = ClassicalPredictor()
        for _ in range(12):
            predictor.update(22.0, 0.5, 25.0, 50.0, 400.0)
        forecast = np.zeros((24, 3))
        forecast[:, 0] = 0.55
        forecast[:, 1] = 26.0
        forecast[:, 2] = 400.0
        pred = predictor.predict(22.0, forecast)
        self.assertEqual(len(pred), 24)
        self.assertTrue(np.all((pred >= 15.0) & (pred <= 35.0)))

    def test_hybrid_qml_vqc(self):
        """Verify PennyLane Variational Quantum Circuit loads weights and evaluates 4 qubits."""
        qml_pred = QMLPredictor()
        for _ in range(12):
            qml_pred.update(22.0, 0.5, 25.0, 50.0, 400.0)
        forecast = np.zeros((24, 3))
        forecast[:, 0] = 0.55
        forecast[:, 1] = 26.0
        forecast[:, 2] = 400.0
        pred = qml_pred.predict(22.0, forecast)
        self.assertEqual(len(pred), 24)
        self.assertTrue(np.all((pred >= 15.0) & (pred <= 35.0)))


class TestMPCOptimization(unittest.TestCase):
    def test_analytical_gradient_mpc(self):
        """Verify analytical-gradient L-BFGS-B controller solves within horizon bounds."""
        ctrl = make_mpc_controller_fast(predictor_fn=None, horizon=24)
        util_fore = np.full(24, 0.5)
        tamb_fore = np.full(24, 25.0)
        carbon_fore = np.full(24, 400.0)
        u_opt = ctrl(22.0, step_idx=0, util_fore=util_fore,
                     tamb_fore=tamb_fore, carbon_fore=carbon_fore)
        self.assertTrue(0.0 <= u_opt <= 1.0)


class TestReliabilityAndCausalModels(unittest.TestCase):
    def test_arrhenius_reference_point(self):
        """Verify Arrhenius thermal aging factor AF = 1.00 at nominal reference (22 C, 50% load)."""
        T_room = np.array([22.0])
        util = np.array([0.50])
        res = compute_reliability_from_scratch(T_room, util)
        self.assertAlmostEqual(res["Mean_Tj_C"], 53.5, places=3)
        self.assertAlmostEqual(res["Mean_Arrhenius_AF"], 1.0, places=3)
        self.assertAlmostEqual(res["Relative_MTBF"], 1.0, places=3)

    def test_arrhenius_elevated_temperature(self):
        """Verify higher temperatures cause exponential acceleration in aging (AF > 1)."""
        T_room = np.array([27.0])
        util = np.array([0.80])
        res = compute_reliability_from_scratch(T_room, util)
        self.assertGreater(res["Mean_Tj_C"], 53.5)
        self.assertGreater(res["Mean_Arrhenius_AF"], 1.0)
        self.assertLess(res["Relative_MTBF"], 1.0)


class TestSpatialWorkloadPlacement(unittest.TestCase):
    def setUp(self):
        self.plant = SpatialThermalPlant()
        self.classifier = WorkloadClassifier(crit_fraction=0.35)
        self.dispatcher = ThermalAwareOptimalDispatcher(n_racks=4)

    def test_recirculation_matrix(self):
        """Verify 4x4 heat recirculation matrix and Rack 4 penalty."""
        self.assertEqual(self.plant.N_RACKS, 4)
        self.assertEqual(self.plant.D_MATRIX.shape, (4, 4))
        # Rack 4 suffers highest total heat entrainment
        entrainment = np.sum(self.plant.D_MATRIX, axis=1)
        self.assertEqual(np.argmax(entrainment), 3)

    def test_convex_qp_dispatch(self):
        """Verify optimal convex QP directs critical workloads away from hot racks."""
        w_crit, w_batch = self.classifier.split_demand(total_util=0.60, n_racks=4)
        T_inlet = np.array([21.0, 22.0, 23.5, 25.5])  # Rack 4 is severely overheating
        res = self.dispatcher.dispatch(w_crit, w_batch, T_inlet)
        w_c = res["w_crit"]
        w_b = res["w_batch"]
        w_tot = res["w_total"]

        self.assertEqual(len(w_tot), 4)
        self.assertAlmostEqual(np.sum(w_c), w_crit, places=3)
        self.assertAlmostEqual(np.sum(w_b), w_batch, places=3)
        # Critical workload on Rack 4 should be substantially less than on cold Rack 1
        self.assertLess(w_c[3], w_c[0])


if __name__ == "__main__":
    unittest.main()

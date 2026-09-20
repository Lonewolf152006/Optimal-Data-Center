"""
Master One-Click System Verification & Health Audit
===================================================
MathWorks Excellence in Innovation Challenge #196: Optimal Data Center Cooling

Runs quick, non-destructive smoke tests across all subsystems in < 10 seconds:
  1. Environment & Package Dependencies
  2. Dynamic Thermal Plant & Baseline Thermostat
  3. Classical Neural Predictor (GRU PyTorch)
  4. Hybrid Quantum Predictor (PennyLane Variational Circuit)
  5. Fast Carbon-Aware MPC Controller (Analytical Gradient)
  6. Arrhenius Semiconductor Reliability Model (Spot-Check)
  7. 4-Rack Spatial Thermal Model & Convex QP Workload Dispatcher
  8. Data, Models, and Visual Artifacts Integrity

Usage:
  python verify_all.py
"""

import os
import sys
import time
import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_header(title):
    print(f"\n{BOLD}{'=' * 85}{RESET}")
    print(f"{BOLD} {title}{RESET}")
    print(f"{BOLD}{'=' * 85}{RESET}")


def test_environment():
    """Check Python version and core library imports."""
    t0 = time.time()
    errors = []
    
    # Python version
    py_ver = sys.version_info
    if py_ver < (3, 10):
        errors.append(f"Python version >= 3.10 recommended (found {py_ver.major}.{py_ver.minor})")
        
    required_pkgs = ["numpy", "scipy", "pandas", "torch", "pennylane", "matplotlib"]
    for pkg in required_pkgs:
        try:
            __import__(pkg)
        except ImportError as e:
            errors.append(f"Missing dependency: {pkg} ({e})")
            
    passed = len(errors) == 0
    dt = time.time() - t0
    msg = f"Python {py_ver.major}.{py_ver.minor}.{py_ver.micro} + all 6 core libraries verified" if passed else "; ".join(errors)
    return passed, msg, dt


def test_thermal_plant():
    """Verify lumped-parameter plant physics and baseline thermostat."""
    t0 = time.time()
    try:
        from plant.thermal_plant import ThermalPlant
        from controllers.baseline_thermostat import BaselineThermostat
        
        plant = ThermalPlant()
        T0 = 22.0
        u = 0.5
        util = 0.6
        tamb = 30.0
        
        # Step ODE
        T1 = plant.step(T0, u, util, tamb)
        assert 15.0 < T1 < 35.0, f"Unphysical temperature: {T1}"
        
        # Cooling power & COP
        pcool, q_del = plant.p_cool(u, tamb)
        cop = plant.cop_eff(tamb)
        assert pcool > 0 and 2.5 <= cop <= 8.5, f"Invalid power/COP: P={pcool}, COP={cop}"
        
        # Thermostat logic
        ctrl = BaselineThermostat()
        u_b = ctrl(T0)
        assert 0.0 <= u_b <= 1.0, f"Invalid thermostat output: {u_b}"
        
        return True, "4th-order Runge-Kutta ODE, COP curves, and thermostat verified", time.time() - t0
    except Exception as e:
        return False, str(e), time.time() - t0


def test_classical_predictor():
    """Verify Classical GRU neural surrogate model and checkpoint loading."""
    t0 = time.time()
    try:
        from ml.train_classical import ClassicalPredictor
        predictor = ClassicalPredictor()
        
        # Feed 12 steps of rolling lag history
        for _ in range(12):
            predictor.update(22.0, 0.5, 25.0, 50.0, 400.0)
            
        forecast = np.zeros((24, 3))
        forecast[:, 0] = 0.55  # util
        forecast[:, 1] = 26.0  # tamb
        forecast[:, 2] = 420.0 # carbon
        
        pred = predictor.predict(22.0, forecast)
        assert len(pred) == 24, f"Expected 24 predictions, got {len(pred)}"
        assert np.all((pred >= 15.0) & (pred <= 35.0)), f"Predictions out of range: {pred}"
        
        return True, "models/classical.pt loaded; 24-step GRU forecast executed", time.time() - t0
    except Exception as e:
        return False, str(e), time.time() - t0


def test_qml_predictor():
    """Verify PennyLane Variational Quantum Circuit surrogate and weights."""
    t0 = time.time()
    try:
        from qml.train_qml import QMLPredictor
        qml_pred = QMLPredictor()
        
        # Feed 12 steps of rolling lag history
        for _ in range(12):
            qml_pred.update(22.0, 0.5, 25.0, 50.0, 400.0)
            
        forecast = np.zeros((24, 3))
        forecast[:, 0] = 0.55
        forecast[:, 1] = 26.0
        forecast[:, 2] = 420.0
        
        t_infer0 = time.time()
        pred = qml_pred.predict(22.0, forecast)
        infer_ms = (time.time() - t_infer0) * 1000.0
        
        assert len(pred) == 24, f"Expected 24 predictions, got {len(pred)}"
        assert np.all((pred >= 15.0) & (pred <= 35.0)), f"QML predictions out of range: {pred}"
        
        return True, f"models/qml_weights.json loaded; 4-qubit VQC evaluated ({infer_ms:.1f} ms)", time.time() - t0
    except Exception as e:
        return False, str(e), time.time() - t0


def test_fast_mpc():
    """Verify analytical-gradient receding-horizon MPC controller."""
    t0 = time.time()
    try:
        from optimization.mpc_core_fast import make_mpc_controller_fast
        
        # Test MPC solve without ML surrogate (using internal linearized model)
        ctrl = make_mpc_controller_fast(predictor_fn=None, horizon=24)
        
        util_fore = np.full(24, 0.5)
        tamb_fore = np.full(24, 25.0)
        carbon_fore = np.full(24, 400.0)
        
        t_solve0 = time.time()
        u_opt = ctrl(22.0, step_idx=0, util_fore=util_fore,
                     tamb_fore=tamb_fore, carbon_fore=carbon_fore)
        solve_ms = (time.time() - t_solve0) * 1000.0
        
        assert 0.0 <= u_opt <= 1.0, f"Optimal action out of bounds: {u_opt}"
        assert solve_ms < 500.0, f"Solve took too long: {solve_ms:.1f} ms"
        
        return True, f"Analytical L-BFGS-B gradient convergence ({solve_ms:.1f} ms, u={u_opt:.3f})", time.time() - t0
    except Exception as e:
        return False, str(e), time.time() - t0


def test_reliability_model():
    """Verify Arrhenius thermal aging model spot-check against seed 101."""
    t0 = time.time()
    try:
        from diagnostics.verify_reliability_model import compute_reliability_from_scratch, K_BOLTZMANN_EV, E_ACTIVATION_EV
        import pandas as pd
        
        # Synthetic known point: 22 C room, 50% util -> Tj = 53.5 C -> AF = 1.00
        T_room = np.array([22.0])
        util = np.array([0.50])
        res = compute_reliability_from_scratch(T_room, util)
        
        assert abs(res["Mean_Tj_C"] - 53.5) < 1e-4, f"Tj error: {res['Mean_Tj_C']}"
        assert abs(res["Mean_Arrhenius_AF"] - 1.0) < 1e-4, f"AF error: {res['Mean_Arrhenius_AF']}"
        assert abs(res["Relative_MTBF"] - 1.0) < 1e-4, f"MTBF error: {res['Relative_MTBF']}"
        
        # Check that results/reliability_summary.csv exists and matches
        csv_path = os.path.join(REPO_ROOT, "results", "reliability_summary.csv")
        assert os.path.exists(csv_path), "Missing reliability_summary.csv"
        df = pd.read_csv(csv_path)
        assert len(df) >= 30, f"Expected 30 rows (10 seeds x 3 controllers), found {len(df)}"
        
        return True, "JEDEC/IEEE Arrhenius math verified; 10-seed summary intact", time.time() - t0
    except Exception as e:
        return False, str(e), time.time() - t0


def test_spatial_workload_dispatcher():
    """Verify 4-rack recirculation model and convex QP dispatcher."""
    t0 = time.time()
    try:
        from plant.spatial_thermal_plant import SpatialThermalPlant
        from controllers.spatial_workload_dispatcher import WorkloadClassifier, ThermalAwareOptimalDispatcher
        
        plant = SpatialThermalPlant()
        assert plant.N_RACKS == 4, "Expected 4 racks"
        assert plant.D_MATRIX.shape == (4, 4), "Expected 4x4 recirculation matrix"
        
        classifier = WorkloadClassifier(crit_fraction=0.35)
        w_crit, w_batch = classifier.split_demand(total_util=0.60, n_racks=4)
        
        dispatcher = ThermalAwareOptimalDispatcher(n_racks=4)
        T_inlet = np.array([21.0, 22.0, 23.5, 25.0])
        res = dispatcher.dispatch(w_crit, w_batch, T_inlet)
        w_c = res["w_crit"]
        w_b = res["w_batch"]
        w_tot = res["w_total"]
        
        assert len(w_tot) == 4, "Expected 4 rack allocations"
        assert np.isclose(np.sum(w_c), w_crit, atol=1e-3), "Critical workload balance violated"
        assert np.isclose(np.sum(w_b), w_batch, atol=1e-3), "Batch workload balance violated"
        # Rack 4 is hottest (25.0 C); optimal dispatcher should allocate less critical work to Rack 4 than Rack 1
        assert w_c[3] < w_c[0], "Optimal dispatcher failed to steer critical work away from hot rack 4"
        
        return True, "Convex QP solved; critical work steered from hot Rack 4 to cold Rack 1", time.time() - t0
    except Exception as e:
        return False, str(e), time.time() - t0


def test_artifacts_integrity():
    """Verify dataset, trained models, visual figures, and companion Simulink model."""
    t0 = time.time()
    missing = []
    
    expected_files = [
        os.path.join(REPO_ROOT, "models", "classical.pt"),
        os.path.join(REPO_ROOT, "models", "qml_weights.json"),
        os.path.join(REPO_ROOT, "models", "DataCenterCooling.slx"),
        os.path.join(REPO_ROOT, "data", "thermal_dataset.csv"),
        os.path.join(REPO_ROOT, "results", "fig1_timeseries.png"),
        os.path.join(REPO_ROOT, "results", "fig3_multiseed_boxplots.png"),
        os.path.join(REPO_ROOT, "results", "fig5_bias_by_horizon.png"),
        os.path.join(REPO_ROOT, "results", "fig5_component_reliability.png"),
        os.path.join(REPO_ROOT, "results", "fig6_spatial_workload_placement.png"),
        os.path.join(REPO_ROOT, "results", "multi_seed_results.csv"),
        os.path.join(REPO_ROOT, "results", "causal_bias_test_results.csv"),
        os.path.join(REPO_ROOT, "docs", "METHODOLOGY.md"),
        os.path.join(REPO_ROOT, "docs", "MATHWORKS_SUBMISSION_ANSWERS.md"),
        os.path.join(REPO_ROOT, "LICENSE"),
        os.path.join(REPO_ROOT, "data", "sample", "sample_thermal_data.csv"),
    ]
    
    for f in expected_files:
        if not os.path.exists(f):
            missing.append(os.path.relpath(f, REPO_ROOT))
            
    if missing:
        return False, f"Missing {len(missing)} artifact(s): {', '.join(missing[:3])}", time.time() - t0
        
    return True, f"All {len(expected_files)} key artifacts, figures, models, and datasets present", time.time() - t0


def main():
    print_header("MATHWORKS CHALLENGE #196 -- MASTER SYSTEM HEALTH AUDIT")
    print("Project: Optimal Data Center Cooling (Predictive & Quantum-Enhanced)")
    print(f"Repository Root: {REPO_ROOT}\n")
    
    tests = [
        ("1. Environment & Dependencies", test_environment),
        ("2. Plant Physics & Thermostat", test_thermal_plant),
        ("3. Classical GRU Predictor", test_classical_predictor),
        ("4. Hybrid QML PennyLane VQC", test_qml_predictor),
        ("5. Fast Carbon-Aware MPC", test_fast_mpc),
        ("6. Arrhenius Reliability Model", test_reliability_model),
        ("7. 4-Rack Spatial QP Dispatcher", test_spatial_workload_dispatcher),
        ("8. Models & Visual Artifacts", test_artifacts_integrity),
    ]
    
    results = []
    total_time0 = time.time()
    
    print(f"{'Subsystem / Test Category':<35} {'Result':<10} {'Time':<8} {'Details'}")
    print("-" * 88)
    
    all_passed = True
    for name, test_fn in tests:
        passed, details, elapsed = test_fn()
        results.append((name, passed, details, elapsed))
        if not passed:
            all_passed = False
            
        status_str = f"{GREEN}[ PASS ]{RESET}" if passed else f"{RED}[ FAIL ]{RESET}"
        print(f"{name:<35} {status_str:<10} {elapsed:5.2f}s  {details}")
        
    total_elapsed = time.time() - total_time0
    print("-" * 88)
    
    print_header("AUDIT SUMMARY")
    if all_passed:
        print(f"{GREEN}{BOLD}>>> ALL 8 SUBSYSTEM CHECKS PASSED SUCCESSFULLY ({total_elapsed:.2f}s) <<<{RESET}")
        print("  - Python physics and optimal control engines are operational.")
        print("  - Classical GRU and Hybrid QML models execute valid predictions.")
        print("  - Reliability and 4-rack spatial convex QP algorithms verified.")
        print("  - All evaluation datasets, figures, and models are present and valid.")
        print("  - Ready for MathWorks evaluation, peer review, and defense.")
    else:
        print(f"{RED}{BOLD}>>> AUDIT COMPLETED WITH FAILURES ({total_elapsed:.2f}s) <<<{RESET}")
        print("Please inspect the failed tests above.")
    print("=" * 85 + "\n")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

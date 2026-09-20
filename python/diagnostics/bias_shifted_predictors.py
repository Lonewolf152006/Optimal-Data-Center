"""
Bias-Shifted Predictor Wrappers for Causal Experiment
======================================================
Defines wrapper predictor classes that intercept an existing trained
predictor's output trajectory and add (or subtract) a fixed per-step
bias offset BEFORE the MPC optimizer sees it.

This enables causal interventions:
  - "If we artificially give GRU the same near-term positive bias that
     QML naturally exhibits, does GRU's compliance jump to QML's level?"
  - "If we artificially remove QML's near-term positive bias, does QML's
     compliance drop to GRU's level?"

All wrappers expose the same (predict, update, reset) interface as
ClassicalPredictor / QMLPredictor, so they can be used as drop-in
replacements inside PredictorWrapper and make_mpc_controller_fast.

NO files are modified. NO models are retrained. This is a read-only
diagnostic that loads bias_by_horizon_step.csv at import time.
"""

import os
import numpy as np
import pandas as pd

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
BIAS_CSV = os.path.join(REPO_ROOT, "results", "bias_by_horizon_step.csv")


def _load_bias_offsets():
    """Load the measured bias difference (QML - GRU) from CSV.

    Returns
    -------
    diff_qml_minus_gru : ndarray (24,)
        Per-horizon-step offset (QML bias - GRU bias).
    qml_bias : ndarray (24,)
        Per-horizon-step QML signed bias.
    """
    df = pd.read_csv(BIAS_CSV)
    diff = df["Difference_QML_minus_GRU"].values.astype(np.float64)
    qml_bias = df["Hybrid_QML_Bias"].values.astype(np.float64)
    return diff, qml_bias


# Load once at module import
_DIFF_QML_MINUS_GRU, _QML_BIAS = _load_bias_offsets()
HORIZON = len(_DIFF_QML_MINUS_GRU)  # 24


class BiasShiftedPredictor:
    """Generic wrapper: adds a fixed offset vector to an inner predictor's output.

    Parameters
    ----------
    inner_predictor : object
        Must have predict(current_state, forecast_features), update(...), reset().
    offset : ndarray (HORIZON,)
        Fixed offset to ADD to the inner predictor's output trajectory.
    label : str
        Human-readable label for logging.
    """

    def __init__(self, inner_predictor, offset, label="BiasShifted"):
        self.inner = inner_predictor
        self.offset = np.array(offset, dtype=np.float64)
        self.label = label

    def predict(self, current_state, forecast_features):
        raw = self.inner.predict(current_state, forecast_features)
        return raw + self.offset

    def update(self, *args, **kwargs):
        return self.inner.update(*args, **kwargs)

    def reset(self):
        return self.inner.reset()


# ------------------------------------------------------------------
# Factory functions for the four experimental conditions
# ------------------------------------------------------------------

def make_gru_plus_qml_near_bias(gru_predictor):
    """GRU + QML-minus-GRU bias offset on steps 1-4 only, zero elsewhere.

    Hypothesis: If near-term (5-20 min) positive bias is the causal driver,
    grafting QML's near-term bias onto GRU should boost compliance toward 99.99%.
    """
    offset = np.zeros(HORIZON)
    offset[:4] = _DIFF_QML_MINUS_GRU[:4]  # steps 1-4 (indices 0-3)
    return BiasShiftedPredictor(
        gru_predictor, offset,
        label="GRU + QML Near Bias (Steps 1-4)"
    )


def make_gru_plus_qml_full_bias(gru_predictor):
    """GRU + QML-minus-GRU bias offset across ALL 24 steps.

    Tests whether the causal effect is distributed across the full horizon
    rather than concentrated in steps 1-4.
    """
    offset = _DIFF_QML_MINUS_GRU.copy()
    return BiasShiftedPredictor(
        gru_predictor, offset,
        label="GRU + QML Full Bias (Steps 1-24)"
    )


def make_qml_minus_qml_near_bias(qml_predictor):
    """QML with its OWN steps 1-4 bias subtracted out, zero elsewhere.

    Removes QML's near-term over-prediction, pushing it toward GRU-like
    behavior in the receding-horizon control window. If near-term bias is
    causal, compliance should drop toward GRU's 87.81%.
    """
    offset = np.zeros(HORIZON)
    offset[:4] = -_QML_BIAS[:4]  # subtract QML's own bias at steps 1-4
    return BiasShiftedPredictor(
        qml_predictor, offset,
        label="QML - Own Near Bias (Steps 1-4)"
    )


def make_qml_minus_qml_full_bias(qml_predictor):
    """QML with its OWN bias subtracted across ALL 24 steps.

    Removes QML's bias profile entirely. Tests full-horizon vs near-term.
    """
    offset = -_QML_BIAS.copy()
    return BiasShiftedPredictor(
        qml_predictor, offset,
        label="QML - Own Full Bias (Steps 1-24)"
    )

"""
Hybrid Quantum-Classical (QML) Temperature Predictor
=====================================================
Uses PennyLane for a variational quantum circuit sandwiched between
classical dimensionality-reduction and regression layers.

Architecture
------------
  Input (132 features: 12×5 lag + 24×3 forecast)
    → Classical encoder: Linear(132→32) → ReLU → Linear(32→8) → ReLU → Linear(8→4) → Tanh·π
    → 4-qubit variational circuit (angle encoding + 3 entangling layers)
    → 4 expectation values ⟨Z⟩
    → Classical decoder: Linear(4→32) → ReLU → Linear(32→24)

IDENTICAL input features, lag window, and prediction horizon as
``train_classical.py`` for a fair comparison.

Reports MAE / RMSE / R² plus inference latency.
Saves frozen parameters to ``models/qml_weights.json``.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pennylane as qml

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CSV_PATH = os.path.join(REPO_ROOT, "data", "thermal_dataset.csv")
WEIGHTS_PATH = os.path.join(REPO_ROOT, "models", "qml_weights.json")

LAG_WINDOW = 12
HORIZON = 24
LAG_FEATURES = 5
FORECAST_FEATURES = 3
INPUT_DIM = LAG_WINDOW * LAG_FEATURES + HORIZON * FORECAST_FEATURES  # 132
N_QUBITS = 4
N_LAYERS = 3


# ======================================================================
# Quantum circuit
# ======================================================================
dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    """4-qubit variational circuit with angle encoding and entangling layers.

    Supports batched inputs via parameter broadcasting.

    Parameters
    ----------
    inputs : tensor (batch, 4) or (4,)   Encoded features (angles).
    weights : tensor (n_layers, n_qubits, 2)  Variational parameters.
    """
    # Angle encoding (vectorized across batch)
    for i in range(N_QUBITS):
        if inputs.ndim > 1:
            qml.RY(inputs[:, i], wires=i)
        else:
            qml.RY(inputs[i], wires=i)

    # Variational layers
    for layer in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        # Ring entanglement
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# ======================================================================
# Hybrid model
# ======================================================================
class HybridQMLModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Classical encoder → 4 angle values
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, N_QUBITS),
            nn.Tanh(),  # output in [-1, 1], scaled by π later
        )
        # Variational quantum weights
        self.q_weights = nn.Parameter(0.1 * torch.randn(N_LAYERS, N_QUBITS, 2))

        # Classical decoder
        self.decoder = nn.Sequential(
            nn.Linear(N_QUBITS, 32),
            nn.ReLU(),
            nn.Linear(32, HORIZON),
        )

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        encoded = self.encoder(x) * np.pi  # scale to [-π, π]
        q_res = quantum_circuit(encoded, self.q_weights)
        q_out = torch.stack(q_res, dim=-1).float()  # (batch, N_QUBITS), ensure float32
        return self.decoder(q_out)  # (batch, HORIZON)


# ======================================================================
# Dataset (same logic as train_classical — exact parity)
# ======================================================================
def build_sequences(df, max_samples=None):
    lag_cols = ["room_temp", "utilization", "ambient_temp",
                "cooling_power", "carbon_intensity"]
    fc_cols = ["utilization", "ambient_temp", "carbon_intensity"]
    target_col = "room_temp"

    X_all, Y_all = [], []
    for _, grp in df.groupby("run_id"):
        grp = grp.sort_values("time").reset_index(drop=True)
        lag_data = grp[lag_cols].values
        fc_data = grp[fc_cols].values
        tgt_data = grp[target_col].values
        n = len(grp)
        for i in range(LAG_WINDOW, n - HORIZON):
            lag_flat = lag_data[i - LAG_WINDOW: i].flatten()
            fc_flat = fc_data[i: i + HORIZON].flatten()
            X_all.append(np.concatenate([lag_flat, fc_flat]))
            Y_all.append(tgt_data[i: i + HORIZON] - tgt_data[i - 1])

    X_all = np.array(X_all, dtype=np.float32)
    Y_all = np.array(Y_all, dtype=np.float32)

    if max_samples is not None and len(X_all) > max_samples:
        idx = np.random.default_rng(42).choice(len(X_all), max_samples, replace=False)
        X_all, Y_all = X_all[idx], Y_all[idx]

    return X_all, Y_all


# ======================================================================
# Training
# ======================================================================
def train():
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)

    train_ids = list(range(80))
    test_ids = list(range(80, 100))
    df_train = df[df["run_id"].isin(train_ids)]
    df_test = df[df["run_id"].isin(test_ids)]

    # QML trains on a subsample for tractable runtime
    QML_TRAIN_SAMPLES = 5000
    QML_TEST_SAMPLES = 2000
    print(f"Building sequences (train <= {QML_TRAIN_SAMPLES}, test <= {QML_TEST_SAMPLES})...", flush=True)
    X_tr, Y_tr = build_sequences(df_train, max_samples=QML_TRAIN_SAMPLES)
    X_te, Y_te = build_sequences(df_test, max_samples=QML_TEST_SAMPLES)
    print(f"  train: {len(X_tr)}, test: {len(X_te)}")

    # Normalisation
    x_mean, x_std = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-8
    y_mean, y_std = Y_tr.mean(), Y_tr.std() + 1e-8

    X_tr_n = (X_tr - x_mean) / x_std
    Y_tr_n = (Y_tr - y_mean) / y_std
    X_te_n = (X_te - x_mean) / x_std

    train_ds = TensorDataset(torch.from_numpy(X_tr_n), torch.from_numpy(Y_tr_n))
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = HybridQMLModel()
    optim = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    EPOCHS = 30
    print(f"\nTraining hybrid QML for {EPOCHS} epochs...", flush=True)
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        n_samples = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * len(yb)
            n_samples += len(yb)
        epoch_loss /= n_samples
        scheduler.step(epoch_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{EPOCHS}  loss={epoch_loss:.6f}", flush=True)

    # Evaluate
    model.eval()
    with torch.no_grad():
        pred_te = model(torch.from_numpy(X_te_n)).numpy()
    pred_te = pred_te * y_std + y_mean

    mae = mean_absolute_error(Y_te.flatten(), pred_te.flatten())
    rmse = np.sqrt(mean_squared_error(Y_te.flatten(), pred_te.flatten()))
    r2 = r2_score(Y_te.flatten(), pred_te.flatten())

    # Inference latency
    single_input = torch.from_numpy(X_te_n[:1])
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(single_input)
        latencies.append(time.perf_counter() - t0)
    avg_latency_ms = np.mean(latencies) * 1000

    print(f"\n--- Hybrid QML Test Metrics ---", flush=True)
    print(f"  MAE     = {mae:.4f} deg C", flush=True)
    print(f"  RMSE    = {rmse:.4f} deg C", flush=True)
    print(f"  R2      = {r2:.4f}", flush=True)
    print(f"  Latency = {avg_latency_ms:.2f} ms / prediction", flush=True)

    # Save weights
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    save_dict = {
        "encoder": {k: v.tolist() for k, v in model.encoder.state_dict().items()},
        "q_weights": model.q_weights.detach().numpy().tolist(),
        "decoder": {k: v.tolist() for k, v in model.decoder.state_dict().items()},
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2,
                    "latency_ms": avg_latency_ms},
    }
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(save_dict, f, indent=2)
    print(f"  Weights saved to {WEIGHTS_PATH}")
    return model, save_dict["metrics"]


# ======================================================================
# Predictor wrapper (for MPC integration)
# ======================================================================
class QMLPredictor:
    """Wraps the trained hybrid QML model and exposes ``predict()``."""

    def __init__(self, weights_path=WEIGHTS_PATH):
        with open(weights_path) as f:
            ckpt = json.load(f)

        self.model = HybridQMLModel()
        # Restore encoder
        enc_sd = {k: torch.tensor(v, dtype=torch.float32) for k, v in ckpt["encoder"].items()}
        self.model.encoder.load_state_dict(enc_sd)
        # Restore quantum weights
        self.model.q_weights = nn.Parameter(torch.tensor(ckpt["q_weights"], dtype=torch.float32))
        # Restore decoder
        dec_sd = {k: torch.tensor(v, dtype=torch.float32) for k, v in ckpt["decoder"].items()}
        self.model.decoder.load_state_dict(dec_sd)
        self.model.eval()

        self.x_mean = np.array(ckpt["x_mean"], dtype=np.float32)
        self.x_std = np.array(ckpt["x_std"], dtype=np.float32)
        self.y_mean = ckpt["y_mean"]
        self.y_std = ckpt["y_std"]
        self.history = []

    def update(self, room_temp, utilization, ambient_temp,
               cooling_power, carbon_intensity):
        self.history.append([room_temp, utilization, ambient_temp,
                             cooling_power, carbon_intensity])

    def predict(self, current_state, forecast_features):
        if len(self.history) < LAG_WINDOW:
            return np.full(HORIZON, current_state)

        lag = np.array(self.history[-LAG_WINDOW:], dtype=np.float32).flatten()
        fc = forecast_features[:HORIZON].flatten().astype(np.float32)
        x = np.concatenate([lag, fc])
        x_n = (x - self.x_mean) / self.x_std

        with torch.no_grad():
            pred = self.model(torch.from_numpy(x_n).unsqueeze(0)).numpy().flatten()
        delta = pred * self.y_std + self.y_mean
        return current_state + delta

    def reset(self):
        self.history = []


if __name__ == "__main__":
    train()

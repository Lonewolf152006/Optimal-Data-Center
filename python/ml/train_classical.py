"""
Classical (GRU) Temperature Predictor
=======================================
Trains a GRU to predict room temperature over the next ``horizon`` steps
given a lag window of past observations and forecast features.

Architecture
------------
  Lag window (12 × 5)  →  GRU(hidden=64, layers=2)  →  h_gru (64)
  Forecast (24 × 3 = 72)  →  MLP  →  h_fc (64)
  Concat(h_gru, h_fc)  →  MLP  →  24 predicted temperatures

Reports MAE / RMSE / R² on a held-out test split.
Saves to ``models/classical.pt``.
Exposes ``predict(current_state, forecast_features)`` matching the
MPC predictor_fn signature.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CSV_PATH = os.path.join(REPO_ROOT, "data", "thermal_dataset.csv")
MODEL_PATH = os.path.join(REPO_ROOT, "models", "classical.pt")

LAG_WINDOW = 12
HORIZON = 24
LAG_FEATURES = 5   # room_temp, utilization, ambient_temp, cooling_power, carbon_intensity
FORECAST_FEATURES = 3   # utilization, ambient_temp, carbon_intensity


# ======================================================================
# Model
# ======================================================================
class TempPredictorGRU(nn.Module):
    def __init__(self, lag_feat=LAG_FEATURES, fc_feat=FORECAST_FEATURES,
                 lag_win=LAG_WINDOW, horizon=HORIZON, hidden=64, layers=2):
        super().__init__()
        self.gru = nn.GRU(lag_feat, hidden, layers,
                          batch_first=True, dropout=0.1)
        self.fc_net = nn.Sequential(
            nn.Linear(fc_feat * horizon, hidden),
            nn.ReLU(),
        )
        self.out_net = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizon),
        )

    def forward(self, lag_seq, forecast_flat):
        _, h_n = self.gru(lag_seq)
        h_gru = h_n[-1]                         # (batch, hidden)
        h_fc = self.fc_net(forecast_flat)        # (batch, hidden)
        return self.out_net(torch.cat([h_gru, h_fc], dim=1))


# ======================================================================
# Dataset preparation
# ======================================================================
def build_sequences(df):
    """Extract (lag_window, forecast, target) triples from the dataframe."""
    lag_cols = ["room_temp", "utilization", "ambient_temp",
                "cooling_power", "carbon_intensity"]
    fc_cols = ["utilization", "ambient_temp", "carbon_intensity"]
    target_col = "room_temp"

    X_lag, X_fc, Y = [], [], []

    for run_id, grp in df.groupby("run_id"):
        grp = grp.sort_values("time").reset_index(drop=True)
        data = grp[lag_cols].values
        fc_data = grp[fc_cols].values
        tgt_data = grp[target_col].values
        n = len(grp)

        for i in range(LAG_WINDOW, n - HORIZON):
            X_lag.append(data[i - LAG_WINDOW: i])       # (lag, 5)
            X_fc.append(fc_data[i: i + HORIZON].flatten())  # (horizon*3,)
            Y.append(tgt_data[i: i + HORIZON] - tgt_data[i - 1])  # (horizon,) delta from current state

    return (np.array(X_lag, dtype=np.float32),
            np.array(X_fc, dtype=np.float32),
            np.array(Y, dtype=np.float32))


# ======================================================================
# Training
# ======================================================================
def train():
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)

    # Split by run_id: first 80 runs train, last 20 test
    train_ids = list(range(80))
    test_ids = list(range(80, 100))
    df_train = df[df["run_id"].isin(train_ids)]
    df_test = df[df["run_id"].isin(test_ids)]

    print("Building sequences...")
    X_lag_tr, X_fc_tr, Y_tr = build_sequences(df_train)
    X_lag_te, X_fc_te, Y_te = build_sequences(df_test)
    print(f"  train samples: {len(Y_tr)}, test samples: {len(Y_te)}")

    # Compute normalization stats from training data
    lag_mean = X_lag_tr.reshape(-1, LAG_FEATURES).mean(axis=0)
    lag_std = X_lag_tr.reshape(-1, LAG_FEATURES).std(axis=0) + 1e-8
    fc_mean = X_fc_tr.mean(axis=0)
    fc_std = X_fc_tr.std(axis=0) + 1e-8
    y_mean = Y_tr.mean()
    y_std = Y_tr.std() + 1e-8

    # Normalize
    def norm_lag(x): return (x - lag_mean) / lag_std
    def norm_fc(x): return (x - fc_mean) / fc_std
    def norm_y(y): return (y - y_mean) / y_std
    def denorm_y(y): return y * y_std + y_mean

    X_lag_tr_n = norm_lag(X_lag_tr)
    X_fc_tr_n = norm_fc(X_fc_tr)
    Y_tr_n = norm_y(Y_tr)
    X_lag_te_n = norm_lag(X_lag_te)
    X_fc_te_n = norm_fc(X_fc_te)

    train_ds = TensorDataset(
        torch.from_numpy(X_lag_tr_n),
        torch.from_numpy(X_fc_tr_n),
        torch.from_numpy(Y_tr_n),
    )
    loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TempPredictorGRU().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    EPOCHS = 25
    print(f"\nTraining GRU for {EPOCHS} epochs on {device}...", flush=True)
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for lag_b, fc_b, y_b in loader:
            lag_b, fc_b, y_b = lag_b.to(device), fc_b.to(device), y_b.to(device)
            pred = model(lag_b, fc_b)
            loss = criterion(pred, y_b)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * len(y_b)
        epoch_loss /= len(train_ds)
        scheduler.step(epoch_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{EPOCHS}  loss={epoch_loss:.6f}", flush=True)

    # Evaluate
    model.eval()
    with torch.no_grad():
        pred_te = model(
            torch.from_numpy(X_lag_te_n).to(device),
            torch.from_numpy(X_fc_te_n).to(device),
        ).cpu().numpy()
    pred_te = denorm_y(pred_te)

    mae = mean_absolute_error(Y_te.flatten(), pred_te.flatten())
    rmse = np.sqrt(mean_squared_error(Y_te.flatten(), pred_te.flatten()))
    r2 = r2_score(Y_te.flatten(), pred_te.flatten())
    print(f"\n--- Classical GRU Test Metrics ---")
    print(f"  MAE  = {mae:.4f} °C")
    print(f"  RMSE = {rmse:.4f} °C")
    print(f"  R²   = {r2:.4f}")

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "lag_mean": lag_mean.tolist(),
        "lag_std": lag_std.tolist(),
        "fc_mean": fc_mean.tolist(),
        "fc_std": fc_std.tolist(),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
    }, MODEL_PATH)
    print(f"  Model saved to {MODEL_PATH}")
    return model, {"mae": mae, "rmse": rmse, "r2": r2}


# ======================================================================
# Predictor wrapper (for MPC integration)
# ======================================================================
class ClassicalPredictor:
    """Wraps the trained GRU and exposes the ``predict()`` interface
    expected by ``mpc_core.make_mpc_controller(predictor_fn=...)``.

    Call ``update()`` after each simulation step to feed the rolling
    lag window, then ``predict()`` to get a temperature trajectory.
    """

    def __init__(self, model_path=MODEL_PATH, device="cpu"):
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        self.device = device
        self.model = TempPredictorGRU().to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.lag_mean = np.array(ckpt["lag_mean"], dtype=np.float32)
        self.lag_std = np.array(ckpt["lag_std"], dtype=np.float32)
        self.fc_mean = np.array(ckpt["fc_mean"], dtype=np.float32)
        self.fc_std = np.array(ckpt["fc_std"], dtype=np.float32)
        self.y_mean = ckpt["y_mean"]
        self.y_std = ckpt["y_std"]
        self.history = []

    def update(self, room_temp, utilization, ambient_temp,
               cooling_power, carbon_intensity):
        self.history.append([room_temp, utilization, ambient_temp,
                             cooling_power, carbon_intensity])

    def predict(self, current_state, forecast_features):
        """
        Parameters
        ----------
        current_state : float   Current room temperature.
        forecast_features : ndarray (horizon, 3)
            [utilization, ambient_temp, carbon_intensity] per step.

        Returns
        -------
        ndarray (horizon,)  Predicted room temperatures.
        """
        if len(self.history) < LAG_WINDOW:
            return np.full(HORIZON, current_state)

        lag = np.array(self.history[-LAG_WINDOW:], dtype=np.float32)
        fc = forecast_features[:HORIZON].flatten().astype(np.float32)

        lag_n = (lag - self.lag_mean) / self.lag_std
        fc_n = (fc - self.fc_mean) / self.fc_std

        with torch.no_grad():
            pred = self.model(
                torch.from_numpy(lag_n).unsqueeze(0).to(self.device),
                torch.from_numpy(fc_n).unsqueeze(0).to(self.device),
            ).cpu().numpy().flatten()
        delta = pred * self.y_std + self.y_mean
        return current_state + delta

    def reset(self):
        self.history = []


if __name__ == "__main__":
    train()

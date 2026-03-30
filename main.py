from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

from dataset import CSVDataset
from model import CNN1DModel, CSVModel, TransformerEncoderModel
from train_test import test, train, unnormalize

WINDOW = 72 * 2  # 72 hours of past data at 30-min intervals = 3 days
HORIZON = 12 * 2  # 24 steps ahead at 30-min intervals = 12 hours into the future
BATCH_SIZE = 128
LR = 0.0005
NUM_EPOCHS = 100
HIDDEN_SIZE = 32
NUM_LAYERS = 4
TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2
DATASET_PATH = "carbon_data/uk_carbon_intensity_2011-04-02_to_2026-03-29_with_weather.csv"

FEATURES = [
    # "temperature",
    # "wind_speed_10m",
    "wind_speed_100m",
    # "wind_direction",
    # "cloud_cover",
    "solar_radiation",
    "precipitation",
    "pressure",
    # "humidity",
]
TARGET = "carbon_intensity"
RESIDUAL_TARGET = "carbon_intensity_residual"

torch.manual_seed(42)
np.random.seed(42)

df = pd.read_csv(DATASET_PATH)
df = df.dropna()
df[TARGET] = df["carbon_intensity"]

lag_roll_features = [col for col in df.columns if ("_lag_" in col or "_rolling_mean_" in col)]
MODEL_FEATURES = FEATURES + lag_roll_features

if "carbon_intensity_lag_1" not in df.columns:
    raise KeyError("carbon_intensity_lag_1 is required for persistence baseline comparison.")

# Train the model on residuals: residual = actual - persistence baseline.
df[RESIDUAL_TARGET] = df[TARGET] - df["carbon_intensity_lag_1"]

# Persistence baseline over full dataframe: predict current value using lag-1.
baseline_mae_full = np.mean(np.abs(df[TARGET] - df["carbon_intensity_lag_1"]))
print(f"Persistence baseline MAE (full data): {baseline_mae_full:.4f}")


n = len(df)
train_end = int(n * TRAIN_SPLIT)
val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))

X_train = df[MODEL_FEATURES].iloc[:train_end]
y_train = df[RESIDUAL_TARGET].iloc[:train_end]
X_val = df[MODEL_FEATURES].iloc[train_end:val_end]
y_val = df[RESIDUAL_TARGET].iloc[train_end:val_end]
X_test = df[MODEL_FEATURES].iloc[val_end:]
y_test = df[RESIDUAL_TARGET].iloc[val_end:]
y_val_target_raw = df[TARGET].iloc[train_end:val_end].to_numpy(dtype=np.float32)
y_test_target_raw = df[TARGET].iloc[val_end:].to_numpy(dtype=np.float32)

# Align persistence baseline to the same test targets used by the windowed dataset.
val_baseline_pred_raw = (
    df["carbon_intensity_lag_1"].iloc[train_end:val_end].to_numpy(dtype=np.float32)
)
test_baseline_pred_raw = df["carbon_intensity_lag_1"].iloc[val_end:].to_numpy(dtype=np.float32)
aligned_start = WINDOW + HORIZON - 1
if len(y_test_target_raw) <= aligned_start:
    raise ValueError("Test split too small for baseline alignment with current window/horizon.")
baseline_mae_test = np.mean(
    np.abs(y_test_target_raw[aligned_start:] - test_baseline_pred_raw[aligned_start:])
)
print(f"Persistence baseline MAE (test-aligned): {baseline_mae_test:.4f}")

val_target_aligned = y_val_target_raw[aligned_start:]
val_baseline_aligned = val_baseline_pred_raw[aligned_start:]

sc = RobustScaler()
X_train = sc.fit_transform(X_train).astype(np.float32)
X_val = sc.transform(X_val).astype(np.float32)
X_test = sc.transform(X_test).astype(np.float32)

sc_target = RobustScaler()
y_train = (
    sc_target.fit_transform(y_train.to_numpy(dtype=np.float32).reshape(-1, 1))
    .flatten()
    .astype(np.float32)
)
y_val = (
    sc_target.transform(y_val.to_numpy(dtype=np.float32).reshape(-1, 1))
    .flatten()
    .astype(np.float32)
)
y_test = (
    sc_target.transform(y_test.to_numpy(dtype=np.float32).reshape(-1, 1))
    .flatten()
    .astype(np.float32)
)

train_dataset = CSVDataset(X_train, y_train, window=WINDOW, horizon=HORIZON, normalize=False)
val_dataset = CSVDataset(X_val, y_val, window=WINDOW, horizon=HORIZON, normalize=False)
test_dataset = CSVDataset(X_test, y_test, window=WINDOW, horizon=HORIZON, normalize=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Device selection
if torch.cuda.is_available():
    print("✓ CUDA is available. Using GPU for training.")
    device = torch.device("cuda")
elif torch.mps.is_available():
    print("✓ MPS is available. Using Apple Silicon GPU for training.")
    device = torch.device("mps")
else:
    print("✓ No GPU available. Using CPU for training.")
    device = torch.device("cpu")


MODEL_TYPE = "transformer"
TRANSFORMER_POOLING = "last"  # options: 'last', 'mean', 'attention', or 'sweep'

train_dataset = CSVDataset(X_train, y_train, window=WINDOW, horizon=HORIZON, normalize=False)
val_dataset = CSVDataset(X_val, y_val, window=WINDOW, horizon=HORIZON, normalize=False)
test_dataset = CSVDataset(X_test, y_test, window=WINDOW, horizon=HORIZON, normalize=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

if MODEL_TYPE == "transformer" and TRANSFORMER_POOLING == "sweep":
    pooling_modes = ["last", "mean", "attention"]
elif MODEL_TYPE == "transformer":
    pooling_modes = [TRANSFORMER_POOLING]
else:
    pooling_modes = [None]

results = []
for pooling_mode in pooling_modes:
    if MODEL_TYPE == "lstm":
        model = CSVModel(
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            input_size=X_train.shape[1],
        )
    elif MODEL_TYPE == "cnn":
        model = CNN1DModel(
            input_size=X_train.shape[1],
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
        )
    elif MODEL_TYPE == "transformer":
        transformer_pooling_mode = pooling_mode if pooling_mode is not None else "last"
        print(f"\n--- Transformer pooling mode: {pooling_mode} ---")
        model = TransformerEncoderModel(
            input_size=X_train.shape[1],
            d_model=HIDDEN_SIZE,
            nhead=4,
            num_layers=NUM_LAYERS,
            pooling=transformer_pooling_mode,
        )
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    pool_suffix = f"_{pooling_mode}" if pooling_mode is not None else ""
    model_path = Path(
        f"best_model_{MODEL_TYPE}{pool_suffix}_h{HIDDEN_SIZE}_l{NUM_LAYERS}_w{WINDOW}_lr{LR}.pth"
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
    )
    loss_fn = torch.nn.SmoothL1Loss(beta=0.5)
    train(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        NUM_EPOCHS,
        device,
        path_to_model=model_path,
        target_scaler=sc_target,
        val_baseline_raw=val_baseline_aligned,
        val_target_raw=val_target_aligned,
        scheduler=scheduler,
        grad_clip=1.0,
        patience=10,
    )

    # Evaluate the best checkpoint saved during early stopping.
    best_checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(best_checkpoint, torch.nn.Module):
        model = best_checkpoint.to(device)
    elif isinstance(best_checkpoint, dict):
        model.load_state_dict(best_checkpoint)
        model.to(device)
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(best_checkpoint)}")

    print("Evaluating on test set...")
    mean_test_loss, residual_mae = test(
        model,
        test_loader,
        loss_fn,
        device,
        target_scaler=sc_target,
    )

    # Reconstruct carbon intensity prediction from baseline + predicted residual.
    model.eval()
    all_pred_residuals = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            pred = model(x).squeeze()
            all_pred_residuals.append(pred.cpu().numpy())

    pred_residual_raw = unnormalize(np.concatenate(all_pred_residuals), sc_target)
    baseline_aligned = test_baseline_pred_raw[aligned_start:]
    target_aligned = y_test_target_raw[aligned_start:]
    if len(pred_residual_raw) != len(target_aligned):
        raise ValueError(
            "Prediction and aligned target lengths do not match for residual evaluation."
        )
    pred_target_raw = baseline_aligned + pred_residual_raw
    reconstructed_mae = float(np.mean(np.abs(pred_target_raw - target_aligned)))
    print(f"Reconstructed target MAE (baseline + residual): {reconstructed_mae:.4f}")

    results.append(
        (
            pooling_mode if pooling_mode is not None else MODEL_TYPE,
            mean_test_loss,
            residual_mae,
            reconstructed_mae,
        )
    )

if len(results) > 1:
    print("\n=== Pooling Comparison ===")
    print(f"{'mode':<12} {'test_loss':>10} {'res_mae':>10} {'recon_mae':>10}")
    for mode, test_loss_value, residual_mae_value, recon_mae_value in results:
        print(
            f"{mode:<12} {test_loss_value:>10.4f} {residual_mae_value:>10.4f} "
            f"{recon_mae_value:>10.4f}"
        )

if results:
    print("\n=== Model vs Persistence Baseline (test-aligned) ===")
    print(f"{'mode':<12} {'model_mae':>10} {'baseline':>10} {'delta':>10}")
    for mode, _, _, recon_mae_value in results:
        delta = recon_mae_value - baseline_mae_test
        print(f"{mode:<12} {recon_mae_value:>10.4f} {baseline_mae_test:>10.4f} {delta:>+10.4f}")

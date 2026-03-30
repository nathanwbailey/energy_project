from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def unnormalize(values: np.ndarray, scaler: Optional[Any]) -> np.ndarray:
    """Convert normalised values back to original scale using a StandardScaler.

    Args:
        values: Normalised predictions or targets (1D array).
        scaler: Fitted StandardScaler used during normalisation, or None to skip.

    Returns:
        Values in original (raw) units.
    """
    if scaler is None:
        return values
    # inverse_transform expects 2D, so reshape and squeeze back
    return scaler.inverse_transform(values.reshape(-1, 1)).squeeze()


def train(  # noqa: C901
    model: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    valloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    path_to_model: Path,
    target_scaler: Optional[Any] = None,
    val_baseline_raw: Optional[np.ndarray] = None,
    val_target_raw: Optional[np.ndarray] = None,
    scheduler: Optional[Any] = None,
    grad_clip: Optional[float] = None,
    patience: int = 10,
) -> None:
    """Train the model with early stopping to prevent overfitting.

    Args:
        target_scaler: Fitted scaler for the target variable (for unnormalisation).
        val_baseline_raw: Optional baseline values aligned to validation targets.
        val_target_raw: Optional raw validation targets aligned to validation targets.
        patience: Number of epochs with no improvement to wait before stopping.
    """
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = []
        for batch in trainloader:
            optimizer.zero_grad()
            x = batch[0].to(device)
            y = batch[1].to(device)
            pred = model(x).squeeze()
            loss = loss_fn(pred, y)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            train_loss.append(loss.item())

        # Validation pass: collect predictions and targets for raw MAE
        model.eval()
        val_loss = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in valloader:
                x = batch[0].to(device)
                y = batch[1].to(device)
                pred = model(x).squeeze()
                loss = loss_fn(pred, y)
                val_loss.append(loss.item())
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        # Unnormalise and compute raw MAE
        preds_raw = unnormalize(np.concatenate(all_preds), target_scaler)
        targets_raw = unnormalize(np.concatenate(all_targets), target_scaler)
        raw_mae = np.mean(np.abs(preds_raw - targets_raw))

        reconstructed_mae = None
        if val_baseline_raw is not None and val_target_raw is not None:
            if len(val_baseline_raw) != len(preds_raw) or len(val_target_raw) != len(preds_raw):
                raise ValueError("Validation baseline/target arrays must match prediction length.")
            pred_target_raw = val_baseline_raw + preds_raw
            reconstructed_mae = np.mean(np.abs(pred_target_raw - val_target_raw))

        mean_train_loss = np.mean(train_loss)
        mean_val_loss = np.mean(val_loss)

        if scheduler is not None:
            scheduler.step(mean_val_loss)

        metric_msg = (
            f"Epoch: {epoch}, Train Loss: {mean_train_loss:.4f}, "
            f"Valid Loss: {mean_val_loss:.4f}, Raw MAE: {raw_mae:.4f}"
        )
        if reconstructed_mae is not None:
            metric_msg += f", Reconstructed target MAE: {reconstructed_mae:.4f}"
        print(metric_msg)

        # Early stopping
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            patience_counter = 0
            torch.save(model, path_to_model)
            print(f"  → Saving best model (val_loss: {mean_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs (patience={patience})")
                break


def test(
    model: torch.nn.Module,
    testloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    target_scaler: Optional[Any] = None,
) -> tuple[float, float]:
    """Evaluate the model on the test set.

    Args:
        target_scaler: Fitted scaler for the target variable (for unnormalisation).
    """
    model.eval()
    test_loss = []
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in testloader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            pred = model(x).squeeze()
            loss = loss_fn(pred, y)
            test_loss.append(loss.item())
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    # Unnormalise predictions and targets
    preds_raw = unnormalize(np.concatenate(all_preds), target_scaler)
    targets_raw = unnormalize(np.concatenate(all_targets), target_scaler)
    raw_mae = np.mean(np.abs(preds_raw - targets_raw))

    mean_test_loss = float(np.mean(test_loss))
    print(f"Test Loss: {mean_test_loss:.4f}, Raw MAE: {raw_mae:.4f}")
    return mean_test_loss, float(raw_mae)

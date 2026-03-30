from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray


class CSVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        features: NDArray[Any],
        labels: NDArray[Any],
        window: int,
        horizon: int,
        normalize: bool = True,
        feature_means: NDArray[Any] | None = None,
        feature_stds: NDArray[Any] | None = None,
    ) -> None:
        super().__init__()
        self.features = features
        self.labels = labels
        self.window = window
        self.horizon = horizon
        # Normalize
        if normalize and feature_means is not None and feature_stds is not None:
            self.feature_means = feature_means
            self.feature_stds = feature_stds
            self.features = (features - self.feature_means) / self.feature_stds
        elif normalize:
            self.feature_means = features.mean(axis=0)
            self.feature_stds = features.std(axis=0) + 1e-8
            self.features = (features - self.feature_means) / self.feature_stds
        else:
            self.features = features

    def __len__(self) -> int:
        return self.features.shape[0] - self.window - self.horizon + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        context_window = torch.from_numpy(self.features[idx : idx + self.window])
        target_value = torch.from_numpy(np.array(self.labels[idx + self.window + self.horizon - 1]))
        return (context_window, target_value)

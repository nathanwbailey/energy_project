import torch


# Transformer Encoder for time-series regression
class TransformerEncoderModel(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        pooling: str = "last",
    ):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_size, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = pooling
        self.attn_pool = torch.nn.Linear(d_model, 1)
        self.fc = torch.nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        if self.pooling == "last":
            x = x[:, -1, :]
        elif self.pooling == "mean":
            x = x.mean(dim=1)
        elif self.pooling == "attention":
            weights = torch.softmax(self.attn_pool(x), dim=1)
            x = torch.sum(x * weights, dim=1)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling}")
        return self.fc(x)  # (batch, 1)


class CSVModel(torch.nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, input_size: int) -> None:
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
        )  # Increased dropout from 0.1
        self.dense_layer = torch.nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (_, _) = self.lstm(x)
        out = out[:, -1, :]
        pred = self.dense_layer(out)
        return pred


class CNN1DModel(torch.nn.Module):
    """1D CNN for time-series regression (e.g., carbon intensity forecasting).

    Expects input shape: (batch, seq_len, input_size).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        # Build a stack of 1D conv layers with increasing channels
        layers = []
        in_channels = input_size
        out_channels = hidden_size

        for i in range(num_layers):
            layers.append(
                torch.nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,  # same-ish padding
                )
            )
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(out_channels))
            in_channels = out_channels
            out_channels = min(out_channels * 2, 256)  # cap channel growth

        self.conv_layers = torch.nn.Sequential(*layers)

        # Global average pooling collapses the time dimension
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)

        # Final linear layer produces a single output value
        self.fc = torch.nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size) -> transpose to (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)  # (batch, channels, seq_len)
        x = self.global_pool(x)  # (batch, channels, 1)
        x = x.squeeze(-1)  # (batch, channels)
        return self.fc(x)  # (batch, 1)

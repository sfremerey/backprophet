import datetime
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import backprophet_utils as bpu


TENSORBOARD = False  # Use tensorboard for logging, but you need to start it manually in the background
RENDER_PLOTS = False


# This class is written by ChatGPT 5
class CNNModel(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128, k: int = 3, num_classes: int = 1, pdrop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=k, padding=k//2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=k, padding=k//2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(pdrop),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.net(x)

def main():
    device = bpu.set_torch_device()
    end_date = pd.Timestamp.today()
    end_date = pd.to_datetime(end_date).date()
    df = bpu.get_df(end_date)
    df_scaled = bpu.scale_df(df)
    input_dim, n_features, X_train, X_test, Y_train, Y_test = bpu.get_train_test_set(df_scaled)

    # Timestamp for Tensorboard
    now = datetime.datetime.now()
    date_time = now.strftime("%Y%m%d-%H%M%S")

    # Hyperparameters
    learning_rate = 0.001
    training_epochs = 300
    batch_size = 128
    hidden_size = 256

    model = CNNModel(n_features, hidden=hidden_size, k=3, num_classes=1, pdrop=0.1).to(device)
    model_name = "cnn"
    bpu.train_eval_model(model, model_name, device, df, X_train, X_test, Y_train, Y_test, learning_rate, training_epochs, batch_size, hidden_size, end_date, RENDER_PLOTS, TENSORBOARD)


if __name__ == "__main__":
    main()

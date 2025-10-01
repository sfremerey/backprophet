import datetime
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import backprophet_utils as bpu


TENSORBOARD = False  # Use tensorboard for logging, but you need to start it manually in the background
RENDER_PLOTS = False


# GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # self.gru2 = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.act = nn.LeakyReLU(0.01)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        # out, _ = self.gru2(out, h0)
        out = self.fc(out[:, -1, :])
        out = self.act(out)
        out = self.fc2(out)
        return out


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
    num_layers = 3

    model = GRUModel(n_features, hidden_size, num_layers, 1).to(device)
    model_name = "gru"
    bpu.train_eval_model(model, model_name, device, df, X_train, X_test, Y_train, Y_test, learning_rate, training_epochs, batch_size, hidden_size, end_date, RENDER_PLOTS, TENSORBOARD)


if __name__ == "__main__":
    main()

import datetime
import os
import backprophet_utils as bpu
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class AvgEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        for m in self.models: m.eval()

    @torch.no_grad()
    def forward(self, x):
        outs = [m(x) for m in self.models]
        return torch.stack(outs, dim=0).mean(dim=0)

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity="relu")
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        # out, _ = self.gru2(out, h0)
        out = self.fc(out[:, -1, :])
        out = nn.LeakyReLU()(out)
        out = self.fc2(out)
        return out

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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        # out, _ = self.gru2(out, h0)
        out = self.fc(out[:, -1, :])
        out = self.act(out)
        out = self.fc2(out)
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def main():
    # Set torch device
    num_cpus = os.cpu_count() or 1
    print(f"Torch will use {num_cpus} CPUs for computation...")
    torch.set_num_threads(num_cpus)
    # Enable hardware accelerator if available
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon
    if torch.cuda.is_available() or torch.version.hip is not None:  # nvidia or AMD
        device = torch.device("cuda")  # NVIDIA
    if "COLAB_TPU_ADDR" in os.environ:  # TPU available on Google Colab only
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()  # TPU (Tensor Processing Unit)
    print(f"Using device: {device}")

    # end_date = pd.Timestamp.today() - pd.DateOffset(days=1)  # Only use if you want to run the model again for e.g. yesterday
    end_date = pd.Timestamp.today()
    end_date = pd.to_datetime(end_date).date()

    df = pd.read_csv(f"data/{end_date}.csv")
    df.set_index("DATE", inplace=True)  # Set index of df to "DATE" column so that scaler doesn't scale date

    scaler = MinMaxScaler()  # Other possible scalers would be: StandardScaler, RobustScaler
    scaled_values = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)

    X, y = bpu.create_dataset_multivariate(df_scaled, target_col="META_CLOSE", look_back=60)
    print("X shape:", X.shape)  # (num_samples, 60, num_features)
    print("y shape:", y.shape)  # (num_samples,)
    # Shuffle=False important for time series data
    # Cf. e.g. https://stackoverflow.com/questions/74025273/is-train-test-splitshuffle-false-appropriate-for-time-series
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    look_back = X_train.shape[1]
    n_features = X_train.shape[2]
    input_dim = look_back * n_features

    # Reshape X to 2D tensor for MLP
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    # Regression targets need to be float and (same, 1)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).reshape(-1, 1)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).reshape(-1, 1)

    # Timestamp for Tensorboard
    now = datetime.datetime.now()
    date_time = now.strftime("%Y%m%d-%H%M%S")

    m1 = torch.load(f"models/{end_date}_rnn_4layers.pth", weights_only=False).to(device)
    m2 = torch.load(f"models/{end_date}_gru_3layers.pth", weights_only=False).to(device)
    m3 = torch.load(f"models/{end_date}_lstm_4layers.pth", weights_only=False).to(device)
    ensemble = AvgEnsemble([m1, m2, m3]).to(device)
    scaler_y = MinMaxScaler().fit(df[["META_CLOSE"]])

    print("\nATTENTION: The following output is no financial advice!!!")
    rnn_pred = bpu.predict_next_day(df_scaled, scaler_y, look_back, m1, "RNN", device)
    gru_pred = bpu.predict_next_day(df_scaled, scaler_y, look_back, m2, "GRU", device)
    lstm_pred = bpu.predict_next_day(df_scaled, scaler_y, look_back, m3, "LSTM", device)
    ensemble_pred = bpu.predict_next_day(df_scaled, scaler_y, look_back, ensemble, "Ensemble", device)

    # bpu.plot_preds_time_and_xy(
    #     df=df, target_col="META_CLOSE", look_back=look_back,
    #     X_train=X_train, Y_train=Y_train,
    #     X_test=X_test, Y_test=Y_test,
    #     model=ensemble, save_name=f"{end_date}_ensemble",
    #     scY=scaler_y, title_prefix="META"
    # )

    new_row = {"DATE": end_date, "RNN_PRED": rnn_pred, "GRU_PRED": gru_pred,
               "LSTM_PRED": lstm_pred, "ENSEMBLE_PRED": ensemble_pred}
    df = pd.DataFrame([new_row])
    file_path = "data/META_predictions.csv"
    print(f"Save data to {file_path}")
    if os.path.isfile(file_path):
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, mode="w", header=True, index=False)


if __name__ == "__main__":
    main()

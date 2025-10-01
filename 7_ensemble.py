import datetime
import os
import backprophet_utils as bpu
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler


RENDER_PLOTS = False


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
    device = bpu.set_torch_device()
    # end_date = pd.Timestamp.today() - pd.DateOffset(days=1)  # Only use if you want to run the model again for e.g. yesterday
    end_date = pd.Timestamp.today()
    end_date = pd.to_datetime(end_date).date()

    df = bpu.get_df(end_date)
    df_scaled = bpu.scale_df(df)
    input_dim, n_features, X_train, X_test, Y_train, Y_test = bpu.get_train_test_set(df_scaled)

    # Timestamp for Tensorboard
    now = datetime.datetime.now()
    date_time = now.strftime("%Y%m%d-%H%M%S")

    m1 = torch.load(f"models/{end_date}_rnn.pth", weights_only=False).to(device)
    m2 = torch.load(f"models/{end_date}_gru.pth", weights_only=False).to(device)
    m3 = torch.load(f"models/{end_date}_lstm.pth", weights_only=False).to(device)
    ensemble = AvgEnsemble([m1, m2, m3]).to(device)
    scaler_y = MinMaxScaler().fit(df[["META_CLOSE"]])

    print("\nATTENTION: The following output is no financial advice!!!")
    rnn_pred = bpu.predict_next_day(df_scaled, scaler_y, look_back, m1, "RNN", device)
    gru_pred = bpu.predict_next_day(df_scaled, scaler_y, look_back, m2, "GRU", device)
    lstm_pred = bpu.predict_next_day(df_scaled, scaler_y, look_back, m3, "LSTM", device)
    ensemble_pred = bpu.predict_next_day(df_scaled, scaler_y, look_back, ensemble, "Ensemble", device)

    if RENDER_PLOTS:
        bpu.plot_preds_time_and_xy(
            df=df, target_col="META_CLOSE", look_back=look_back,
            X_train=X_train, Y_train=Y_train,
            X_test=X_test, Y_test=Y_test,
            model=ensemble, save_name=f"{end_date}_ensemble",
            scY=scaler_y, title_prefix="META"
        )

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

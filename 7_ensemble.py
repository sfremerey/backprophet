import os
import torch
import pandas as pd
import backprophet_models as bpm
import backprophet_utils as bpu
from sklearn.preprocessing import MinMaxScaler


RENDER_PLOTS = False
LOOK_BACK_PERIOD = 60  # Number of days to look back for building "sliding window", cf. bpu.get_train_test_set()
MODEL_NAME = "ensemble"

def main():
    device = bpu.set_torch_device()
    # end_date = pd.Timestamp.today() - pd.DateOffset(days=1)  # Only use if you want to run the model again for e.g. yesterday
    end_date = pd.Timestamp.today()
    end_date = pd.to_datetime(end_date).date()

    backprophet_data = bpu.BackprophetData(end_date, MODEL_NAME, LOOK_BACK_PERIOD)

    m1 = torch.load(f"models/{end_date}_rnn.pth", weights_only=False).to(device)
    m2 = torch.load(f"models/{end_date}_gru.pth", weights_only=False).to(device)
    m3 = torch.load(f"models/{end_date}_lstm.pth", weights_only=False).to(device)
    ensemble_model = bpm.AvgEnsemble([m1, m2, m3]).to(device)
    scaler_y = MinMaxScaler().fit(backprophet_data.df[["META_CLOSE"]])

    print("\nATTENTION: The following output is no financial advice!!!")
    rnn_pred = bpu.predict_next_day(backprophet_data.df_scaled, scaler_y, backprophet_data.look_back_period, m1, "RNN", device)
    gru_pred = bpu.predict_next_day(backprophet_data.df_scaled, scaler_y, backprophet_data.look_back_period, m2, "GRU", device)
    lstm_pred = bpu.predict_next_day(backprophet_data.df_scaled, scaler_y, backprophet_data.look_back_period, m3, "LSTM", device)
    ensemble_pred = bpu.predict_next_day(backprophet_data.df_scaled, scaler_y, backprophet_data.look_back_period, ensemble_model, "Ensemble", device)

    if RENDER_PLOTS:
        bpu.plot_preds_time_and_xy(
            df=backprophet_data.df, target_col="META_CLOSE", look_back=backprophet_data.look_back_period,
            X_train=backprophet_data.X_train, Y_train=backprophet_data.Y_train,
            X_test=backprophet_data.X_test, Y_test=backprophet_data.Y_test,
            model=ensemble_model, save_name=f"{backprophet_data.end_date}_{backprophet_data.model_name}",
            scY=scaler_y, title_prefix="META"
        )

    new_row = {"DATE": backprophet_data.end_date, "RNN_PRED": rnn_pred, "GRU_PRED": gru_pred,
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

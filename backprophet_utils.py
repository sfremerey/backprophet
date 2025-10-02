import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.ticker import MaxNLocator, NullLocator, NullFormatter
from pandas.tseries.offsets import BDay
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  #, StandardScaler, RobustScaler
from tqdm import tqdm
np.random.seed(42)
torch.manual_seed(42)


# Inspired by https://medium.com/thedeephub/rnns-in-action-predicting-stock-prices-with-recurrent-neural-networks-9155a33c4c3b
# "Given the past 60 days, predict tomorrow’s target value."
def create_dataset_multivariate(df, target_col, look_back=60):
    X = []
    Y = []

    feature_cols = [c for c in df.columns if c != "DATE"]  # DATE anyway is index
    for i in range(len(df) - look_back):
        X.append(df[feature_cols].iloc[i:i+look_back].values)      # (look_back, n_feat)
        Y.append(df[target_col].iloc[i+look_back])               # value for next day
    return np.array(X), np.array(Y)


# Set torch device
def set_torch_device():
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
    return device


def scale_df(df):
    scaler = MinMaxScaler()  # Other possible scalers would be: StandardScaler, RobustScaler
    scaled_values = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)
    return df_scaled


def get_train_test_set(df_scaled, model_name, look_back_period):
    X, y = create_dataset_multivariate(df_scaled, target_col="META_CLOSE", look_back=look_back_period)
    print("X shape:", X.shape)  # (num_samples, look_back_period, num_features)
    print("y shape:", y.shape)  # (num_samples,)
    # Shuffle=False important for time series data
    # Cf. e.g. https://stackoverflow.com/questions/74025273/is-train-test-splitshuffle-false-appropriate-for-time-series
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    look_back = X_train.shape[1]
    n_features = X_train.shape[2]
    input_dim = look_back * n_features

    # Reshape X to 2D tensor in case of MLP model
    if model_name == "simplemlp":
        X_train = torch.tensor(X_train, dtype=torch.float32).reshape(len(X_train), input_dim)
        X_test = torch.tensor(X_test, dtype=torch.float32).reshape(len(X_test), input_dim)
    else:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)

    # Regression targets need to be float and (same, 1)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).reshape(-1, 1)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).reshape(-1, 1)
    return input_dim, n_features, X_train, X_test, Y_train, Y_test


class BackprophetData:
    def __init__(self, end_date, model_name, look_back_period):
        super(BackprophetData, self).__init__()
        # Initialize df
        self.end_date = end_date
        self.model_name = model_name
        self.look_back_period = look_back_period
        self.df = pd.read_csv(f"data/{end_date}.csv")
        self.df.set_index("DATE", inplace=True)  # Set index of df to "DATE" column so that scaler doesn't scale date
        # Dropping of columns is only for testing purposes
        # The more rows dropped, the worse the results on the train set, but the better the results on the test set
        # self.df.drop(columns=["FEARANDGREED", "^SPX_CLOSE", "^SPX_VOLUME", "^DJI_CLOSE", "^DJI_VOLUME", "GPRD", "META_VOLUME"], inplace=True)
        self.df_scaled = scale_df(self.df)
        (self.input_dim, self.n_features, self.X_train, self.X_test, self.Y_train, self.Y_test) = get_train_test_set(self.df_scaled, self.model_name, self.look_back_period)


def train_eval_model(model, backprophet_data, device, learning_rate, training_epochs, batch_size, RENDER_PLOTS, TENSORBOARD):
    criterion = nn.MSELoss()
    mae_loss = nn.L1Loss(reduction="mean")  # MAE
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if TENSORBOARD:
        from torch.utils.tensorboard import SummaryWriter
        # Timestamp for Tensorboard
        now = datetime.datetime.now()
        date_time = now.strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(
            f"runs/{backprophet_data.model_name}_{date_time}"
        )

    # Add a small graph once (dummy input to avoid pushing full test set)
    try:
        writer.add_graph(model, torch.randn(1, backprophet_data.input_dim).to(device))
    except Exception as e:
        # Some environments may not support add_graph; safe to continue
        print(f"add_graph skipped: {e}")

    def evaluate(epoch: int):
        model.eval()
        with (torch.no_grad()):
            xtest = backprophet_data.X_test.to(device)
            ytest = backprophet_data.Y_test.to(device)
            y_pred_test = model(xtest)
            loss_test = criterion(y_pred_test, ytest).item()
            mae_test = mae_loss(y_pred_test, ytest).item()

            xtrain = backprophet_data.X_train.to(device)
            ytrain = backprophet_data.Y_train.to(device)
            y_pred_train = model(xtrain)
            loss_train = criterion(y_pred_train, ytrain).item()
            mae_train = mae_loss(y_pred_train, ytrain).item()
        if TENSORBOARD:
            writer.add_scalar("Loss/X_test", loss_test, epoch)
            writer.add_scalar("Loss/X_train", loss_train, epoch)
            writer.add_scalar("Metric/MAE/X_test", mae_test, epoch)
            writer.add_scalar("Metric/MAE/X_train", mae_train, epoch)

        print(f"[E{epoch:03d}] "
              f"loss_test≈{loss_test:.4f} | loss_train≈{loss_train:.4f}"
              f"mae_test≈{mae_test:.4f} | mae_train≈{mae_train:.4f}")

        model.train()

    # Training loop
    model.train()
    for epoch in tqdm(range(training_epochs)):
        # Shuffle once per epoch
        perm = torch.randperm(len(backprophet_data.X_train))
        Xs = backprophet_data.X_train[perm]
        Ys = backprophet_data.Y_train[perm]

        running_loss = 0.0
        n_batches = 0

        for i in range(0, len(Xs), batch_size):
            xb = Xs[i:i + batch_size].to(device, non_blocking=False)
            yb = Ys[i:i + batch_size].to(device, non_blocking=False)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(1, n_batches)
        if TENSORBOARD:
            writer.add_scalar("Loss/train_epoch_avg", avg_loss, epoch)

        # Evaluate every 2 epochs
        if (epoch + 1) % 2 == 0:
            evaluate(epoch + 1)

    # Final eval + close TB
    evaluate(training_epochs)
    scaler_y = MinMaxScaler().fit(backprophet_data.df[["META_CLOSE"]])
    if RENDER_PLOTS:
        plot_preds_time_and_xy(
            df=backprophet_data.df, target_col="META_CLOSE", look_back=backprophet_data.look_back_period,
            X_train=backprophet_data.X_train, Y_train=backprophet_data.Y_train,
            X_test=backprophet_data.X_test, Y_test=backprophet_data.Y_test,
            model=model, save_name=f"{backprophet_data.end_date}_{backprophet_data.model_name}",
            scY=scaler_y, title_prefix="META"
        )
    if TENSORBOARD:
        writer.close()

    # Save model
    Path("models").mkdir(parents=True, exist_ok=True)
    torch.save(model, f"models/{backprophet_data.end_date}_{backprophet_data.model_name}.pth")


# Predict next business day's close from the last available data row
def predict_next_day(df_scaled, scaler_y, look_back, model, model_name, device):
    model.eval()
    with torch.no_grad():  # Disables gradient calculation, hence dropout is disabled
        # Build the latest window (same scaling as during training)
        feature_cols = list(df_scaled.columns)  # DATE is index already
        last_window_scaled = df_scaled[feature_cols].iloc[-look_back:].values  # (look_back, n_features)
        x_latest = torch.tensor(last_window_scaled, dtype=torch.float32).unsqueeze(0).to(device)  # (1, look_back, n_features)

        # run model, get on CPU and in numpy form, ravel() flattens the array to 1D, prediction in scaled space
        yhat_scaled = model(x_latest).cpu().numpy().ravel()[0]

    # Inverse-transform to original price scale
    yhat = scaler_y.inverse_transform(np.array([[yhat_scaled]])).ravel()[0]

    # Compute next business day
    last_date = pd.to_datetime(df_scaled.index[-1]).date()
    next_date = (pd.Timestamp(last_date) + BDay(1)).date()

    print(f"Predicted META close for {model_name} on {next_date}: {yhat:.2f} USD")
    return yhat


# This function is mainly written by ChatGPT 5
def plot_preds_time_and_xy(df, target_col, look_back,
                           X_train, Y_train, X_test, Y_test,
                           model, save_name, scY=None, title_prefix="META"):
    """
    df: original DataFrame with DATE as index (chronologically sorted)
    target_col: e.g. "META_CLOSE"
    look_back: int, your window size
    X_train/X_test: tensors of shape (N, input_dim)
    Y_train/Y_test: tensors of shape (N, 1)  -- scaled if you used scY
    scY: sklearn scaler fitted on df[[target_col]]  (or None if not scaled)
    """
    model.eval()
    with torch.no_grad():
        yhat_train_s = model(X_train).detach().cpu().numpy()   # (Ntr,1)
        yhat_test_s  = model(X_test ).detach().cpu().numpy()   # (Nte,1)

    ytrain_s = Y_train.detach().cpu().numpy()                  # (Ntr,1)
    ytest_s  = Y_test.detach().cpu().numpy()                   # (Nte,1)

    # Inverse-transform to original units (USD) if scaler provided
    if scY is not None:
        yhat_train = scY.inverse_transform(yhat_train_s).ravel()
        yhat_test  = scY.inverse_transform(yhat_test_s ).ravel()
        ytrain     = scY.inverse_transform(ytrain_s    ).ravel()
        ytest      = scY.inverse_transform(ytest_s     ).ravel()
    else:
        yhat_train, yhat_test = yhat_train_s.ravel(), yhat_test_s.ravel()
        ytrain,     ytest     = ytrain_s.ravel(),     ytest_s.ravel()

    # Build date index aligned to targets (first target is at index look_back)
    y_dates = df.index[look_back:]                 # length == len(ytrain)+len(ytest)
    n_tr = len(ytrain)
    dates_train = y_dates[:n_tr]
    dates_test  = y_dates[n_tr:]

    # Metrics
    def mae(a,b): return float(np.mean(np.abs(a-b)))
    def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
    mae_tr, rmse_tr = mae(ytrain, yhat_train), rmse(ytrain, yhat_train)
    mae_te, rmse_te = mae(ytest,  yhat_test ), rmse(ytest,  yhat_test )

    print(f"[Train] MAE={mae_tr:.4f}, RMSE={rmse_tr:.4f}")
    print(f"[ Test ] MAE={mae_te:.4f}, RMSE={rmse_te:.4f}")

    # ---------- Time-series overlay ----------
    fig = plt.figure(figsize=(12,5))
    ax = plt.gca()
    plt.plot(dates_train, ytrain,     label="Train Actual")
    plt.plot(dates_train, yhat_train, label="Train Pred")
    plt.plot(dates_test,  ytest,      label="Test Actual")
    plt.plot(dates_test,  yhat_test,  label="Test Pred")
    plt.title(f"{title_prefix}: {target_col} — Actual vs Predicted")
    plt.xlabel("Date"); plt.ylabel(target_col)
    plt.legend(); plt.tight_layout()
    # ==== keep labels sparse ====
    if isinstance(df.index, pd.DatetimeIndex):
        y_dates = df.index[look_back:]
        span_days = (y_dates[-1] - y_dates[0]).days

        if span_days >= 3 * 365:
            major = mdates.YearLocator()  # yearly
            fmt = mdates.DateFormatter("%Y")
        elif span_days >= 365:
            major = mdates.MonthLocator(bymonth=[1, 4, 7, 10])  # quarterly
            fmt = mdates.DateFormatter("%Y-%m")
        elif span_days >= 90:
            major = mdates.MonthLocator(interval=1)  # monthly
            fmt = mdates.DateFormatter("%Y-%m")
        elif span_days >= 14:
            major = mdates.WeekdayLocator(byweekday=mdates.MO)  # weekly (Mondays)
            fmt = mdates.DateFormatter("%Y-%m-%d")
        else:
            step = max(1, span_days // 8)  # ≈8 ticks
            major = mdates.DayLocator(interval=step)
            fmt = mdates.DateFormatter("%m-%d")

        ax.xaxis.set_major_locator(major)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_locator(NullLocator())  # no minor ticks
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.figure.autofmt_xdate()  # neat rotation/alignment
    else:
        # Non-date x: cap ticks to ~8
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, prune="both"))

    plt.tight_layout()
    # plt.show()
    fig.savefig(f"plots/{save_name}_pricetrend.pdf")

    # ---------- Scatter x vs y (predicted vs actual) ----------
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    # Train
    ax = axes[0]
    ax.scatter(ytrain, yhat_train, s=8, alpha=0.6)
    mn, mx = np.min(ytrain), np.max(ytrain)
    ax.plot([mn, mx], [mn, mx])
    ax.set_title(f"Train: Pred vs Actual\nMAE={mae_tr:.3f}, RMSE={rmse_tr:.3f}")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    # Test
    ax = axes[1]
    ax.scatter(ytest, yhat_test, s=8, alpha=0.6)
    mn, mx = np.min(ytest), np.max(ytest)
    ax.plot([mn, mx], [mn, mx])
    ax.set_title(f"Test: Pred vs Actual\nMAE={mae_te:.3f}, RMSE={rmse_te:.3f}")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")

    plt.tight_layout()
    # plt.show()
    fig.savefig(f"plots/{save_name}_scatter.pdf")

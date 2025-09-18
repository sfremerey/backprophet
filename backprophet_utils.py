import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.tseries.offsets import BDay
from matplotlib.ticker import MaxNLocator, NullLocator, NullFormatter
np.random.seed(42)


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
    return None


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

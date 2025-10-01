import datetime
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import backprophet_utils as bpu
torch.manual_seed(42)
np.random.seed(42)
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


TENSORBOARD = False  # Use tensorboard for logging, but start manually in the background
RENDER_PLOTS = False
if TENSORBOARD:
    from torch.utils.tensorboard import SummaryWriter


def main():
    device = bpu.set_torch_device()
    # end_date = pd.Timestamp.today() - pd.DateOffset(days=1)  # Only use if you want to run the model again for e.g. yesterday
    end_date = pd.Timestamp.today()
    end_date = pd.to_datetime(end_date).date()
    df = bpu.get_df(end_date)
    df_scaled = bpu.scale_df(df)
    input_dim, n_features, X_train, X_test, Y_train, Y_test = bpu.get_train_test_set(df_scaled, mlp_model=True)

    # Timestamp for Tensorboard
    now = datetime.datetime.now()
    date_time = now.strftime("%Y%m%d-%H%M%S")

    # Hyperparameters
    learning_rate = 0.001
    dropout_rate = 0.1
    training_epochs = 300
    batch_size = 128
    hidden_size = 256

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_size, bias=True),
        nn.LeakyReLU(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(hidden_size, 1, bias=True),
    ).to(device)

    criterion = nn.MSELoss()
    mae_loss = nn.L1Loss(reduction="mean")  # MAE
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # For TensorBoard
    if TENSORBOARD:
        writer = SummaryWriter(
            f"runs/simplemlp_{date_time}"
        )

    # Add a small graph once (dummy input to avoid pushing full test set)
    try:
        writer.add_graph(model, torch.randn(1, input_dim).to(device))
    except Exception as e:
        # Some environments may not support add_graph; safe to continue
        print(f"add_graph skipped: {e}")

    def evaluate(epoch: int):
        model.eval()
        with (torch.no_grad()):
            xtest = X_test.to(device)
            ytest = Y_test.to(device)
            y_pred_test = model(xtest)
            loss_test = criterion(y_pred_test, ytest).item()
            mae_test = mae_loss(y_pred_test, ytest).item()

            xtrain = X_train.to(device)
            ytrain = Y_train.to(device)
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
        perm = torch.randperm(len(X_train))
        Xs = X_train[perm]
        Ys = Y_train[perm]

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
    scaler_y = MinMaxScaler().fit(df[["META_CLOSE"]])
    if RENDER_PLOTS:
        bpu.plot_preds_time_and_xy(
            df=df, target_col="META_CLOSE", look_back=look_back,
            X_train=X_train, Y_train=Y_train,
            X_test=X_test, Y_test=Y_test,
            model=model, save_name=f"{end_date}_simple_mlp",
            scY=scaler_y, title_prefix="META"
        )
    if TENSORBOARD:
        writer.close()

    # Save model
    Path("models").mkdir(parents=True, exist_ok=True)
    torch.save(model, f"models/{end_date}_simple_mlp.pth")


if __name__ == "__main__":
    main()

# Some results for comparison - for the "interested reader" ;-)
# For file 2025-09-16.csv

# data including everything
# [Train] MAE=4.8579, RMSE=6.4967
# [ Test ] MAE=35.9516, RMSE=45.8093

# data not including FEARANDGREED, ^SPX_CLOSE, ^SPX_VOLUME, ^DJI_CLOSE, ^DJI_VOLUME, GPRD, META_VOLUME
# [Train] MAE=6.3206, RMSE=9.1589
# [ Test ] MAE=13.1977, RMSE=18.2137

# data not including FEARANDGREED, ^SPX_CLOSE, ^SPX_VOLUME, ^DJI_CLOSE, ^DJI_VOLUME, GPRD
# [Train] MAE=5.1914, RMSE=7.3961
# [ Test ] MAE=15.9410, RMSE=21.2048

# data not including FEARANDGREED, ^SPX_CLOSE, ^SPX_VOLUME, ^DJI_CLOSE, ^DJI_VOLUME
# [Train] MAE=4.9931, RMSE=6.7476
# [ Test ] MAE=17.5549, RMSE=22.3178

# data not including FEARANDGREED, ^SPX_CLOSE, ^SPX_VOLUME
# [Train] MAE=4.7165, RMSE=6.2816
# [ Test ] MAE=29.9856, RMSE=39.5723

# data not including FEARANDGREED
# [Train] MAE=10.3572, RMSE=12.5853
# [ Test ] MAE=31.2990, RMSE=38.9286

# data not including FEARANDGREED, GPRD
# [Train] MAE=5.1702, RMSE=7.1684
# [ Test ] MAE=37.7971, RMSE=49.5859

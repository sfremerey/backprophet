import datetime
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import backprophet_utils as bpu
torch.manual_seed(42)
np.random.seed(42)
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


TENSORBOARD = True  # Use tensorboard for logging, but start manually in the background


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

    end_date = pd.Timestamp.today()
    end_date = pd.to_datetime(end_date).date()
    df = pd.read_csv(f"data/{end_date}.csv")
    df.set_index("DATE", inplace=True)  # Set index of df to "DATE" column so that scaler doesn't scale date
    # Dropping of columns is only for testing purposes
    # The more rows dropped, the worse the results on the train set, but the better the results on the test set
    # df.drop(columns=["FEARANDGREED", "^SPX_CLOSE", "^SPX_VOLUME", "^DJI_CLOSE", "^DJI_VOLUME", "GPRD", "META_VOLUME"], inplace=True)

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
    X_train = torch.tensor(X_train, dtype=torch.float32).reshape(len(X_train), input_dim)
    X_test = torch.tensor(X_test, dtype=torch.float32).reshape(len(X_test), input_dim)

    # Regression targets need to be float and (same, 1)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).reshape(-1, 1)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).reshape(-1, 1)

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
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
from sklearn.model_selection import train_test_split
from tqdm import tqdm

TENSORBOARD = False  # Use tensorboard for logging, but start manually in the background
RENDER_PLOTS = False
if TENSORBOARD:
    from torch.utils.tensorboard import SummaryWriter


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

    # Hyperparameters
    learning_rate = 0.001
    training_epochs = 300
    batch_size = 128
    hidden_size = 256

    model = CNNModel(n_features, hidden=hidden_size, k=3, num_classes=1, pdrop=0.1).to(device)

    criterion = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss(reduction="mean")  # MAE
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # For TensorBoard
    if TENSORBOARD:
        writer = SummaryWriter(
            f"runs/cnn_{date_time}"
        )

    # Add a small graph once (dummy input to avoid pushing full test set)
    try:
        writer.add_graph(model, torch.randn(1, input_dim).to(device))
    except Exception as e:
        # Some environments may not support add_graph; safe to continue
        print(f"add_graph skipped: {e}")

    def evaluate(epoch: int):
        """Evaluate on the full test set (10k) and of train for speed; log to TB."""
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
            model=model, save_name=f"{end_date}_cnn",
            scY=scaler_y, title_prefix="META"
        )
    if TENSORBOARD:
        writer.close()

    # Save model
    Path("models").mkdir(parents=True, exist_ok=True)
    torch.save(model, f"models/{end_date}_cnn.pth")


if __name__ == "__main__":
    main()

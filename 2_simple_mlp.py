import datetime
import os
import torch
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# Inspired by https://medium.com/thedeephub/rnns-in-action-predicting-stock-prices-with-recurrent-neural-networks-9155a33c4c3b
def create_dataset_multivariate(df, target_col, look_back=60):
    X, Y = [], []
    feature_cols = [c for c in df.columns if c not in ["DATE", target_col]]

    for i in range(len(df) - look_back - 1):
        # Slice look_back rows for all feature columns
        features = df[feature_cols].iloc[i:(i + look_back)].values
        X.append(features)

        # Target is e.g. META_CLOSE at i+look_back+1
        Y.append(df[target_col].iloc[i + look_back + 1])

    X = np.array(X)  # Shape: (samples, look_back, n_features)
    Y = np.array(Y)  # Shape: (samples,)
    return X, Y

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

    X, y = create_dataset_multivariate(df, target_col="META_CLOSE", look_back=60)
    print("X shape:", X.shape)  # (num_samples, 60, num_features)
    print("y shape:", y.shape)  # (num_samples,)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    look_back = X_train.shape[1]
    n_features = X_train.shape[2]
    input_dim = look_back * n_features

    # Convert to torch Tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    Y_test = torch.tensor(Y_test, dtype=torch.long)

    learning_rate = 0.001
    dropout_rate = 0.1
    training_epochs = 200
    batch_size = 256
    hidden_size = 128

    # Time stamp for TB run
    now = datetime.datetime.now()
    date_time = end_date.strftime("%Y%m%d-%H%M%S")

    # Model1
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_size, bias=True),
        torch.nn.LeakyReLU(),
        torch.nn.Dropout(p=dropout_rate),
        torch.nn.Linear(hidden_size, 1, bias=True),
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # For TensorBoard
    writer = SummaryWriter(
        f"runs/simplelinear_{date_time}_m1-ep{training_epochs}-lr{learning_rate}-dr{dropout_rate}-bat{batch_size}-hid{hidden_size}"
    )

    # Add a small graph once (dummy input to avoid pushing full test set)
    try:
        writer.add_graph(model, torch.randn(1, input_dim).to(device))
    except Exception as e:
        # Some environments may not support add_graph; safe to continue
        print(f"add_graph skipped: {e}")

    global_step = 0

    def evaluate(epoch: int, log_hist: bool = False):
        """Evaluate on the full test set (10k) and of train for speed; log to TB."""
        model.eval()
        with torch.no_grad():
            # Test
            xtrain = X_test.to(device)
            ytrain = Y_test.to(device)
            logits_test = model(xtrain)
            loss_test = criterion(logits_test, ytrain).item()
            acc_test = (logits_test.argmax(1) == ytrain).float().mean().item()

            # Train
            xtrain = X_train.to(device)
            ytrain = Y_train.to(device)
            logits_train = model(xtrain)
            loss_train = criterion(logits_train, ytrain).item()
            acc_train = (logits_train.argmax(1) == ytrain).float().mean().item()
        if tensorboard:
            writer.add_scalar("Accuracy/X_test", acc_test, epoch)
            writer.add_scalar("Accuracy/X_train", acc_train, epoch)
            writer.add_scalar("Loss/X_test", loss_test, epoch)
            writer.add_scalar("Loss/X_train", loss_train, epoch)

            if log_hist:
                for name, param in model.named_parameters():
                    writer.add_histogram(f"params/{name}", param, epoch)

        print(f"[E{epoch:03d}] acc_test={acc_test:.4f} loss_test={loss_test:.4f} | "
              f"acc_train≈{acc_train:.4f} loss_train≈{loss_train:.4f}")

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
            global_step += 1

            # Lightweight per-step logging: only a scalar (kept cheap)
            if global_step % 50 == 0 and tensorboard:
                writer.add_scalar("Loss/train_step", loss.item(), global_step)

        avg_loss = running_loss / max(1, n_batches)
        if tensorboard:
            writer.add_scalar("Loss/train_epoch_avg", avg_loss, epoch)

        # print(f"Epoch {epoch + 1:03d}/{training_epochs} | avg_loss={avg_loss:.4f} | {epoch_time:.2f}s")

        # Evaluate every 2 epochs; add histograms every 10 epochs
        if (epoch + 1) % 2 == 0:
            evaluate(epoch + 1, log_hist=((epoch + 1) % 10 == 0))

    # Final eval + close TB
    evaluate(training_epochs, log_hist=True)
    if tensorboard:
        writer.close()

    # Save model
    # Path("models").mkdir(parents=True, exist_ok=True)
    # torch.save(model,
    #            f"models/gold_m1-ep{training_epochs}-lr{learning_rate}-dr{dropout_rate}-bat{batch_size}-hid{hidden_size}.pth")


if __name__ == "__main__":
    main()
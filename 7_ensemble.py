import os
import torch
import torch.nn as nn


class AvgEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        for m in self.models: m.eval()

    @torch.no_grad()
    def forward(self, x):
        outs = [m(x) for m in self.models]
        return torch.stack(outs, dim=0).mean(dim=0)

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.act = nn.LeakyReLU(0.01)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        out, _ = self.gru2(out, h0)
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

    m1 = torch.load("models/simple_mlp.pth", weights_only=False).to(device)
    m2 = torch.load("models/gru_3layers.pth", weights_only=False).to(device)
    m3 = torch.load("models/lstm_4layers.pth", weights_only=False).to(device)

    ensemble = AvgEnsemble([m1, m2, m3]).to(device)
    # todo build xb
    y_ensemble = ensemble(xb)


if __name__ == "__main__":
    main()

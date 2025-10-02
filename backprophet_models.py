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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
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

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity="relu")
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        # out, _ = self.gru2(out, h0)
        out = self.fc(out[:, -1, :])
        out = nn.LeakyReLU()(out)
        out = self.fc2(out)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModule(nn.Module):
    def __init__(self, n_features: int, hidden_size=None):
        super().__init__()
        self.n_features = n_features
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, num_layers=2
        )
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, X, **kwargs):
        output, hn = self.lstm(X)  # lstm with input, hidden, and internal state
        return self.fc(output[-1, :])

class NewLstmModule(nn.Module):

    def __init__(self, n_features, hidden_size=64):
        super().__init__()
        self.n_features=n_features
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=1, bidirectional=False)
        self.activation = get_activation("linear")
        self.fc = nn.Linear(in_features=hidden_size,out_features=1) #Dense

    def forward(self, X, **kwargs):
        output, (hn, cn) = self.lstm(X)  # lstm with input, hidden, and internal state
        x = self.fc(output[-1, :])
        x = self.activation(x)
        return x


def get_activation(activation_name):
    """Returns a callable activation function given its name."""
    name = activation_name.lower()
    if name == "relu":
        return F.relu
    elif name == "tanh":
        return torch.tanh
    elif name == "sigmoid":
        return torch.sigmoid
    elif name == "linear":
        return lambda x: x
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")

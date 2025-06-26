import torch


class MlpClassifier(torch.nn.Module):
    def __init__(self, feature_size=256, n_labels=4, hidden_size=128, n_layers=3):
        super().__init__()
        self.n_layers = n_layers
        self.hidden = torch.nn.ModuleList(
            [torch.nn.Linear(feature_size, hidden_size), torch.nn.ReLU()],
        )
        for _ in range(self.n_layers - 2):
            self.hidden.append(torch.nn.Linear(hidden_size, hidden_size))
            self.hidden.append(torch.nn.ReLU())
        self.output = torch.nn.Linear(hidden_size, n_labels)

    def forward(self, x):
        for layer in self.hidden:
            x = layer(x)
        x = self.output(x)
        x = torch.sigmoid(x).squeeze(-1)
        return x
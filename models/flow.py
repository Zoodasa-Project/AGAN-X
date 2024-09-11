import torch
import torch.nn as nn

class ContinuousNormalizingFlow(nn.Module):
    def __init__(self, dim, hidden_dim=64, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            CNFLayer(dim, hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x, t):
        for layer in self.layers:
            x = layer(x, t)
        return x

class CNFLayer(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x, t):
        input = torch.cat([x, t.unsqueeze(-1)], dim=-1)
        return x + self.net(input)

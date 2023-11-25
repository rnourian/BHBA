import torch.nn as nn


class ResidualBlockMain(nn.Module):
    def __init__(self, dim):
        super(ResidualBlockMain, self).__init__()
        self.fc = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(0.5)

        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        y = self.norm(x)
        y = self.relu(y)
        y = self.drop(y)
        y = self.fc(y)
        y = self.norm(y)
        y = self.relu(y)
        y = self.fc(y)

        out = y + residual
        return out

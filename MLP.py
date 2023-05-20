import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(in_channels)
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.fc4 = nn.Linear(hidden_channels, hidden_channels)
        self.bn4 = nn.BatchNorm1d(hidden_channels)
        self.fc5 = nn.Linear(hidden_channels, hidden_channels)
        self.bn5 = nn.BatchNorm1d(hidden_channels)
        self.fc6 = nn.Linear(hidden_channels, out_channels)
        self.act = nn.ReLU()

    def forward(self, data, selection=None):
        x = data.pos
        if data.norm is not None:
            x = torch.cat([x, data.norm], dim=1)
        if data.dha is not None:
            x = torch.cat([x, data.dha], dim=1)
        if data.x is not None:
            x = torch.cat([x, data.x], dim=1)
        if selection is not None:
            x = (x * selection).float()
        x = self.bn0(x)
        x = self.bn1(self.act(self.fc1(x)))
        x = self.bn2(self.act(self.fc2(x)))
        x = self.bn3(self.act(self.fc3(x)))
        x = self.bn4(self.act(self.fc4(x)))
        x = gnn.global_mean_pool(x, data.batch)
        # x = torch.cat([x, data.confound], dim=1)
        x = self.bn5(self.act(self.fc5(x)))
        x = self.fc6(x)
        return x


class actor_MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(in_channels)
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.fc4 = nn.Linear(hidden_channels, hidden_channels)
        self.bn4 = nn.BatchNorm1d(hidden_channels)
        self.fc5 = nn.Linear(hidden_channels, hidden_channels)
        self.bn5 = nn.BatchNorm1d(hidden_channels)
        self.fc6 = nn.Linear(hidden_channels, out_channels)
        self.act = nn.ReLU()

    def forward(self, data):
        x = data.pos
        if data.norm is not None:
            x = torch.cat([x, data.norm], dim=1)
        if data.dha is not None:
            x = torch.cat([x, data.dha], dim=1)
        if data.x is not None:
            x = torch.cat([x, data.x], dim=1)
        x = self.bn0(x)
        x = self.bn1(self.act(self.fc1(x)))
        x = self.bn2(self.act(self.fc2(x)))
        x = self.bn3(self.act(self.fc3(x)))
        x = self.bn4(self.act(self.fc4(x)))
        x = self.bn5(self.act(self.fc5(x)))
        x = self.fc6(x)
        return torch.sigmoid(x)

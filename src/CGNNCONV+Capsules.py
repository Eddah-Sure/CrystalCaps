import math
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter, to_dense_batch

class CartesianAwareCrystalGNNCapsNet(nn.Module):
    def __init__(self, node_features, edge_features, hidden_channels,
                 num_conv_layers, primary_caps, primary_dim,
                 secondary_caps, secondary_dim, dropout_rate=0.0):
        super().__init__()

        # Node embedding with position
        self.node_embedding = nn.Sequential(
            nn.Linear(node_features + 3, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )

        # Edge embedding
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU()
        )

        # Graph convolution layers with residual connections
        self.convs = nn.ModuleList([
            CartesianAwareCGCNNConv(hidden_channels)
            for _ in range(num_conv_layers)
        ])

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_conv_layers)
        ])


        self.primary_caps = PrimaryCapsuleLayer(
            in_features=hidden_channels,
            out_caps=primary_caps,
            caps_dim=primary_dim
        )
        self.secondary_caps = SecondaryCapsuleLayer(
            in_dim=primary_dim,
            out_caps=secondary_caps,
            out_dim=secondary_dim,
            routing_iterations=3
        )


        self.attn_net = nn.Sequential(
            nn.Linear(secondary_dim, 1)
        )

        self.predictor = nn.Sequential(
            nn.Linear(secondary_dim, hidden_channels//2),
            nn.BatchNorm1d(hidden_channels//2),
            nn.ReLU(),
            nn.Linear(hidden_channels//2, hidden_channels//4),
            nn.BatchNorm1d(hidden_channels//4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels//4, 1)
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.l2_reg = 1e-7

    def forward(self, data):
        x, edge_index, edge_attr, pos = data.x, data.edge_index, data.edge_attr, data.pos
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = torch.cat([x, pos], dim=1)
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)

        for i, conv in enumerate(self.convs):
            x_res = x
            x = conv(x, edge_index, edge_attr, pos)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            if i > 0:
                x = x + x_res
        primary_caps = self.primary_caps(x)
        secondary_caps = self.secondary_caps(primary_caps, batch)
        attn_weights = F.softmax(self.attn_net(secondary_caps), dim=1)
        weighted_caps = (attn_weights * secondary_caps).sum(dim=1)
        return self.predictor(weighted_caps)

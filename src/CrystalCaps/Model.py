"""
Main E(3)-equivariant crystal graph network with capsule networks.
This contains the complete model architecture that combines
E(3)-equivariant convolutions with capsule networks for crystal
property prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter, to_dense_batch
import torch_geometric.transforms as T
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.nn import Gate, BatchNorm
from e3nn.o3 import Irreps, spherical_harmonics
from GNNBase import EquivariantGNN, LayerNormalization
from CapsuleNetwork import PrimaryCapsuleLayer, SecondaryCapsuleLayer


class CGNe3(nn.Module):
    def __init__(self, node_features, edge_features, hidden_channels, num_conv_layers, primary_caps, primary_dim, secondary_caps, secondary_dim, dropout_rate):
        super().__init__()
        #Node embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(node_features, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU(),
        )
        #Edge embedding
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_features, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU()
        )
        self.init_vector_features = nn.Linear(hidden_channels, hidden_channels)
        self.convs = nn.ModuleList([
            EquivariantGNN(hidden_channels)
            for _ in range(num_conv_layers)
        ])
        self.layer_norms = nn.ModuleList([
            LayerNormalization(hidden_channels)
            for _ in range(num_conv_layers)
        ])
        #Capsules
        self.primary_caps = PrimaryCapsuleLayer(
            scalar_features=hidden_channels,
            vector_features=hidden_channels,
            out_caps=primary_caps,
            caps_dim=primary_dim
        )
        self.secondary_caps = SecondaryCapsuleLayer(
            in_dim=primary_dim,
            out_caps=secondary_caps,
            out_dim=secondary_dim,
            routing_iterations=2
        )
        #Attention
        self.attn_net = nn.Sequential(
            nn.Linear(secondary_dim // 2, 1)
        )
        self.predictor = nn.Sequential(
            nn.Linear(secondary_dim, hidden_channels//2),
            nn.BatchNorm1d(hidden_channels//2),
            nn.SiLU(),
            nn.Linear(hidden_channels//2, hidden_channels//4),
            nn.BatchNorm1d(hidden_channels//4),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels//4, 1)
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.scalar_act = nn.SiLU()
        self.gate_act = nn.Sigmoid()

    def forward(self, data):
        x, edge_index, edge_attr, pos = data.x, data.edge_index, data.edge_attr, data.pos
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x_scalar = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        vector_weights = self.init_vector_features(x_scalar)
        x_vector = torch.zeros(x_scalar.size(0), vector_weights.size(1), 3, device=x_scalar.device)

        for i in range(x_scalar.size(0)):
            x_vector[i] = vector_weights[i].unsqueeze(-1) * pos[i].unsqueeze(0)
        #Convolutions
        for i, (conv, norm) in enumerate(zip(self.convs, self.layer_norms)):
            x_scalar_res = x_scalar
            x_vector_res = x_vector
            x_scalar, x_vector = conv(x_scalar, x_vector, edge_index, edge_attr, pos)
            x_scalar, x_vector = norm(x_scalar, x_vector)
            x_scalar_act = self.scalar_act(x_scalar)
            gates = self.gate_act(x_scalar)
            x_vector = x_vector * gates.unsqueeze(-1)
            if i > 0:
                x_scalar = x_scalar_act + x_scalar_res
                x_vector = x_vector + x_vector_res
            else:
                x_scalar = x_scalar_act

        primary_caps, primary_vectors = self.primary_caps(x_scalar, x_vector)
        if primary_vectors is not None:
            if primary_vectors.dim() == 3:  
                primary_caps_size = primary_caps.size(1)
                primary_vectors = primary_vectors.unsqueeze(1).expand(-1, primary_caps_size, -1, -1)
        secondary_caps, _ = self.secondary_caps(primary_caps, primary_vectors, batch)
        scalar_part = secondary_caps[:, :, :secondary_caps.size(2)//2]
        attn_weights = F.softmax(self.attn_net(scalar_part), dim=1)
        #Weighted capsules
        weighted_caps = (attn_weights * secondary_caps).sum(dim=1)

        return self.predictor(weighted_caps)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from e3nn import o3
from e3nn.o3 import Irreps, spherical_harmonics, FullyConnectedTensorProduct

class EquivariantGNN(nn.Module):
    def __init__(self, channels, num_rbf=16, cutoff=10.0, lmax=1):
        super().__init__()
        self.channels = channels
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.lmax = lmax

        self.rbf = RadialBasisLayer(num_rbf, cutoff)

        self.scalar_mlp = nn.Sequential(
            nn.Linear(channels * 2 + num_rbf, channels),
            nn.SiLU(),
            nn.Linear(channels, channels)
        )
        irreps_in1 = o3.Irreps(f"{channels}x0e + {channels}x1e")
        irreps_in2 = o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])
        irreps_out = o3.Irreps(f"{channels}x0e + {channels}x1e")
        self.tp = FullyConnectedTensorProduct(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=irreps_out,
            internal_weights=True
        )
        self.scalar_act = nn.SiLU()
        self.gate_act = nn.Sigmoid()

    def forward(self, x_scalar, x_vector, edge_index, edge_attr, pos):
        row, col = edge_index
        dist = torch.norm(pos[row] - pos[col], dim=-1, keepdim=True)
        edge_vec = pos[row] - pos[col]
        #RBF
        rbf_output = self.rbf(dist)
        scalar_message_input = torch.cat([
            x_scalar[row],
            x_scalar[col],
            rbf_output
        ], dim=-1)
        scalar_message = self.scalar_mlp(scalar_message_input)
        #Spherical harmonics
        edge_sh = spherical_harmonics(
            list(range(self.lmax + 1)),
            edge_vec / (dist + 1e-8),
            normalize=True
        )
        src_features = torch.cat([x_scalar[row], x_vector[row].reshape(x_vector[row].shape[0], -1)], dim=-1)

        message = self.tp(src_features, edge_sh)
        scalar_out = scatter(scalar_message, col, dim=0, dim_size=x_scalar.size(0), reduce='add')
        vector_out = scatter(message[:, self.channels:].view(-1, self.channels, 3), col, dim=0, dim_size=x_vector.size(0), reduce='add')
        #Activation
        scalar_out = self.scalar_act(scalar_out)
        #Gating 
        gates = self.gate_act(scalar_out)
        gated_vectors = vector_out * gates.unsqueeze(-1)
        x_scalar_new = x_scalar + scalar_out
        x_vector_new = x_vector + gated_vectors
        return x_scalar_new, x_vector_new

class RadialBasisLayer(nn.Module):
    def __init__(self, num_rbf, cutoff):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.centers = nn.Parameter(torch.linspace(0, cutoff, num_rbf), requires_grad=False)
        self.widths = nn.Parameter((cutoff / num_rbf) * torch.ones(num_rbf), requires_grad=False)

    def forward(self, dist):
        dist = dist.clamp(max=self.cutoff)
        dist_expanded = dist.expand(-1, self.num_rbf)
        rbf = torch.exp(-((dist_expanded - self.centers.view(1, -1)) / self.widths.view(1, -1))**2)
        envelope = self._envelope(dist)
        envelope_expanded = envelope.expand_as(rbf)
        return rbf * envelope_expanded

    def _envelope(self, dist):
        return 1 - (dist / self.cutoff)**2
class LayerNormalization(nn.Module):
    def __init__(self, channels, scalar_only=False):
        super().__init__()
        self.channels = channels
        self.scalar_only = scalar_only
        self.scalar_scale = nn.Parameter(torch.ones(channels))
        self.scalar_bias = nn.Parameter(torch.zeros(channels))

        if not scalar_only:
            self.vector_scale = nn.Parameter(torch.ones(channels))

    def forward(self, x_scalar, x_vector=None):
        mean = x_scalar.mean(dim=1, keepdim=True)
        var = x_scalar.var(dim=1, keepdim=True, unbiased=False)
        x_scalar = (x_scalar - mean) / (var + 1e-5).sqrt()
        x_scalar = x_scalar * self.scalar_scale.view(1, -1) + self.scalar_bias.view(1, -1)

        if x_vector is not None and not self.scalar_only:
            vec_norm = torch.norm(x_vector, dim=2, keepdim=True)
            vec_mean = vec_norm.mean(dim=1, keepdim=True)
            vec_var = vec_norm.var(dim=1, keepdim=True, unbiased=False)
            scale = (vec_norm / (vec_var + 1e-5).sqrt()) * self.vector_scale.view(1, -1, 1)
            x_vector = x_vector * scale

        return x_scalar, x_vector

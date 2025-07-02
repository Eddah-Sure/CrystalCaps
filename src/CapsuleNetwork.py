import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct

class PrimaryCapsuleLayer(nn.Module):
    def __init__(self, scalar_features, vector_features, out_caps, caps_dim):
        super().__init__()
        self.out_caps = out_caps
        self.caps_dim = caps_dim
        #Scalar feature projection
        self.scalar_projection = nn.Linear(scalar_features, out_caps * (caps_dim // 2))
        self.vector_weights = nn.Linear(scalar_features, out_caps)
        irreps_in1 = o3.Irreps(f"{vector_features}x1e")  
        irreps_in2 = o3.Irreps("0e")  
        irreps_out = o3.Irreps(f"{caps_dim//2}x0e")  
        self.vector_tp = FullyConnectedTensorProduct(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=irreps_out,
            internal_weights=True
        )

    def forward(self, x_scalar, x_vector):
        batch_size = x_scalar.size(0)
        scalar_out = self.scalar_projection(x_scalar)
        scalar_out = scalar_out.view(batch_size, self.out_caps, self.caps_dim // 2)
        vector_weights = self.vector_weights(x_scalar)
        vector_weights = vector_weights.view(batch_size, self.out_caps, 1)
        vector_caps = []

        for i in range(self.out_caps):
            weighted_vectors = vector_weights[:, i:i+1, :] * x_vector
            weighted_vectors_flat = weighted_vectors.reshape(batch_size, -1)
            invariants = self.vector_tp(weighted_vectors_flat, torch.ones(batch_size, 1, device=x_scalar.device))
            invariants = invariants.view(batch_size, self.caps_dim // 2)
            vector_caps.append(invariants)
        vector_caps = torch.stack(vector_caps, dim=1)
        #Concatenate scalars, vectors
        capsules = torch.cat([scalar_out, vector_caps], dim=2)

        return capsules, x_vector

class SecondaryCapsuleLayer(nn.Module):
    def __init__(self, in_dim, out_caps, out_dim, routing_iterations):
        super().__init__()
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.routing_iterations = routing_iterations
        self.scalar_dim = out_dim // 2
        self.vector_dim = out_dim // 2
        self.in_dim = in_dim

        self.W_scalar = nn.Parameter(torch.randn(out_caps, in_dim // 2, self.scalar_dim))
        self.W_vector = nn.Parameter(torch.randn(out_caps, in_dim // 2, self.vector_dim))
        self.bias = nn.Parameter(torch.zeros(out_caps, out_dim))
    #Squashing
    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / (torch.sqrt(squared_norm) + 1e-8)

    def forward(self, x, x_vectors, batch):
        batch_size = batch.max().item() + 1
        secondary_capsules = []
        secondary_vectors = []

        for b in range(batch_size):
            mask = (batch == b)
            if mask.sum() == 0:
                continue

            x_b = x[mask]  
            x_vectors_b = x_vectors[mask] if x_vectors is not None else None
            nodes, primary_caps, primary_dim = x_b.size()
            scalar_part = x_b[:, :, :primary_dim//2]  
            vector_part = x_b[:, :, primary_dim//2:] 
            u_hat = torch.zeros(nodes, primary_caps, self.out_caps, self.out_dim, device=x.device)

            for i in range(self.out_caps):
                scalar_transformed = torch.matmul(scalar_part, self.W_scalar[i])  
                vector_transformed = torch.matmul(vector_part, self.W_vector[i]) 
                u_hat[:, :, i, :self.scalar_dim] = scalar_transformed
                u_hat[:, :, i, self.scalar_dim:] = vector_transformed
            u_hat_flat = u_hat.view(-1, self.out_caps, self.out_dim)  
            num_inputs = u_hat_flat.size(0)  
            #Initialize routing logits
            b_ij = torch.zeros(num_inputs, self.out_caps, device=x.device)
            #Routing
            for _ in range(self.routing_iterations):
                c_ij = F.softmax(b_ij, dim=1)  
                c_ij = c_ij.unsqueeze(2)
                s_j = (c_ij * u_hat_flat).sum(dim=0) + self.bias 
                v_j = self.squash(s_j, dim=1)  
                if _ < self.routing_iterations - 1: 
                    u_scalar = u_hat_flat[:, :, :self.scalar_dim] 
                    v_scalar = v_j[:, :self.scalar_dim].unsqueeze(0) 
                    #Agreement
                    agreement = (u_scalar * v_scalar).sum(dim=2) 
                    b_ij = b_ij + agreement

            if x_vectors_b is not None:
                c_ij_reshaped = c_ij.view(nodes, primary_caps, self.out_caps, 1)
                vector_channels = x_vectors_b.size(2)
                routed_vectors = torch.zeros(self.out_caps, vector_channels, 3, device=x.device)
                for k in range(self.out_caps):
                    weights = c_ij_reshaped[:, :, k, 0].view(nodes, primary_caps, 1, 1)
                    if x_vectors_b.dim() == 3: 
                        expanded_vectors = x_vectors_b.unsqueeze(1).expand(-1, primary_caps, -1, -1)
                        routed_vectors[k] = (weights * expanded_vectors).sum(dim=(0, 1))
                    else:  
                        routed_vectors[k] = (weights * x_vectors_b).sum(dim=(0, 1))
                secondary_vectors.append(routed_vectors.unsqueeze(0))
            secondary_capsules.append(v_j.unsqueeze(0))
        if not secondary_capsules:
            return torch.zeros((batch_size, self.out_caps, self.out_dim), device=x.device), None
        return torch.cat(secondary_capsules, dim=0), torch.cat(secondary_vectors, dim=0) if secondary_vectors else None


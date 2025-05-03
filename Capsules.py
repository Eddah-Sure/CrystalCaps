
class PrimaryCapsuleLayer(nn.Module):
    def __init__(self, in_features, out_caps, caps_dim):
        super(PrimaryCapsuleLayer, self).__init__()
        self.out_caps = out_caps
        self.caps_dim = caps_dim
        self.projection = nn.Sequential(
            nn.Linear(in_features, out_caps * caps_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.projection(x)
        return x.view(x.size(0), self.out_caps, self.caps_dim)

class SecondaryCapsuleLayer(nn.Module):
    def __init__(self, in_dim, out_caps, out_dim, routing_iterations=3):

        super(SecondaryCapsuleLayer, self).__init__()
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.routing_iterations = routing_iterations
        self.W = nn.Parameter(torch.randn(1, in_dim, out_caps * out_dim))
        self.bias = nn.Parameter(torch.zeros(out_caps, out_dim))

    def squash(self, tensor, dim=-1):

        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / (torch.sqrt(squared_norm) + 1e-8)

    def forward(self, x, batch):
        batch_size = batch.max().item() + 1
        secondary_capsules = []

        # Process each graph in the batch independently.
        for b in range(batch_size):
            mask = (batch == b)
            if mask.sum() == 0:
                continue
            x_b = x[mask]

            N, primary_caps, primary_dim = x_b.size()
            x_flat = x_b.contiguous().view(-1, primary_dim)
            u_hat = torch.matmul(x_flat.unsqueeze(1), self.W)
            u_hat = u_hat.view(-1, self.out_caps, self.out_dim)
            num_input = u_hat.size(0)
            b_ij = torch.zeros(num_input, self.out_caps, device=x.device)

            # Dynamic routing iterations:
            for _ in range(self.routing_iterations):
                c_ij = F.softmax(b_ij, dim=1)
                s_j = (c_ij.unsqueeze(2) * u_hat).sum(dim=0) + self.bias
                v_j = self.squash(s_j, dim=-1)
                b_ij = b_ij + (u_hat * v_j.unsqueeze(0)).sum(dim=2)

            secondary_capsules.append(v_j.unsqueeze(0))

        if not secondary_capsules:
            return torch.zeros((batch_size, self.out_caps, self.out_dim), device=x.device)
        return torch.cat(secondary_capsules, dim=0)


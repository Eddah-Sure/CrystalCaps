class CartesianAwareCGCNNConv(nn.Module):
    def __init__(self, channels, aggr="add"):
        super().__init__()
        self.channels = channels
        self.aggr = aggr
        self.edge_gate = nn.Sequential(
            nn.Linear(channels * 3 + 3, channels),
            nn.Sigmoid()
        )
        self.update_fn = nn.Sequential(
            nn.Linear(channels * 2 + 3, channels),
            nn.ReLU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x, edge_index, edge_attr, pos):
        row, col = edge_index
        pos_diff = pos[row] - pos[col]
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, pos_diff=pos_diff)
        out = self.update_fn(torch.cat([x, out, pos], dim=1))
        return out

    def propagate(self, edge_index, x, edge_attr, pos_diff):

        row, col = edge_index

        edge_features = torch.cat([x[row], x[col], edge_attr, pos_diff], dim=1)
        gates = self.edge_gate(edge_features)
        messages = gates * x[col]

        out = torch.zeros_like(x)
        out.index_add_(0, row, messages)
        return out

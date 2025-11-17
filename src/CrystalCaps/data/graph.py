"""
Graph data structure for crystal materials.

This module contains the Graph class that handles the conversion of raw
crystal data into a format suitable for  gnns.
"""

import numpy as np
import torch


class Graph:
    def __init__(self, graph_data):
        try:
            self.nodes = graph_data['node_features']
            self.type_counts = graph_data['type_counts']
            self.neighbor_counts = graph_data['neighbor_counts']
            self.neighbors = graph_data['neighbors']
            self.bond_lengths = graph_data['bond_lengths']
            self.cart_coords = graph_data['cart_coords']


            self.nodes = np.array(self.nodes)
            self.type_counts = np.array(self.type_counts)
            self.neighbor_counts = np.array(self.neighbor_counts)
            self.neighbors = np.array(self.neighbors)
            self.bond_lengths = np.array(self.bond_lengths)
            self.cart_coords = np.array(self.cart_coords)

        except KeyError as e:
            raise ValueError(f"Missing required graph data field: {str(e)}")

        if len(self.nodes) != len(self.cart_coords):
            raise ValueError(f"Number of nodes ({len(self.nodes)}) doesn't match coordinate count ({len(self.cart_coords)})")
        if self.cart_coords.shape[1] != 3:
            raise ValueError("Coordinates must be 3-dimensional")
        if len(self.bond_lengths) != len(self.neighbors):
            raise ValueError(f"Bond lengths count ({len(self.bond_lengths)}) must match neighbor count ({len(self.neighbors)})")

        self.edge_attr = self._create_edge_attributes()
        self.edge_index = self._create_edge_index()

    def _create_edge_attributes(self):
        edge_types = []
        for edge_type, count in enumerate(self.type_counts):
            edge_types.extend([edge_type] * count)

        if len(edge_types) != len(self.bond_lengths):
            raise ValueError(f"Edge type count ({len(edge_types)}) doesn't match bond lengths count ({len(self.bond_lengths)})")

        return torch.tensor(
            np.column_stack([self.bond_lengths, edge_types]),
            dtype=torch.float32
        )

    def _create_edge_index(self):
        edge_sources = []
        num_edge_labels = len(self.type_counts)
        neighbor_counts = self.neighbor_counts.reshape(num_edge_labels, -1)
        for edge_type in range(num_edge_labels):
            for node_idx, count in enumerate(neighbor_counts[edge_type]):
                edge_sources.extend([node_idx] * count)

        edge_targets = []
        start_idx = 0
        for count in self.type_counts:
            end_idx = start_idx + count
            edge_targets.extend(self.neighbors[start_idx:end_idx])
            start_idx = end_idx

        if len(edge_sources) != len(edge_targets):
            raise ValueError(f"Edge sources count ({len(edge_sources)}) and targets count ({len(edge_targets)}) mismatch")
        if len(edge_sources) != len(self.bond_lengths):
            raise ValueError(f"Edge count ({len(edge_sources)}) doesn't match bond length count ({len(self.bond_lengths)})")

        return torch.tensor([edge_sources, edge_targets], dtype=torch.long).contiguous()

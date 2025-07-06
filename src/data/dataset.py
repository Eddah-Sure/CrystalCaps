"""
Dataset class for crystal materials data.
This module contains the CartesianGraphDataset class that loads and processes
crystal structure data.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .graph import Graph


class CartesianGraphDataset(Dataset):
   
    
    def __init__(self, path, target_name):
        super().__init__()
        self.path = path
        self.target_name = target_name

        # Load all required data
        self._load_graph_data()
        self._load_config()
        self._load_targets()

        # Validate data consistency
        if len(self.graph_data) != len(self.targets):
            raise ValueError(
                f"Graph count ({len(self.graph_data)}) doesn't match "
                f"target count ({len(self.targets)})"
            )

    def _load_graph_data(self):
        """Load and validate graph data from NPZ file."""
        npz_path = os.path.join(self.path, "BandgapTargets.npz")
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                graph_dict = data['graph_dict'].item()
                self.graph_names = list(graph_dict.keys())
                self.graph_data = []

                for name, graph in graph_dict.items():
                    try:
                        if 'cart_coords' not in graph:
                            raise ValueError(f"Missing cart_coords in graph {name}")
                        self.graph_data.append(Graph(graph))
                    except ValueError as e:
                        print(f"Skipping invalid graph {name}: {str(e)}")
                        continue

                if not self.graph_data:
                    raise ValueError("No valid graphs found in NPZ file")
                    
        except Exception as e:
            raise RuntimeError(f"Error loading graph data: {str(e)}")

    def _load_config(self):
        """Load and validate configuration."""
        config_path = os.path.join(self.path, "BandgapTargets_config.json")
        try:
            with open(config_path) as f:
                config = json.load(f)

            self.atomic_numbers = config["atomic_numbers"]
            self.node_vectors = np.array(config["node_vectors"])
            self.n_node_feat = len(self.node_vectors[0])
            self.pos_dim = config.get("pos_dim", 3)
            
            # Create atomic number to index mapping
            self.atomic_to_idx = {
                num: idx for idx, num in enumerate(self.atomic_numbers)
            }
            
            if len(self.atomic_to_idx) != len(self.atomic_numbers):
                raise ValueError("Duplicate atomic numbers in config")

        except Exception as e:
            raise RuntimeError(f"Error loading config: {str(e)}")

    def _load_targets(self):
        """Load target values from CSV file."""
        targets_path = os.path.join(self.path, "BandgapTargets.csv")
        try:
            df = pd.read_csv(targets_path)
            
            if self.target_name not in df.columns:
                raise ValueError(
                    f"Target column '{self.target_name}' not found in CSV"
                )

            self.targets = df[self.target_name].values
            if len(self.targets) == 0:
                raise ValueError("No targets found in CSV file")

            # Update graph names from CSV (more reliable than NPZ keys)
            self.graph_names = df['mpid'].values.tolist()

        except Exception as e:
            raise RuntimeError(f"Error loading targets: {str(e)}")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.graph_data)

    def __getitem__(self, index):
        """
        Create PyTorch Geometric Data object for the given index.
        
        Args:
            index (int): Index of the sample to retrieve
            
        Returns:
            Data: PyTorch Geometric Data object containing:
                - x: Node features
                - edge_index: Edge connectivity
                - edge_attr: Edge attributes
                - pos: Node positions
                - y: Target value
                - material_id: Material identifier
        """
        graph = self.graph_data[index]

        # Create node features from atomic numbers
        node_features = np.zeros((len(graph.nodes), self.n_node_feat))
        for i, atomic_num in enumerate(graph.nodes):
            idx = self.atomic_to_idx[atomic_num]
            node_features[i] = self.node_vectors[idx]

        # Handle position data (ensure it's a tensor)
        if isinstance(graph.cart_coords, torch.Tensor):
            pos = graph.cart_coords.clone().detach()
        else:
            pos = torch.tensor(graph.cart_coords, dtype=torch.float32)

        # Create PyTorch Geometric Data object
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            pos=pos,
            y=torch.tensor([[self.targets[index]]], dtype=torch.float32),
            material_id=self.graph_names[index]
        )
        
        return data

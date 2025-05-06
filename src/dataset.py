import os
import json
import numpy as np
import pandas as pd

class Graph:
    def __init__(self, graph_data):
        try:
            self.nodes = graph_data['node_features']
            self.type_counts = graph_data['type_counts']
            self.neighbor_counts = graph_data['neighbor_counts']
            self.neighbors = graph_data['neighbors']
            self.bond_lengths = graph_data['bond_lengths']
            self.cart_coords = graph_data['cart_coords']

            # To numpy arrays
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

        # Build graph components
        self.edge_attr = self._create_edge_attributes()
        self.edge_index = self._create_edge_index()

    def _create_edge_attributes(self):
        """Create edge attributes with bond lengths and types"""
        edge_types = []
        for edge_type, count in enumerate(self.type_counts):
            edge_types.extend([edge_type] * count)

        if len(edge_types) != len(self.bond_lengths):
            raise ValueError(f"Edge type count ({len(edge_types)}) doesn't match bond lengths count ({len(self.bond_lengths)})")

        # Stack bond lengths and edge types
        return torch.tensor(
            np.column_stack([self.bond_lengths, edge_types]),
            dtype=torch.float32
        )

    def _create_edge_index(self):
        """Create edge index with source and target nodes"""
        edge_sources = []
        num_edge_labels = len(self.type_counts)
        neighbor_counts = self.neighbor_counts.reshape(num_edge_labels, -1)

        # Create source nodes for each edge type
        for edge_type in range(num_edge_labels):
            for node_idx, count in enumerate(neighbor_counts[edge_type]):
                edge_sources.extend([node_idx] * count)

        # Create target nodes
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

class CartesianGraphDataset(Dataset):
    def __init__(self, path, target_name):
        super().__init__()
        self.path = path
        self.target_name = target_name

        self._load_graph_data()
        self._load_config()
        self._load_targets()

        if len(self.graph_data) != len(self.targets):
            raise ValueError(
                f"Graph count ({len(self.graph_data)}) doesn't match "
                f"target count ({len(self.targets)})"
            )

    def _load_graph_data(self):
        """Load and validate graph data from NPZ file"""
        npz_path = os.path.join(self.path, "file.npz")
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
        """Load and validate configuration"""
        config_path = os.path.join(self.path, "file.json")
        try:
            with open(config_path) as f:
                config = json.load(f)

            self.atomic_numbers = config["atomic_numbers"]
            self.node_vectors = np.array(config["node_vectors"])
            self.n_node_feat = len(self.node_vectors[0])
            self.pos_dim = config.get("pos_dim", 3)
            self.atomic_to_idx = {num: idx for idx, num in enumerate(self.atomic_numbers)}
            if len(self.atomic_to_idx) != len(self.atomic_numbers):
                raise ValueError("Duplicate atomic numbers in config")

        except Exception as e:
            raise RuntimeError(f"Error loading config: {str(e)}")

    def _load_targets(self):
        targets_path = os.path.join(self.path, "file.csv")
        try:
            df = pd.read_csv(targets_path)
            if self.target_name not in df.columns:
                raise ValueError(f"Target column '{self.target_name}' not found in CSV")

            self.targets = df[self.target_name].values
            if len(self.targets) == 0:
                raise ValueError("No targets found in CSV file")

            self.graph_names = df['mpid'].values.tolist()

        except Exception as e:
            raise RuntimeError(f"Error loading targets: {str(e)}")

    def __len__(self):
        return len(self.graph_data)

    def __getitem__(self, index):
        graph = self.graph_data[index]

        # Create one-hot node features
        node_features = np.zeros((len(graph.nodes), self.n_node_feat))
        for i, atomic_num in enumerate(graph.nodes):
            idx = self.atomic_to_idx[atomic_num]
            node_features[i] = self.node_vectors[idx]

        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            pos=graph.cart_coords.clone().detach() if isinstance(graph.cart_coords, torch.Tensor)
                else torch.tensor(graph.cart_coords, dtype=torch.float32),
            y=torch.tensor([[self.targets[index]]], dtype=torch.float32),
            material_id=self.graph_names[index]
        )
        return data


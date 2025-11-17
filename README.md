# CrystalCaps: Capsule Graph Networks for Accurate and Interpretable Crystalline Materials Property Predictions (CGN-e3)

This repository presents the very first implementation of **Equivariant Capsule Graph Networks (CGN-e3)**, a novel architecture that integrates capsule networks with graph neural networks for crystalline materials representation  (first to integrate equivariance into capsules network). The CGN-e3 model processes atomic graphs by encoding neighbor distances via radial basis functions, angles via spherical harmonics, and aggregates messages via Clebsch–Gordan tensor products, satisfying equivariance under 3D reflections, rotations, and translations.


<p align="center">
<img width="600" alt="image" src="https://github.com/user-attachments/assets/0aafda89-a2f0-4d30-b143-212178b01a62" />

</p>

The model processes a crystal’s atomic graph through a sequence of equivariant graph convolutions, capsule routing layers and an attention mechanism to predict material properties. First, atomic numbers are embedded via one-hot and distances are encoded by a radial basis function (RBF) expansion. In each equivariant convolutional layer, for each central atom and neighbor the relative vector is computed and decomposed it into its length and direction. The distance is expanded into a vector of Gaussian RBFs and direction expanded in spherical harmonics. A learnable multilayer perceptron (MLP) is applied to the RBF vector to produce radial filter coefficients, which are then multiplied with the spherical harmonics and tensor-producted with the neighbor’s feature tensor. Clebsch–Gordan tensor product aggregates spherical-harmonic order with neighbor features to produce output of order. The summed messages are then assembled back into irreducible feature vectors for atom. Each layer output is passed through suitable nonlinearities; that is SiLU on scalars and gated-sigmoid on vectors and an equivariant normalization for normalization and rescaling. After convolutions, each atom has a tuple of scalar and vector features encoding local geometry and chemistry. We treat these as primary capsules at the node level. These primary node capsules are then aggregated into a smaller set of graph capsules via an attention-and-routing mechanism. An attention weights nodes to balance different graph sizes, and then dynamic routing iteratively refines coupling coefficients between each node’s capsule and each graph-level capsule. Finally, to produce the property prediction, the graph capsule outputs are either further routed with attention into final output capsules corresponding to target properties.

<p align="center">
  <img width="600" alt="image" src="https://github.com/user-attachments/assets/27cadc44-1eea-4637-9e81-330b5827d3b4" />
</p>
Salency maps for interpretation
<p align="center">
<img width="600" alt="image" src="https://github.com/user-attachments/assets/3267c22c-404d-4aad-96ea-baa8f6a8ca85" />

</p>
## Quick Start

```bash
# Clone the repository
git clone https://github.com/Eddah-Sure/CrystalCaps.git
cd CrystalCaps

# Install dependencies
pip install -r requirements.txt

# Run training (from repo root)
python -m src.crystalcaps.Train
```
## Requirements

### Requirements
- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric
- e3nn
- NumPy
- Pandas
- Scikit-learn

### Installation

```bash
# Install PyTorch with CUDA support (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all other dependencies
pip install -r requirements.txt
```

For CPU-only installation:
```bash
pip install -r requirements.txt
```

## Data Preparation

### Dataset Structure
The model requires three main files for each dataset:

1. **`targets.csv`**: Contains target values (e.g., formation energy, band gap)
   - Must include columns: `mpid` (material ID) and the target property from the Material Project Database

2. **`graph_data.npz`**: Contains crystal graph attributes
   - Generated using the provided graph coordinator ( check the our Graph Coordinator )

3. **`config.json`**: Defines node feature vectors
   - Contains atomic numbers and their corresponding feature vectors, idealy we generate this fom our graph coordinator.

### Data Sources
- Material IDs are provided in the `data/` directory
- Use the graph coordinator in `Materials Project/Graph coordinator.py` to generate graph files
- **API Key Required**: Get the Materials Project API key [here](https://next-gen.materialsproject.org/api)

### Example Dataset Structure
```
dataset/
├── targets.csv          # Target properties
├── graph_data.npz       # Graph representations
└── config.json          # Node feature definitions
```

##  Training


```python
model, results = run_CGNe3(
    dataset_path="path/to/dataset",
    target_name="",
    epochs=200,
    batch_size=32,
    hidden_channels=128,
    num_conv_layers=3,
    primary_caps=16,
    primary_dim=64,
    secondary_caps=8,
    secondary_dim=32,
    dropout_rate=0.1,
    early_stopping_patience=20
)
```


### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 64 | Training batch size |
| `hidden_channels` | 128 | Hidden layer dimensions |
| `num_conv_layers` | 2 | Number of graph convolution layers |
| `primary_caps` | 8 | Number of primary capsules |
| `primary_dim` | 16 | Primary capsule dimensions |
| `secondary_caps` | 6 | Number of secondary capsules |
| `secondary_dim` | 16 | Secondary capsule dimensions |
| `dropout_rate` | 0.1 | Dropout probability |
| `early_stopping_patience` | 20 | Early stopping patience |


## Project Structure

```
CrystalCaps/
├── src/
│   ├── crystalcaps/
│   │   ├── CapsuleNetworkLayers.py      # Capsule network components
│   │   ├── GNNBase.py                  # Equivariant GNN layers
│   │   ├── Model.py                    # CGNe3 model definition
│   │   ├── Train.py                    # Training pipeline
│   │   └── data/                       # Data utilities
│   │       ├── dataset.py
│   │       └── graph.py
│   └── graph_coordinator.py            # Graph construction utility
├── data/                               # Dataset files
│   ├── e_form.csv
│   └── Metalclasses.csv
├── figures/                            # Documentation figures
├── examples/                           # Example notebooks/scripts
├── tests/                              # Unit tests
├── docs/                               # Full documentation
├── requirements.txt                    # Dependencies
├── README.md                           # This file
├── LICENSE                             # License
└── .gitignore                          # Git ignores
```


## Authorship

This work was primarily written by **Eddah K. Sure**, advised by **Prof. Wu Xing** and **Prof.Qian Quan**.

## Citation

If you use this code in research, please cite us as:

```bibtex
@article{sure2025crystalcaps,
  title={Capsule Graph Networks for Accurate and
 Interpretable Crystalline Materials Property
 Prediction},
  author={Sure, Eddah K. Xing, Wu and Quan, Qian},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
        

        









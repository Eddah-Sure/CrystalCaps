# CrystalCaps: Equivariant Capsule Graph Networks (CGN-e3)

This repository presents the original implementation of **Equivariant Capsule Graph Networks (CGN-e3)**, a novel architecture that integrates capsule networks with graph neural networks for crystalline materials representation. The CGN-e3 model processes atomic graphs by encoding neighbor distances via radial basis functions, angles via spherical harmonics, and aggregates messages via Clebsch–Gordan tensor products, satisfying equivariance under 3D reflections, rotations, and translations.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Eddah-Sure/CrystalCaps.git
cd CrystalCaps

# Install dependencies
pip install -r requirements.txt

# Run training
cd src
python Train.py
```
<p align="center">
<img width="532" alt="image" src="https://github.com/user-attachments/assets/0aafda89-a2f0-4d30-b143-212178b01a62" />

</p>

The model processes a crystal’s atomic graph through a sequence of equivariant graph convolutions, capsule routing layers and an attention mechanism to predict material properties. First, atomic numbers are embedded via one-hot and distances are encoded by a radial basis function (RBF) expansion. In each equivariant convolutional layer, for each central atom and neighbor the relative vector is computed and decomposed it into its length and direction. The distance is expanded into a vector of Gaussian RBFs and direction expanded in spherical harmonics. A learnable multilayer perceptron (MLP) is applied to the RBF vector to produce radial filter coefficients, which are then multiplied with the spherical harmonics and tensor-producted with the neighbor’s feature tensor. Clebsch–Gordan tensor product aggregates spherical-harmonic order with neighbor features to produce output of order. The summed messages are then assembled back into irreducible feature vectors for atom. Each layer output is passed through suitable nonlinearities; that is SiLU on scalars and gated-sigmoid on vectors and an equivariant normalization for normalization and rescaling. After convolutions, each atom has a tuple of scalar and vector features encoding local geometry and chemistry. We treat these as primary capsules at the node level. These primary node capsules are then aggregated into a smaller set of graph capsules via an attention-and-routing mechanism. An attention weights nodes to balance different graph sizes, and then dynamic routing iteratively refines coupling coefficients between each node’s capsule and each graph-level capsule. Finally, to produce the property prediction, the graph capsule outputs are either further routed with attention into final output capsules corresponding to target properties.

<p align="center">
  <img width="533" alt="image" src="https://github.com/user-attachments/assets/27cadc44-1eea-4637-9e81-330b5827d3b4" />
</p>
Salency maps for interpretation
<p align="center">
<img width="550" alt="image" src="https://github.com/user-attachments/assets/3267c22c-404d-4aad-96ea-baa8f6a8ca85" />

</p>

## Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Dependencies
All dependencies are listed in `requirements.txt`. Key packages include:

- **PyTorch Ecosystem**: `torch`, `torch-geometric`, `torch-scatter`
- **Equivariant Networks**: `e3nn`
- **Scientific Computing**: `numpy`, `pandas`, `scipy`
- **Machine Learning**: `scikit-learn`
- **Visualization**: `matplotlib`, `seaborn` (optional)

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
   - Must include columns: `mpid` (material ID) and your target property

2. **`graph_data.npz`**: Contains crystal graph attributes
   - Generated using the provided graph coordinator

3. **`config.json`**: Defines node feature vectors
   - Contains atomic numbers and their corresponding feature vectors

### Data Sources
- Material IDs are provided in the `data/` directory
- Use the graph coordinator in `Materials Project/Graph coordinator.py` to generate graph files
- **API Key Required**: Get your Materials Project API key [here](https://next-gen.materialsproject.org/)

### Example Dataset Structure
```
your_dataset/
├── targets.csv          # Target properties
├── graph_data.npz       # Graph representations
└── config.json          # Node feature definitions
```

##  Training


```python
model, results = run_CGNe3(
    dataset_path="path/to/your/dataset",
    target_name="property_value",
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

### Command Line Training

```bash
cd src
python Train.py
```

**Note**: Update the dataset path and target name in the `if __name__ == "__main__"` section of `Train.py`

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Training batch size |
| `hidden_channels` | 128 | Hidden layer dimensions |
| `num_conv_layers` | 3 | Number of graph convolution layers |
| `primary_caps` | 16 | Number of primary capsules |
| `primary_dim` | 64 | Primary capsule dimensions |
| `secondary_caps` | 8 | Number of secondary capsules |
| `secondary_dim` | 32 | Secondary capsule dimensions |
| `dropout_rate` | 0.1 | Dropout probability |
| `early_stopping_patience` | 20 | Early stopping patience |

##  Architecture Overview

The CGN-e3 model consists of several key components:

### 1. **Equivariant Graph Convolutions** (`GNNBase.py`)
- **EquivariantGNN**: Processes atomic graphs with E(3) equivariance
- **RadialBasisLayer**: Encodes distances using Gaussian RBFs
- **LayerNormalization**: Normalizes scalar and vector features

### 2. **Capsule Networks** (`CapsuleNetwork.py`)
- **PrimaryCapsuleLayer**: Converts node features to primary capsules
- **SecondaryCapsuleLayer**: Aggregates primary capsules via dynamic routing

### 3. **Main Model** (`Model.py`)
- **CGNe3**: Complete model integrating GNN and capsule components
- Attention mechanism for capsule aggregation
- Final prediction layers

### 4. **Data Processing** (`Dataloader.py`)
- **GraphDataset**: Loads and processes crystal structure data
- **Graph**: Handles individual crystal graph representations

### 5. **Training Pipeline** (`Train.py`)
- Training and evaluation functions
- Model checkpointing and metrics logging
- Hyperparameter management

## Project Structure

```
CrystalCaps/
├── src/
│   ├── Train.py           # Training pipeline
│   ├── Model.py           # CGNe3 model definition
│   ├── GNNBase.py         # Equivariant GNN layers
│   ├── CapsuleNetwork.py  # Capsule network components
│   └── Dataloader.py      # Data loading utilities
├── data/                  # Dataset files
├── Materials Project/     # Graph generation tools
├── requirements.txt       # Dependencies
└── README.md             # This file
```

##  Model Features

- **E(3) Equivariance**: Invariant to rotations, reflections, and translations
- **Capsule Routing**: Dynamic routing for hierarchical feature learning
- **Attention Mechanism**: Weighted aggregation of capsule features
- **Multi-scale Representation**: From atomic to graph-level features

## Results and Outputs

The model generates several output files:

- `results_{target_name}/training_metrics.csv`: Training progress
- `results_{target_name}/best_model.pth`: Best model checkpoint
- `results_{target_name}/test_metrics.csv`: Test performance metrics
- `results_{target_name}/predictions.csv`: Detailed predictions

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Authorship

This work was primarily written by **Eddah K. Sure**, advised by **Prof. Wu Xing**.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{sure2024crystalcaps,
  title={CrystalCaps: Equivariant Capsule Graph Networks for Crystalline Materials},
  author={Sure, Eddah K. and Xing, Wu},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
        









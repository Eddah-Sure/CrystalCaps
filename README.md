# CrystalCaps
This repository presents the original implementetion of Equivariant Capsule Graph Networks, CGN-e3, a novel architecture that integrates capsule networks with graph neural networks for crystalline materials representation. CGN-e3 model processes atomic graphs by encoding neighbor distances via radial basis functions, angles via spherical harmonics, and aggregates messages via the Clebsch–Gordan tensor products, satisfying equivariance under 3D reflections, rotations, and translations. 
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

**Requirements:**
- Python
- Pymatgen
- e3nn
- scikit-learn
- NetworkX, Graphviz, Matplot - For plots

**Data**

To reproduce this work, we have provided the material IDs used for each of the tasks. Check the Data file.

We have also provided a graph coordinator to create the graph data files for each of the datasets. You will require an API Key for this, check here for details [Materials Project](https://next-gen.materialsproject.org/)

**Dataset Files**
We have only provided the material IDs used. However this model does require 3 mail files for each of the tasks. We provide clarity on what the 3 files are;
- targetfile.csv: contains all the target values, for insance the continous values of bandgap incase of the bandgap prediction task and so on
- graphfile.npz: contains all the crystal graph attributes generated
- cofigfile.json: defines the node vectors

**Training**
<pre>dataset = \,
target_name = \,
batch-size=\,
node_features=\,
edge_features=\,
hidden_channels=\,
num_conv_layers=\,
primary_caps=8\,
primary_dim=\,
secondary_caps=\,
secondary_dim=\,
epochs=\,
early_stopping_patience=\,
lr=\ </pre>

**Authorship**
This was primarily written by Eddah K. Sure, advised by Prof. Wu Xing

**Cite**
        









# CrystalCaps
This repository contains the original implimentation of GNN + Capsules enhanced architecture for crystalline material representation. The architecture has been used to predict continous material properties including band gap and formation energy. We also introduce a cross entroppy loss for a binary classification task


![mp-4651](https://github.com/user-attachments/assets/4f9590f7-c593-4a67-a965-b30613454723)



GNN + Capsules for crystalline materials introduces a novel architecture, Capsule Crystal Graph Networks*, that integrates capsule networks into GNNs to extract node features. By adopting the use capsules to generate multiple embeddings for each atom species, CapsCGNets captures a crystals structural and chemical properties property from different aspects. The use of vector capsules explicitly models part-whole hierarchies between the individual atoms and the crystal in periodic crystalline systems, enabling robust learning of local atomic environments and their interactions. We demonstrate state-of-the-art performance on a number of prediction tasks on the Materials project data and provide interpretable insights into the learned capsule representations.

![image](https://github.com/user-attachments/assets/86ca1c86-cbcd-4333-a259-c37146c67d02)

**Requirements:**
- Python
- Pymatgen - for graph coordinator
- NetworkX, Graphviz, Matplot - For plots
- scikit-learn

**Data**
To reproduce this work, we have provided the material IDs used for each of the tasks. Check the Data file.
We have also provided a graph coordinator to create the graph data files for each of the datasets. You will require an API Key for this, check here for details [Link Text](https://next-gen.materialsproject.org/)








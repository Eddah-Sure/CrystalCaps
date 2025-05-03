# CrystalCaps
This repository contains the original implimentation of GNN + Capsules enhanced architecture for crystalline material representation. The architecture has been used to predict continous material properties including band gap and formation energy. We also introduce a cross entroppy loss for a binary classification task

![image](https://github.com/user-attachments/assets/33641282-0108-4a62-b1cd-8025534ff91a)


GNN + Capsules for crystalline materials
This paper introduces a novel architecture, Capsule Crystal Graph Networks*, that integrates capsule networks into GNNs to extract node features. By adopting the use capsules to generate multiple embeddings for each atom species, CapsCGNets captures a crystals structural and chemical properties property from different aspects. The use of vector capsules explicitly models part-whole hierarchies between the individual atoms and the crystal in periodic crystalline systems, enabling robust learning of local atomic environments and their interactions. We demonstrate state-of-the-art performance on a number of prediction tasks on the Materials project data and provide interpretable insights into the learned capsule representations.

![image](https://github.com/user-attachments/assets/86ca1c86-cbcd-4333-a259-c37146c67d02)



# Graph Neural Networks for Fast Prediction of Plastic Metal Deformation in Hot Forging

This repository contains code and sample data to demonstrate GNN capability to predict deformation in hot metal forging. The code uses methods presented in the original MeshGraphNets implementation as well as meshGraphNets_PyTorch.

Original MeshGraphNets research paper can be found at: [meshgraphnets research](https://arxiv.org/pdf/2010.03409)

Original MeshGraphNets code can be found at: [meshgraphnets code](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)

Subsequent PyTorch implementation of this code can be found at: [meshGraphNets_PyTorch](https://github.com/echowve/meshGraphNets_pytorch)

# Authors

## GNN Data Pre-Processing and Model Development
- Sam St. John
- Faezeh Bagheri

## Input Data Generation


# Usage

1. Download code and input files
2. Execute code in numerical order (e.g., 01_identify_exterior, 02_create_triangular_mesh, 03_create_tensors, 04_run_model)

Note that the first three code files are performing data preparation to transform source simulation data to tensors. The final code file demonstrates GNN learning based on this limited source data set.

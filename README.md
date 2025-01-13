# GhostMLP_GAT

## Overview

This project focuses on 3D object classification using lightweight machine learning approaches. The proposed pipeline integrates **GhostMLP** for efficient feature extraction and **GAT (Graph Attention Network)** for enhanced feature classification. The project leverages benchmark datasets and features extracted from PointNet++ and GhostMLP architectures for graph-based classification.

## Key Features
- Lightweight feature extraction using **GhostMLP**.
- Feature classification using graph-based methods (**GCN** and **GAT**).
- Benchmarked against **PointNet++** for performance comparisons.
- Optimized for resource-constrained environments.

## How to Use

1. **Dataset Preparation**
   - **Raw Dataset**: 
     The dataset with partial single-view Point Cloud Data (PCDs) was generated from the benchmark dataset **ModelNet40** using Pointview-GCN. Please download the dataset and follow these steps:
     - Create a directory named `single_view_modelnet`.
     - Place the dataset under your project folder’s `data` directory.
   - **Feature Extracted Dataset**:
     - Features extracted using PointNet++ are stored in `data/modelnet_trained_feature`.
     - Features extracted using GhostMLP are stored in `data/modelnet_trained_ghostfeature`.

2. **Run Feature Classification**
   - As the feature extraction process has already been completed, you can directly run the classification scripts:
     - For **GCN** classification, execute the corresponding classification file (e.g., `GCN_Point.py`) under the folder GCN.
     - For **GAT** classification, execute the corresponding classification file (e.g., `GAT_Ghost.py`) under the folder GAT.

3. **Results**
   - The results will include validation accuracy, mean class accuracy, and overall performance metrics for the chosen classification model.

## Project Structure
![Uploading image.png…]()

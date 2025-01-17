# GhostMLP_GAT

## Overview

This project focuses on 3D object classification using lightweight machine-learning approaches. The proposed pipeline integrates **GhostMLP** for efficient feature extraction and **GAT (Graph Attention Network)** for enhanced feature classification. The project leverages benchmark datasets and features extracted from PointNet++ and GhostMLP architectures for graph-based classification.

## Key Features
- Lightweight feature extraction using **GhostMLP**.
- Feature classification using graph-based methods (**GCN** and **GAT**).
- Benchmarked against **PointNet++** for performance comparisons.
- Optimized for resource-constrained environments.

## Installation and Environment Setup

To set up the environment for this project, follow these steps:

1. **Python Version**:
   Ensure you have **Python 3.9.19** installed.

2. **Dependencies**:
   Use the provided `requirements.txt` file to install all necessary dependencies. Run the following command in your terminal:

   ```bash
   pip install -r requirements.txt

## How to Use

1. **Dataset Preparation**
   - **Raw Dataset**: 
     The dataset with partial single-view Point Cloud Data (PCDs) was generated from the benchmark dataset **ModelNet40** using Pointview-GCN. Please download the dataset from this link and follow these steps:
     - Unzip the folder `modelnetdata` and rename it: `single_view_modelnet`.
     - Place the dataset under your project folder’s `data` directory.
   - **Feature Extracted Dataset**:
     - Features extracted using PointNet++ are stored in `data/modelnet_trained_feature`.
     - Features extracted using GhostMLP are stored in `data/modelnet_trained_ghostfeature`.

2. **Run Feature Classification**
   - As the feature extraction process has already been completed, you can directly run the classification scripts:
     - For **GCN** classification on PointNet++ extracted features, execute the corresponding classification file (e.g., `GCN_Point.py`) under the folder GCN.
     - For **GCN** classification on GhostMLP extracted features, execute the corresponding classification file (e.g., `GCN_Ghost.py`) under the folder GCN.
     - For **GAT** classification on GhostMLP extracted features, execute the corresponding classification file (e.g., `GAT_Ghost.py`) under the folder GAT.
     - For **GAT** classification on PointNet++ extracted features, execute the corresponding classification file (e.g., `GAT_Point.py`) under the folder GAT.

3. **Results**
   - The results will include validation accuracy, mean class accuracy, and overall performance metrics for the chosen classification model.
  
Optional **Run Feature Extraction**
   - For **PointNet++** feature extraction, execute the corresponding file (e.g., `main.py`) under the folder Feature_extraction_PointNet.
   - For **GhostMLP** feature extraction, execute the corresponding file (e.g., `main.py`) under the folder Feature_extraction_Ghost.



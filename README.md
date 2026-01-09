Handwritten Multi-Digit Recognition

## Overview
This project implements an **end-to-end Optical Character Recognition (OCR)** system for **variable-length handwritten digit sequences** using a **CRNN (Convolutional Recurrent Neural Network)** architecture optimized with **CTC loss**. The model is trained and evaluated on a **synthetic multi-digit MNIST dataset** with controlled digit overlap.

## Key Features
- Alignment-free sequence recognition using **CTC loss**
- Handles **variable-length digit sequences (≤5 digits)**
- Robust to **digit overlap up to 50%**
- Modular training pipeline built with **PyTorch Lightning**

## Architecture
- **CNN**: Extracts spatial features from digit images  
- **BiLSTM (2 layers)**: Models sequential dependencies across digits  
- **CTC Loss**: Enables training without explicit character-level alignment  
- **Greedy Decoding**: Converts network outputs to digit sequences  

## Dataset
- Synthetic **multi-digit MNIST** generated from single-digit MNIST
- Total samples: **1.4k+ (train/val/test)**
- Configurable digit overlap and sequence length
- Padding and blank tokens used for CTC compatibility

## Training Details
- Framework: PyTorch + PyTorch Lightning
- Optimizer: Adam
- Learning rate scheduling: ReduceLROnPlateau
- Metrics:  
  - Digit Accuracy  
  - Character Error Rate (CER)

## Results
- Achieved **>99% digit-level accuracy**
- Low **Character Error Rate (CER)** on validation and test sets
- Stable convergence under CTC-based optimization


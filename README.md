# Mini PyTorch Learner

A modular PyTorch project demonstrating how a neural network learns an underlying relationship directly from data using gradient descent.

---

# Project Overview

This project builds a simple neural network from scratch using PyTorch to learn a linear relationship between input and output data.

Instead of manually programming the rule, the model gradually adjusts its internal parameters (weights and bias) through optimization until its predictions closely match the training data.

The goal of this project is to understand the complete supervised learning pipeline—from data to prediction, loss computation, backpropagation, parameter optimization, and convergence.

---

# Features

- Built a neural network using `torch.nn.Linear`
- Generated a simple supervised learning dataset
- Implemented forward propagation
- Computed prediction error using Mean Squared Error (MSE)
- Optimized parameters using Stochastic Gradient Descent (SGD)
- Visualized training loss over time
- Compared model predictions against ground truth
- Animated the learning process through a training GIF
- Organized the project into modular components

---

# Project Structure

```text
mini-pytorch-learner/

├── main.py
├── data.py
├── model.py
├── train.py
├── plot.py

├── outputs/
│   ├── loss_curve.png
│   ├── prediction_vs_truth.png
│   └── training.gif

├── requirements.txt
├── README.md
└── .gitignore
```

---

# Learning Pipeline

```text
Dataset
   │
   ▼
Neural Network
   │
   ▼
Predictions
   │
   ▼
Loss (MSE)
   │
   ▼
Gradients
   │
   ▼
Optimizer (SGD)
   │
   ▼
Updated Parameters
   │
   └──────────────┐
                  ▼
             Next Epoch
```

---

# Results

## Training Loss

Shows how prediction error decreases as the model learns.

<img width="305" height="201" alt="image" src="https://github.com/user-attachments/assets/918d1798-3616-45a3-98e6-eeba35399a5f" />


---

## Prediction vs Ground Truth

Comparison between the learned predictions and the true data.

![Prediction](outputs/prediction_vs_truth.png)

---

## Learning Animation

Visualization of the optimization process across training epochs.

![Training GIF](outputs/training.gif)

---

# Concepts Learned

- Representing data using tensors
- Building neural networks with PyTorch
- Understanding weights and biases
- Forward propagation
- Loss computation using Mean Squared Error
- Automatic differentiation (Autograd)
- Gradient descent optimization
- Learning rate and parameter updates
- Visualizing model convergence
- Structuring machine learning projects into reusable modules

---

# Installation

```bash
pip install -r requirements.txt
```

---

# Run

```bash
python main.py
```

---

# CUDA vs PyTorch — MLP Backpropagation Study

A low-level implementation study of Multi Layer Perceptron (MLP) training using:

* Raw Python + NumPy
* PyTorch
* Custom CUDA Kernels

The project explores how neural network training works internally across different abstraction levels by implementing:

* forward propagation
* backpropagation
* gradient accumulation
* SGD updates
* GPU execution

without relying entirely on high-level frameworks.

---

# Architecture

The same neural network architecture is implemented across all versions:

```text id="ep5yzx"
Input(784)
    ↓
Linear(784 → 256)
    ↓
ReLU
    ↓
Linear(256 → 128)
    ↓
ReLU
    ↓
Linear(128 → 10)
```

Dataset:

* MNIST handwritten digit classification

---

# Implementations

## 1. PyTorch Implementation

Implemented using:

* `torch.nn`
* `autograd`
* SGD optimizer
* CrossEntropyLoss

Purpose:

* establish framework baseline
* understand high-level training abstractions

Features:

* batched training
* GPU support
* loss and accuracy tracking

---

## 2. Raw Python + NumPy Implementation

A complete from-scratch implementation of:

* forward propagation
* manual backpropagation
* SGD optimization
* softmax + cross entropy gradient

Implemented without:

* autograd
* tensor frameworks
* deep learning libraries

Features:

* manually derived gradients
* numerical gradient checking using finite differences
* explicit computational graph logic

This implementation serves as the mathematical reference for the CUDA implementation.

---

## 3. CUDA + pybind11 Implementation

Low-level GPU implementation using:

* CUDA C++
* custom CUDA kernels
* pybind11 bindings
* Python orchestration

Implemented components:

* forward propagation kernels
* ReLU activation kernels
* softmax computation
* backpropagation kernels
* gradient accumulation
* SGD parameter updates

The CUDA implementation explores:

* GPU memory management
* thread indexing
* low-level gradient propagation
* CUDA kernel execution

---

# Project Goals

This project aims to study:

* how backpropagation works internally
* differences between abstraction layers
* manual gradient computation
* CUDA kernel execution flow
* GPU-based neural network training

The focus of this repository is correctness and implementation understanding rather than performance optimization.

---

# Repository Structure

```text id="hh2prf"
MLP
├── pytorch_mlp.py
│
├── raw_mlp.py
│
├── cuda_mlp/
│   ├── kernel.cu
│   ├── cuda_mlp.cu
│   └── mlp.py
│
└── README.md
```

---

# Technologies Used

* Python
* NumPy
* PyTorch
* CUDA C++
* pybind11

---

# Current Status

Implemented:

* MLP forward propagation
* manual backpropagation
* CUDA kernels for training
* Python ↔ CUDA integration

Future work may include:

* benchmarking
* kernel optimization
* Batch processing in CUDA implementaion

---

# Build Instructions

## PyTorch Version

```bash id="mp3hnf"
python pytorch_mlp.py
```

---

## NumPy Version

```bash id="mgbc6q"
python raw_mlp.py
```

---

## CUDA Version

Build the CUDA extension:

```bash id="n1zwgo"
nvcc -O3 -shared -Xcompiler -fPIC $(python -m pybind11 --includes) ./MLP/cuda_mlp/cuda_mlp.cu -o ./MLP/cuda_mlp/cuda_mlp.so
```

Run:

```bash id="st4v74"
python ./MLP/cuda_mlp/mlp.py
```

---

# Learning Objectives

This project explores:

* reverse-mode autodiff mechanics
* manual gradient propagation
* CUDA kernel programming
* neural network training internals
* Python ↔ C++ interoperability
* GPU execution fundamentals

The objective is to understand how modern deep learning systems work underneath high-level frameworks.
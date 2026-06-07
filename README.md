# backprop-system-study
This repo explores the efficiency of Pytorch with simple python and custom CUDA backpropagation.

## Following sections have been completed:

### MLP-
This directory explores the working of simple Multi Layer perceptron.
- mlp.py : pytorch's version
- raw_mlp.py : only python, mostly numpy, version of the MLP. Pytorch is not utilized here. The correctness of gradients have been verified comparing backprop with finite differnce method. Currently it is very unoptimized. Optimizations to be done later.
- cuda_mlp/ : contains the code base for cuda implementation of MLP computation. Backpropagation and Feed Forward paas are all implemented in CUDA/C++.

### tiny_CNN-
This directory explores the working of the CNN. The architecture in consideration is shallow.
- pytorch_cnn.py : pytorch implementation of CNN.
- raw_cnn.py : Numpy implementation of CNN. Work in progress....
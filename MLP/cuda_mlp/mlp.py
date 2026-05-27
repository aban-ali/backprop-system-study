# This file contains the raw CUDA + Python implementation of 
# multi layer percetron, i.e. without using pytorch and autograd.
# The architecture of MLP remain same as - 
# Inputs->Linear->ReLU->Linear->ReLU->Linear

from MLP.cuda_mlp.mlp import NeuralNet

import numpy as np
import random



def load_data(train=True):
    """Extract the data from the binary files downloaded through
    torchvision's datasets module. This also separates image data from
    respective labels and train data from test data.
    
    """
    filename = "train-images-idx3-ubyte" if train else "t10k-images-idx3-ubyte"
    label_file = "train-labels-idx1-ubyte" if train else "t10k-labels-idx1-ubyte"
    n = int(6e4) if train else int(1e4)
    images = []
    labels = []

    f = open(f"./dataset/MNIST/raw/{filename}", "rb")
    l = open(f"./dataset/MNIST/raw/{label_file}", "rb")

    f.read(16)
    l.read(8)

    for _ in range(n):
        labels.append(ord(l.read(1)))
        image = []
        for __ in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels)

    data = np.column_stack((images, labels))
    np.random.shuffle(data)

    return data


def main():
    nn_size = np.array([784, 256, 128, 10], dtype=np.int32)
    net = NeuralNet(sizes=nn_size)
    data = load_data()
    net.SGD(data)

    test_data = load_data(train=False)
    net.validate_model(test_data)

    del net


if __name__ == "__main__":
    main()
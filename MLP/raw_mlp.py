# This file contains the raw implementation of 
# multi layer percetron, i.e. without using pytorch and autograd.
# The architecture of MLP remain same as - 
# Inputs->Linear->ReLU->Linear->ReLU->Linear

import numpy as np
import random

class CrossEntropyCost():
    @staticmethod
    def func(a, y):
        pass

    @staticmethod
    def func_delta():
        pass


class HelperFunc():

    @staticmethod
    def relu(z):
        """ReLU Value."""
        return np.maximum(z, 0)
    
    @staticmethod
    def relu_prime(z):
        """Derivative of ReLU."""
        return (z > 0).astype(float)


class NeuralNet():

    def __init__(self, sizes, cost=CrossEntropyCost) -> None:
        """The parameter `sizes` is a list that contains the number of neurons
        in each layer of NN. The weights and biases are initialized accordingly.
        The parameter `cost` takes in the cost function for calculating the Loss.

        """
        self.layers = len(sizes)
        self.sizes = sizes
        self.weights_init()
        self.cost = cost
    
    def weights_init(self):
        """Random initialization of weights and biases according to 
        the `sizez` parameter. 
        
        """
        self.biases = [ np.random.randn(size, 1) for size in self.sizes[1:] ]
        self.weights = [ np.random.randn(y, x) 
                                for x, y in zip(self.sizes[:-1], self.sizes[1:]) ]
        
    def feedforward(self, x):
        """Calculate and return the value obtained 
        from Neural Network after a forward pass.
        
        """
        for w,b in zip(self.weights[:-1], self.biases[:-1]):
            x = HelperFunc.relu( np.dot(w, x) + b )
        x = np.dot(self.weights[-1], x) + self.biases[-1]
        return x

    def SGD(self, data, epochs=3, batch_size=20, lr=0.005):
        """"""
        for epoch in range(epochs):
            print(f"Epoch {epoch} started....")
            images, labels = data
            size = len(images)
            mini_batches = []

            for i in range(0, size, batch_size):
                mini_batches.append((images[i:i+batch_size], labels[i:i+batch_size]))

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)

    def update_mini_batch(self, mini_batch, lr=0.005):
        images, labels = mini_batch
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in zip(images, labels):
            delta_w, delta_b = self.backprop(x, y)
            grad_w += delta_w
            grad_b += delta_b
        self.weights -= ( lr / len(mini_batch) ) * grad_w
        self.biases -= ( lr / len(mini_batch) ) * grad_b
        
    def backprop(self, x, y):
        delta_w = [ np.zeros(w.shape) for w in self.weights ]
        delta_b = [ np.zeros(b.shape) for b in self.biases ]



    def validate_model(self):
        pass






def load_data(train=True):
    """Extract the """
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

    return (np.array(images), np.array(labels))



def main():
    nn_size = [784, 256, 128, 10]
    net = NeuralNet(sizes=nn_size)
    pass



if __name__ == "__main__":
    main()

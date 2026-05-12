# This file contains the raw implementation of 
# multi layer percetron, i.e. without using pytorch and autograd.
# The architecture of MLP remain same as - 
# Inputs->Linear->ReLU->Linear->ReLU->Linear->Softmax

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
    def sigmoid(z):
        """Sigmoid Value."""
        return 1.0 / ( 1.0 + np.exp(-z) )
    
    @staticmethod
    def sigmoid_prime():
        """Derivative of sigmoid."""
        pass




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
        self.weights = [ np.random.randn(x, y) 
                                for x, y in zip(self.sizes[:-1], self.sizes[1:]) ]
        
    def feedforward(self, x):
        """Calculate and return the value obtained 
        from Neural Network after a forward pass.
        
        """
        for w,b in zip(self.weights, self.biases):
            x = HelperFunc.sigmoid( np.dot(w, x) + b )
        return x

    def SGD(self):
        pass

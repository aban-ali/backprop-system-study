# This file contains the raw implementation of 
# multi layer percetron, i.e. without using pytorch and autograd.
# The architecture of MLP remain same as - 
# Inputs->Linear->ReLU->Linear->ReLU->Linear

import numpy as np
import random

class CrossEntropyCost():
    @staticmethod
    def func(z, y):
        """Calculates Cross Entropy Loss. 
        Implemented using fused kernel of Cross Entropy and Softmax.
        It uses logits from the models calculation.
        
        """
        max_val = np.max(z)
        log_softmax = max_val + np.log(np.sum(np.exp(z - max_val)))
        return log_softmax - z[y, 0]

    @staticmethod
    def func_delta(z, y):
        """Calculates the derivative of the fused kernel 
        of Cross Entropy and Softmax.
        
        """
        shifted_z = z - np.max(z)
        probs = np.exp(shifted_z) / np.sum(np.exp(shifted_z))
        probs[y] -= 1
        return probs

class Activation():
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
        self.weights = [ np.random.randn(y, x) * np.sqrt(2/x)
                                for x, y in zip(self.sizes[:-1], self.sizes[1:]) ]
        
    def feedforward(self, x):
        """Calculate and return the value obtained 
        from Neural Network after a forward pass.
        
        """
        for w,b in zip(self.weights[:-1], self.biases[:-1]):
            x = Activation.relu( np.dot(w, x) + b )
        x = np.dot(self.weights[-1], x) + self.biases[-1]
        return x

    def SGD(self, data, epochs=3, batch_size=32, lr=0.005):
        """Calculates the gradients and updates the weights and
        biases of neural network. 
        
        """
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
        """Updates the weights and biases of the model
        for every batch. It implement simple SGD model.
        
        """
        images, labels = mini_batch
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in zip(images, labels):
            delta_w, delta_b = self.backprop(x, y)
            grad_w = [ gw + dw for gw, dw in zip(grad_w, delta_w) ]
            grad_b = [ gb + db for gb, db in zip(grad_b, delta_b) ]

        self.weights = [ w - ( lr / len(images) ) * gw
                        for w, gw in zip(self.weights, grad_w) ]
        self.biases = [ b - ( lr / len(images) ) * gb
                        for b, gb in zip(self.biases, grad_b) ]
        
    def backprop(self, x, y):
        """Implements the backpropogation for the model.
        Computes the gradient of a single vaule as of now.
        
        """
        delta_w = [ np.zeros(w.shape) for w in self.weights ]
        delta_b = [ np.zeros(b.shape) for b in self.biases ]
        
        activation = x.reshape(-1, 1)
        activations = [activation]
        zs = []
        for w,b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = Activation.relu(z)
            activations.append(activation)
        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z)
        activations.append(z)

        
        delta = CrossEntropyCost.func_delta(activations[-1], y)
        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.layers):
            z = zs[-l]
            rp = Activation.relu_prime(z)
            delta = np.dot(self.weights[-l+1].T, delta) * rp
            delta_b[-l] = delta
            delta_w[-l] = np.dot(delta, activations[-l-1].T)
        return (delta_w, delta_b)



    def validate_model(self, test_data):
        """Model's correctness can be validated from here after training is done.
        It will print out the total average loss occured for test data and 
        the accuracy of model for each label.
        
        """
        images, labels = test_data
        count = {}
        correct_preds = {}
        total_loss = 0

        for x, y in zip(images, labels):
            x = x.reshape(-1, 1)
            count[y] = count.get(y, 0) + 1

            x = self.feedforward(x)
            label = np.argmax(x)
            if label == y:
                correct_preds[y] = correct_preds.get(y, 0) + 1
            
            total_loss += CrossEntropyCost.func(x, y)
        
        print(f"Total Average Loss = {total_loss/len(images)}")
        for i in range(10):
            pred = correct_preds.get(i, 0)
            c = count[i]
            print(f"Label:{i}\t Correct/Total predictions: {pred}/{c}\t Accuracy:{(pred*100.0/c):.2f}%")



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

    return (
        np.array(images, dtype=np.float32) / 255.0, 
        np.array(labels)
    )



def main():
    nn_size = [784, 256, 128, 10]
    net = NeuralNet(sizes=nn_size)
    data = load_data()
    net.SGD(data)

    test_data = load_data(train=False)
    net.validate_model(test_data)
    

if __name__ == "__main__":
    main()
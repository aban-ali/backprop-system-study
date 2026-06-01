# This file contains the raw CUDA + Python implementation of 
# multi layer percetron, i.e. without using pytorch and autograd.
# The architecture of MLP remain same as - 
# Inputs->Linear->ReLU->Linear->ReLU->Linear

from MLP.cuda_mlp.cuda_mlp import NeuralNet

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
    data = []

    f = open(f"./dataset/MNIST/raw/{filename}", "rb")
    l = open(f"./dataset/MNIST/raw/{label_file}", "rb")

    f.read(16)
    l.read(8)

    for k in range(n):
        data_point = []
        for __ in range(28*28):
            data_point.append(ord(f.read(1)) / 255.0)
        data_point.append(ord(l.read(1)))
        data.append(data_point)
        if k==3: break

    data = np.array(data, dtype=np.float32)
    print("Loaded data of shape: ", data.shape)
    np.random.shuffle(data)

    return data


def SGD(model, data, epochs=3, batch_size=32, lr=0.005):
    """Orchestrates the backpropogation performed in CUDA.
    Acts as a helper function.
    
    """
    for epoch in range(epochs):
        # print(f"Epoch {epoch} started...")

        size = len(data)
        mini_batches = []

        for i in range(0, size, batch_size):
            mini_batches.append(data[i:i+batch_size])
        
        for mini_batch in mini_batches:
            model.update_miniBatch(mini_batch, lr)
     

def validate_model(model, test_data):
    """Model's correctness can be validated from here after training is done.
    It will print out the total average loss occured for test data and 
    the accuracy of model for each label.
    
    """
    count = {}
    correct_preds = {}
    total_loss = 0

    for data in test_data:
        x = data[:-1]
        y = data[-1]

        count[y] = count.get(y, 0) + 1
        logits = model.feedforward(x)
        label = np.argmax(logits)
        if label == y:
            correct_preds[y] = correct_preds.get(y, 0) + 1

        total_loss += crossEntropyLoss(logits, y)

    print(f"Total Average Loss = {total_loss/len(test_data)}")
    for i in range(10):
        pred = correct_preds.get(i, 0)
        c = count[i]
        print(f"Label:{i}\t Correct/Total predictions: {pred}/{c}\t Accuracy:{(pred*100.0/c):.2f}%")


def crossEntropyLoss(logits, y):
    """Implements standard fused Cross Entropy Loss,
    i.e. Softmax + Cross Entropy
    
    """
    max_val = np.max(logits)
    log_softmax = max_val + np.log(np.sum(np.exp(logits - max_val)))
    return log_softmax - logits[y]


def main():
    nn_size = np.array([784, 256, 128, 10], dtype=np.int32)
    net = NeuralNet(nn_size)
    data = load_data()
    SGD(net, data)

    test_data = load_data(train=False)
    validate_model(test_data)

    del net

if __name__ == "__main__":
    main()



nn_size = np.array([784, 256, 128, 10], dtype=np.int32)
net = NeuralNet(nn_size)
data = load_data()
d = data[0].reshape(1, -1)
x,y = d[:, :-1], d[:, -1]
print(x.shape, y[0])
print("Loss: \n\t", crossEntropyLoss(net.feedforward(x), int(y[0])))
print("Logits:\n\t", net.feedforward(x))
print("="*50)
for _ in range(10):
    SGD(net, d, 1, 1)
    # print("="*50)
    loss = crossEntropyLoss(net.feedforward(x), int(y[0]))
    print("Loss: \t", loss)
    # if(loss == 0):
    #     break
print("Logits:\n\t", net.feedforward(x))

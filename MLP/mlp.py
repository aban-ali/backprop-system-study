# This file contains the pytorch implementation
# of simple multi layer perceptron architecture as follows - 
# Inputs->Linear->ReLU->Linear->ReLU->Linear

from torchvision import datasets
from torchvision import transforms

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

def model_init(device):
    """Initialize the Neural Network, Loss function
    and SGD optimizer.
    """
    net = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005)

    return net, criterion, optimizer

def load_data(train=True):
    """Load the training data with batch size 20,
    and load testing data with batch size 1 (helps in calculating 
    total loss and per label accuracy)
    """
    batch_size = 32 if train else 1

    dataset = datasets.MNIST(root="./dataset", train=train, 
                        download=False, transform=transforms.ToTensor())
    data = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data

def start_training(net, criterion, optimizer, data, device, epochs=3):
    """This function trains the Neural Net with simple hyperparamets."""
    for epoch in range(epochs):

        print(f"Epoch {epoch+1} started....")
        running_loss = 0

        for i, (images, labels) in enumerate(data):
            images = images.view(images.shape[0], -1).to(device)

            optimizer.zero_grad()
            preds = net(images)
            loss = criterion(preds, labels.to(device))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                print(f"Loss at {i+1}th iteration is: {running_loss/1000.0}")
                running_loss = 0
    
    return net

def validate_model(model, data, criterion, device):
    """Here we validate Neural Net model on test data."""
    count = {}
    correct_preds = {}
    total_loss = 0

    with torch.no_grad():
        for image, label in data:
            image = image.view(-1).to(device)
            label = label.to(device)
            count[label.item()] = count.get(label.item(), 0) + 1

            pred = model(image)
            total_loss += criterion(pred, label).item()
            pred_val = torch.argmax(pred).item()

            if pred_val == label.item():
                correct_preds[label.item()] = correct_preds.get(label.item(), 0) + 1

    print(f"Total Average Loss = {total_loss/len(data[0])}")
    for i in range(10):
        pred = correct_preds[i]
        c = count[i]
        print(f"Label:{i}\t Correct/Total predictions: {pred}/{c}\t Accuracy:{(pred*100.0/c):.2f}%")
            



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net, criterion, optimizer = model_init(device)
    data = load_data(train=True)
    net = start_training(net, criterion, optimizer, data, device)
    data = load_data(train=False)
    validate_model(net, data, criterion, device)

if __name__ == "__main__":
    main()
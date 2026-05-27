# This file contains the pytorch implementation
# of TinyCNN. Its architecture is as follows: 
# Inputs->Conv2D->ReLU->MaxPool->Conv2D->ReLU->MaxPool->(Flatten)->Linear->ReLU->Linear

from torchvision import datasets
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def model_init(device):
    """Initialize the Convolutional Neural Network,
    Loss function and SGD optimizer.
    """
    cnn = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.05)

    return cnn, criterion, optimizer


def load_data(train=True):
    """Load the training data with batch size 32,
    and load testing data with batch size 1 (helps in calculating 
    total loss and per label accuracy)
    """
    batch_size = 32 if train else 1

    dataset = datasets.CIFAR10(root="./dataset", train=train, 
                        download=False, transform=transforms.ToTensor())
    data = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data

def start_training(
        model, criterion, optimizer, 
        data, device, epochs=3
                   ):
    """This function trains the CNN model with simple hyperparameters."""
    for epoch in range(epochs):

        print(f"Epoch {epoch+1} started...")
        running_loss = 0

        for i, (images, labels) in enumerate(data):
            images = images.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels.to(device))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i%100 == 99:
                print(f"Loss at {i+1}th iteration is: {running_loss/1000.0}")
                running_loss = 0

    return model



def validate_model(model, data, criterion, device):
    """Here we validate CNN model on test data."""
    count = {}
    correct_preds = {}
    total_loss = 0

    with torch.no_grad():
        for image, label in data:
            image = image.to(device)
            label = label.to(device)
            count[label.item()] = count.get(label.item(), 0) + 1

            pred = model(image)
            total_loss += criterion(pred, label).item()
            pred_val = torch.argmax(pred).item()

            if pred_val == label.item():
                correct_preds[label.item()] = correct_preds.get(label.item(), 0) + 1

    print(f"Total Average Loss = {total_loss/len(data.dataset)}")
    for i in range(10):
        pred = correct_preds.get(i, 0)
        c = count[i]
        print(f"Label:{i}\t Correct/Total predictions: {pred}/{c}\t Accuracy:{(pred*100.0/c):.2f}%")
            



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cnn, criterion, optimizer = model_init(device)

    train_data = load_data()
    test_data = load_data(train = False)
    
    cnn = start_training(cnn, criterion, optimizer, train_data, device)
    validate_model(cnn, test_data, criterion, device)



if __name__ == "__main__":
    main()
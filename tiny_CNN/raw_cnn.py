# This file contains the raw python implementation
# of TinyCNN. Its architecture is as follows: 
# Inputs->Conv2D->ReLU->MaxPool->Conv2D->ReLU->MaxPool->(Flatten)->Linear->ReLU->Linear

import numpy as np
import pickle
from os import path


class CNN():
    def __init__(self):
        pass

class Dataset():
    def __init__(self):
        self.classes = []
        self.metadata_filepath = path.join("dataset", "cifar-10-batches-py", "batches.meta")
        self.trainData_filepath = path.join("dataset", "cifar-10-batches-py", "data_batch_")
        self.num_trainData_batch = 5
        self.testData_filepath = path.join("dataset", "cifar-10-batches-py", "test_batch")

    async def get_data_classes(self):
        """Gets the classes of images"""

        with open(self.metadata_filepath, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        
        self.classes = list(map(lambda x: x.decode(), batch[b'label_names']))

    def load_trainData(self):
        """Loads the training data from 5 binary files."""

        all_images = []
        all_labels = []
        self.get_data_classes()

        for i in range(1, self.num_trainData_batch):
            with open(self.trainData_filepath+i, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
            
            images = batch[b'data']
            labels = batch[b'labels']

            images = images.reshape(-1, 3, 32, 32)

            all_images.append(images)
            all_labels.extend(labels)

        return (
            np.concatenate(all_images, axis=0), np.array(all_labels)
            )
    
    def load_testData(self):
        """Loads the test data from the binary file."""
        with open(self.testData_filepath, "rb") as f:
            data = pickle.load(f, encoding="bytes")
        
        images = data[b'data']
        labels = data[b'labels']

        images = images.reshape(-1, 3, 32, 32)

        return (
            images, np.array(labels)
        )

def main():
    # # size layout = [ Conv_layer:(in_channels, out_channels, kernel_size, padding), 
    # #                 Linear:(in_dim, out_dim) ]
    # cnn_size = [(3, 32, 3, 1), (32, 64, 3, 1), ]
    
    dataset = Dataset()

if __name__ == "__main__":
    main()
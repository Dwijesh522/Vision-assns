import cv2
import os
from os import path
import csv
from os import listdir
from os.path import isfile, join
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import random
import shutil
import matplotlib.pyplot as plt 

class customDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.label.iloc[idx, 0])
        # w * h
        image = cv2.imread(img_name,0)
        labels = self.label.iloc[idx, 1]
        labels = np.array([labels])
        labels = labels.astype('float').reshape(-1, 1)
        sample = {'image': image, 'label': labels}

        if self.transform:
            sample = self.transform(sample)
        
        # label size must have been compatible with output of the network
        image = np.expand_dims(sample['image'], axis=0)
        return image, labels[0][0]
        # return sample

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.bn1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3, 7, 3)
        self.bn2 = nn.BatchNorm2d(7)
        self.conv3 = nn.Conv2d(7, 12, 3)
        self.bn3 = nn.BatchNorm2d(12)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(12 * 4 * 4, 40)  
        # self.bn4 = nn.BatchNorm1d(40)
        self.dr1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(40, 3)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(torch.relu(self.bn1(self.conv1(x))), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(torch.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(torch.relu(self.bn3(self.conv3(x))), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = torch.relu(self.dr1(self.fc1(x)))
        x = self.fc2(x)
        x = self.soft(x)
        return x

    def num_flat_features(self, x):
        # print(x.size())
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        # print(num_features)
        return num_features

def train_network(net, dataloader, testloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=0.00621, centered = True)
    mini_batch = 1559
    loss_values = []
    training_loss = []
    validation_loss = []
    epochs = []
    for epoch in range(25):  #28 loop over the dataset multiple times

        running_loss = 0.0
    
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # make the parameter gradients zero
            optimizer.zero_grad()

            # forward + backward + optimize
            inputs = inputs.float()
            outputs = net(inputs)

            labels = labels.type(torch.LongTensor)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
        training_loss.append((running_loss*32)/2328)                                                    ####
        print(" training loss: " + str((running_loss*32)/2328))                                         ####

        running_loss = 0
        # validating on test data set
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # make the parameter gradients zero
                optimizer.zero_grad()

                # forward + backward + optimize
                inputs = inputs.float()
                outputs = net(inputs)

                labels = labels.type(torch.LongTensor)
                loss = criterion(outputs, labels)

                # print statistics
                running_loss += loss.item()
        validation_loss.append((running_loss*32)/503)                                                   ####
        epochs.append(epoch)
        print("validation loss: " + str((running_loss*32)/503))                                         ####
        print("epoch " + str(epoch) + " completed...")
    print('Finished Training...')
    # drawing plot for training and validation loss
    plt.plot(epochs, training_loss, label = "Training loss")
    plt.plot(epochs, validation_loss, label = "Validation loss")
    # naming the x axis 
    plt.xlabel('epochs') 
    # naming the y axis 
    plt.ylabel('loss') 
    # show a legend on the plot 
    plt.legend() 
    plt.savefig('loss.png')
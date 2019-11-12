import cv2
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
import random
# -------------------------------
import video_to_dataset
import cnn
#--------------------------------
if __name__ == '__main__':
    random.seed(711)
    # getting the path to three types of videos
    print("\ndefault path: /home/dwijesh/Documents/sem5/vision/assns/assn4_data/videos/\n")
    path = input("Enter the path to stop, prev, next folders: ")
    video_to_dataset.store_dataset_from_video(path, "train_dataset.csv", "train_dataset", 15)
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
    custom_dataset = cnn.customDataset(csv_file='train_dataset/train_dataset.csv', 
                                    root_dir='/home/dwijesh/Documents/sem5/vision/assns/Vision-assns/assn4/train_dataset/')
    # in each batch there will be "batch_size" number of elements: image1, image2, ... imagek -> label1, label2,... labelk     if batch_size = k
    # train_loader = DataLoader(dataset=custom_dataset, batch_size=3, shuffle=True)
    net = cnn.Net()
    # print(net)
    cnn.train_network(net, custom_dataset)
    
    # storing the network
    path = './lenet.pth'
    torch.save(net.state_dict(), path)
    # calculating the training accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in custom_dataset:
            # images, labels = data
            images = data['image']
            labels = data['label']
            # labels is 2D matrix
            label_val = labels[0][0]
            images = images.float()
            outputs = net(images)
            # predicted will be a tensor
            _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            total += label_val
            # extracting the value of the tensor for comparision
            predicted_value = (predicted.data.cpu().numpy()[0])
            correct += (predicted_value == label_val).sum().item()

    print('Accuracy of the network on the train images: %d %%' % (100 * correct / total))

    # adding the test dataset
    print("\ndefault path: /home/dwijesh/Documents/sem5/vision/assns/assn4_data/test_videos/\n")
    test_dataset_path = input("Enter the path to test_dataset folders: ")
    video_to_dataset.store_dataset_from_video(test_dataset_path, "test_dataset.csv", "test_dataset", 20)
    custom_dataset = cnn.customDataset(csv_file='test_dataset/test_dataset.csv', 
                                    root_dir='/home/dwijesh/Documents/sem5/vision/assns/Vision-assns/assn4/test_dataset/')
    
    # calculating the test accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in custom_dataset:
            # images, labels = data
            images = data['image']
            labels = data['label']
            # labels is 2D matrix
            label_val = labels[0][0]
            images = images.float()
            outputs = net(images)
            # predicted will be a tensor
            _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            total += label_val
            # extracting the value of the tensor for comparision
            predicted_value = (predicted.data.cpu().numpy()[0])
            correct += (predicted_value == label_val).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
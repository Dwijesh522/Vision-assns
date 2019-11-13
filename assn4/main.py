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
import cnn_bn_do
import inference
#--------------------------------
if __name__ == '__main__':
    random.seed(711)
    # getting the path to three types of videos
    # print("\ndefault path: /home/dwijesh/Documents/sem5/vision/assns/assn4_data/videos/\n")
    # path = input("Enter the path to stop, prev, next folders: ")
    # video_to_dataset.store_dataset_from_video(path, "train_dataset.csv", "train_dataset", 15)
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
    custom_dataset = cnn_bn_do.customDataset(csv_file='train_dataset/train_dataset.csv', 
                                    root_dir='/home/dwijesh/Documents/sem5/vision/assns/Vision-assns/assn4/train_dataset/')
    # in each batch there will be "batch_size" number of elements: image1, image2, ... imagek -> label1, label2,... labelk     if batch_size = k
    train_loader = DataLoader(dataset=custom_dataset, batch_size=32, shuffle=True)
    # test loader
    test_dataset = cnn_bn_do.customDataset(csv_file='test_dataset/test_dataset.csv', 
                                    root_dir='/home/dwijesh/Documents/sem5/vision/assns/Vision-assns/assn4/test_dataset/')
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)
    #########################
    it = iter(train_loader)
    image, label = it.next()
    #########################
    net = cnn_bn_do.Net()
    cnn_bn_do.train_network(net, train_loader, test_loader)
    
    # storing the network
    path = './lenet.pth'
    torch.save(net.state_dict(), path)
    # calculating the training accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_loader:
            images, labels = data
            images = images.float()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # print(predicted)
            # print(labels.long())
            # exit()
            labels = labels.long()
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the train images: %d %%' % (
    100 * correct / total))

    # adding the test dataset
    # print("\ndefault path: /home/dwijesh/Documents/sem5/vision/assns/assn4_data/test_videos/\n")
    # test_dataset_path = input("Enter the path to test_dataset folders: ")
    # video_to_dataset.store_dataset_from_video(test_dataset_path, "test_dataset.csv", "test_dataset", 15)
    # calculating test accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.float()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # print(predicted)
            # print(labels.long())
            # exit()
            labels = labels.long()
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

    # real time inference
    inference.infer(net)
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
        # 1 * w * h
        image = image[np.newaxis,:,:]
        # 1 * 1 * w * h
        image = image[np.newaxis,:,:,:]
        # numpy array to long tensor
        image = torch.from_numpy(image).long()
        labels = self.label.iloc[idx, 1]
        labels = np.array([labels])
        labels = labels.astype('float').reshape(-1, 1)
        sample = {'image': image, 'label': labels}

        if self.transform:
            sample = self.transform(sample)

        # return sample['image'], sample['label']
        return sample
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.conv2 = nn.Conv2d(3, 6, 3)
        self.conv3 = nn.Conv2d(6, 10, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(10 * 4 * 4, 40)  
        self.fc2 = nn.Linear(40, 3)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(torch.sigmoid(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(torch.sigmoid(self.conv2(x)), 2)
        x = F.max_pool2d(torch.sigmoid(self.conv3(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = torch.sigmoid(self.fc1(x))
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

def train_network(net, custom_dataset):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.Adadelta(net.parameters(), lr=0.01)
    mini_batch = 1405
    loss_values = []
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        # rand_num = random.randrange(30,100,3)        
        for i, data in enumerate(custom_dataset, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['image']
            labels = data['label']
            if(labels[0][0] == 0):
                ans = torch.zeros([1], dtype = torch.long)
                ans[0] = 0
            elif(labels[0][0] == 1):
                ans = torch.zeros([1], dtype = torch.long)
                ans[0] = 1
            else:
                ans = torch.zeros([1], dtype = torch.long)
                ans[0] = 2
            inputs = inputs.float()
            # print(ans)
            # break
            # make the parameter gradients zero
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, ans)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % mini_batch == mini_batch-1:    # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / mini_batch))
                loss_values.append(running_loss/mini_batch)
                running_loss = 0.0
        print("epoch " + str(epoch) + " completed...")
    plt.plot(loss_values)
    # print(loss_values)
    print('Finished Training...')



if __name__ == '__main__':
    random.seed(711)
    # getting the path to three types of videos
    # print("\ndefault path: /home/dwijesh/Documents/sem5/vision/assns/assn4_data/videos/\n")
    # path = input("Enter the path to stop, prev, next folders: ")
    # stop_path = path + "stop/"
    # next_path = path + "next/"
    # prev_path = path + "prev/"
    
    # stop_video_names = [f for f in listdir(stop_path) if isfile(join(stop_path, f))]
    # prev_video_names = [f for f in listdir(prev_path) if isfile(join(prev_path, f))]
    # next_video_names = [f for f in listdir(next_path) if isfile(join(next_path, f))]

    # # changing the dir
    # if(os.path.isdir("train_dataset")):
    #     shutil.rmtree("train_dataset")

    # os.mkdir("train_dataset")
    # os.chdir("train_dataset")

    # # deleting csv file if already exists
    # if(os.path.exists("training_dataset.csv")):
    #     os.remove("training_dataset.csv")

    # # printing header row in csv file
    # # printing into csv file
    # row = ["image_names", "label"]
    # with open('training_dataset.csv', 'a') as csvFile:
    #     writer = csv.writer(csvFile)
    #     writer.writerow(row)

    # # iterating over sstop video frames
    # i=0
    # image_count = 0
    # while(i < len(stop_video_names)):
    #     cap = cv2.VideoCapture(stop_path + stop_video_names[i])
    #     fgbg = cv2.createBackgroundSubtractorKNN(5000, 400, False)
    #     ret = 1
    #     while(ret):
    #         ret, frame = cap.read()
    #         if(not ret): break
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         canny_frame = cv2.Canny(frame,20, 50, apertureSize = 3)
    #         canny_frame = cv2.resize(canny_frame, (50, 50), interpolation = cv2.INTER_AREA)
    #         if(image_count%15 == 0):
    #             rand1 = random.randint(0, 1000000)
    #             rand2 = random.randint(0, 1000000)
    #             cv2.imwrite(str(rand1)+ "_" + str(rand2) + ".jpg", canny_frame)
    #             # printing into csv file
    #             row = [str(rand1)+ "_" + str(rand2) + ".jpg", str(1)]
    #             # cv2.imwrite("stop_" + str(image_count) + ".jpg", canny_frame)
    #             with open('training_dataset.csv', 'a') as csvFile:
    #                 writer = csv.writer(csvFile)
    #                 writer.writerow(row)
    #         image_count += 1
    #     i += 1
    # # iterating over next video frames
    # i=0
    # image_count = 0
    # while(i < len(next_video_names)):
    #     cap = cv2.VideoCapture(next_path + next_video_names[i])
    #     fgbg = cv2.createBackgroundSubtractorKNN(5000, 400, False)
    #     ret = 1
    #     while(ret):
    #         ret, frame = cap.read()
    #         if(not ret): break
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         canny_frame = cv2.Canny(frame,20, 50, apertureSize = 3)
    #         canny_frame = cv2.resize(canny_frame, (50, 50), interpolation = cv2.INTER_AREA)
    #         if(image_count%15 == 0):
    #             rand1 = random.randint(0, 1000000)
    #             rand2 = random.randint(0, 1000000)
    #             cv2.imwrite(str(rand1)+ "_" + str(rand2) + ".jpg", canny_frame)
    #             # printing into csv file
    #             row = [str(rand1)+ "_" + str(rand2) + ".jpg", str(2)]
    #             with open('training_dataset.csv', 'a') as csvFile:
    #                 writer = csv.writer(csvFile)
    #                 writer.writerow(row)
    #         image_count += 1
    #     i += 1
    # # iterating over prev video frames
    # i=0
    # image_count = 0
    # while(i < len(prev_video_names)):
    #     cap = cv2.VideoCapture(prev_path + prev_video_names[i])
    #     fgbg = cv2.createBackgroundSubtractorKNN()
    #     ret = 1
    #     while(ret):
    #         ret, frame = cap.read()
    #         if(not ret): break
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         canny_frame = cv2.Canny(frame,20, 50, apertureSize = 3)
    #         canny_frame = cv2.resize(canny_frame, (50, 50), interpolation = cv2.INTER_AREA)
    #         if(image_count%15 == 0):
    #             rand1 = random.randint(0, 1000000)
    #             rand2 = random.randint(0, 1000000)
    #             cv2.imwrite(str(rand1)+ "_" + str(rand2) + ".jpg", canny_frame)
    #             # printing into csv file
    #             row = [str(rand1)+ "_" + str(rand2) + ".jpg", str(0)]
    #             # cv2.imwrite("prev_" + str(image_count) + ".jpg", canny_frame)
    #             # printing into csv file
    #             # row = ["prev_" + str(image_count) + ".jpg", str(0)]
    #             with open('training_dataset.csv', 'a') as csvFile:
    #                 writer = csv.writer(csvFile)
    #                 writer.writerow(row)
    #         image_count += 1
    #     i += 1
    # #shuffling the csv file

    # df = pd.read_csv('training_dataset.csv')
    # df = df.sample(frac = 1).reset_index(drop = True)

    # export_csv = df.to_csv(r'train_dataset.csv', index = None, header = True)
    # # coming out of dir
    # os.chdir("..")
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
    custom_dataset = customDataset(csv_file='train_dataset/train_dataset.csv', 
                                    root_dir='/home/dwijesh/Documents/sem5/vision/assns/Vision-assns/assn4/train_dataset/')
    # in each batch there will be "batch_size" number of elements: image1, image2, ... imagek -> label1, label2,... labelk     if batch_size = k
    # train_loader = DataLoader(dataset=custom_dataset, batch_size=3, shuffle=True)
    net = Net()
    # print(net)
    train_network(net, custom_dataset)
    
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
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

class customDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
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
#         print(img_name)
        image = cv.imread(img_name)
        labels = self.label.iloc[idx, 1]
        labels = np.array([labels])
        labels = labels.astype('float').reshape(-1, 1)
        sample = {'image': image, 'label': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == '__main__':
    # getting the path to three types of videos
    print("\ndefault path: /home/dwijesh/Documents/sem5/vision/assns/assn4_data/videos/\n")
    path = input("Enter the path to stop, prev, next folders: ")
    stop_path = path + "stop/"
    next_path = path + "next/"
    prev_path = path + "prev/"
    
    stop_video_names = [f for f in listdir(stop_path) if isfile(join(stop_path, f))]
    prev_video_names = [f for f in listdir(prev_path) if isfile(join(prev_path, f))]
    next_video_names = [f for f in listdir(next_path) if isfile(join(next_path, f))]

    # changing the dir
    if(os.path.isdir("train_dataset")):
        os.chdir("train_dataset")
    else:
        os.mkdir("train_dataset")
        os.chdir("train_dataset")

    # deleting csv file if already exists
    if(os.path.exists("training_dataset.csv")):
        os.remove("training_dataset.csv")

    # printing header row in csv file
    # printing into csv file
    row = ["image_names", "label"]
    with open('training_dataset.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)

    # iterating over sstop video frames
    i=0
    image_count = 0
    while(i < len(stop_video_names)):
        cap = cv2.VideoCapture(stop_path + stop_video_names[i])
        fgbg = cv2.createBackgroundSubtractorKNN(5000, 400, False)
        ret = 1
        while(ret):
            ret, frame = cap.read()
            if(not ret): break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            canny_frame = cv2.Canny(frame,20, 50, apertureSize = 3)
            canny_frame = cv2.resize(canny_frame, (50, 50), interpolation = cv2.INTER_AREA)
            if(image_count%15 == 0):
                cv2.imwrite("stop_" + str(image_count) + ".jpg", canny_frame)
                # printing into csv file
                row = ["stop_" + str(image_count) + ".jpg", str(1)]
                with open('training_dataset.csv', 'a') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
            image_count += 1
        i += 1
    # iterating over next video frames
    i=0
    image_count = 0
    while(i < len(next_video_names)):
        cap = cv2.VideoCapture(next_path + next_video_names[i])
        fgbg = cv2.createBackgroundSubtractorKNN(5000, 400, False)
        ret = 1
        while(ret):
            ret, frame = cap.read()
            if(not ret): break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            canny_frame = cv2.Canny(frame,20, 50, apertureSize = 3)
            canny_frame = cv2.resize(canny_frame, (50, 50), interpolation = cv2.INTER_AREA)
            if(image_count%15 == 0):
                cv2.imwrite("next_" + str(image_count) + ".jpg", canny_frame)
                # printing into csv file
                row = ["next_" + str(image_count) + ".jpg", str(2)]
                with open('training_dataset.csv', 'a') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
            image_count += 1
        i += 1
    # iterating over prev video frames
    i=0
    image_count = 0
    while(i < len(prev_video_names)):
        cap = cv2.VideoCapture(prev_path + prev_video_names[i])
        fgbg = cv2.createBackgroundSubtractorKNN()
        ret = 1
        while(ret):
            ret, frame = cap.read()
            if(not ret): break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            canny_frame = cv2.Canny(frame,20, 50, apertureSize = 3)
            canny_frame = cv2.resize(canny_frame, (50, 50), interpolation = cv2.INTER_AREA)
            if(image_count%15 == 0):
                cv2.imwrite("prev_" + str(image_count) + ".jpg", canny_frame)
                # printing into csv file
                row = ["prev_" + str(image_count) + ".jpg", str(0)]
                with open('training_dataset.csv', 'a') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
            image_count += 1
        i += 1

    # coming out of dir
    os.chdir("..")
    custom_dataset = customDataset(csv_file='train_dataset/training_dataset.csv', 
                                    root_dir='/home/dwijesh/Documents/sem5/vision/assns/Vision-assns/assn4/train_dataset/')
    print(len(custom_dataset))

#Try Canny Edge for plain background!

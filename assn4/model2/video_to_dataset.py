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
import preprocessing

# given a path to videos extract the frames and apply the preprocessing to the images and store it in a given directory
def store_dataset_from_video(path, csv_file_name,store_it_here, one_of_how_many):
    stop_path = path + "stop/"
    next_path = path + "next/"
    prev_path = path + "prev/"
    
    stop_video_names = [f for f in listdir(stop_path) if isfile(join(stop_path, f))]
    prev_video_names = [f for f in listdir(prev_path) if isfile(join(prev_path, f))]
    next_video_names = [f for f in listdir(next_path) if isfile(join(next_path, f))]

    # changing the dir
    if(os.path.isdir(store_it_here)):
        shutil.rmtree(store_it_here)

    os.mkdir(store_it_here)
    os.chdir(store_it_here)

    # deleting csv file if already exists
    if(os.path.exists("unordered_dataset.csv")):
        os.remove("unordered_dataset.csv")

    # printing header row in csv file
    # printing into csv file
    row = ["image_names", "label"]
    with open('unordered_dataset.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)

    # iterating over sstop video frames
    i=0
    image_count = 0
    while(i < len(stop_video_names)):
        cap = cv2.VideoCapture(stop_path + stop_video_names[i])
        fgbg = cv2.createBackgroundSubtractorKNN()
        ret = 1
        while(ret):
            ret, frame = cap.read()
            if(not ret): break
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # canny_frame = cv2.Canny(frame,20, 50, apertureSize = 3)
            # canny_frame = cv2.resize(canny_frame, (50, 50), interpolation = cv2.INTER_AREA)
            # -----------------------------  code of apply canny  -----------------------------------
            # canny_frame = preprocessing.apply_canny(frame, 20, 50, 50, 50)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            canny_frame = fgbg.apply(frame)
            # canny_frame = cv2.Canny(frame,20, 50, apertureSize = 3)
            canny_frame = cv2.resize(canny_frame, (50, 50), interpolation = cv2.INTER_AREA)
            # ----------------------------------------------------------------------------------------
            if(image_count%one_of_how_many == 0):
                rand1 = random.randint(0, 1000000)
                rand2 = random.randint(0, 1000000)
                cv2.imwrite(str(rand1)+ "_" + str(rand2) + ".jpg", canny_frame)
                # printing into csv file
                row = [str(rand1)+ "_" + str(rand2) + ".jpg", str(1)]
                # cv2.imwrite("stop_" + str(image_count) + ".jpg", canny_frame)
                with open('unordered_dataset.csv', 'a') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
            image_count += 1
        i += 1
    # iterating over next video frames
    i=0
    image_count = 0
    while(i < len(next_video_names)):
        print("video " + str(i) + " done")
        cap = cv2.VideoCapture(next_path + next_video_names[i])
        fgbg = cv2.createBackgroundSubtractorKNN()
        ret = 1
        while(ret):
            ret, frame = cap.read()
            if(not ret): break
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # canny_frame = cv2.Canny(frame,20, 50, apertureSize = 3)
            # canny_frame = cv2.resize(canny_frame, (50, 50), interpolation = cv2.INTER_AREA)
            # -----------------------------  code of apply canny  -----------------------------------
            # canny_frame = preprocessing.apply_canny(frame, 20, 50, 50, 50)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            canny_frame = fgbg.apply(frame)
            # canny_frame = cv2.Canny(frame,20, 50, apertureSize = 3)
            canny_frame = cv2.resize(canny_frame, (50, 50), interpolation = cv2.INTER_AREA)
            # ----------------------------------------------------------------------------------------
            if(image_count%one_of_how_many == 0):
                rand1 = random.randint(0, 1000000)
                rand2 = random.randint(0, 1000000)
                cv2.imwrite(str(rand1)+ "_" + str(rand2) + ".jpg", canny_frame)
                # printing into csv file
                row = [str(rand1)+ "_" + str(rand2) + ".jpg", str(2)]
                with open('unordered_dataset.csv', 'a') as csvFile:
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
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # canny_frame = cv2.Canny(frame,20, 50, apertureSize = 3)
            # canny_frame = cv2.resize(canny_frame, (50, 50), interpolation = cv2.INTER_AREA)
            # -----------------------------  code of apply canny  -----------------------------------
            # canny_frame = preprocessing.apply_canny(frame, 20, 50, 50, 50)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            canny_frame = fgbg.apply(frame)
            # canny_frame = cv2.Canny(frame,20, 50, apertureSize = 3)
            canny_frame = cv2.resize(canny_frame, (50, 50), interpolation = cv2.INTER_AREA)
            # ----------------------------------------------------------------------------------------
            if(image_count%one_of_how_many == 0):
                rand1 = random.randint(0, 1000000)
                rand2 = random.randint(0, 1000000)
                cv2.imwrite(str(rand1)+ "_" + str(rand2) + ".jpg", canny_frame)
                # printing into csv file
                row = [str(rand1)+ "_" + str(rand2) + ".jpg", str(0)]
                with open('unordered_dataset.csv', 'a') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
            image_count += 1
        i += 1
    #shuffling the csv file

    df = pd.read_csv('unordered_dataset.csv')
    df = df.sample(frac = 1).reset_index(drop = True)

    export_csv = df.to_csv(csv_file_name, index = None, header = True)
    os.remove("unordered_dataset.csv")
    # coming out of dir
    os.chdir("..")
#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_folder = "../data/"
train_folder = data_folder + "train_data/"
train_roi_folder = data_folder + "train_data_roi/"
test_folder = data_folder + "test_data/"
test_roi_folder = data_folder + "test_data_roi/"


os.makedirs(train_roi_folder)
os.makedirs(test_roi_folder)


train_videos = glob.glob(train_folder + "*.mp4")
test_videos = glob.glob(test_folder + "*.mp4")


len(train_videos)


len(test_videos)


C = 5

def get_coords(frame):
    cframe = frame.copy()
    
    cframe[cframe[:,:,2] < 200] = 0
    cframe[cframe[:,:,0] > 150] = 0
    cframe[cframe[:,:,1] > 150] = 0
    pos = np.where(cframe != 0)
    
    x_min = max(0, min(pos[0]) - C)
    x_max = min(cframe.shape[0], max(pos[0]) + C)
    y_min = max(0, min(pos[1]) - C)
    y_max = min(cframe.shape[1], max(pos[1]) + C)
    
    return x_min, x_max, y_min, y_max


def get_mask(crop):
    cframe = crop.copy()
    cframe[cframe[:,:,2] < 200] = 0
    cframe[cframe[:,:,0] > 150] = 0
    cframe[cframe[:,:,1] > 150] = 0
    
    pos = np.where(cframe != 0)
    pts = []
    for i in range(len(pos[0])):
        pt = [pos[1][i], pos[0][i]]
        if pt not in pts:
            pts.append(pt)
    pts = np.array(pts)
    
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    
    return mask


dim = (112, 112)

def make_roi(videos, destination_folder):
    for i, video in enumerate(videos):

        if i % 200 == 0:
            print(i)

        cap = cv2.VideoCapture(video)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(destination_folder + video.split("\\")[-1], fourcc, 20, dim)

        if (cap.isOpened()== False): 
            print("Error opening video stream or file")

        first = True
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                if first:
                    x_min, x_max, y_min, y_max = get_coords(frame)
                    first = False
                crop = frame[x_min:x_max, y_min:y_max, :]
                resized = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)
                out.write(resized)
            else: 
                break

        cap.release()
        out.release()


make_roi(train_videos, train_roi_folder)


make_roi(test_videos, test_roi_folder)


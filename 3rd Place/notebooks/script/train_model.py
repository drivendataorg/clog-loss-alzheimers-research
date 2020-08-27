#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import math
import time
import random
import pandas as pd
import numpy as np

import torch
import torchvision
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, accuracy_score, log_loss


class VideoIterator(Dataset):
    def __init__(self, df, transforms, device, aug=False):
        self.df = df
        print(self.df["stalled"].value_counts())
        self.transforms = transforms
        self.device = device
        self.aug = aug
        self.rotations = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
        self.flips = [0, 1, -1]
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        x = []
        video = cv2.VideoCapture(row["path"])
        if not video.isOpened():
            print("Error opening video file.")
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                x.append(frame)
            else:
                break
        video.release()
        
        if self.aug:    
            if random.random() <= 0.25:
                rot = random.choice(self.rotations)
                x = [cv2.rotate(img, rot) for img in x]
            if random.random() <= 0.25:
                fl = random.choice(self.flips)
                x = [cv2.flip(img, fl) for img in x]
        
        x = [self.transforms(frame) for frame in x]
        x = torch.stack(x)
        x = x.permute(1, 0, 2, 3)
        x = x.unsqueeze(0)
        x = x.to(self.device, dtype=torch.float)
        y = torch.FloatTensor([[row["stalled"]]]).to(self.device)
        return x, y

    def __len__(self):
        return len(self.df)

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)


class R2plus1dModel(nn.Module):
    def __init__(self):
        super(R2plus1dModel, self).__init__()

        self.cnn = torchvision.models.video.r2plus1d_18(pretrained=True)
        self.cnn.fc = nn.Linear(in_features=512,
                                out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        x = self.cnn(input)
        x = self.sig(x)
        return x
    

class Rmc3Model(nn.Module):
    def __init__(self):
        super(Rmc3Model, self).__init__()

        self.cnn = torchvision.models.video.mc3_18(pretrained=True)
        self.cnn.fc = nn.Linear(in_features=512,
                                out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        x = self.cnn(input)
        x = self.sig(x)
        return x


class R3dModel(nn.Module):
    def __init__(self):
        super(R3dModel, self).__init__()

        self.cnn = torchvision.models.video.r3d_18(pretrained=True)
        self.cnn.fc = nn.Linear(in_features=512,
                                out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        x = self.cnn(input)
        x = self.sig(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


data_folder = "../data/"
train_data_folder = data_folder + "train_data_roi/"
train_csv = data_folder + "train_data.csv"


transformations = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
])
    
df = pd.read_csv(train_csv)
df["path"] = train_data_folder + df["filename"]
    
train_df, val_df = train_test_split(df, 
                                    test_size=0.1, 
                                    shuffle=True, 
                                    stratify=df["stalled"].values, 
                                    random_state=42)

print("Training data")
train_data_iterator = VideoIterator(train_df, transformations, device, True)
print("\nValidation data")
val_data_iterator = VideoIterator(val_df, transformations, device, False)

w = len(train_df[train_df["stalled"] == 0]) / len(train_df[train_df["stalled"] == 1])
print("\nStalled class weight: " + str(w))


model_folder = "../models/model_v1/"
os.makedirs(model_folder)

no_epochs = 30
log_interval = 100

# Choose model architecture here:
# model = R3dModel().to(device)
# model = Rmc3Model().to(device)
model = R2plus1dModel().to(device)

criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters())
# this is actually cosine annealing
lr_scheduler = OneCycleLR(optimizer, pct_start=0.0001, max_lr=1e-4, epochs=no_epochs, steps_per_epoch=len(train_data_iterator))

summary_writer = SummaryWriter(model_folder + "tensorlogs/")


best_mcc = -1
best_loss = math.inf

for i in range(no_epochs):
    
    model.train()
    partial_loss_sum = 0
    print("\nEpoch " + str(i))
    start_time = time.time()
    train_data_iterator.shuffle()
    train_losses = []
    for j, (x, y) in enumerate(train_data_iterator):
        outputs = model(x)
        optimizer.zero_grad()
        loss = criterion(outputs, y)
        train_losses.append(loss.item())
        partial_loss_sum += loss.item()
        if y[0].item() == 1:
            loss = loss * w
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if (j + 1) % log_interval == 0:
            end_time = time.time()
            total_time = round(end_time - start_time, 2)
            partial_loss = round(partial_loss_sum / log_interval, 6)
            lr = optimizer.param_groups[0]["lr"]
            print("    " + str(j + 1) + "/" + str(len(train_data_iterator)) + " | loss " + str(partial_loss) + " | lr " + str(lr) + " | " + str(total_time) + "s")
            start_time = time.time()
            partial_loss_sum = 0
    train_loss = round(sum(train_losses) / len(train_losses), 4)
                    
    print("\nEvaluating...")
    y_true = []
    y_pred = []
    y_pred_p = []

    with torch.no_grad():
        for j, (x, y) in enumerate(val_data_iterator):
            outputs = model(x)
            p = outputs[0].item()
            y_pred_p.append(p)
            if p < 0.5:
                p = 0
            else:
                p = 1
            y_pred.append(p)
            y_true.append(y[0].item())

    val_mcc = round(matthews_corrcoef(y_true, y_pred), 4)
    val_acc = round(accuracy_score(y_true, y_pred), 4)
    val_loss = round(log_loss(y_true, y_pred_p), 4)

    if val_mcc >= best_mcc or val_loss <= best_loss: 
        model_path = model_folder + "model_epoch:" + str(i) + "_mcc:" + str(val_mcc) + "_acc:" + str(val_acc) + "_loss" + str(val_loss) + ".pth"
        torch.save(model.state_dict(), model_path)
        if val_mcc >= best_mcc:
            best_mcc = val_mcc
        if val_loss <= best_loss:
            best_loss = val_loss

    print("Train loss: " + str(train_loss))
    print("Val loss: " + str(val_loss))
    print("Val MCC: " + str(val_mcc))
    print("Val accuracy: " + str(val_acc))    
    
    summary_writer.add_scalar("train_loss", train_loss, i)
    summary_writer.add_scalar("val_loss", val_loss, i)
    summary_writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, i)
    summary_writer.add_scalar("val_acc", val_acc, i)
    summary_writer.add_scalar("val_mcc", val_mcc, i)
    summary_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], i)


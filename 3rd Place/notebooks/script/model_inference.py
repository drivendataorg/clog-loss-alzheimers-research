#!/usr/bin/env python
# coding: utf-8

import cv2
import time
import numpy as np
import pandas as pd

import torch
import torchvision
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data.dataset import Dataset


test_data_dir = "../data/test_data_roi/"
submission_format_file = "../submissions/submission_format.csv"


models_dir = "../models/"

# model_name = "R3D"
# model_name = "MC3"
model_name = "R(2+1)D_v1"
# model_name = "R(2+1)D_v2"

model_path_dict = {
    "R3D": models_dir + "R3D/model_v67_epoch_27_mcc_0.7801_acc_0.9635_loss0.1248.pth",
    "MC3": models_dir + "MC3/model_v66_epoch_28_mcc_0.7989_acc_0.9673_loss0.1125.pth",
    "R(2+1)D_v1": models_dir + "R(2+1)D_v1/model_v65_epoch_28_mcc_0.769_acc_0.9635_loss0.1323.pth",
    "R(2+1)D_v2": models_dir + "R(2+1)D_v2/model_v72_epoch_28_mcc_0.7306_acc_0.9572_loss0.1684.pth",
}
model_path = model_path_dict[model_name]


class VideoIterator(Dataset):
    def __init__(self, data_dir, data_csv, transforms, device):
        self.df = pd.read_csv(data_csv)
        self.df["path"] = data_dir + self.df["filename"]
        self.transforms = transforms
        self.device = device

    def __getitem__(self, index):
        row = self.df.iloc[index]
        x = []
        video = cv2.VideoCapture(row["path"])
        if not video.isOpened():
            print("Error opening video file!")
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                x.append(self.transforms(frame))
            else:
                break
        video.release()
        x = torch.stack(x)
        x = x.permute(1, 0, 2, 3)
        x = x.unsqueeze(0)
        x = x.to(self.device, dtype=torch.float)
        return x

    def __len__(self):
        return len(self.df)


class R2plus1dModel(nn.Module):
    def __init__(self):
        super(R2plus1dModel, self).__init__()

        self.cnn = torchvision.models.video.r2plus1d_18(pretrained=False)
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

        self.cnn = torchvision.models.video.mc3_18(pretrained=False)
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

        self.cnn = torchvision.models.video.r3d_18(pretrained=False)
        self.cnn.fc = nn.Linear(in_features=512,
                                out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        x = self.cnn(input)
        x = self.sig(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


transformations = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
])

data_iterator = VideoIterator(test_data_dir, submission_format_file, transformations, device)
print(len(data_iterator))


# Choose model architecture here:
model_class_dict = {
    "R3D": R3dModel,
    "MC3": Rmc3Model,
    "R(2+1)D_v1": R2plus1dModel,
    "R(2+1)D_v2": R2plus1dModel,
}

model = model_class_dict[model_name]().to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.train()


y_pred = []

with torch.no_grad():
    for j, x in enumerate(data_iterator):
        if j % 200 == 0:
            print(j)
        outputs = model(x)
        p = outputs[0].item()
        y_pred.append(p)


subm_file =  f"../submissions/inference_{model_name}.csv"

df = data_iterator.df.copy(deep=False)

df["stalled"] = y_pred
df = df[["filename", "stalled"]]

df.to_csv(subm_file, index=False)


#!/usr/bin/env python
# coding: utf-8

import pandas as pd


sub = "../submissions/"

T = 0.95
file = sub + "inference_R(2+1)D_v1.csv"

# T = 0.80
# file = sub + "inference_R(2+1)D_v2.csv"

# T = 0.90
# file = sub + "inference_MC3.csv"

# T = 0.80
# file = sub + "inference_R3D.csv"


df = pd.read_csv(file)


df


df = df.rename(columns={"stalled": "p"})


df


s = [0 if p < T else 1 for p in df["p"].values]


df["stalled"] = s


sum(s) / len(s) * 100


df = df[["filename", "stalled"]]


df


df.to_csv(sub + "submission_R(2+1)D_v1_95%.csv", index=False)


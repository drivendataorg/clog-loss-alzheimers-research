#!/usr/bin/env python
# coding: utf-8

import pandas as pd


sub = "../submissions/"
file1 = sub + "inference_R(2+1)D_v1.csv"
file2 = sub + "inference_MC3.csv"
# file3 = sub + "inference_R3D.csv"
file3 = sub + "inference_R(2+1)D_v2.csv"


df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)


df = df1.merge(df2, on="filename")
df = df.merge(df3, on="filename")


df = df.rename(columns={"stalled_x": "s1", "stalled_y": "s2", "stalled": "s3"})


df


s = []
T1 = 0.95
T2 = 0.90
T3 = 0.80
for i, row in df.iterrows():
    if (row["s1"] >= T1 and row["s2"] >= T2) or (row["s1"] >= T1 and row["s3"] >= T3) or (row["s2"] >= T2 and row["s3"] >= T3):
        s.append(1)
    else:
        s.append(0)


df["stalled"] = s


sum(s) / len(s) * 100


df = df[["filename", "stalled"]]


df


df.to_csv(sub + "submission_R(2+1)D(v1)_MC3_R(2+1)D(v2)_ensemble.csv", index=False)


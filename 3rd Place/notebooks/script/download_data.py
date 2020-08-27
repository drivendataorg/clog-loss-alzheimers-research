#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd

import boto3
import botocore
from botocore.handlers import disable_signing
s3 = boto3.resource('s3')
s3.meta.client.meta.events.register("choose-signer.s3.*", disable_signing)


data = "../data/"
train_df = pd.read_csv(data + "train_data.csv")
test_df = pd.read_csv(data + "test_data.csv")


train_df


test_df


train_data = data + "train_data/"
os.makedirs(train_data)

BUCKET_NAME = "drivendata-competition-clog-loss"

for i, image_name in enumerate(train_df["filename"].values):
    if i % 200 == 0:
        print(i)
        
    from_image = "train/" + image_name
    to_image = train_data + image_name
    try:
        s3.Bucket(BUCKET_NAME).download_file(from_image, to_image)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("The object does not exist.")
        else:
            raise


test_data = data + "test_data/"
os.makedirs(test_data)

BUCKET_NAME = "drivendata-competition-clog-loss"

for i, image_name in enumerate(test_df["filename"].values):
    if i % 200 == 0:
        print(i)
        
    from_image = "test/" + image_name
    to_image = test_data + image_name
    try:
        s3.Bucket(BUCKET_NAME).download_file(from_image, to_image)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("The object does not exist.")
        else:
            raise


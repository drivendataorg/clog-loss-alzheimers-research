[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

![Banner Image](https://www.drivendata.co/images/stallcatchers-vessels.jpg)

# Clog Loss: Advance Alzheimer’s Research with Stall Catchers

## Goal of the Competition

5.8 million Americans [live with Alzheimer’s dementia](https://www.alz.org/alzheimers-dementia/facts-figures), including 10% of all seniors 65 and older. Scientists at Cornell have discovered links between “stalls,” or clogged blood vessels in the brain, and Alzheimer’s. Stalls can reduce overall blood flow in the brain by 30%. The ability to prevent or remove stalls may transform how Alzheimer’s disease is treated.

[Stall Catchers](https://stallcatchers.com/main) is a citizen science project that crowdsources the analysis of Alzheimer’s disease research data provided by Cornell University’s Department of Biomedical Engineering. It resolves a pressing analytic bottleneck: for each hour of data collection it would take an entire week to analyze the results in the lab, which means an entire experimental dataset would take 6-12 months to analyze. Today, the Stall Catchers players are collectively analyzing data 5x faster than the lab while exceeding data quality requirements.

Through the Stall Catchers project, there is now a one-of-a-kind dataset that can be used to train and evaluate ML models on this task. Each exemplar is an image stack (a 3D image) taken from a live mouse brain showing blood vessels and blood flow. Each stack has an outline drawn around a target vessel segment and has been converted to an mp4 video file. **The objective of this competition is to classify the outlined vessel segment as flowing—if blood is moving through the vessel—or stalled if the vessel has no blood flow.**

## What's in this Repository

This repository contains code from winning competitors in the [Clog Loss: Advance Alzheimer’s Research with Stall Catchers](https://www.drivendata.org/competitions/65/clog-loss-alzheimers-research/) DrivenData challenge.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

Place | Team or User | Public Score | Private Score | Summary of Model
----- | ------------ | ---   | ---   | ---
1     | ZFTurbo      | 0.844 | 0.856 | Neural network built on the principles of the DenseNet121 network, which is used to classify images, but in a 3D version. At the output of the network after GlobalAveragePooling, an additional classification fully connected layer with 512 elements was added, followed by a Dropout layer with a probability of 0.5, to reduce overfitting. At the very end, there was a fully connected layer with one neuron (since we only have one class) and sigmoid activation (see img. 1). The final output was the combined predictions with test-time augmentation for five models from five-fold cross-validation.
2     | kbrodt       | 0.846 | 0.838 | Heavy ResNet101 model was trained with binary cross entropy loss with standard spatial augmentations like horizontal and vertical flips, distortions and noise. Model was trained on full tier 1 dataset and all stalled samples with crowd score> 0.6 from tier 2 dataset. The predictions of five different snapshots on various epochs of the same model were averaged.
3     | LauraOnac    | 0.777 | 0.810 | Two types of data augmentations are performed on an entire video with a 25% chance each: rotations and flips. The final train set consisted of 7,931 videos, all from tier1, with 91.1% of them being flowing. The highest performance was achieved using an ensemble of two R(2+1)D models and an MC3 model, spatio-temporal neural network architectures based on ResNet-18 and pretrained for action recognition on the Kinetics-400 data set.
Bonus | AZ_KA        | 0.467 | 0.536 | <Description from the 3rd place's writeup>

Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Benchmark Blog Post: ["Advance Alzheimer’s Research with Stall Catchers - Benchmark"](https://www.drivendata.co/blog/stall-catchers-alzheimers-benchmark/)**

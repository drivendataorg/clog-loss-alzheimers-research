# Clog Loss: Advance Alzheimer’s Research with Stall Catchers 

[Clog Loss: Advance Alzheimer’s Research with Stall Catchers](https://www.drivendata.org/competitions/65/clog-loss-alzheimers-research/page/207/).

[2nd place](https://www.drivendata.org/competitions/65/clog-loss-alzheimers-research/leaderboard/)
out of 922 with 0.8389 Matthew's correlation coefficient (MCC) (top 1 -- 0.8555).

### Prerequisites

- GPU with 32Gb RAM (e.g. Tesla V100)
- [NVIDIA apex](https://github.com/NVIDIA/apex)

```bash
pip install -r requirements.txt
```

### Usage

#### Download

First download the train and test data from the competition link into `/home/data/clog-loss` folder.

Then you have to prepare train and test datasets.

```bash
sh ./preprocess.sh
```

This will download whole dataset (1.4Tb), crop the region of interest from video using code provided by `@Moshel`
on the [forum](https://community.drivendata.org/t/python-code-to-find-the-roi/4499) and save it in
[lmdb](https://github.com/jnwatson/py-lmdb) format for fast reading of 3D arrays. Note, it will take around 1 day
and consume 200Gb of RAM, hence disk. So, if you have not enough RAM you can easily rewrite the code to process the data by chunks.

#### Train

To train the model run

```bash
sh ./train.sh
```

On 1 GPU Tesla V100 it will take around 2-3 weeks. If you have more GPUs, you can train the model in a distributed manner.
For example, if you have 4 GPUs rewrite code in [`train.sh`](./train.sh)

```bash
GPU=0,1,2,3
N_GPUS=4

CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=$N_GPUS ./src/train.py <OTHER_ARGUMENTS>
```

#### Test

Once the model trained run the following command to do inference on test.

```bash
sh ./test.sh
```

On 1 GPU Tesla V100 it takes around 40m.

You can also download the trained models from [yandex disk](https://yadi.sk/d/2GGRsM-ac5CaKQ), unzip and run

```bash
sh ./predict.py
```

Last two scripts will generate submission file with probabilities. Use 0.5 threshold to convert them into binary format.

### Approach

3D Convolutional network on full tier 1 and tier 2 datasets.

#### First observations

Baseline model based on 2D CNN as feature extractor for video frames and LSTM as classifier reaches 0.59 LB.
Single 3D ResNet34 on `Micro` dataset already gives 0.68 on LB. With 2 folds and TTAs one can reach 0.76.
Heavier models like 3D ResNet50 and ResNet101 has 0.71 and 0.76 respectively. On full tier 1 dataset single
3D ResNet34 gives 0.81. Interestingly, height and width of crops have a positive signal. Gradient boosting
on these 2 features gives 0.16 on local validation. I only tried to add them into video features, but the score
was getting worse on LB. So I discarded this idea on early stage of model development.

#### Summary

The best models for 2D image recognition are 2D convolutional neural networks (CNNs). Most likely this will be true for 3D.
So the 3D CNNs were chosen, namely ResNets. Thanks to provided region of interest on video one can crop it to reduce significantly
disk and, hence, ram usage. To deal with different spatial resolutions of frames they were resized to 160x160. Depth dimension
was zero-padded till the longest depth of all samples in the batch. As dataset is highly imbalanced balanced sampler with ¼ ratio
of positive (stalled) classes was applied. This also makes training faster. Heavy ResNet101 model was trained with binary cross entropy
loss with standard spatial augmentations like horizontal and vertical flips, distortions and noise. Finally, deep learning is data-hungry
so the model was trained on full tier 1 dataset and all stalled samples with crowd score> 0.6 from tier 2 dataset. The predictions of
five different snapshots on various epochs of the same model were averaged.

#### Highlights

- 3D ResNet101
- 160x160xF resized crops of ROIs, where F is a video depth
- Full tier 1 dataset and all stalled examples from tier 2 dataset with crowd score > 0.6
- Binary Cross Entropy
- Batch size 4
- AdamW with learning rate `1e-4`
- CosineAnealing scheduler
- Spatial augmentations like horizontal and vertical flips, rotate on 90, distortions, noise. Depths didn't touch. See [`dataset.py`](./src/dataset.py#L10)
- Balanced sampler with 1/4 ratio of stalled and non-stalled samples
- Mean of 5 predictions of 5 different snapshots of the same model 3D ResNet101

#### Tried

I tried to improve second part of baseline model 2D CNN + LSTM. LSTM was replaced by 1D CNN and Transformer, but the score was the same as with LSTM (0.59 score). New neural network ResNeSt works not better than Efficientnets. Surprisingly, but test time augmentations worsen the score. One thing I find is interesting that width and height has positive signal, i. e. if we train some model (linear or gradient boosting) with these two features we can get 0.16 Matthew correlation coefficient on local validation. But I didn’t try it on leader board. Furthermore it is useless in production.

- 2D CNN + LSTM (0.59 LB)
- 2D CNN + Transformer is not better than LSTM
- 2D CNN + 1D CNN is not better than LSTM
- 3D CNN Efficientnet doesn't train
- Test time augmentations doesn't improve score
- Focal and Lovasz losses are not better than BCE
- Crowd score instead binary label in loss, but the results are alomost the same
- 2nd level model on out of fold predictions with additional features of crop (width and height) worsen local validation and public LB
- [AdaTune](https://github.com/awslabs/adatune) works same as `lr=1e-4`
- 1 round of pseudo-labeling (not enough investigated)

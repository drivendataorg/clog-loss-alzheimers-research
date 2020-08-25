Clog Loss: Advance Alzheimer’s Research with Stall Catchers - 3rd Place
==============================

Project organization
------------

    │
    ├── data                 <- Video data.
    │    │    
    │    ├── test_data.csv
    │    │
    │    └── train_data.csv
    │
    ├── models               <- Trained models.
    │
    ├── notebooks            <- Jupyter notebooks.
    │    │    
    │    ├── download_data.ipynb
    │    │
    │    ├── ensemble_submission.ipynb
    │    │
    │    ├── model_inference.ipynb
    │    │
    │    ├── preprocess_data.ipynb
    │    │
    │    ├── single_model_submision.ipynb
    │    │
    │    └── train_model.ipynb
    │
    ├── submissions          <- Submission files.
    │    │
    │    └── submission_format.csv
    |
    ├── README.md            <- Instructions for developers using this project.
    |
    └── requirements.txt     <- Python libraries needed to reproduce the analysis environment.


Python environment
------------
- create a Python 3.6 environment
- install all dependencies from requirements.txt


Steps to replicate the submission
------------

### 1. Download data
- run notebooks/download_data.ipynb
- this will create the raw data folders: **test_data** and **train_data**
- it will download the corresponding files, as specified in test_data.csv and train_data.csv
    
### 2. Preprocess data
- run notebooks/preprocess_data.ipynb
- this will create the processed data folders: **test_data_roi** and **train_data_roi**
- it will extract the region of interest from the videos in the raw data folders, resize them, and save them to the processed data folders
    
### 3. Train models
- run notebooks/train_model.ipynb
- without any modification, an R(2+1)D model will be trained for 30 epochs
- to change the model architecture, code cell #7 needs to be modified
- the model weights will be saved to the specified path
    
### 4. Model inference
- run notebooks/model_inference.ipynb
- the path to the model weights and the model architecture can be specified in code cells #3 and #8, respectively
- this will generate a *csv* file with the raw model outputs
    
### 5. Make submission
#### 5.1. Single model submission
- run notebooks/single_model_submission.ipynb
- here, you need to specify the model inference file and a decision threshold
- this will generate a submission file
    
#### 5.2. Ensemble submission
- run notebooks/ensemble_submission.ipynb
- here, you need to specify three model inference files and a decision threshold for each of them
- this will generate a submission file
- the best submission was made with an ensemble of two R(2+1)D models and an MC3 model
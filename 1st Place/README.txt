* Software Requirements

Main requirements: Python 3.5 or Python 3.6, keras 2.2+, Tensorflow 1.13+
Other requirements: numpy, pandas, opencv-python, scipy, sklearn
You need to have CUDA 10.0 installed. You can get it from conda: `conda install cudatoolkit=10.0`
Solution was tested on Anaconda3-2019.10-Linux-x86_64.sh: https://www.anaconda.com/distribution/

* Hardware requirements

* All batch sizes for Neural nets are tuned to be used on NVIDIA GTX 1080 Ti 11 GB card. To use code with other GPUs with less memory - decrease batch size accordingly.
* For fast validation 3D volumes during training are read in memory. So training will require ~64GB of RAM.

* How to run:

Code expects all input files in "../input/" directory. Fix paths in a00_common_functions.py if needed.
All r*.py files must be run one by one. All intermediate folders will be created automatically.

Only inference part:
python preproc_data/r01_extract_roi_parts.py test
python net_v20_d121_only_tier1_finetune/r42_process_test.py

There is file run_inference.sh - which do all the stuff including pip installation of required modules etc.

Full pipeline including training of models:
python3 preproc_data/r01_extract_roi_parts.py
# Uncomment if you need to create new KFold split
# python3 preproc_data/r03_gen_kfold_split.py
python3 net_v13_3D_roi_regions_densenet121/r31_train_3D_model_dn121.py
python3 net_v14_d121_auc_large_valid/r31_train_3D_model_dn121.py
python3 net_v20_d121_only_tier1_finetune/r31_train_3D_model_dn121.py
python3 net_v20_d121_only_tier1_finetune/r42_process_test.py

There is file run_train.sh - which do all the stuff including pip installation of required modules etc.


You need to change run_inference.sh and run_train.sh for your environment:

If needed, explicitly add your Python environment to your PATH. Change this variable to location of your python (Anaconda)
export PATH="/var/anaconda3-temp/bin/"

Change this variable to location of your code
export PYTHONPATH="$PYTHONPATH:/<path-to-repo>/clog-loss-alzheimers-research/1st Place/src/"

After you run inference or train final submission file will be located in ../subm/submission.csv file.

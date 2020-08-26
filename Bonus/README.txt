Requirements:

•	MATLAB 2020a (Deep Learning Toolbox, Computer Vision Toolbox, Image Processing Toolbox)
•	A system with Nvidia GPU (with 4GB memory) and CUDA toolkit installed


Steps for running the codes:

Step.1:  Selecting the videos for training from main dataset
-	Open “preparing_training_data.mlx”  
-	In section 1.1, provide the location of "train_metadata.csv" and "train_labels.csv"
-	In section 1.7, provide the location of main dataset
-	The code creates a folder (“(Input_train”) containing selected videos for training and csv file (“train_metadata_V1.csv”) 
        containing specifications of those videos under column “milli”

Step.2: Training CNN
-	Open the file “main_file.mlx”
-	In section 2.1, provide the location of “train_metadata_V1.csv”, “train_metadata.csv" and "train_labels.csv"
-	In section 2.2.1, provide the location of selected training data
-	In section 2.2.2, provide the location of the test set
-	In section 2.5, it is possible to define the preferred accuracy
-	In section 2.6, it possible to choose using pretrained network or not (with changing “train_ind” value to “false” or “true”) 
-	In section 2.7, the test set labels are being predicted and saved as csv file ('testResults.csv')

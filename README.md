# CS184 Final Project
A Mask R-CNN model applied in `Sartorius - Cell Instance Segmentation` Kaggle competition. Reference: https://www.kaggle.com/julian3833/sartorius-starter-torch-mask-r-cnn-lb-0-273#Train-loop
##  src:
A directory with the changed code used to generate prediction
#### 1_Import.py
All the necessary imports  and randomness fixation to replicate the same result after rerunning the code.
#### 2_Configuration.py
Data load and parameter setup (e.g. thresholds, number of epochs) for training and testing purpose.
#### 3_Preprocess.py
Data transformation and annotations(train.csv) mask as the ground truth.
#### 4_Model.py
Use Torch package to create a Mask R-CNN model.
#### 5_Training.py
The 8-epoch training loop of the model.
#### 6_Analyze.py
Analysis of prediction on training data.
#### 7_Testdata.py
Transformation and normalization of the test data. 
#### 8_Prediction
Final prediction on the test data and final submission.

## A Jupyter Notebook
#### project.ipynb
A jupyter notebook that use the trained model to generate predictions on a sample of the data and the test data, including the loss curve.
#### project.html
A .html file that contains both codes and output of project (1).ipynb.
#### train
A directory that contains a sample image.
#### train1.csv
A .csv file that contains the annotation of the sample data.
#### stats.csv
A .csv file that store the batch train loss and mask-only loss for 8-epoch training.
#### submission.csv
A .csv file that can be submitted to kaggle, which contains all the predicted pixels range of neurons in the  test data. 

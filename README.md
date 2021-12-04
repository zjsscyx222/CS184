# CS184 Final Project
A Mask R-CNN model applied in `Sartorius - Cell Instance Segmentation` Kaggle competition. Reference: https://www.kaggle.com/julian3833/sartorius-starter-torch-mask-r-cnn-lb-0-273#Train-loop
##  final:
Directory with all the files used to generate prediction
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
This is the file that we can see how well the model from  predicts.
#### 7_Testdata.py
This is the file that transforms and normalizes the test data. 
#### 8_Prediction
This is the file that we make final prediction on the test data and get final submission.

## A Jupyter Notebook
#### project.ipynb
A jupyter notebook that use the trained model to generate predictions on a sample of the data and the test data.
#### project.html
A .html file that contains both codes and output of project (1).ipynb.
#### train
Directory that contains a sample image.
#### train1.csv
A .csv file that contains the annotation of the sample data.
#### stats.csv
This is the file that collects loss value during each batch in each epoch when we are in the training loop.
#### submission.csv
This is the file that can be submitted to kaggle, which contains all pixels that are predicted to be neurons in test data by our model. Each neuron lies in one row so there might be many rows that have the same id, which means there's many neurons in one test image.

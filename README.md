# CS184 Final Project
A Mask R-CNN model applied in `Sartorius - Cell Instance Segmentation` Kaggle competition. Reference: https://www.kaggle.com/julian3833/sartorius-starter-torch-mask-r-cnn-lb-0-273#Train-loop
##  `final`:Files Used to Generate Prediction

### `import111.py`
All the necessary imports  and randomness fixation to replicate the same result after rerunning the code.
### `comfig.py` 
Load data and set up the parameters(like thresholds, number of epochs) for training and testing purpose.
### `training.py`
This is the file that transforms the dataset and gives annotations as the ground truth.
### `Model.py`
This is the file that builds the frame of model.
### `Training Dataset and DataLoader.py`
This is the training loop code. We will finish and save 8 models from 8 epoch here.
### `Analyze prediction results for train set.py`
This is the file that we can see how well the model from  predicts.
### `test_dataset.py`
This is the file that transforms and normalizes the test data. 
### `Utilities.py` 
This is the file that we get final submission.

## project.ipynb
This is the jupyter notebook file that can run and quickly show the performance of one of the 8 models.

## project.html
This is the file that contains both codes and output of project (1).ipynb.

## Files used in project.ipynb:

### train
This is the folder contains a sample image.
### train1.csv
This is the .csv file that contains annotation for the sample data.
### stats.csv
This is the file that collects loss value during each batch in each epoch when we are in the training loop.
### submission.csv
This is the file that can be submitted to kaggle, which contains all pixels that are predicted to be neurons in test data by our model. Each neuron lies in one row so there might be many rows that have the same id, which means there's many neurons in one test image.

# CS184
## final
This is the folder that contains all codes that we will use for this problem.(Already sorted by run order here)
#### import111.py
This is the file that contains all the basic imports within the whole process and fix the random inorder to replicate the same result by rerunning the code after many times.
#### comfig.py 
This is the file that loads data and set the fundemental variables(like thresholds, number of epochs) for training and testing later.
#### training.py
This is the file that transforms the dataset and gives annotations as the ground truth.
#### Model.py
This is the file that builds the frame of model.
#### Training Dataset and DataLoader.py
This is the training loop code. We will finish and save 8 models from 8 epoch here.
#### Analyze prediction results for train set.py 
This is the file that we can see how well the model from  predicts.
#### test_dataset.py
This is the file that transforms and normalizes the test data. 
#### Utilities.py 
This is the file that we get final submission.

## project (1).ipynb
This is the jupyter notebook file that can run and quickly show the performance of one of the 8 models.

## project_html.pdf
This is the file that contains both codes and output of project (1).ipynb.

## stats.csv
This is the file that collects loss value during each batch in each epoch when we are in the training loop.

## submission.csv
This is the file that can be submitted to kaggle, which contains all pixels that are predicted to be neurons in test data by our model. Each neuron lies in one row so there might be many rows that have the same id, which means there's many neurons in one test image.

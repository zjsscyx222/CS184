import os
import torch
import random
import numpy as np




# Configuration

TRAIN_CSV = "D:/CS18444/sartorius-cell-instance-segmentation/train.csv"
TRAIN_PATH = "D:/CS18444/sartorius-cell-instance-segmentation/train"
TEST_PATH = "../sartorius-cell-instance-segmentation/test"

WIDTH = 704
HEIGHT = 520

# Reduced the train dataset to 5000 rows
TEST = False

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE = torch.device('cpu')

RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)

BATCH_SIZE = 5

# No changes tried with the optimizer yet.
MOMENTUM = 0.9
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005

# Changes the confidence required for a pixel to be kept for a mask.
# Only used 0.5 till now.
MASK_THRESHOLD = 0.5

# Normalize to resnet mean and std if True.
NORMALIZE = False


# Use a StepLR scheduler if True. Not tried yet.
USE_SCHEDULER = False

# Amount of epochs
NUM_EPOCHS = 8


BOX_DETECTIONS_PER_IMG = 539


MIN_SCORE = 0.59


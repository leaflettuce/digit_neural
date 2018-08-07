# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

#import kaggle data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sub = pd.read_csv("data/sample_submission.csv")

#initiate classifier
model = Sequential() 
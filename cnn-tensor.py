# -*- coding: utf-8 -*-
"""
CNN for recognizing digits form handwriten script.
Kaggle already have images split out into pixel matrix
Simple test driving keras and CNN's for school project
(Much help from yassineghouzam intro to CNN Kernal in Kaggle)
"""
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

#import kaggle data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sub = pd.read_csv("data/sample_submission.csv")

#split training data
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1) 
del train 

#Normalize the data - Tranform to Grayscale
X_train = X_train / 255.0
test = test / 255.0

#reshape in 3 dimensions
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

#Encode labels 
Y_train = to_categorical(Y_train, num_classes = 10)

#split up train and test
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, 
                                                  test_size = 0.1, 
                                                  random_state=42)


#initiate classifier
model = Sequential() 

#convolution - add layers
model.add(Convolution2D(filters = 32, kernel_size = (3, 3), 
                        input_shape = (28, 28, 1), activation = 'relu'))
model.add(Convolution2D(filters = 32, kernel_size = (3, 3), 
                        input_shape = (28, 28, 1), activation = 'relu'))

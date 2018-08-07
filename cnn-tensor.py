# -*- coding: utf-8 -*-
"""
CNN for recognizing digits form handwriten script.
Kaggle already have images split out into pixel matrix
Simple test driving keras and CNN's for school project
(Much help from yassineghouzam intro to CNN Kernal in Kaggle)
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import ReduceLROnPlateau

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
model.add(Convolution2D(filters = 32, kernel_size = (5, 5), 
                        input_shape = (28, 28, 1), activation = 'relu'))

model.add(MaxPool2D(pool_size = (2, 2)))

#more complex layers                                  ###COMMENTED OUT TO SAVE TIME
#model.add(Convolution2D(filters = 64, kernel_size = (3,3), 
#                 activation ='relu'))
#model.add(MaxPool2D(pool_size=(2,2)))

#flatten
model.add(Flatten())

#Full Connection 
model.add(Dense(256, activation = "relu"))
model.add(Dense(10, activation = "softmax"))

# Compile //try RMSprop?
model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])


# Set a learning rate annealer -- drops LR by half is not improving after 3 epochs
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

epochs = 12   #test number  set higher for final
batch_size = 100

#FIT TIME
model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
          validation_data = (X_test, Y_test), verbose = 1)


#confusion matrix
Y_pred = model.predict(X_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_test, axis = 1) 
# compute the confusion matrix
confusion_matrix = confusion_matrix(Y_true, Y_pred_classes) 
print(confusion_matrix)


###Run on test and write out for Kaggle submission
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

sub.Label = results

sub.to_csv("subs/cnn_32conv-adam-12epoch.csv",index=False)
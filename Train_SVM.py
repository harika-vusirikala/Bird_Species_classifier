import numpy as np
#import pandas as pd
import keras
from keras.preprocessing import image
from keras.models import Sequential
from sklearn import svm
from keras.layers import *
from sklearn.metrics import accuracy_score

x_train = np.load("data/train_data_BW.npy")
x_test = np.load("data/test_data_BW.npy")
y_train = np.load("data/train_labels_BW.npy")
y_test = np.load("data/test_labels_BW.npy")

num_classes = 200
#print(y_train)

seed = 5
np.random.seed(seed)

s=np.arange(x_train.shape[0])
np.random.shuffle(s)
x_train = x_train[s]
y_train = y_train[s]

#x_train.reshape(len(x_train),-1)
print(x_train.shape)

svm_classifier = svm.SVC(C=100.0,gamma = 'auto',degree=4)
training_history = svm_classifier.fit(x_train,y_train)

y_pred = svm_classifier.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)




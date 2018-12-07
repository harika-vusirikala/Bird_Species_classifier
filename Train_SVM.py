'''Name         : Train_SVM.npy
   Date         : 11/27/18
   Author       : Harika Vusirikala
   Description  : This program loads training and testing data from the local device and trains and tests an SVM
                 classifier.
   Dependencies : data/train_data_BW.npy,
                  data/test_data_BW.npy,
                  data/train_labels_BW.npy,
                  data/test_labels_BW.npy
'''

import numpy as np
import keras
from sklearn import svm
from keras.layers import *
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,classification_report

'''Loading data from the local device'''
x_train = np.load("data/train_data_BW.npy")
x_test = np.load("data/test_data_BW.npy")
y_train = np.load("data/train_labels_BW.npy")
y_test = np.load("data/test_labels_BW.npy")

'''Number of classes is set to 201'''
num_classes = 200
#print(y_train)

'''Shuffling the input data'''
s=np.arange(x_train.shape[0])
np.random.shuffle(s)
x_train = x_train[s]
y_train = y_train[s]

'''Training SVM classifier'''
svm_classifier = svm.SVC(C=100.0,gamma = 'auto',degree=4)
training_history = svm_classifier.fit(x_train,y_train)

'''Testing the SVM classifier'''
y_pred = svm_classifier.predict(x_test)
accuracy = svm_classifier.score(x_test,y_test)

print(accuracy)

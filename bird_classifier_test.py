'''Name        : bird_classifier_test.py
   Date        : 12/19/18
   Author      : Harika Vusirikala
   Description : This program tests the trained model with test images
   Dependencies : data/test_data.npy,
                  data/test_labels.npy,
                Model/Bird_classifier_adam_v8.h5(Model to be tested)
   '''
import numpy as np
import keras
from keras.preprocessing import image
from keras.models import load_model
from keras.layers import *
import matplotlib.pyplot as plt

'''Loading test data and test labels from local disk'''
x_test = np.load("data/test_data.npy")
y_test = np.load("data/test_labels.npy")

'''Number of classes is set to 200'''
num_classes = 200
#y_test = y_test.astype('float32')-1

'''Encoding the test samples with one-hot encoding'''
y_test = keras.utils.to_categorical(y_test,num_classes)

'''Loading the model from the disk'''
model = load_model("Model/Bird_classifier_adam_v8.h5")
model.summary()

'''Testing the model with test data and their labels'''
score = model.evaluate(x_test,y_test,verbose=2)
'''Printing the test accuracy'''
print('Test accuracy:\n ',score)

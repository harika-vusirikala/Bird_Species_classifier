import numpy as np
import keras
from keras.preprocessing import image
from keras.models import load_model
from keras.layers import *
import matplotlib.pyplot as plt

x_test = np.load("data/test_data.npy")
y_test = np.load("data/test_labels.npy")

'''s=np.arange(x_test.shape[0])
np.random.shuffle(s)
x_test = x_test[s]
y_test = y_test[s]
print(y_test.shape)'''

num_classes = 201

y_test = keras.utils.to_categorical(y_test,num_classes)

input_shape = (100,100,3)
model = load_model("Bird_classifier_model3.h5")

score = model.evaluate(x_test,y_test,verbose=2)
print('Test accuracy: ',score[1])
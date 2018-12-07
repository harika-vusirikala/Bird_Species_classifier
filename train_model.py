'''Name         : train_model.py
   Date         : 12/19/18
   Author       : Harika Vusirikala
   Description  : This program trains an existing model.
   Dependencies : data/train_data.npy,
                  data/train_labels.npy,
                  Model/Bird_classifier_adam_v8.h5(Model to be tested)
   '''
import numpy as np
#import pandas as pd
import keras
from keras.preprocessing import image
from keras.models import load_model
from keras.layers import *
from keras.optimizers import RMSprop,adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

'''Loads Training data from the local disk'''
x_train = np.load("data/train_data.npy")
y_train = np.load("data/train_labels.npy")

'''Number of classes in the dataset'''
num_classes = 200

'''Shuffling the training data'''
s=np.arange(x_train.shape[0])
np.random.shuffle(s)
x_train = x_train[s]
y_train = y_train[s]
print(y_train)

'''Labels are encoded using one-hot encoding'''
y_train = keras.utils.to_categorical(y_train,num_classes)

'''Shape of each sample input given to the CNN'''
input_shape = (100,100,3)
'''Load the saved model from local disk'''
model = load_model("Bird_classifier_adam_v2.h5")

adam = adam(lr = 0.0001)

'''Image Data Generator for data augmentation as there are only 30 images per class'''
datagenerator = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)


datagenerator.fit(x_train)

'''Compiling the model'''
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy','top_k_categorical_accuracy'])

'''Training the model'''
training_history = model.fit_generator(datagenerator.flow(x_train,y_train,batch_size = 200),epochs = 100,verbose = 2,
                    steps_per_epoch = x_train.shape[0]//200)
'''Save the model to disk'''
model.save('Bird_classifier_adam_v3.h5')
print('Model saved to disk')

'''Plotting training accuracy and loss'''
plt.plot(training_history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

'''Name         : Image_processing.py
   Date         : 11/12/18
   Author       : Harika Vusirikala
   Description  : This program loads the training data from the local disk and trains the model.
   Dependencies : data/train_data.npy,
                  data/train_labels.npy
   '''

import numpy as np
#import pandas as pd
import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

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
#print(y_train)
#print(x_train.shape)
y_train = keras.utils.to_categorical(y_train,num_classes)

'''Shape of each sample input given to the CNN'''
input_shape = (100,100,3)

'''Creating a sequential Model'''
model = Sequential()

'''Adding 8 convolution and max-pooling layers with activation function as RELU'''
model.add(Conv2D(32,kernel_size=(3,3),activation='linear',input_shape=input_shape))
model.add(LeakyReLU(alpha=0.01))
model.add(MaxPooling2D(pool_size= (2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(64,kernel_size=(3,3),activation='linear'))
model.add(LeakyReLU(alpha=0.01))
model.add(MaxPooling2D(pool_size= (2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(32,kernel_size=(3,3),activation='linear'))
model.add(LeakyReLU(alpha=0.01))
model.add(MaxPooling2D(pool_size= (2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(64,kernel_size=(3,3),activation='linear'))
model.add(LeakyReLU(alpha=0.01))
model.add(MaxPooling2D(pool_size= (2,2)))

model.add(Dropout(0.25))

'''Filtered response is flattened'''
model.add(Flatten())
'''A fully connected layer with 256 nodes is added'''
model.add(Dense(256,input_dim=1024, activation='linear',kernel_regularizer=regularizers.l2(0.0001)))
model.add(LeakyReLU(alpha=0.01))
model.add(Dropout(0.5))

'''Softmax calssifier with 200 classes'''
model.add(Dense(num_classes,activation="softmax"))

model.summary()

#optimizer = RMSprop(lr=0.001,rho=0.9,epsilon = 1e-08,decay = 0.0)
adam = Adam(lr=0.01)
'''Image Data Generator for data augmentation as there are only 30 images per class'''
datagenerator = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=40,
        zoom_range = 0.1,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)

datagenerator.fit(x_train)

#model.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['accuracy'])
'''Compiling the model'''
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
'''Training the model'''
training_history = model.fit_generator(datagenerator.flow(x_train,y_train,batch_size = 50),epochs = 200,verbose = 2,
                    steps_per_epoch = x_train.shape[0]//50)
'''Save the model to disk'''
model.save('Model/Bird_classifier_adam_v9.h5')
print('Model saved to disk')

'''Plotting training accuracy and loss'''
plt.plot(training_history.history['acc'])
plt.title('Training Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

plt.plot(training_history.history['loss'])
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.show()

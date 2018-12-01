import numpy as np
#import pandas as pd
import keras
from keras.models import Model,Sequential
from keras.applications import inception_v3,vgg16
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.optimizers import adam,SGD
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt

x_train = np.load("data/train_data_224X224.npy")
x_train = vgg16.preprocess_input(x_train)
y_train = np.load("data/train_labels_224X224.npy")

#number of classes in your dataset e.g. 20
num_classes = 200

s=np.arange(x_train.shape[0])
np.random.shuffle(s)
x_train = x_train[s]
y_train = y_train[s]
#print(y_train)
#print(x_train.shape)
y_train = keras.utils.to_categorical(y_train,num_classes)

base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (224,224,3))

x = Flatten()(base_model.output)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
predictions = Dense(num_classes, activation = 'softmax')(x)

#create graph of your new model
head_model = Model(input = base_model.input, output = predictions)

#compile the model
head_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

head_model.summary()

sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
head_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

training_history = head_model.fit(x_train,y_train,batch_size=16,shuffle=True,epochs=20,verbose=2,validation_split=0.2)

head_model.save('bird_classifier_vgg16.h5')

print('Model saved to disk')

plt.plot(training_history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
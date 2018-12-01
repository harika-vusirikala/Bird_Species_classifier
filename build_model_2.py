import numpy as np
#import pandas as pd
import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import RMSprop,adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

x_train = np.load("data/train_data.npy")
y_train = np.load("data/train_labels.npy")
x_test = np.load("data/test_data.npy")
y_test = np.load("data/test_labels.npy")

num_classes = 201
#print(y_train)

#seed = 5
#np.random.seed(seed)

s=np.arange(x_train.shape[0])
np.random.shuffle(s)
x_train = x_train[s]
y_train = y_train[s]

s=np.arange(x_test.shape[0])
np.random.shuffle(s)
x_test = x_test[s]
y_test = y_test[s]

y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)

print(x_train.shape)
print(x_test.shape)

input_shape = (100,100,3)
model = Sequential()
'''model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size= (2,2)))

model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size= (2,2)))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size= (2,2)))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size= (2,2)))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256,input_dim=1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))

model.add(Dense(200,activation="softmax"))

model.summary()'''

model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(100,100,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(num_classes,activation="softmax"))

adam = adam(lr = 0.01)
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
#datagenerator.fit(x_train)

#model.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['accuracy'])
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
training_history = model.fit(x_train,y_train,batch_size = 100,epochs = 50, verbose = 2)
model.save('Bird_classifier_model4.h5')
print('Model saved to disk')

plt.plot(training_history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()


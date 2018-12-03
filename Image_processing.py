import numpy as np
#import pandas as pd
import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization

x_train = np.load("data/train_data.npy")
y_train = np.load("data/train_labels.npy")

num_classes = 201
#print(y_train)

seed = 5
np.random.seed(seed)

s=np.arange(x_train.shape[0])
np.random.shuffle(s)
x_train = x_train[s]
y_train = y_train[s]
#print(y_train)
#print(x_train.shape)
y_train = keras.utils.to_categorical(y_train,num_classes)

input_shape = (100,100,3)
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size= (2,2)))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size= (2,2)))

model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size= (2,2)))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size= (2,2)))

#model.add(Conv2D(64,kernel_size=(5,5),activation='relu'))
#model.add(MaxPooling2D(pool_size= (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256,input_dim=1024, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))

model.add(Dense(num_classes,activation="softmax"))

model.summary()

optimizer = RMSprop(lr=0.001,rho=0.9,epsilon = 1e-08,decay = 0.0)

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

#model.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['accuracy'])
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
training_history = model.fit_generator(datagenerator.flow(x_train,y_train,batch_size = 200),epochs = 350,verbose = 2,
                    steps_per_epoch = x_train.shape[0]//200)
model.save('Model/Bird_classifier_adam_v8.h5')
print('Model saved to disk')

plt.plot(training_history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()




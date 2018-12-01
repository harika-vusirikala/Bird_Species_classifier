import numpy as np
#import pandas as pd
import keras
from keras.preprocessing import image
from keras.models import load_model
from keras.layers import *
from keras.optimizers import RMSprop,adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

x_train = np.load("data/train_data.npy")
y_train = np.load("data/train_labels.npy")

num_classes = 200

s=np.arange(x_train.shape[0])
np.random.shuffle(s)
x_train = x_train[s]
y_train = y_train[s]
print(y_train)

y_train = keras.utils.to_categorical(y_train,num_classes)

input_shape = (100,100,3)
model = load_model("Bird_classifier_adam_v2.h5")

adam = adam(lr = 0.0001)
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

datagenerator.fit(x_train)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy','top_k_categorical_accuracy'])

training_history = model.fit_generator(datagenerator.flow(x_train,y_train,batch_size = 200),epochs = 100,verbose = 2,
                    steps_per_epoch = x_train.shape[0]//200)
model.save('Bird_classifier_adam_v3.h5')
print('Model saved to disk')

plt.plot(training_history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

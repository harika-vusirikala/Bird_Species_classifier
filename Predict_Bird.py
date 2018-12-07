'''Name         : Predict_Bird.py
   Date         : 12/04/18
   Author       : Harika Vusirikala
   Description  : This module predicts the bird species present in the image
   Usage        : python Predict_Bird.py <image_path relative to this file>
   Dependencies : CUB_200_2011/classes.txt,
                  Model/Bird_classifier_adam_v8.h5 (model to be used to predict)
'''
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
import sys

'''Retrieves image path given by the user through command line'''
image_path = sys.argv[1]

'''Retrieves the labels for each class from classes.txt file'''
classes ={}
with open("CUB_200_2011"+'/classes.txt') as file:
    for line in file:
        split_line = line.rstrip().split(' ')
        classes[int(split_line[0])] = split_line[1]

im_rows=100
im_cols= 100
'''Input image is converted to a numpy array'''
image =cv2.imread(image_path)
image = Image.fromarray(image,'RGB')
resize_img = image.resize((im_rows,im_cols))
image = np.array(resize_img).astype('float32')/255
image = image.reshape([-1,100,100,3])

'''Model is loaded from the local disk'''
model = load_model("Model/Bird_classifier_adam_v8.h5")

'''Predicting the bird species in the image'''
pred= model.predict_classes(image)
'''Printing the prediction'''
print(classes[pred[0]])

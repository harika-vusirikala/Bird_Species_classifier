'''This program loads the data and saves them to the local machine for further use
    Base Directory: CUB_200_2011
    Images Directory : CUB_200_2011/images
'''

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
#import os

base_dir = "CUB_200_2011"
images_dir = base_dir+"/images/"
im_rows = 100
im_cols = 100

# Reading the file classes.txt
# File Structure: class_id class_name
classes = {}
with open(base_dir+'/classes.txt') as file:
    for line in file:
        split_line = line.rstrip().split(' ')
        classes[split_line[0]] = split_line[1]

# Reading the file images.txt
# File Structure: image_id image_path
images_set = {}
with open(base_dir+"/images.txt") as file:
    for line in file:
        split_line = line.rstrip().split(' ')
        images_set[split_line[0]] = split_line[1]

# Reading the file image_class_set.txt. This file connects each image to its class.
# File Structure: image_id class_id
image_class_set = {}
with open(base_dir+'/image_class_labels.txt') as file:
    for line in file:
        split_line = line.rstrip().split(' ')
        image_class_set[split_line[0]] = split_line[1]
# Splitting the image ids for train and test datasets
train_set=[]
test_set = []

# Reading the file train_test_split.txt. This file tells whether an image is a
# training image or not. 1 represents training image, 0 represents testing image
# File Structure: image_id   is training image?
with open(base_dir+'/train_test_split.txt') as file:
    for line in file:
        split_line = line.rstrip().split(' ')
        if split_line[1] == '1':
            train_set.append(split_line[0])
        elif split_line[1] == '0':
            test_set.append(split_line[0])

train_data = []
train_labels = []
test_data = []
test_labels = []

# Reading each image in the training set and croping it to size 224 X 224
# and adding it to the training data. Corresponding label is added to
# training labels
'''for train_image_id in train_set:
    image =cv2.imread(images_dir+images_set[train_image_id])
    image = Image.fromarray(image,'RGB')
    resize_img = image.resize((im_rows,im_cols))
    normalize_array = np.array(resize_img).astype('float32')/255
    train_data.append(normalize_array)
    train_labels.append(image_class_set[train_image_id])

train_array_data = np.array(train_data)
print(train_array_data.shape)
train_labels = np.array(train_labels)
print("Saving Training data")
np.save("data/train_data",train_array_data)
print("Saving Training labels")
np.save("data/train_labels",train_labels)'''
# Reading each image in the testing set and croping it to size 244 X 244
# and adding it to the testing data. Corresponding label is added to
# testing labels
for test_image_id in test_set:
    image =cv2.imread(images_dir+images_set[test_image_id])
    image = Image.fromarray(image, 'RGB')
    resize_img = image.resize((im_rows, im_cols))
    normalize_array = np.array(resize_img).astype('float32') / 255
    test_data.append(normalize_array)
    test_labels.append(image_class_set[test_image_id])

# converting each array to a numpy array
test_data = np.array(test_data)
test_labels = np.array(test_labels)
print(test_data.shape)
# Saving the data local directory so that it can be loaded later.
print("Saving Testing data")
np.save("data/test_data",test_data)
print("Saving Testing labels")
np.save("data/test_labels",test_labels)
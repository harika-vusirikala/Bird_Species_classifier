# Bird_Species_classifier
A Deep Neural Network to classify 200 bird species 

Loading_data.py - Loads the data from CUB-200-2011 dataset, splits the data into training and test data. Saves this data to local disk. Generates 4 files train_set.npy, train_labels.npy, test_set.npy, test_labels.npy

Image_Processing.py - Loads the training data and labels from the local disk. Creates Model_1 and trains it with training data. The trained model is saved to the disk.

Build_model_2.py - Loads the training data and labels from the local disk. Creates Model_2 and trains it with training data. The trained model is saved to the disk.

train_model.py - Trains a saved model with training data. This file is created to further train a model.

bird_classifier.test - Loads and tests a saved model. Prints the test accuracy.

Predict_Bird.py - Predicts the bird in the input image and prints it.

Loading_data_SVM.py - Loads the dataset to be used for SVM classifier.

Train_SVM.py - Trains and tests SVM classifier.


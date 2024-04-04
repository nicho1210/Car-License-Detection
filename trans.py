# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 13:58:57 2022
@author: 2540817538(有问题请联系此QQ)
python3.8.8
"""
import cv2
import os
import struct
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import cv2 as cv

 
def trans(image, label, save):#image位置，label位置和转换后的数据保存位置
    if 'train' in os.path.basename(image):
        prefix = 'train'
    else:
        prefix = 'test'
 
    labelIndex = 0
    imageIndex = 0
    i = 0
    lbdata = open(label, 'rb').read()
    magic, nums = struct.unpack_from(">II", lbdata, labelIndex)
    labelIndex += struct.calcsize('>II')
 
    imgdata = open(image, "rb").read()
    magic, nums, numRows, numColumns = struct.unpack_from('>IIII', imgdata, imageIndex)
    imageIndex += struct.calcsize('>IIII')
 
    for i in range(nums):
        label = struct.unpack_from('>B', lbdata, labelIndex)[0]
        labelIndex += struct.calcsize('>B')
        im = struct.unpack_from('>784B', imgdata, imageIndex)
        imageIndex += struct.calcsize('>784B')
        im = np.array(im, dtype='uint8')
        img = im.reshape(28, 28)
        #Flip the image horizontally and then rotate it 90 degrees anti-clockwise.
        img = cv2.flip(img, 1)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        save_name = os.path.join(save, '{}_{}_{}.jpg'.format(prefix, i, label))
        cv2.imwrite(save_name, img)
 
if __name__ == '__main__':
    #需要更改的文件路径！！！！！！
    #此处是原始数据集位置
    train_images = 'emnist-letters-train-images-idx3.ubyte'
    train_labels = 'emnist-letters-train-labels-idx1.ubyte'
    test_images ='emnist-letters-test-images-idx3.ubyte'
    test_labels = 'emnist-letters-test-labels-idx1.ubyte'
    #此处是我们将转化后的数据集保存的位置
    save_train ='train_images'
    save_test ='test_images'
    
    if not os.path.exists(save_train):
        os.makedirs(save_train)
    if not os.path.exists(save_test):
        os.makedirs(save_test)
    """
    trans(test_images, test_labels, save_test)
    trans(train_images, train_labels, save_train)
    """
    
    #training the model
    train_images_folder = 'train_images'
    test_images_folder = 'test_images'

    train_images = os.listdir(train_images_folder)
    train_labels = [int(image.split('_')[-1].split('.')[0]) for image in train_images]

    test_images = os.listdir(test_images_folder)
    test_labels = [int(image.split('_')[-1].split('.')[0]) for image in test_images]
    # Load the images and labels from the train_images folder
    train_images_data = []
    train_labels_data = []
    for image_name in train_images:
        image_path = os.path.join(train_images_folder, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        train_images_data.append(image)
        label = int(image_name.split('_')[-1].split('.')[0])
        train_labels_data.append(label)

    # Load the images and labels from the test_images folder
    test_images_data = []
    test_labels_data = []
    for image_name in test_images:
        image_path = os.path.join(test_images_folder, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        test_images_data.append(image)
        label = int(image_name.split('_')[-1].split('.')[0])
        test_labels_data.append(label)

    # Convert the data to numpy arrays
    train_images_data = np.array(train_images_data)
    train_labels_data = np.array(train_labels_data)
    test_images_data = np.array(test_images_data)
    test_labels_data = np.array(test_labels_data)

    # Normalize the pixel values
    train_images_data = train_images_data / 255.0
    test_images_data = test_images_data / 255.0

    # Reshape the data to match the input shape of the DNN model
    train_images_data = train_images_data.reshape(train_images_data.shape[0], 28, 28, 1)
    test_images_data = test_images_data.reshape(test_images_data.shape[0], 28, 28, 1)

    # Convert the labels to categorical format
    train_labels_data = to_categorical(train_labels_data)
    test_labels_data = to_categorical(test_labels_data)

    # Define the DNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(27, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_images_data, train_labels_data, epochs=10, batch_size=32, validation_data=(test_images_data, test_labels_data))
    
    model.save('car_license_model_en_ver1.h5')
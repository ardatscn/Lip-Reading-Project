# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:01:24 2022

@author: PCA
"""

import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
from keras.layers.normalization import *
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import os
import seaborn as sns
from keras.applications.vgg16 import VGG16

words = ['afiyetolsun', 'basla', 'bitir', 'gorusmekuzere', 'gunaydin','hosgeldiniz','ozurdilerim','tesekkurederim','selam','merhaba']
labels = ['Hoş geldiniz', 'Özür dilerim', 'Teşekkür ederim', 'Merhaba', 'Selam', 'Günaydın']
n_labels = len(words)
root_path = 'dataset/frames/'
shape_pred_path = 'dataset/shape_predictor_68_face_landmarks.dat' 
# words = os.listdir(root_path)

root_path_mouth = 'dataset/mouth2/'

# Read input images and assign labels based on folder names
print(os.listdir("dataset"))

SIZE = 50  #Resize images

#Capture training data and labels into respective lists
train_images = []
train_labels = [] 
root_path_train = 'dataset/collage/train/'

for word in words:    
    image_list = os.listdir(root_path_train + word)
    for im in image_list:
        img = cv2.imread(root_path_train + word + '/' + im)
        img = cv2.resize(img, (SIZE, SIZE))
        train_images.append(img)
        train_labels.append(word)
#Convert lists to arrays        
train_images = np.array(train_images)
train_labels = np.array(train_labels)


# Capture test/validation data and labels into respective lists

test_images = []
test_labels = [] 
root_path_test = 'dataset/collage/test/'
for word in words:    
    image_list = os.listdir(root_path_test + word)
    for im in image_list:
        img = cv2.imread(root_path_test + word + '/' + im)
        img = cv2.resize(img, (SIZE, SIZE))
        test_images.append(img)
        test_labels.append(word)
        
#Convert lists to arrays                
test_images = np.array(test_images)
test_labels = np.array(test_labels)

#Encode labels from text to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

#Split data into test and train datasets
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

#One hot encode y values for neural network. 
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

#Load VGG16
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

#Make the layers nontrainable.
for layer in VGG_model.layers:
	layer.trainable = False
    
VGG_model.summary()
x = Flatten()(VGG_model.output)

prediction = Dense(10, activation='softmax')(x)
model = Model(inputs=VGG_model.input, outputs=prediction)

model.summary()

model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=15)
model.evaluate(x_test,y_test)

















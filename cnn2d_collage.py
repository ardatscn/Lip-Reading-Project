# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 13:23:25 2022

@author: PCA
"""
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import seaborn as sns
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import cv2
from imutils import face_utils
from sklearn.metrics import confusion_matrix
# import imutils
import os
# from sklearn.preprocessing import MinMaxScaler
# from google.colab.patches import cv2_imshow
from skimage.transform import resize
import imageio

np.random.seed(3)

# In[ ]:


# speakers = ['F01','F02','F04','F05','F06','M01','M02','M04','M07','M08'] 
# word_folder = ['01','02','03']
# varieties = ['01','02','03','04','05','06','07','08', '09', '10']


words = ['afiyetolsun', 'basla', 'bitir', 'gorusmekuzere', 'gunaydin','hosgeldiniz','ozurdilerim','tesekkurederim','selam','merhaba']
labels = ['afiyetolsun', 'basla', 'bitir', 'gorusmekuzere', 'gunaydin','hosgeldiniz','merhaba','ozurdilerim','selam','tesekkurederim']
n_labels = len(words)
root_path = 'dataset/frames/'
shape_pred_path = 'dataset/shape_predictor_68_face_landmarks.dat' 
# words = os.listdir(root_path)

root_path_mouth = 'dataset/mouth2/'

# Read input images and assign labels based on folder names
print(os.listdir("dataset"))

SIZE = 256  #Resize images

#Capture training data and labels into respective lists
train_images = []
train_labels = [] 
root_path_train = 'dataset/collage/train/'

for word in words:    
    image_list = os.listdir(root_path_train + word)
    for im in image_list:
        img = cv2.imread(root_path_train + word + '/' + im,0)
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
        img = cv2.imread(root_path_test + word + '/' + im,0)
        img = cv2.resize(img, (SIZE, SIZE))
        test_images.append(img)
        test_labels.append(word)
#Convert lists to arrays                
test_images = np.array(test_images)
test_labels = np.array(test_labels)

valid_images = []
valid_labels = []
root_path_valid = 'dataset/collage/valid/'
for word in words:    
    image_list = os.listdir(root_path_valid + word)
    for im in image_list:
        img = cv2.imread(root_path_valid + word + '/' + im,0)
        img = cv2.resize(img, (SIZE, SIZE))
        valid_images.append(img)
        valid_labels.append(word)
#Convert lists to arrays                
valid_images = np.array(valid_images)
valid_labels = np.array(valid_labels)

#Encode labels from text to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)
le.fit(valid_labels)
valid_labels_encoded = le.transform(valid_labels)

#Split data into test and train datasets (already split but assigning to meaningful convention)
x_train, y_train, x_test, y_test,x_valid,y_valid = train_images, train_labels_encoded, test_images, test_labels_encoded,valid_images,valid_labels_encoded

###################################################################
# Normalize pixel values to between 0 and 1
x_train, x_test,x_valid = x_train / 255.0, x_test / 255.0, x_valid / 255.0

#One hot encode y values for neural network. 
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
y_valid_one_hot = to_categorical(y_valid)

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(5,5), activation = 'relu', input_shape = (256,256,1)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(32,(5,5), activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(64,(5,5), activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64, activation = 'relu'),
                                    keras.layers.Dropout(0.4),  
                                    tf.keras.layers.Dense(10, activation = 'softmax')])

model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = 'adam',
              metrics = ['accuracy'])

callbacks = [EarlyStopping(monitor='val_accuracy', patience=4)] # Early Stopping
history = model.fit(x_train, y_train, epochs=100, validation_data = (x_valid,y_valid), callbacks=callbacks)
y_pred = model.predict(x_test)
y_classes = [np.argmax(element) for element in y_pred]
cm = confusion_matrix(test_labels_encoded, y_classes)
cm_plot_labels = ['afiyetolsun', 'basla', 'bitir', 'gorusmekuzere', 'gunaydin','hosgeldiniz','merhaba','ozurdilerim','selam','tesekkurederim']
sns.heatmap(cm,xticklabels =['afiyetolsun', 'basla', 'bitir', 'gorusmekuzere', 'gunaydin','hosgeldiniz','merhaba','ozurdilerim','selam','tesekkurederim'], yticklabels=['afiyetolsun', 'basla', 'bitir', 'gorusmekuzere', 'gunaydin','hosgeldiniz','merhaba','ozurdilerim','selam','tesekkurederim'], annot=True)

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

model.evaluate(x_test,y_test)
# model = Sequential()

# model.add(Conv2D(256, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(256, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

# model.add(Dense(64))

# model.add(Dense(1))
# model.add(Activation('sigmoid'))

# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# model.fit(train_dataset, batch_size=32, epochs=3)
# # one_hot_encoded_labels = to_categorical(words)
# # features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels,
# #                                                                             test_size = 0.25, shuffle = True,
# #                                                                             random_state = seed_constant)












# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:13:09 2022

@author: PCA
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 13:23:25 2022

@author: PCA
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
import os

words = ['afiyetolsun', 'basla', 'bitir', 'gorusmekuzere', 'gunaydin','hosgeldiniz','ozurdilerim','tesekkurederim','selam','merhaba']
labels = ['afiyetolsun', 'basla', 'bitir', 'gorusmekuzere', 'gunaydin','hosgeldiniz','merhaba','ozurdilerim','selam','tesekkurederim']
n_labels = len(words)
root_path = 'dataset/frames/'
shape_pred_path = 'dataset/shape_predictor_68_face_landmarks.dat' 
root_path_mouth = 'dataset/mouth2/'

print(os.listdir("dataset"))

SIZE = 256  #Resize images

train_images = []
train_labels = [] 
root_path_train = 'dataset/collage_notmodified/train/'

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

test_images = []
test_labels = [] 
root_path_test = 'dataset/collage_notmodified/test/'
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
root_path_valid = 'dataset/collage_notmodified/valid/'
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

#Encode labels to integers
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)
le.fit(valid_labels)
valid_labels_encoded = le.transform(valid_labels)

#Split data
x_train, y_train, x_test, y_test,x_valid,y_valid = train_images, train_labels_encoded, test_images, test_labels_encoded,valid_images,valid_labels_encoded

# Normalize pixel values
x_train, x_test,x_valid = x_train / 255.0, x_test / 255.0, x_valid / 255.0

#One hot encode values
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
                                    tf.keras.layers.Dense(128, activation = 'relu'),
                                    keras.layers.Dropout(0.2),  
                                    tf.keras.layers.Dense(10, activation = 'softmax')])

model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = 'adam',
              metrics = ['accuracy'])

callbacks = [EarlyStopping(monitor='val_accuracy', patience=4)]
history = model.fit(x_train, y_train, epochs=100, validation_data = (x_valid,y_valid), callbacks=callbacks)
y_pred = model.predict(x_test)
y_classes = [np.argmax(element) for element in y_pred]
cm = confusion_matrix(test_labels_encoded, y_classes)
cm_plot_labels = ['afiyetolsun', 'basla', 'bitir', 'gorusmekuzere', 'gunaydin','hosgeldiniz','merhaba','ozurdilerim','selam','tesekkurederim']
sns.heatmap(cm,xticklabels =['afiyetolsun', 'basla', 'bitir', 'gorusmekuzere', 'gunaydin','hosgeldiniz','merhaba','ozurdilerim','selam','tesekkurederim'], yticklabels=['afiyetolsun', 'basla', 'bitir', 'gorusmekuzere', 'gunaydin','hosgeldiniz','merhaba','ozurdilerim','selam','tesekkurederim'], annot=True)
model.evaluate(x_test,y_test)










# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:27:12 2022

@author: PCA
"""

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
from keras.layers import TimeDistributed
# import imutils
import os
# from sklearn.preprocessing import MinMaxScaler
# from google.colab.patches import cv2_imshow
from skimage.transform import resize
import imageio
# import mlflow.tensorflow
np.random.seed(3)

# In[ ]:


# speakers = ['F01','F02','F04','F05','F06','M01','M02','M04','M07','M08'] 
# word_folder = ['01','02','03']
# varieties = ['01','02','03','04','05','06','07','08', '09', '10']

# mlflow.tensorflow.autolog()
words = ['afiyetolsun', 'basla', 'bitir', 'gorusmekuzere', 'gunaydin','hosgeldiniz','ozurdilerim','tesekkurederim','selam','merhaba']
labels = ['afiyetolsun', 'basla', 'bitir', 'gorusmekuzere', 'gunaydin','hosgeldiniz','merhaba','ozurdilerim','selam','tesekkurederim']
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
root_path_train = 'dataset/mouth_CNN+BiGRU/train/'

for word in words:    
    video_list = os.listdir(root_path_train + word)
    for video in video_list:
        im_list = os.listdir(root_path_train + word + '/' + video)
        for im in im_list:
            img = cv2.imread(root_path_train + word + '/' + video + '/' + im,0)
            img = cv2.resize(img, (SIZE, SIZE))
            train_images.append(img)
            train_labels.append(word)
#Convert lists to arrays        
train_images = np.array(train_images)
train_labels = np.array(train_labels)


# Capture test/validation data and labels into respective lists

test_images = []
test_labels = [] 
root_path_test = 'dataset/mouth_CNN+BiGRU/test/'
for word in words:    
    video_list = os.listdir(root_path_test + word)
    for video in video_list:
        im_list = os.listdir(root_path_test + word + '/' + video)
        for im in im_list:
            img = cv2.imread(root_path_test + word + '/' + video + '/' + im,0)
            img = cv2.resize(img, (SIZE, SIZE))
            test_images.append(img)
            test_labels.append(word)
#Convert lists to arrays                
test_images = np.array(test_images)
test_labels = np.array(test_labels)

valid_images = []
valid_labels = []
root_path_valid = 'dataset/mouth_CNN+BiGRU/valid/'
for word in words:    
    video_list = os.listdir(root_path_valid + word)
    for video in video_list:
        im_list = os.listdir(root_path_valid + word + '/' + video)
        for im in im_list:
            img = cv2.imread(root_path_valid + word + '/' + video + '/' + im,0)
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

x_train = x_train.reshape(len(x_train),50,50,1)
x_test = x_test.reshape(len(x_test),50,50,1)
x_valid = x_valid.reshape(len(x_valid),50,50,1)

model = Sequential()
model.add(tf.keras.layers.Conv2D(16,(5,5), input_shape= (50,50,1), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2)),
model.add(tf.keras.layers.Conv2D(32,(5,5), activation='relu')),
model.add(tf.keras.layers.MaxPool2D(2,2)),
model.add(tf.keras.layers.Conv2D(64,(5,5), activation='relu')),
model.add(tf.keras.layers.MaxPool2D(2,2)),
model.add(Flatten(name = 'FFF')),


model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = 'adam',
              metrics = ['accuracy'])

modelFeatured = models.Model(inputs = model.input,
                             outputs = model.get_layer('FFF').output)

train_featured = modelFeatured.predict(x_train)
test_featured = modelFeatured.predict(x_test)
valid_featured = modelFeatured.predict(x_valid)

temp = train_featured[:25]
temp = temp.reshape(1,temp.shape[0],temp.shape[1])
for i in range(25,21725,25):
    temp = np.append(temp, train_featured[i:i+25].reshape(1,25,train_featured.shape[1]), axis = 0)
trainX = temp

temp = test_featured[:25]
temp = temp.reshape(1,temp.shape[0],temp.shape[1])
for i in range(25,12775,25):
    temp = np.append(temp, test_featured[i:i+25].reshape(1,25,test_featured.shape[1]), axis = 0)
testX = temp

temp = valid_featured[:25]
temp = temp.reshape(1,temp.shape[0],temp.shape[1])
for i in range(25,7600,25):
    temp = np.append(temp, valid_featured[i:i+25].reshape(1,25,valid_featured.shape[1]), axis = 0)
validX = temp

temp_train = []
for i in range (869):
    n = i * 25
    temp_train.append(y_train[n])

temp_train = np.array(temp_train)

temp_test = []
for i in range (511):
    n = i * 25
    temp_test.append(y_test[n])

temp_test = np.array(temp_test)

temp_valid = []
for i in range (304):
    n = i * 25
    temp_valid.append(y_valid[n])

temp_valid = np.array(temp_valid)

n_steps = 25
n_features = 256

model1 = Sequential()
model1.add(Bidirectional(GRU(64, return_sequences = True)))
model1.add(Bidirectional(GRU(64)))
model1.add(Dense(128, activation='relu'))
model1.add(Dense(10, activation='softmax'))

model1.compile(loss = "sparse_categorical_crossentropy",
              optimizer = 'adam',
              metrics = ['accuracy'])
callbacks = [EarlyStopping(monitor='val_accuracy', patience=5)] # Early Stopping
history = model1.fit(trainX,temp_train, epochs=50, validation_data = (validX,temp_valid), callbacks = callbacks)

y_pred = model1.predict(testX)
y_classes = [np.argmax(element) for element in y_pred]
# cm = confusion_matrix(temp_test, y_classes)
# cm_plot_labels = ['afiyetolsun', 'basla', 'bitir', 'gorusmekuzere', 'gunaydin','hosgeldiniz','merhaba','ozurdilerim','selam','tesekkurederim']
# sns.heatmap(cm,xticklabels =['afiyetolsun', 'basla', 'bitir', 'gorusmekuzere', 'gunaydin','hosgeldiniz','merhaba','ozurdilerim','selam','tesekkurederim'], yticklabels=['afiyetolsun', 'basla', 'bitir', 'gorusmekuzere', 'gunaydin','hosgeldiniz','merhaba','ozurdilerim','selam','tesekkurederim'], annot=True)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

model1.evaluate(testX,temp_test)









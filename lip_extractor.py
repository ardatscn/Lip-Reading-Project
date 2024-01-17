# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 13:23:25 2022

@author: PCA
"""

import numpy as np
import cv2
from imutils import face_utils
import dlib 
import os

words = ['hosgeldiniz', 'merhaba', 'ozurdilerim', 'tesekkurederim', 'selam']
labels = ['Hoş geldiniz', 'Özür dilerim', 'Teşekkür ederim', 'Merhaba', 'Selam', 'Günaydın']
n_labels = len(words)
root_path = 'dataset/basla50/'
shape_pred_path = 'dataset/shape_predictor_68_face_landmarks.dat' 

root_path_mouth = 'dataset/basla50mouth/'

os.mkdir(root_path_mouth)
for word in words:    
    videos = os.listdir(root_path + word)
    os.mkdir(root_path_mouth + word)
    for video in videos:
        print(video) #001, 002, 003
        os.mkdir(root_path_mouth + word + '/' + video)
        sequence = []
        image_list = os.listdir(root_path + word + '/' + video)
        for im in image_list:
            image = cv2.imread(root_path + word + '/' + video + '/' + '/' + im, 0)
            face=dlib.get_frontal_face_detector()(image, 1)
            for each in face:
                face_points=dlib.shape_predictor(shape_pred_path)(image,each)
                face_points = face_utils.shape_to_np(face_points)
                (x, y, w, h) = cv2.boundingRect(np.array([face_points[49:68]])) # 48-68 mouth points
                (a,b) = face_points[49]
                mouth = image[y:y+h, x:x+w]
                try:
                    mouth = cv2.resize(mouth, (50, 50))
                    print(video)
                    cv2.imwrite(root_path_mouth + word + '/' + video + '/' + im, mouth)
                except Exception as e:
                    print(e)
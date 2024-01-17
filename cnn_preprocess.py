# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 11:40:44 2022

@author: PCA
"""

import numpy as np
import cv2
import os

words = ['afiyetolsun', 'basla', 'bitir', 'gorusmekuzere', 'gunaydin','hosgeldiniz','ozurdilerim','tesekkurederim','selam','merhaba']
labels = ['Hoş geldiniz', 'Özür dilerim', 'Teşekkür ederim', 'Merhaba', 'Selam', 'Günaydın']
root_path = 'dataset/mouth/'
shape_pred_path = 'dataset/shape_predictor_68_face_landmarks.dat' 

word_length = 15
input_dim = 50
channel = 1
stretch_seq =[]
images = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
l = 0
images_last=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

for word in words:    
    videos = os.listdir(root_path + word)
    for video in videos:
        image_list = os.listdir(root_path + word + '/' + video)
        for i in range(25):
            images[i] = image_list[int(i*len(image_list)/25)]
            if (i == 24):
                for im in images:
                    images_last[l] = cv2.imread(root_path + word + '/' + video + '/'  + im, 0)
                    l = l+1
                images_last = np.array(images_last)
                Horizontal1=np.hstack([images_last[0],images_last[1],images_last[2],images_last[3],images_last[4]])
                Horizontal2=np.hstack([images_last[5],images_last[6],images_last[7],images_last[8],images_last[9]])
                Horizontal3=np.hstack([images_last[10],images_last[11],images_last[12],images_last[13],images_last[14]])
                Horizontal4=np.hstack([images_last[15],images_last[16],images_last[17],images_last[18],images_last[19]])
                Horizontal5=np.hstack([images_last[20],images_last[21],images_last[22],images_last[23],images_last[24]])
                Vertical_attachment=np.vstack([Horizontal1,Horizontal2,Horizontal3,Horizontal4,Horizontal5])
                cv2.imwrite('dataset/collage_notmodified/'+ word+ '/' + im ,Vertical_attachment)
                l = 0
        







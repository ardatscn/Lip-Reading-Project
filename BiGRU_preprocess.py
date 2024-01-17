import cv2
import os

words = ['afiyetolsun', 'basla', 'bitir', 'gorusmekuzere', 'gunaydin','hosgeldiniz','ozurdilerim','tesekkurederim','selam','merhaba']
labels = ['Hoş geldiniz', 'Özür dilerim', 'Teşekkür ederim', 'Merhaba', 'Selam', 'Günaydın']
root_path = 'dataset/mouth/'
root_path_mouth = 'dataset/mouth_CNN+BiGRU/'

images = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
l = 0
images_last=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

for word in words:    
    videos = os.listdir(root_path + word)
    os.mkdir(root_path_mouth + word)
    for video in videos:
        image_list = os.listdir(root_path + word + '/' + video)
        os.mkdir(root_path_mouth + word + '/' + video)
        for i in range(25):
            images[i] = image_list[int(i*len(image_list)/25)]
            if (i == 24):
                n = 0
                for im in images:
                    n = str(n)
                    visual = cv2.imread(root_path + word + '/' + video  + '/' + im, 0)
                    cv2.imwrite('dataset/mouth_CNN+BiGRU/'+ word + '/' + video + '/' + n + '.jpg' ,visual)
                    n = int(n)
                    n = n + 1
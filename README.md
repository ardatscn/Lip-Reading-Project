# Lip-Reading-Project
There were videos of thousands of Turkish speakers 
saying 10 different words which were gathered from YouTube videos. The purpose 
was to acquire the highest accuracy rate of classification of these videos to these 10 
classes (10 words) using machine learning or deep learning models. This project was 
conducted on python and it involved detailed research on articles for detecting facial
landmarks from images, data pre-processing, feature extraction, deep learning 
models (CNN, LSTM, BiGRU, etc.), neural networks, overfitting, underfitting, 
regularization and related topics. Some models were created that can actually 
achieve lip reading to classify unlabeled data to 10 different classes. These models 
were evaluated and compared with each other to find the most powerful model that 
serves this task. For comparison and adjustment of hyperparameters of the models, 
MlFlow was used that is responsible for keeping track of the hyperparameters. The 
motivation behind this work was to acquire the highest accuracy of lip reading. The 
work done for Lip Reading Project was significant because every engineer in the 
project was focusing on different models, approaches, pre-processing methods etc. 
that could best achieve the lip-reading purpose and in this sense, my work had some 
unique methods and it helped them to get the results of various models that can be 
compared and evaluated with other engineers’ approaches.

A technical report was also included that summarizes the findings. This project was
conducted in an internship, therefore this report contains some parts related to
the company or my mentors. The report also gives brief information about another
AI project which is "Avionic Maintenance Prediction".

To also give a brief summary here:
Lip reading is an important concept for enhancing speech recognition in noisy or loud 
environments. Some of the significant aspects of the topic is that it could be used to 
enhance security or to help people with hearing loss to communicate. Although lip 
reading has been performed by professionals for decades, with the improvements in 
Computer Vision and Deep Learning, machines have the potential to perform 
automated lip reading (ALR) which is the process of understanding what is being 
spoken based solely on the visuals. This project uses machine learning to 
accomplish ALR by applying deep learning concepts to a non-synthetic dataset of 
commonly used Turkish words gathered from YouTube videos.
The goal of the project is to use implement several deep learning models for pattern 
recognition and classification on the given dataset that consists of frames of 10
Turkish words cut from YouTube videos that has separated to train, test and 
validation datasets. Several deep learning approaches was experimented through project 
to find the method that gives the highest accuracy for classification of the frames in 
the test dataset to the correct class (“Afiyet Olsun”, “Günaydın”, etc.). The purpose of 
this project can be summarized as to find the deep learning model that give the 
highest test accuracy rate on the given dataset.


Initially the dataset contained 10 Turkish words: “Afiyet olsun”, “Başla”, “Bitir”, 
“Görüşmek Üzere”, “Günaydın”, “Hoş Geldiniz”, “Merhaba”, “Özür Dilerim”, “Selam”, 
“Teşekkür Ederim”. Every word contributed to a separate class. For the preparation 
of the dataset, 5-25 frames were cut from YouTube videos that demonstrated the 
pronunciation of the words (the dataset was already created and summer training did 
not involve creating a dataset at any point). Each word had 200-250 videos assigned 
and a total of 2000-3500 frames were gathered. Initially the dataset was divergent in 
terms of images per word since longer words like “Teşekkür Ederim” contained more 
images per video. After the pre-processing techniques applied this situation was solved for 
every deep learning model created.
Three deep learning models (or methods) were used on the dataset. They were 
CNN, CNN + BiGRU (pre-processing was different than just CNN), Transfer 
Learning of VGG16.

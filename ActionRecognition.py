#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D, Attention, Dense, Flatten, MaxPool3D, MaxPool2D,BatchNormalization, Conv3D
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import os
from scipy.io import loadmat
from tensorflow.keras import Model
import cv2
import pickle

labels_path = './Labels_MERL_Shopping_Dataset/'
results_path = './Results_MERL_Shopping_Dataset/DetectedActions/'
videos_path = './Videos_MERL_Shopping_Dataset/'

x_train_path = videos_path+'train/'
y_train_path = 'train_y.pkl'

x_test_path = videos_path + 'test/'
y_test_path = 'test_y.pkl'

x_val_path = videos_path + '/val/'
y_val_path = 'val_y.pkl'


class newGen(Sequence):
    
    def __init__(self, x_path, folder_name, y_path, to_fit = True, batch_size = 2, seq_len = 15):
        self.x_path = x_path + folder_name
        self.folder_name = folder_name
        self.y_path = y_path
        self.to_fit = to_fit
        self.all_frames = self.get_all_frames(self.x_path)
        self.targets = self.get_Y(y_path, folder_name)
        self.series_data = TimeseriesGenerator(self.all_frames, self.targets, length = seq_len, batch_size=batch_size)
        self.len = len(self.series_data)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        X, Y = self.series_data[index]
        return tf.cast(X, tf.float16), Y

    def get_all_frames(self, x_path):
        images_list = sorted(os.listdir(self.x_path))
        all_frames = []
        for img in images_list:
            all_frames.append(cv2.imread(x_path+'/'+img))
        return np.stack(all_frames)
    
    def get_Y(self, y_path, images_folder):
        with open(y_path, 'rb') as pickle_file:
            y_dict = pickle.load(pickle_file)
        
        key = images_folder.split('_')[:2]
        key = '_'.join(key)
        return np.array(y_dict[key])


class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv3d_1 = Conv3D(4, kernel_size = (1,7,7), padding = 'same')
        self.batchnorm_1 = BatchNormalization()
        self.batchnorm_2 = BatchNormalization()
        self.batchnorm_3 = BatchNormalization()
        self.maxpool3d = MaxPool3D(pool_size=(1,2,2))
        self.conv3d_2 = Conv3D(8, kernel_size = (1,5,5), padding = 'same')
        self.conv3d_3 = Conv3D(16, kernel_size = (1,3,3), padding = 'same')


    def call(self, inputs):
        x = self.conv3d_1(inputs)
        x = self.batchnorm_1(x)
        x = self.maxpool3d(x)
        x = self.conv3d_2(x)
        x = self.batchnorm_2(x)
        x = self.maxpool3d(x)
        x = self.conv3d_3(x)
        x = self.batchnorm_3(x)
        x = self.maxpool3d(x)
        return x


class MyCL_Model(Model):
    
    def __init__(self):
        super(MyCL_Model, self).__init__()
        self.encoder = Encoder()
        self.convlstm_1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), strides = (2, 2) ,padding='valid', return_sequences=False)
        self.batchnorm = BatchNormalization()
        self.maxpool2d = MaxPool2D()
        self.flatten = Flatten()
        self.dense_1 = Dense(100, activation = 'relu')
        self.classifier = Dense(6, activation = 'softmax') 

        
    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.convlstm_1(x)
        x = self.batchnorm(x)
        x = self.maxpool2d(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        # x = self.dense_2(x)
        # x = self.dense_3(x)
        # x = self.dense_4(x)
        return self.classifier(x)



model = MyCL_Model()
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])



epochs = 10
for i in range(epochs):
    folders = os.listdir(x_train_path)
    for folder in folders:
        training_generator = newGen(x_train_path, folder, y_train_path)
        model.fit_generator(training_generator)

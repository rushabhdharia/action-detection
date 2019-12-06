#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D, Attention, Dense, Flatten, MaxPool3D, MaxPool2D,BatchNormalization
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import os
from scipy.io import loadmat
from tensorflow.keras import Model
import cv2
import pickle


# In[2]:


labels_path = './Labels_MERL_Shopping_Dataset/'
results_path = './Results_MERL_Shopping_Dataset/DetectedActions/'
videos_path = './Videos_MERL_Shopping_Dataset/'


# In[3]:


x_train_path = videos_path+'train/'
y_train_path = 'train_y.pkl'


# In[4]:


x_test_path = videos_path + 'test/'
y_test_path = 'test_y.pkl'


# In[5]:


x_val_path = videos_path + '/val/'
y_val_path = 'val_y.pkl'


# In[13]:


class DataGenerator(Sequence):
    
    def __init__(self, x_path, y_path = None, to_fit = True,  seq_len = 30):
        self.x_path = x_path        
#         self.batch_size = batch_size
        self.to_fit = to_fit
        self.list_X = os.listdir(self.x_path)
        self.seq_len = seq_len
        if to_fit:
            self.y_path = y_path
            self.dict_Y = self.get_y(y_path)
    
    
    def __len__(self):
        return len(self.list_X)
    
    
    def __getitem__(self, index):
        images_folder = self.list_X[index]
        images_list = sorted(os.listdir(self.x_path + images_folder))
        all_frames = []
        for img in images_list:
            all_frames.append(np.array(cv2.imread(x_train_path+images_folder+'/'+img)))
        
        X = self.stack_frames(all_frames)
        
        if self.to_fit:
            key = images_folder.split('_')[:2]
            key = '_'.join(key)
            Y = np.array(self.dict_Y(key))
            return X, Y[30:]
        
        return X
    
    def get_y(self, path):
        with open(path, 'rb') as pickle_file:
            y_dict = pickle.load(pickle_file)
        return y_dict 
    
    def stack_frames(self, frames):
        stacked_frames = []
        for i in range(len(frames) - self.seq_len):
            end = i + 30
            stacked_frames.append(frames[i:end])
        
        return np.stack(stacked_frames)


# In[14]:


training_generator = DataGenerator(x_train_path ,y_path = y_train_path)
validation_generator = DataGenerator(x_val_path ,y_path = y_val_path)
testing_generator = DataGenerator(x_test_path ,y_path = y_test_path)


# In[ ]:





# In[15]:


class MyCL_Model(Model):
    
    def __init__(self):
        super(MyCL_Model, self).__init__()
        self.convlstm_1 = ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', return_sequences=True)
        self.batchnorm = BatchNormalization()
        self.maxpool3d = MaxPool3D(pool_size=(1,2,2))
        self.maxpool2d = MaxPool2D()
        self.convlstm_2 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)
        self.convlstm_3 = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False)
        self.flatten = Flatten()
        self.dense_1 = Dense(10000)
        self.dense_2 = Dense(1000)
        self.dense_3 = Dense(100)
        self.dense_4 = Dense(10)
        self.classifier = Dense(1) 

        
    def call(self, inputs):
        x = self.convlstm_1(inputs)
        x = self.batchnorm(x)
        x = maxpool3d(x)
        x = convlstm_2(x)
        x = self.batchnorm(x)
        x = maxpool3d(x)
        x = convlstm_3(x)
        x = maxpool2d(x)
        x = flatten(x)
        x = dense_1(x)
        x = dense_2(x)
        x = dense_3(x)
        x = dense_4(x)
        return self.classifier(x)


# In[16]:


model = MyCL_Model()
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])


# In[ ]:


model.fit_generator(generator = training_generator, validation_data=validation_generator)


# ip shape = (30, 680, 920, 3)
# convlstm return sequences true 5d else 4d
# batchnorm
# maxpool3d
# 
# attention
# then flatten

# In[ ]:


model.evaluate_generator(testing_generator)


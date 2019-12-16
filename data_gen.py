import os
import cv2
import pickle

import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


class DataGenerator(Sequence):
    
    def __init__(self, x_path, y_path = None, to_fit = True,  seq_len = 30, batch_size = 4):
        self.x_path = x_path        
        self.batch_size = batch_size
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
            all_frames.append(np.array(cv2.imread(self.x_path+images_folder+'/'+img)))
        
        all_frames = np.stack(all_frames).astype(np.float16)
        
        key = images_folder.split('_')[:2]
        key = '_'.join(key)
        Y = np.array(self.dict_Y[key])
        all_frames, targets = self.check(all_frames, Y)
        series_data = TimeseriesGenerator(all_frames, targets, length = self.seq_len, batch_size=self.batch_size)
        
        return series_data
    
    def get_y(self, path):
        with open(path, 'rb') as pickle_file:
            y_dict = pickle.load(pickle_file)
        return y_dict 
    
    def check(self, all_frames, targets):
        if all_frames.shape[0] < targets.shape[0]:
            targets = targets[:-1]
        return all_frames, targets
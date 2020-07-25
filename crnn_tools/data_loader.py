import os
import cv2
import numpy as np
import random
import math
import tensorflow as tf
from crnn_tools import libs
from crnn_tools import config
from glob import glob
from tqdm import tqdm
print("It takes a few minutes to initialize data loading.")
BATCH_LIST = glob("data/img_batches/part_*/batch_*")
BATCH_LIST = sorted(BATCH_LIST)
TRAIN_LIST = BATCH_LIST[:-1]
TEST_LIST = [BATCH_LIST[-1]]


def convert_to_gray(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    return img_gray


def parser(result):
    images = np.array([convert_to_gray(items[0]) for items in result],dtype=np.float32)
    images = np.expand_dims(images, axis=3)
    labels = np.array([items[1] for items in result],dtype=np.int32)
    return images,labels
    
    
class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self,batch_list,batch_size=config.BATCH_SIZE):
        self.batch_list = batch_list
        self.datas = []
        for items in batch_list:
            self.datas = self.datas+[items+"/"+it for it in os.listdir(items)]
        self.datas = np.array(self.datas)
        np.random.shuffle(self.datas)
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.datas))
        np.random.shuffle(self.indexes)
        
    def __len__(self):
        # The number of steps of each epoch
        return math.ceil(len(self.datas) / float(self.batch_size))
    
    def __getitem__(self,index):
        batch_index = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]  
        result = []
        
        name_list = list(self.datas[batch_index])
        name_list = sorted(name_list)
        for name in name_list:
            img = cv2.imread(name)
            if np.random.randint(0, 100) > 50:
                img = 255-img
            label = libs.total_dic[name.split("/")[-1]]
            result.append([img,label])
        random.shuffle(result) 
            
        parsered_result = parser(result)
        input_length = np.array([config.WIDTH]*parsered_result[0].shape[0])
        input_length = input_length.reshape(-1,1)
        label_length = np.array([config.LABEL_LENGTH]*parsered_result[0].shape[0])
        label_length = label_length.reshape(-1,1)
        X = [parsered_result[0], parsered_result[1], input_length, label_length]
        y = parsered_result[1]

        return X,y
    
    def on_epoch_end(self):
        print("Epoch ends! Start shuffling")
        np.random.shuffle(self.indexes)
 

       

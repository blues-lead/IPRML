# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 10:52:47 2019

Class for reading the dataset CIFAR-10. In the taget folder there must be
7 files: data_batch_[1..5], batches_meta, test_butch

@author: Anton
"""

import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import os

class DataMatrix:
    def __init__(self, path):
        self.__path = path
        self.__datafiles = []
        self.__testfiles = []
        self.__get_fileList()
        self.__trainbatch = []
        self.__testbatch = []
        self.__train_lbls = []
        self.__test_lbls = []
        self.__names = {}
        self.__unpickle("\\test_batch")
        
        
    def get_lblvector(self):
        pass
        #return self.__batch_lbl
        
            
    def show_random_picture(self,n=0):
        pic = np.array((32,32,3),dtype=np.uint8)
        if n==0:
            row = random.randint(0,10001)
        else:
            row = n
        red = np.array((32,32),dtype=np.uint8)
        rw = self.__testbatch[row]
        red = rw[0:1024].reshape(32,32)
        green = rw[1024:2048].reshape(32,32)
        blue = rw[2048:3072].reshape(32,32)
        pic = np.dstack((red,green,blue))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(pic,interpolation='nearest')
        return self.__test_lbls[row]
    
    def get_trainingdata_row(self,ind):
        return self.__trainbatch[ind]
    
    def get_testdata_row(self,ind):
        return self.__testbatch[ind]
    
    def get_name(self, idx):
        return self.__names[idx]
    
    def get_training_label(self,idx):
        return self.__train_lbls[idx]
    
    def get_test_label(self,idx):
        return self.__test_lbls[idx]
    
    def get_training_data(self):
        return self.__trainbatch
    
    def get_test_data(self):
        return self.__testbatch
    
    def get_training_labels(self):
        return self.__train_lbls
    
    def get_test_labels(self):
        return self.__test_lbls

    def __get_fileList(self):
        for r,d,f in os.walk(self.__path):
            for i in f:
                if "data" in i:
                    self.__datafiles.append(os.path.join(r,i))
                elif "test" in i:
                    self.__testfiles.append(os.path.join(r,i))
        
        
    def __unpickle(self, fname):
        #reading data files
        for file in self.__datafiles:
            with open(file,'rb') as fl:
                btch = pickle.load(fl,encoding='bytes')
                self.__trainbatch.append(btch[b'data']/255)
                self.__train_lbls = np.concatenate((self.__train_lbls,btch[b'labels']))
        self.__trainbatch = np.vstack(self.__trainbatch)
        
        for file in self.__testfiles:
            with open(file, 'rb') as fl:
                btch = pickle.load(fl,encoding='bytes')
                self.__testbatch.append(btch[b'data']/255)
                self.__test_lbls = np.concatenate((self.__test_lbls,btch[b'labels']))
        self.__testbatch = np.vstack(self.__testbatch)
        with open(self.__path + "\\batches.meta",'rb') as file:
            meta = pickle.load(file)
            self.__names = dict(zip(range(0,10),meta['label_names']))
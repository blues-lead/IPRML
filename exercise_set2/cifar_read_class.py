# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 13:09:06 2019

@author: Anton
"""
import os
import numpy as np

# Z:\Documents\TUT\Introduction to Pattern Recognition and Machine Learning\codes\cifar-10-python\cifar-10-batches-py

class DataMatrix:
    def __init__(self, path):
        self.__path = path
        self.__traindata = []
        self.__testdata = None
        self.__test_labels = []
        self.__train_labels = []
        self.__names = {}
        
    def read_files(self):
        import pickle
        datafiles = []
        for r,d,f in os.walk(self.__path):
            for file in f:
                if 'data_batch' in file:
                    datafiles.append(os.path.join(r,file))
                elif 'test' in file:
                    testfile = os.path.join(r,file)
                elif 'meta' in file:
                    metafile = os.path.join(r,file)
        # data files
        for batch in datafiles:
            with open(batch, 'rb') as file:
                batch_i = pickle.load(file,encoding='bytes')
                self.__traindata.append(batch_i[b'data'])
        return np.vstack(self.__traindata)
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:58:28 2019

@author: Anton
"""

import numpy as np
import matplotlib.pyplot as plot

from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from ex2_class import DataMatrix
from keras.utils import to_categorical
from keras.models import model_from_json
import os

def form_averages(data,name): #3 parts of 1024
    z = []
    for i in data:
        z.append( [np.mean(i[0:1024]), np.mean(i[1024:2048]), np.mean(i[2048:3072])] )
    #np.save(name,z)
    return np.array(z)

def form_4x4_averages(data,name): #one layer, 4 subimages
    z = []
    for i in data:
        z.append( [np.mean(i[0:512]), np.mean(i[512:1024]), np.mean(i[1024:1536]), np.mean(i[1536:2048]),
                   np.mean(i[2048:2560]), np.mean(i[2560:3072])] )
    #np.save(name,z)
    return z
    
def train_model(train_data, lbl_train_y): #raw pixels
    model = Sequential()
    
    model.add(Dense(units=64, input_shape=(3072,), activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(optimizer='sgd', loss='mse', metrics=['mse'])
    model.fit(train_data,lbl_train_y, epochs=5000, verbose=1)
    
    model_json = model.to_json()
    with open("model_raw_pixel.json",'w') as json_file:
        json_file.write(model_json)
    model.save_weights('model_raw_pixel_weights')
    
def train_model_avg_3x3(train_data, lbl_train_y): # avg of each channel
    model = Sequential()
    
    model.add(Dense(units=64, input_shape=(3,), activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(optimizer='sgd', loss='mse', metrics=['mse'])
    model.fit(train_data,lbl_train_y, epochs=5000, verbose=1)
    
    model_json = model.to_json()
    with open("model_3x3.json",'w') as json_file:
        json_file.write(model_json)
    model.save_weights('model_3x3_weights')
    
def train_model_avg_6x6(train_data,lbl_train_y): # avg of 4 subimage of each channel
    model = Sequential()
    
    model.add(Dense(units=64, input_shape=(6,), activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(optimizer='sgd', loss='mse', metrics=['mse'])
    model.fit(train_data,lbl_train_y, epochs=5000, verbose=1)
    
    model_json = model.to_json()
    with open("model_6x6.json",'w') as json_file:
        json_file.write(model_json)
    model.save_weights('model_6x6_weights')

def main():
    mydata = DataMatrix("C:\\Users\Anton\Desktop\IntroductionMLcopy\cifar-10-python\cifar-10-batches-py")
    train_data = mydata.get_training_data()
    train_labels = mydata.get_training_labels()
    
    N = 10000 # How much data to use for training
    N_test = 1000 # How much data to use for testing
    
    train_data = train_data[:N]
    train_labels = train_labels[:N]
    
    test_data = mydata.get_test_data()
    test_labels = mydata.get_test_labels()
    
    lbl_test_y = to_categorical(test_labels,10) # 
    lbl_train_y = to_categorical(train_labels,10) #
    
# ========================== Predict with 3 channel avg ========================
    #train_avg_data = form_averages(train_data,'avg_1024_rgb_train_data')
    #test_avg_data = form_averages(test_data,'avg_1024_rgb_test_data')
    
    #train_avg_data = np.load('avg_1024_rgb_train_data.npy')
    #test_avg_data = np.load('avg_1024_rgb_test_data.npy')
    
#    train_model_avg_3x3(train_avg_data,lbl_train_y)
#    json_file = open('model_3x3.json','r')
#    loaded_model_json = json_file.read()
#    json_file.close()
#    loaded_model = model_from_json(loaded_model_json)
#    loaded_model.load_weights('model_3x3_weights')
#    loaded_model.compile(optimizer='sgd', loss='mse', metrics=['mse'])
#    score = loaded_model.evaluate(test_avg_data, lbl_test_y)
#    print(loaded_model.metrics_names[1],score[1]*100)
#    test_prediction_data = form_averages(test_data[:N_test],24)
#    pred = loaded_model.predict(test_prediction_data)
#    predicted = np.argmax(pred,axis=1)
#    test_lbl = np.array([int(a) for a in test_labels[:N_test]])
#    print(np.count_nonzero((test_lbl-predicted)==0))
#=============================================================================
# ========================== Predict with raw pixels =========================    
#    train_model(train_data, lbl_train_y)
#    json_file = open('model_raw_pixel.json','r')
#    loaded_model_json = json_file.read()
#    json_file.close()
#    loaded_model = model_from_json(loaded_model_json)
#    loaded_model.load_weights("model_raw_pixel_weights")
#    loaded_model.compile(optimizer='sgd', loss='mse', metrics=['mse'])
#    score = loaded_model.evaluate(test_data,lbl_test_y)
#    print(loaded_model.metrics_names[1],score[1]*100)
#    pred = loaded_model.predict(test_data[:N_test])
#    predicted = np.argmax(pred,axis=1)
#    test_lbl = np.array([int(a) for a in test_labels[:N_test]])
#    print(np.count_nonzero((test_lbl-predicted)==0))
# ============================================================================
#=============================Predict 4x4 averages============================
    #train_avg_data = form_4x4_averages(train_data,'avg_512_rgb_train_data')
    #test_avg_data = form_4x4_averages(test_data,'avg_512_rgb_test_data')
#    train_avg_data = np.load('avg_512_rgb_train_data.npy')
#    test_avg_data = np.load('avg_512_rgb_test_data.npy')
    
    #train_model_avg_6x6(train_avg_data,lbl_train_y)
#    json_file = open('model_6x6.json','r')
#    loaded_model_json = json_file.read()
#    json_file.close()
#    loaded_model = model_from_json(loaded_model_json)
#    loaded_model.load_weights("model_6x6_weights")
#    loaded_model.compile(optimizer='sgd', loss='mse', metrics=['mse'])
#    score = loaded_model.evaluate(test_avg_data,lbl_test_y)
#    print(loaded_model.metrics_names[1],score[1]*100)
#    pred = loaded_model.predict(test_avg_data[:N_test])
#    predicted = np.argmax(pred,axis=1)
#    test_lbl = np.array([int(a) for a in test_labels[:N_test]])
#    print(np.count_nonzero((test_lbl-predicted)==0))
    
    
    
    
    
    
    
main()
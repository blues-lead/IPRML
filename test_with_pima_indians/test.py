# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 00:08:18 2019

@author: Anton
"""

import csv
import random
import math
import numpy as np

def mean(numbers):
    return sum(numbers)/len(numbers)

def stdev(numbers):
    return math.sqrt(np.var(numbers))

def summarize(dataset):
    summaries = [(mean(attributes), stdev(attributes)) for attributes in dataset]
    return summaries
    

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[-1] not in separated:
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def splitDataset(dataset,splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def load_csv(filename):
    lines = csv.reader(open(filename,'r'))
    dataset = []
    j=0
    for i in lines:
        dataset.append([float(z) for z in i])
        j+=1
    #    dataset[i] = [float(x) for x in dataset]
    return dataset

def main():
    mydata = load_csv("pima-indians-diabetes.data.csv")
    numbers=[1,2,3,4,5]
    dataset = [[1,20,0], [2,21,1], [3,22,0]]
    summary = summarize(dataset)
    print('Attribute summaries:', summary)
    
    
main()
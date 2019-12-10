# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 18:31:14 2019

@author: Anton
"""

import numpy as np
import math
from ex2_class import DataMatrix
from keras.datasets import cifar10

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
print(x_train.shape)
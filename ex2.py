from ex2_class import DataMatrix
import numpy as np
import matplotlib.pyplot as plt
import random

def cifar_10_evaluate(pred, gt):
    test = abs(pred-gt)
    return np.count_nonzero(test==0)/len(gt)

def cifar_10_rand(x):
    rnd = random.randint(0,9)
    return rnd

def cifar_10_1NN(x, trdata, trlabel):
    a = []
    for i in trdata:
        a.append(np.linalg.norm(x - i))
    min_val = min(a)
    min_idx = a.index(min_val)
    return trlabel[min_idx]

def cifar_10_3NN(x,trdata,trlabel):
    pass
    """a = []
    min_vals = []
    for i in trdata:
        a.append(np.linalg.norm(x-i))
    for i in range(3):
        min_vals.append(min(a))"""
        
        

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


def main():
    path = "Z:\Documents\TUT\Introduction to Pattern Recognition and Machine Learning\codes\cifar-10-python\cifar-10-batches-py"
    mydata = DataMatrix(path)
    
    A = mydata.get_training_data()
    B = mydata.get_test_data()
    L = mydata.get_test_labels()
    R = mydata.get_training_labels()
    
    #mydata.show_random_picture()
    
    data_to_use = 1000
    
    A = A[0:data_to_use,:]
    L = L[0:data_to_use]
    B = B[0:data_to_use,:]
    R = R[0:data_to_use]
    
    x = []
    for row in B:
        x.append(cifar_10_rand(5))
    accuracy = cifar_10_evaluate(x,L)
    print("Prediction accuracy by random is", accuracy*100,"%")
    
    x = []
    """for i in range(0,1000):
        x.append(cifar_10_1NN(B[i],A,R))#(B[i],A,R)
        printProgressBar(i+1,1000, prefix = 'Progress', suffix = 'Complete',length = 25)"""
    j = 1
    for i in B:
        x.append(cifar_10_1NN(i,A,R))
        printProgressBar(j,data_to_use, prefix = "Progress", suffix = "Complete",length = 25)
        j+=1
    accuracy = cifar_10_evaluate(x,L)
    print("Prediction accuracy by 1NN is", accuracy*100,"%")
    
    
main()
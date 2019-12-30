## Let's calculate first probability of getting a certain class from the bucket, a.k.a
## a priori. Next we COUNT(descrete case)/PDF(non-descrete case) probabilities within each class,
## a.k.a p(feature | class) = ...
from six.moves import cPickle as pickle
import scipy.io as sio
import numpy as np
import math
from ex2_class import DataMatrix
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

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

def cifar_10_evaluate(pred, gt):
    correct = 0
    for i in range(len(pred)):
        if pred[i] == gt[i]:
            correct +=1
    return (correct/len(pred))*100

def normpdf(x,mu,var):
    nom = math.exp(-((x-mu)**2)/(2*var))
    denom = math.sqrt(2*math.pi*var)
    return nom/denom

def cifar_10_features(x):
    r = x[0:1024]
    g = x[1024:2048]
    b = x[2048:3072]
    return [sum(r)/1024, sum(g)/1024, sum(b)/1024]

def cifar_10_bayes_learn(F,label):
    ms = []
    sg = []
    cov = np.matrix
    for i in range(0,10):
        idx = np.where(label==i) #get from label-vector ith entries indecies
        class_i = F[idx] #get data according to gotten indecies
        #calculate mean of each component
        mur = np.mean(class_i[:,0])
        mug = np.mean(class_i[:,1])
        mub = np.mean(class_i[:,2])
        ms.append([mur,mug,mub])
        
        #calculate sigma squared of each component
        sg_r = np.var(class_i[:,0])
        sg_g = np.var(class_i[:,1])
        sg_b = np.var(class_i[:,2])
        sg.append([sg_r,sg_g,sg_b])
    p = len(class_i)/len(F)
    return [np.array(ms),np.array(sg),1/10]

def cifar_10_classify(f,ms,gs,p):
    test = cifar_10_features(f)
    prob = {}
    for i in range(0,10):
        pr = normpdf(test[0],ms[i,0],gs[i,0])
        pg = normpdf(test[1],ms[i,1],gs[i,1])
        pb = normpdf(test[2],ms[i,2],gs[i,2])
        prob[i] = pr*pg*pb*p
    mx = max(prob.values())
    cl = [c for c,nd in prob.items() if mx == nd]
    return cl[0]

def get_covariance_matrix(data):
    t = data.shape
    ns = np.ones((t[0],t[0]))
    a = np.matmul(ns,data)*(1/t[0])
    a = data - a
    dev = np.matmul(np.transpose(a),a)*(1/t[0])
    return dev


def cifar_10_classify_mvn(f,train_data,train_labels, ms, cov_matrices, p):
    #test = cifar_10_features(f)
    prob = []
    for i in range(0,10):
        mus = ms[i]
        idx = np.where(train_labels==i)
        cov_mat = cov_matrices[i]
        mv = mvn.cdf(f,mus,cov_mat)*(1/10)
        prob.append(mv)
    mx = max(prob)
    idx = prob.index(mx)
    return idx
        
    

def save(mus, sigmas):
    np.save('class_mus',mus)
    np.save('class_variances',sigmas)

def main():
    mydata = DataMatrix('Z:\Documents\TUT\Introduction to Pattern Recognition and Machine Learning\codes\cifar-10-python\cifar-10-batches-py')
    T = mydata.get_training_data() #training data
    L = mydata.get_training_labels()
    C = mydata.get_test_data()
    CL = mydata.get_test_labels()
    C = C[0:100,:]
    CL = CL[0:100]
    
    tr_set = np.load('training_mus.npy')
    cov_matrices = np.load('cov_matrices.npy')
    [ms, gs, p] = cifar_10_bayes_learn(tr_set, L)
    
    
        
    prb = []
    for i in range(len(C)):
        test = cifar_10_features(C[i])
        prb.append(cifar_10_classify_mvn(test,tr_set,L,ms,cov_matrices,p))
    print(cifar_10_evaluate(prb,CL))
    print(prb)
    
    #sio.savemat('multi_mus0',mdict={'multi_mus':ms[0]})
    #sio.savemat('multi_covmat0',mdict={'cov_mat':get_covariance_matrix(class_i)})
    #sio.savemat('test_vector',mdict={'feats':cifar_10_features(C[0])})
    
#    prob = []
#    for i in range(len(C)):
#        prob.append(cifar_10_classify(C[i],ms,gs,p))
#    print(cifar_10_evaluate(prob,CL))
    
    
main()
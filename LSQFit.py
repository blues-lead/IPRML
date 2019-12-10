# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 23:41:45 2019

@author: Anton
"""

import numpy as np
import matplotlib.pyplot as plt

class LSQFit:
    def __init__(self,point_count=0):
        self.__line = [0,0]
        self.__point_count = point_count
        self.__counter = 0
        self.__fig = plt.figure()
        self.__cid = self.__fig.canvas.mpl_connect('button_press_event',self)
        self.__ax = self.__fig.add_subplot(111)
        self.__ax.set_ylim([0,10])
        self.__ax.set_xlim([0,10])
        self.__fig.canvas.draw()
        self.__xpoints = []
        self.__ypoints = []
        self.__x = 0
        self.__y = 0
        plt.grid()
        
    def __call__(self, event):
        self.__counter += 1
        if self.__counter == self.__point_count:
            self.__line = self.__determineAB()
            self.__plot_line()
        #print(event.xdata,event.ydata)
        self.__xpoints.append(event.xdata)
        self.__ypoints.append(event.ydata)
        self.__x = event.xdata
        self.__y = event.ydata
        self.__plot_point()
        
    def __plot_point(self):
        plt.plot(self.__x, self.__y, 'o')
        self.__fig.canvas.draw()
        
    def __determineAB(self):
        sum_xi = sum([a for a in self.__xpoints])
        sum_xi_xi = sum([a**2 for a in self.__xpoints])
        sum_xi_yi = sum([a*b for a,b in zip(self.__xpoints, self.__ypoints)])
        sum_yi = sum([a for a in self.__ypoints])
        n = len(self.__xpoints)
        b = (sum_xi*sum_xi_yi - sum_yi*sum_xi_xi)/(sum_xi**2 - n*sum_xi_xi)
        a = (sum_xi*sum_yi-n*sum_xi_yi)/(sum_xi**2-n*sum_xi_xi)
        return [a,b]
    
    def __get_lineEq(self):
        return self.__line
    
    def __plot_line(self):
        print("Got into the function")
        x = np.linspace(0,10,50)
        y = self.__line[0]*x + self.__line[1]
        plt.plot(x,y,'r-')
        self.__fig.canvas.draw()
        #
        print("My result:",self.__line)
        self.__check_my_parameters()
        #
        
    def __check_my_parameters(self):
        if len(self.__xpoints) < 2:
            return
    
        A = np.vstack((self.__xpoints, np.ones(len(self.__xpoints)))).T
        y = np.array(self.__ypoints)
        m,c = np.linalg.lstsq(A,y,rcond=None)[0]
        print("Matrix result",m,c)

def main():
    fit = LSQFit(5)
    #plt.show()
    

    
main()
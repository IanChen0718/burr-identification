# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 20:13:27 2020

@author: IanChen
"""
import numpy as np 
class GetMaxMin():
    def __init__(self, inputArray):
        self.inputArray = inputArray      
        
    def MaxMin(self):
        if len(self.inputArray) > 0:
            valList = []
            # valList = [self.xMin, self.xMax, self.yMin, self.yMax, self.zMin, self.zMax]
            for i in range(self.inputArray.shape[1]):
                _max = np.amax(self.inputArray[:, i])
                _min = np.amin(self.inputArray[:, i])
                # _load = self.inputArray[:, i]
                # _sort = sorted(_load)
                # valList.append(_sort[0])
                # valList.append(_sort[-1])
                valList.append(_min)
                valList.append(_max)
            # _num = 3
            # print("X MIN : {}  X MAX : {}".format(round(valList[0],_num), round(valList[1],_num)))
            # print("Y MIN : {}  Y MAX : {}".format(round(valList[2],_num), round(valList[3],_num)))
            # print("Z MIN : {}  Z MAX : {}".format(round(valList[4],_num), round(valList[5],_num)))
            # print("\n")
            return valList
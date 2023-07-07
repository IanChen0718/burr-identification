# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 16:26:52 2021

@author: IanChen
"""

from .GetMaxMin import GetMaxMin
class Direction():
    def __init__(self, xyzarray, direction, num):
        # self.xyzarray = xyzarray
        self.direction = direction
        self.num = num
        self.Max = 0
        self.Min = 0 
        self.dirIndex = None
        # print("------------Direction------------")
        # print("num :", self.num)
        self.GM = GetMaxMin(xyzarray)
    def getValue(self):
        
        xMin, xMax, yMin, yMax, zMin, zMax = self.GM.MaxMin()
        if self.direction == "x":
            self.Max = round(xMax, self.num)
            self.Min = round(xMin, self.num)
            self.dirIndex = 0
            
        if self.direction == "y":
            self.Max = round(yMax, self.num)
            self.Min = round(yMin, self.num)
            self.dirIndex = 1
            
        if self.direction == "z":
            self.Max = round(zMax, self.num)
            self.Min = round(zMin, self.num)
            self.dirIndex = 2
        return self.Max, self.Min, self.dirIndex        
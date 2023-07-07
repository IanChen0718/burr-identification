# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 23:55:37 2021

@author: IanChen
"""
import numpy as np 
from .GetPlaneEq import GetPlaneEq
from .Tol import Tol

class TrimPlane():
    def __init__(self, xyzarray, p1, p2, p3, step = None, tol = None):
        self.xyzarray  = xyzarray
        self.step = step
        self.p1 = p1 
        self.p2 = p2
        self.p3 = p3
        self.GPE = GetPlaneEq(p1, p2, p3)
        self.a, self.b, self.c, self.d = self.GPE.main()
        self.num, self.tol = Tol(tol).getValue()
        self.xyzFilter = []
    def start(self):
        for i in range(len(self.xyzarray)):
            val = self.a * self.xyzarray[i][0] + \
                  self.b * self.xyzarray[i][1] + \
                  self.c * self.xyzarray[i][2]
            if abs( round((val + self.d), self.num) ) > self.tol:
                self.xyzFilter.append(self.xyzarray[i])
        self.xyzFilter = np.array(self.xyzFilter)
        if self.step is not None:
            if self.step > 1:
                self.step -= 1
                return TrimPlane(self.xyzFilter, self.p1, self.p2, self.p3, self.step).start()
            else:
                return self.xyzFilter       
        else:
            return self.xyzFilter        
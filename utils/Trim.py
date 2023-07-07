# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:06:09 2021

@author: IanChen
"""
import numpy as np 
from .Tol import Tol
from .Direction import Direction

class Trim():
    def __init__(self, xyzarray, direction, flagmin, flagmax, step = None, tol = None):
        self.xyzarray = xyzarray
        self.dir = direction
        self.flagmin = flagmin
        self.flagmax = flagmax
        self.step = step
        self.tol = tol
        #------------------------------------------------------------------------#
        self.condAll1 = False 
        self.condAll2 = False 
        self.xyzFilter = []
        #------------------------------------------------------------------------#
        self.Tol = Tol(self.tol)
        self.num, self.tol = self.Tol.getValue()
        # print("------------  Trim  -------------")
        # print("num : {}, tol : {}".format(self.num, self.tol))
        #------------------------------------------------------------------------#
        self.Dri = Direction(self.xyzarray, self.dir, self.num)
        self.Max, self.Min, self.colIndex = self.Dri.getValue()
    def start(self):
        for i in range(len(self.xyzarray)):
            
            if self.flagmax == 1:
                cond1 = round(self.xyzarray[i][self.colIndex]-self.tol, self.num) <= self.Max
                cond2 = round(self.xyzarray[i][self.colIndex]+self.tol, self.num) >= self.Max
                self.condAll1 = cond1 and cond2
    
            if self.flagmin == 1:
                cond1 = round(self.xyzarray[i][self.colIndex]-self.tol, self.num) <= self.Min
                cond2 = round(self.xyzarray[i][self.colIndex]+self.tol, self.num) >= self.Min
                self.condAll2 = cond1 and cond2
           
            if (self.condAll1 or self.condAll2) == False:
                self.xyzFilter.append(self.xyzarray[i])
    
            self.condAll1 = False    
            self.condAll2 = False   
        self.xyzFilter = np.array(self.xyzFilter)
        if self.step is not None:
            if self.step > 1:
                self.step -= 1
                print("step :", self.step)
                trim = Trim(self.xyzFilter, self.dir, self.flagmin, self.flagmax, step = self.step, tol = self.tol)
                return trim.start()
            else:
                return self.xyzFilter
        else:
            return self.xyzFilter        
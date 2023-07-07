# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 16:59:40 2021

@author: IanChen
"""
import decimal 
class Tol():
    def __init__(self, tol):
        self.tol = tol
        self.d = None
    def getValue(self):
        if self.tol is None or self.tol == 0:
            self.tol = 0
            self.num = 4
        else:
            self.d = decimal.Decimal(str(self.tol))
            self.num = -self.d.as_tuple().exponent
        return self.num, self.tol

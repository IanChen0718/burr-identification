# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:37:36 2022

@author: IanChen
"""

from .GetPlaneEq import GetPlaneEq 
import numpy as np

def Project(Hanger_np, p1, p2, p3):
    gpe = GetPlaneEq(p1, p2, p3)
    a1, b1, c1, d1 = gpe.GetPara()  
    n1 = np.asarray([a1, b1, c1, d1])
    pt_proj = []
    for element in Hanger_np:
        element_proj = Projection(element, n1)
        pt_proj.append(element_proj)
    pt_proj = np.asarray(pt_proj)
    return pt_proj


def Projection(pt, n1):
    pt_1 = np.r_[pt, np.ones(1)]
    _t = -(np.dot(pt_1, n1)) / np.sum(np.square(n1[0:3])) #(n1[0]**2+n1[1]**2+n1[2]**2)
    _const = n1[0:3]*_t
    pt_proj = np.identity(3) @ pt + _const
    return pt_proj
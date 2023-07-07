# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:37:36 2022

@author: IanChen
"""

from .GetPlaneEq import GetPlaneEq 
import numpy as np
import pandas as pd 
def PdProject(df, p1, p2, p3, col_names_list, col_new_names_list):
    df_np = df[col_names_list].to_numpy()
    a1, b1, c1, d1 = GetPlaneEq(p1, p2, p3)
    n1 = np.asarray([a1, b1, c1, d1])
    pt_proj = []
    for element in df_np:
        element_proj = Projection(element, n1)
        pt_proj.append(element_proj)
    pt_proj_np = np.asarray(pt_proj)
    df[col_new_names_list[0]] = pt_proj_np[:, 0].tolist()
    df[col_new_names_list[1]] = pt_proj_np[:, 1].tolist()
    df[col_new_names_list[2]] = pt_proj_np[:, 2].tolist()
    # pt_proj = pd.DataFrame(np.asarray(pt_proj), columns =["X", "Y", "Z"], index = df.index[:]) 
    # return pt_proj


def Projection(pt, n1):
    pt_1 = np.r_[pt, np.ones(1)]
    _t = -(np.dot(pt_1, n1)) / np.sum(np.square(n1[0:3])) #(n1[0]**2+n1[1]**2+n1[2]**2)
    _const = n1[0:3]*_t
    pt_proj = np.identity(3) @ pt + _const
    return pt_proj



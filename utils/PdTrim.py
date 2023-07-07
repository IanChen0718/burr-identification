# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:06:09 2021

@author: IanChen
"""


import numpy as np 
from .GetPlaneEq import GetPlaneEq

def PdTrim(df, direction, min_or_max, _range):
    if min_or_max == 0: ### 0 ->  keep _range from min
        _min = df[direction].min()
        df = df.drop(df[df[direction] >= _min+_range].index)
    elif min_or_max == 1:
        _max = df[direction].max()
        df = df.drop(df[df[direction] <= _max-_range].index)
    return df

def PdTrimThroughPlane(df, p1, p2, p3, offset):
    df_xyz = df[["X", "Y", "Z"]]
    df_xyz.insert(3, "Const", 1)
    
    a1, b1, c1, d1 = GetPlaneEq(p1, p2, p3) 
    abcd = np.asarray([a1, b1, c1, d1])    
    plane_num = df_xyz.dot(abcd)
    plane_num = plane_num.abs()
    pd_trim = df.loc[plane_num[plane_num <= offset].index.values]
    # if pn == "+" or "p":
    #     pd_trim = df.loc[plane_num[plane_num >= 0].index.values]
    # elif pn == "-" or "n":
    #     pd_trim = df.loc[plane_num[plane_num <= 0].index.values]
    
    return pd_trim

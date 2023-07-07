# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:12:01 2022

@author: IanChen
"""

from .GetPlaneEq import GetPlaneEq
import numpy as np
import pandas as pd

def Pd3DPlaneto2D(df, p1, p2, p3, col_names_list, col_new_names_list):

    """Rotate to Z axis"""
    p_a, p_b, p_c ,p_d = GetPlaneEq(p1, p2, p3)
    vec = [p_a, p_b, p_c]
    vec = np.asarray(vec)/np.linalg.norm(vec)
    vec1 = vec.tolist()
    vec2 = [  0,   1,   0]
    a,b = (vec1/ np.linalg.norm(vec1)).reshape(3), (vec2/np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a,b)
    I = np.identity(3)
    r = None
    if [abs(ele) for ele in a] == vec2:
       r = I
    else:
        c = np.dot(a,b)
        s = np.linalg.norm(v)
        vXStr = '{} {} {}; {} {} {}; {} {} {}'.format(0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0)
        k = np.matrix(vXStr)
        r = I + k + k @ k * ((1 -c)/(s**2))
    rm = np.empty((4, 4))
    rm[:3, :3] = r    
    r_pd = pd.DataFrame(r)

    """Projection of df aligns to XZ plane"""
    r_pd = pd.DataFrame(r);
    df_r = r_pd.dot(df[col_names_list].to_numpy().T).T
    df_r.columns = ["X", "Y", "Z"]

    df[col_new_names_list[0]] = df_r["X"].tolist()
    df[col_new_names_list[1]] = df_r["Z"].tolist()

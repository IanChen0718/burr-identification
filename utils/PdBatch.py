# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 10:22:08 2022

@author: IanChen
"""
import copy
import numpy as np 
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import seaborn as sns 
from .draw import *
def PdBatch(df, scan_dir, norm_dir, high_dir, batch_size, min_or_max, batch_dir):
    # df_norm_dir = copy.deepcopy(df[[norm_dir]])
    df_xyz = copy.deepcopy(df[["X", "Y", "Z"]])
    df_xyz[scan_dir] = df_xyz[scan_dir].round(decimals=2)
    df_xyz_groupby = df_xyz.groupby([scan_dir])
    df_xyz_group = df_xyz_groupby.groups
    batch_list = []
    while(len(df_xyz_group) != 0):
        batch_local_list = []
        if len(df_xyz_group) >= batch_size*2:   
            for i in range(batch_size):
                fist_key = list(df_xyz_group.keys())[0]
                batch_local_list.append(df_xyz_group.pop(fist_key).values)
        elif len(df_xyz_group) != 0:
            for i in range(len(df_xyz_group)):
                fist_key = list(df_xyz_group.keys())[0]
                batch_local_list.append(df_xyz_group.pop(fist_key).values)
        batch_local_list = np.concatenate(batch_local_list)  
        batch_list.append(batch_local_list)
    X_label_indices_list = []
    for i,e in enumerate(batch_list):
        if e.shape[0] >=2:
            # draw_one_pd_result(df_xyz.loc[e])
            X = df_xyz.loc[batch_list[i]][[norm_dir, high_dir]]
            kmeans = KMeans(n_clusters = 2, random_state = 0).fit(X)
            m_index = None
            if min_or_max == 0:
                m_index = np.where(X[batch_dir] == X[batch_dir].min())[0]
            elif min_or_max == 1:
                 m_index = np.where(X[batch_dir] == X[batch_dir].max())[0]
            m_label = kmeans.labels_[m_index]
            label_indices = np.where(kmeans.labels_ == m_label[0])
            X_trim = X[batch_dir].iloc[label_indices]
            X_label_indices = list(X_trim.index)
            X_label_indices_list.append(X_label_indices)
    X_label_indices_list = np.concatenate(X_label_indices_list)  
    df_trim = df.loc[X_label_indices_list]    
    exclude_index = df.index.isin(X_label_indices_list)
    df_exclude = df.loc[~exclude_index]    
    return df_exclude, df_trim

def batch_split_visulization(source_plan_np, scan_dir, batch_size):
    source_plan_np_copy = copy.deepcopy(source_plan_np)
    source_plan_np_copy[:, scan_dir] = np.floor(source_plan_np_copy[:, scan_dir]* 100) / 100.0    
    init_dir = source_plan_np_copy[0, scan_dir]
    final_dir = source_plan_np_copy[-1, scan_dir]
    move_dir = init_dir
    split_batch_list = []
    while move_dir <= final_dir:
        result_batch_list = []
        for i in range(batch_size):
            result = np.where(source_plan_np_copy[:, scan_dir] == move_dir)
            result_batch_list.append(list(result[0]))
            # split_list.append(source_plan_np[result[0]])
            if move_dir == final_dir:
                break
            else:
                move_dir = source_plan_np_copy[result[0][-1]+1, scan_dir]
        result_batch_list = sum(result_batch_list, [])
        split_batch_list.append(source_plan_np[result_batch_list])
        if move_dir == final_dir:
            break

    type_list = []
    source_2d = []   

    for i,e in enumerate(split_batch_list):
        if e.shape[0] >=2:
            e_xz = np.delete(e, scan_dir, axis = 1)   
            source_2d.append(e_xz)
            e_xz_type_list = [None] * e_xz.shape[0]
            e_xz_type_np = np.asarray(e_xz_type_list)
            kmeans = KMeans(n_clusters = 2, random_state = 0).fit(e_xz)
            min_index = np.where(e_xz[:, 0] == np.amin(e_xz[:, 0]))
            label_value = kmeans.labels_[min_index]
            label_value_another = None
            if label_value == 0:
                label_value_another = 1
            else:
                label_value_another = 0
            index_ = np.where(kmeans.labels_ == label_value )
            index_another = np.where(kmeans.labels_ == label_value_another)
            e_xz_type_np[index_] = "cluster 1"
            e_xz_type_np[index_another] = "cluster 2"  
            type_list.append(e_xz_type_np)     
    # print(type_list)
    source_2d_np = np.vstack(source_2d)
    type_np = np.hstack(type_list)
    type_list_new = type_np.tolist()
    # print(source_2d_np)
    # print(type_list_new)
    dic = {
    "y distance(mm)": source_2d_np[:,0].tolist(), 
    "z distance(mm)": source_2d_np[:,1].tolist(), 
    "type" : type_list_new
    }        
    df = pd.DataFrame(dic)
    return df
    
def PdGroup(df, scan_dir, norm_dir, high_dir, batch_size, batch_dir):
    # df_norm_dir = copy.deepcopy(df[[norm_dir]])
    df_xyz = copy.deepcopy(df[["X", "Y", "Z"]])
    df_xyz[scan_dir] = df_xyz[scan_dir].round(decimals=1)
    df_xyz_groupby = df_xyz.groupby([scan_dir])
    df_xyz_group = df_xyz_groupby.groups
    batch_list = []
    while(len(df_xyz_group) != 0):
        batch_local_list = []
        if len(df_xyz_group) >= batch_size*2:   
            for i in range(batch_size):
                fist_key = list(df_xyz_group.keys())[0]
                batch_local_list.append(df_xyz_group.pop(fist_key).values)
        elif len(df_xyz_group) != 0:
            for i in range(len(df_xyz_group)):
                fist_key = list(df_xyz_group.keys())[0]
                batch_local_list.append(df_xyz_group.pop(fist_key).values)
        batch_local_list = np.concatenate(batch_local_list)  
        batch_list.append(batch_local_list)
    return batch_list
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 00:08:36 2022

@author: IanChen
"""
import copy
import numpy as np
import pandas as pd
from .PdGeom import PdGeom
from .PdProject import PdProject
from .PdBatch import *

def SourceProfile(df, scan_dir, norm_dir, high_dir):
    geometry = PdGeom(df)
    geometry.estimate(20) 
    
    df_copy = copy.deepcopy(df)
    PdProject(df_copy, [1, 1, 0], [1, 0, 0], [0, 1, 0], ["X", "Y", "Z"], ["PX", "PY", "PZ"])    
    df_copy_proj = df_copy[["PX", "PY", "PZ"]]
    df_copy_proj.columns = ["X", "Y", "Z"]

#     geometry2 = PdGeom(df_copy_proj)
#     df_proj = geometry2.estimate(20)    
#     source_proj_plan, source_proj_corr = geometry2.planarize(0.4)
#     draw_pd_result(source_proj_plan, source_proj_corr)

# #%%    
#     df_copy["Planarity"] = df_proj["Planarity"]
#     source_plan, source_corr = PdPlanarize(df_copy, 0.5)
#     source_plan = df.loc[source_plan.index]
#     draw_pd_result(source_plan, source_corr)
#%%
    df_copy_proj_group = PdGroup(df_copy_proj, scan_dir, norm_dir, high_dir, 1, norm_dir)
    group_size_list = [len(ele) for ele in df_copy_proj_group]
    # sns.histplot(data=group_size_list)

    group_size_list_std = np.std(group_size_list)
    group_size_list_mean = np.mean(group_size_list)
    group_size_pd = pd.DataFrame(group_size_list)
    group_size_index = group_size_pd[group_size_pd[0] > (group_size_list_mean-2*group_size_list_std)].index.values
    df_copy_proj_group_new = [df_copy_proj_group[i] for i in group_size_index]
   
    # sns.kdeplot(group_size_list, shade=True)
    # plt.show()
#%%
    """Group"""
    group_max_min = []
    for ele in df_copy_proj_group_new:
        if len(ele) > 2:
            group_elements = df_copy_proj.loc[ele]
            # print("group_elements :\n", group_elements)
            min_index = group_elements[norm_dir].idxmin()
            # max_index = group_elements[norm_dir].idxmax()
            # group_max_min.append([min_index])
            group_elements_sort = group_elements.sort_values(by=[norm_dir], ascending=True)
            # print("group_elements sort :\n", group_elements_sort)
            # print("min :", group_elements.loc[min_index][norm_dir])
            # print("max :", group_elements.loc[max_index][norm_dir])
            # print("min_index :\n", min_index)
            # print("min_index :\n", group_elements_sort[:2].index.values.tolist())
            # print("type :\n", type(group_elements_sort[:2].index.values.tolist()))
            group_max_min.append(group_elements_sort[:2].index.values.tolist())

    group_max_min = np.concatenate(group_max_min)

    temp = copy.deepcopy(df_copy_proj)
    temp["Y"] = temp["Y"].round(decimals=2)
    df_xyz_groupby = temp.groupby(["Y"])
    df_xyz_group = df_xyz_groupby.groups

    temp = df_copy.loc[group_max_min]
    temp_copy = copy.deepcopy(df_copy.drop(group_max_min))    
    return temp_copy, temp
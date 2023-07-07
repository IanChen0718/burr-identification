# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 17:56:34 2022

@author: IanChen
"""
import numpy as np
# from .Gemo import Geom

# def planarity(source_trim, plan_threshold, corr_threshold, source_trim_pro = None):
#     source_plan_list = []
#     source_corr_list = []
#     if source_trim_pro is None:   
#         geometry = Geom(source_trim)
#         source_normals, source_planarity, idxNN_all_qp, pc1, pc2 = geometry.estimate_normals(20)
#         # fast_histArray, features_list = geometry.calc_alpha_phi_theta(source_normals, idxNN_all_qp)    
#         for i, element in enumerate(source_planarity):
#             if element < plan_threshold:
#                 source_plan_list.append(source_trim[i])
#             elif element > corr_threshold:
#                 source_corr_list.append(source_trim[i])
#     else:
#         geometry = Geom(source_trim)
#         source_normals, _, idxNN_all_qp, pc1, pc2 = geometry.estimate_normals(20)
        
#         geometry = Geom(source_trim_pro)
#         _, source_planarity, _, _, _ = geometry.estimate_normals(20)
#         # fast_histArray, features_list = geometry.calc_alpha_phi_theta(source_normals, idxNN_all_qp)    
#         for i, element in enumerate(source_planarity):
#             if element < plan_threshold:
#                 source_plan_list.append(source_trim[i])
#             elif element > corr_threshold:
#                 source_corr_list.append(source_trim[i])        
#     source_plan_np = np.asarray(source_plan_list)
#     source_corr_np = np.asarray(source_corr_list)
#     return source_plan_np, source_corr_np, source_normals, idxNN_all_qp, pc1, pc2


def planarity(source_trim, plan_threshold, corr_threshold, source_planarity, pc_trim = None):
    source_plan_list = []
    source_corr_list = []
    target_pc1_trim_list = []
    # fast_histArray, features_list = geometry.calc_alpha_phi_theta(source_normals, idxNN_all_qp)    
    for i, element in enumerate(source_planarity):
        if element < plan_threshold:
            source_plan_list.append(source_trim[i])
            if pc_trim is not None:
                target_pc1_trim_list.append(pc_trim[i])
                
        elif element > corr_threshold:
            source_corr_list.append(source_trim[i])   
    source_plan_np = np.asarray(source_plan_list)
    source_corr_np = np.asarray(source_corr_list)
    target_pc1_trim_np = None
    if pc_trim is not None:
        target_pc1_trim_np = np.asarray(target_pc1_trim_list)
    return source_plan_np, source_corr_np, target_pc1_trim_np
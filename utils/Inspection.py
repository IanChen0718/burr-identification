# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 01:12:01 2022

@author: IanChen
"""
import copy
import numpy as np
import pandas as pd
from scipy import spatial

class Inspection:
    def __init__(self, source, target, burr_size, uv, pc2uv, neighbours_number, add_unit = None):
        # self.scan_dir = scan_dir
        # self.norm_dir = norm_dir
        # self.high_dir = high_dir
        
        self.u = uv[0]
        self.v = uv[1]
        self.pc2u = pc2uv[0]
        self.pc2v = pc2uv[1]
        self.target_list = []
        self.dist_list = []
        self.dot_dist_list = []
        self.p2l_dist_list = []
        self.source_corresponding_list = []
        self.burr_size  = burr_size
        
        self.source = copy.deepcopy(source)
        # self.source["Color"] = "Source"
        self.source["Color"] = "<0.2mm"
        self.target = copy.deepcopy(target)
        self.target["Color"] = "Target"
        
        
        
        self.source_2d = source[[self.u, self.v]].to_numpy()
        self.target_2d = target[[self.u, self.v]].to_numpy()
        
        # self.source_pc2_2d = source[[self.pc2u, self.pc2v]].to_numpy()
        self.target_pc2_2d = target[[self.pc2u, self.pc2v]].to_numpy()
        
        kdtree = spatial.cKDTree(self.target_2d)
        self.disNN, self.idxNN = kdtree.query(self.source_2d, k=neighbours_number, p=2, workers=-1) # return distance and index    
        
        # self.source_pc2_2d_unit = self.unitize(self.source_pc2_2d)
        self.target_pc2_2d_unit = self.unitize(self.target_pc2_2d)
        
    
        # self.source["U"+self.pc2u] = list(self.source_pc2_2d_unit[self.pc2u].to_numpy())
        # self.source["U"+self.pc2v] = list(self.source_pc2_2d_unit[self.pc2v].to_numpy())
        if add_unit is None:
            self.target["U"+self.pc2u] = list(self.target_pc2_2d_unit[self.pc2u].to_numpy())
            self.target["U"+self.pc2v] = list(self.target_pc2_2d_unit[self.pc2v].to_numpy())

        # print("self.source: \n", self.source)
        self.index_color = self.source.columns.get_loc("Color")
    def unitize(self, pc2_2d):
        pc2_2d_unit = copy.deepcopy(pc2_2d)
        for i in range(len(pc2_2d)):
            pc2_2d_unit[i] = pc2_2d[i] / np.linalg.norm(pc2_2d[i])
        pc2_2d_unit_pd = pd.DataFrame(pc2_2d_unit)
        pc2_2d_unit_pd.columns = [self.pc2u, self.pc2v]   
        return pc2_2d_unit_pd
    
    def calculate_distance(self):
        for i in range(len(self.source_2d)):
            q = self.source_2d[i, :].reshape((1, -1))
            idx = self.idxNN[i]
            
            q_target_nn = self.target_2d[idx]
            q_target_nn_pc2 = np.asarray(self.target_pc2_2d_unit.iloc[idx])
            
            m = q_target_nn_pc2[:, 1]/q_target_nn_pc2[:, 0]
            b = q_target_nn[:, 1] - m*q_target_nn[:, 0]
            value = q[0, 1] - m*q[0, 0] - b
            
            diff = q - q_target_nn
            dot_dist = np.absolute(np.einsum('ij,ij->i', q_target_nn_pc2, diff))
            index_min = np.argmin(dot_dist)
            self.source_corresponding_list.append(idx[index_min])
            # negative_index = np.where(np.asarray(value) < 0)
            value = np.absolute(value)
            
            # if np.any(negative_index[0] <= 3):
            if np.any(value < 0):
                self.dist_list.append(0)
            else: 
                dist = np.linalg.norm(q_target_nn[index_min]-self.source_2d[i, :])
                self.dist_list.append(dist)
                self.dot_dist_list.append(np.min(dot_dist))
                
                m = q_target_nn_pc2[index_min, 1]/q_target_nn_pc2[index_min, 0]
                b = q_target_nn[index_min, 1] - m*q_target_nn[index_min, 0]
                # p2l = abs(source_2d[i, 0]*m + b - source_2d[i, 1]) / np.linalg.norm([q_target_nn_pc2[index_min, 1], q_target_nn_pc2[index_min, 0]])
                p2l = abs(q_target_nn_pc2[index_min, 1]*self.source_2d[i, 0] - q_target_nn_pc2[index_min, 0]*self.source_2d[i, 1] + b*q_target_nn_pc2[index_min, 0]) / np.linalg.norm([q_target_nn_pc2[index_min, 1], q_target_nn_pc2[index_min, 0]])
                self.p2l_dist_list.append(p2l)
                
                self.target_list.append(idx[index_min])
                # if dist > burr_size:
                # if np.min(dot_dist) > burr_size:
                if p2l > self.burr_size:
                    # print("BURR")
                    self.source.iloc[i, self.index_color] = ">=0.2mm"
                    # self.target.iloc[idx[index_min], self.index_color] = "Corresponding"
                    
        max_dist_index = np.argmax(self.dist_list)  
        # row_indexer = source_corr_trim_xyz2.index[max_dist_index]
        # source_corr_trim_xyz2.loc[row_indexer, "Color"] = "Max"
        # self.source.iloc[max_dist_index, self.index_color] = "Max"
        self.source_corresponding = np.asarray(self.source_corresponding_list)
        return self.source, self.target, self.source_corresponding, self.dist_list
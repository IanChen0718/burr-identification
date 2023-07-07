# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:59:32 2022

@author: IanChen
"""
import math
import numpy as np 
import pandas as pd
import copy 
from scipy import spatial 
import matplotlib.pyplot as plt
import seaborn as sns
from celluloid import Camera

class Manifold:
    def __init__(self, source, target, uv, camera_on_or_off = "OFF", delta = None):
        self.source = source
        self.target = target
        self.u = uv[0]
        self.v = uv[1]

        self.q = target[self.u].idxmin() ## by label
        x_offset = target.loc[self.q, self.u]
        y_offset = target.loc[self.q, self.v]
        
        target[self.u] = target[self.u] - x_offset
        target[self.v] = target[self.v] - y_offset
        
        source[self.u] = source[self.u] - x_offset
        source[self.v] = source[self.v] - y_offset
        

        self.source_2d = source[[self.u, self.v]] ## 2-Dimension
        self.target_2d = target[[self.u, self.v]]        
        
        self.source_idx_list = []
        self.target_idx_list = []
        self.angle_list =[]
        self.r_angle_list = []
        self.source_new_list = []
        self.target_new_list = []
        
        self.target_tangent_list = []
        
        self.source_new_np = np.empty((0,2))
        self.target_new_np = np.empty((0,2))
        self.camera_on_or_off = camera_on_or_off
        if self.camera_on_or_off == "ON":
            fig = plt.figure()
            self.camera = Camera(fig)
        if delta is None:   
            self.delta = self.find_neighbor(self.source_2d, self.target_2d)
        else:
            self.delta = delta 
        self.anim = None
        
    def unfold(self):
        if len(self.target_2d) > 0:
            #------------------------------------------------------------------------------------------------------------#
            if self.camera_on_or_off == "ON":
                sns.scatterplot(x=self.source_2d.iloc[:, 0], y=self.source_2d.iloc[:, 1], color="b", sizes=10)
                sns.scatterplot(x=self.target_2d.iloc[:, 0], y=self.target_2d.iloc[:, 1], color="g", sizes=10)
            #------------------------------------------------------------------------------------------------------------#
            self.target_idx_list.append(self.q)
            query_2d = self.target_2d.loc[self.q, :].to_numpy() 
            self.target_new_np = np.vstack((self.target_new_np, query_2d))
            
            tau = self.corresponding_neighbors(self.q)
            q_next = self.find_next_query(query_2d, self.target_2d)
    
            q_next_2d = self.target_2d.iloc[[q_next], :] 
            q_next_label = q_next_2d.index.values
            
            if len(self.target_2d) > 1:
                start_vec = np.asarray(q_next_2d).tolist()[0]
                goal_vec = [1, 0]
                rm_2d, r_angle = self.rotation_matrix(start_vec, goal_vec)       
                rm_2d_pd = pd.DataFrame(rm_2d)
        
                target_2d_labels = self.target_2d.index
                source_2d_labels = self.source_2d.index
                if len(tau) > 0:
                    source_nn = self.source_2d.loc[tau, :].to_numpy()
                    angle = self.angle_calculate(source_nn, np.asarray(start_vec))
                    self.angle_list.append(angle)
                    self.r_angle_list.append([r_angle]*len(tau))
                    
                self.target_2d = rm_2d_pd.dot(self.target_2d.to_numpy().T).T
                self.source_2d = rm_2d_pd.dot(self.source_2d.to_numpy().T).T
            
                self.target_2d.index = target_2d_labels 
                self.source_2d.index = source_2d_labels
            if len(tau) > 0:
                source_nn = self.source_2d.loc[tau, :]
                self.source_new_np = np.vstack((self.source_new_np, source_nn.to_numpy()))
                self.source_2d.drop(tau.tolist(), inplace = True)            
                self.source_idx_list.append(tau.tolist())
                if len(self.target_2d) == 1:
                    angle = np.asarray([0]*len(tau))
                    self.angle_list.append(angle)
                    self.r_angle_list.append(angle)
                    
            x_offset_local = np.array(self.target_2d.loc[q_next_label, 0])[0]
            y_offset_local = np.array(self.target_2d.loc[q_next_label, 1])[0]
            
            self.target_2d[0] = self.target_2d[0] - x_offset_local
            self.target_2d[1] = self.target_2d[1] - y_offset_local
             
            self.source_2d[0] = self.source_2d[0] - x_offset_local
            self.source_2d[1] = self.source_2d[1] - y_offset_local       
            
            self.source_new_np[:, 0] = self.source_new_np[:, 0] - x_offset_local
            self.source_new_np[:, 1] = self.source_new_np[:, 1] - y_offset_local
            
            self.target_new_np[:, 0] = self.target_new_np[:, 0] - x_offset_local
            self.target_new_np[:, 1] = self.target_new_np[:, 1] - y_offset_local
            
            self.target_2d.drop(self.q, inplace = True)
            self.q = q_next_label.item()  
            #------------------------------------------------------------------------------------------------------------#
            if self.camera_on_or_off == "ON":
                sns.scatterplot(x=self.source_new_np[:, 0], y=self.source_new_np[:, 1], color="b", sizes=10)
                sns.scatterplot(x=self.target_new_np[:, 0], y=self.target_new_np[:, 1], color="g", sizes=10)
                plt.legend(loc='upper right', labels=['Source', 'Target'])
                plt.show()
                self.camera.snap()             
            #------------------------------------------------------------------------------------------------------------#
            return self.unfold()
        else:
            if self.camera_on_or_off == "ON":
                self.anim = self.camera.animate(blit=False)
            self.angle_np = np.hstack(self.angle_list)
            self.r_angle_np = np.hstack(self.r_angle_list)
            return self.source_new_np, self.target_new_np, self.source_idx_list, self.target_idx_list, self.angle_np, self.r_angle_np, self.delta, self.anim
        
                
    def find_neighbor(self, source, target):
        kdtree = spatial.cKDTree(target)
        _, delta = kdtree.query(source, k=1, p=2, workers=-1) ## by index
        return delta
    
    def find_next_query(self, q_2d, target): 
        if len(target) > 1:
            kdtree = spatial.cKDTree(target)     
            _, t_idx = kdtree.query(q_2d, k=2, p=2, workers=-1) ## by index
            q_next = t_idx[1]
            return q_next ## by index
        else:
            q_next = 0
            return q_next ## by index
    
    def corresponding_neighbors(self, q):
        t_idx = np.where(self.target.index == q)[0][0]
        d_idx = np.where(self.delta == t_idx)[0]    
        d_label = self.source.iloc[d_idx, :].index.values 
        return d_label
            
    def rotation_matrix(self, vector1, vector2):
        ### Start
        a = vector1/ np.linalg.norm(vector1)
        
        ### Goal
        b = vector2/ np.linalg.norm(vector2)        
        rm_2d = np.asarray([[a[0]*b[0] + a[1]*b[1], b[0]*a[1]-a[0]*b[1]],
                            [a[0]*b[1] - b[0]*a[1], a[0]*b[0]+a[1]*b[1]]])

        a_angle = math.atan2(vector1[1], vector1[0])*180/np.pi
        b_angle = math.atan2(vector2[1], vector2[0])*180/np.pi
        angle = b_angle - a_angle
        return rm_2d, angle        
            
    def angle_calculate(self, source_vec, start_vec):

        angle_list = []
        start_vec_angle = math.atan2(start_vec[1], start_vec[0])*180/np.pi
        for ele in source_vec:
            source_vec_angle = math.atan2(ele[1], ele[0])*180/np.pi
            angle = source_vec_angle - start_vec_angle
            angle_list.append(angle)
        angle_np = np.asarray(angle_list)
        return angle_np
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
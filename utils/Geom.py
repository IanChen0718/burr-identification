# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 22:01:49 2021

@author: IanChen
"""
import numpy as np
from scipy import spatial
class Geom():
    def __init__(self, pc):
        "pc is an np array"
        self.pc = pc ## Point Cloud
        self.planarity = None
        self.n = self.pc.shape[0] ## number of point cloud
        self.n_neighbors = None
        self._div = 4
    def estimate_normals(self, neighbors):
        self.n_neighbors = neighbors
        # self.nx = np.full(self.no_points, np.nan)
        # self.ny = np.full(self.no_points, np.nan)
        # self.nz = np.full(self.no_points, np.nan)
        normals = np.full([self.n, 3], np.nan)
        self.planarity = np.full(self.n, np.nan)
        
        pc1 = np.full([self.n, 3], np.nan)
        pc2 = np.full([self.n, 3], np.nan)
        
        kdtree = spatial.cKDTree(self.pc)
        query_points = self.pc
        _, idxNN_all_qp = kdtree.query(query_points, k=self.n_neighbors, p=2, n_jobs=-1) # return distance and index 

        for (i, idxNN) in enumerate(idxNN_all_qp):
            
            selected_points = self.pc[idxNN] # np.column_stack((self.x[idxNN], self.y[idxNN], self.z[idxNN]))
            C = np.cov(selected_points.T, bias=False)
            eig_vals, eig_vecs = np.linalg.eig(C)
            idx_sort = eig_vals.argsort()[::-1] # sort from large to small
            eig_vals = eig_vals[idx_sort]
            eig_vecs = eig_vecs[:,idx_sort]
            # self.nx[self.sel[i]] = eig_vecs[0,2]
            # self.ny[self.sel[i]] = eig_vecs[1,2]
            # self.nz[self.sel[i]] = eig_vecs[2,2]
            
            pc1[i] = eig_vecs[:,0]
            pc2[i] = eig_vecs[:,1]
            normals[i] = eig_vecs[:,2]
            self.planarity[i] = (eig_vals[1]-eig_vals[2])/eig_vals[0]
        return normals, self.planarity, idxNN_all_qp, pc1, pc2

    def calc_pfh_hist(self, f):
        """Calculate histogram and bin edges.

        :f: feature vector of f1,f3,f4 (Nx3)
        :returns:
            pfh_hist - array of length div^3, represents number of samples per bin
            bin_edges - range(0, 1, 2, ..., (div^3+1)) 
        """
        # preallocate array sizes, create bin_edges
        pfh_hist, bin_edges = np.zeros(self._div**3), np.arange(0,self._div**3+1)
        
        # find the division thresholds for the histogram
        s = self.calc_thresholds()
        # print("s :", s)
        # Loop for every row in f from 0 to N
        for j in range(0, f.shape[0]):
            # calculate the bin index to increment
            index = 0
            for i in range(1,4):
                index += self.step(s[i-1, :], f[j, i-1]) * (self._div**(i-1))
            
            # Increment histogram at that index
            pfh_hist[index] += 1
        
        return pfh_hist, bin_edges
    
    def calc_alpha_phi_theta(self, normals, idxNN_all_qp):
        idx_neighbor = np.delete(idxNN_all_qp, 0, 1) ## delete first column(itself)
        n_neighbors = self.n_neighbors - 1
        features_list = []
        
        histArray = np.zeros((self.n, self._div**3))
        distArray = np.zeros((n_neighbors))
        distList = []
        
        for i in range(self.n):
            features = np.zeros((n_neighbors, 3))
            print("index: ", i)
            for j in range(len(idx_neighbor[i])):
                pi = self.pc[i]
                pj = self.pc[idx_neighbor[i][j]]                
                if np.arccos(np.dot(normals[i], pj - pi)) <= np.arccos(np.dot(normals[idx_neighbor[i][j]], pi - pj)):
                    ps = pi
                    pt = pj
                    ns = normals[i]
                    nt = normals[idx_neighbor[i][j]]
                else:
                    ps = pj
                    pt = pi
                    ns = normals[idx_neighbor[i][j]]
                    nt = normals[i]      
                u = ns
                difV = pt - ps
                dist = np.linalg.norm(difV)
                difV = difV/dist
                difV = np.asarray(difV).squeeze()
                v = np.cross(difV, u)
                w = np.cross(u, v)

                alpha = np.dot(v, nt)
                phi = np.dot(u, difV)
                theta = np.arctan(np.dot(w, nt) / np.dot(u, nt))
                
                features[j, 0] = alpha
                features[j, 1] = phi
                features[j, 2] = theta
                distArray[j] = dist
                
            features_list.append(features)
            
            distList.append(distArray)
            pfh_hist, bin_edges = self.calc_pfh_hist(features)
            histArray[i, :] = pfh_hist / (len(idx_neighbor[i]))

        fast_histArray = np.zeros_like(histArray)
        for i in range(self.n):
            k = len(idx_neighbor[i])
            for j in range(k):
                spfh_sum = histArray[idx_neighbor[i][j]]*(1/distList[i][j])
            
            fast_histArray[i, :] = histArray[i, :] + (1/k)*spfh_sum
        return fast_histArray, features_list         
                
    def calc_thresholds(self):
        """
        :returns: 3x(div-1) array where each row is a feature's thresholds
        """
        delta = 2./self._div
        s1 = np.array([-1+i*delta for i in range(1,self._div)])
        
        delta = 2./self._div
        s3 = np.array([-1+i*delta for i in range(1,self._div)])
        
        delta = (np.pi)/self._div
        s4 = np.array([-np.pi/2 + i*delta for i in range(1,self._div)])
        
        s = np.array([s1,s3,s4])
        return s                 
                                

               
    def step(self, si, fi):
        """Helper function for calc_pfh_hist. Depends on selection of div

        :si: TODO
        :fi: TODO
        :returns: TODO

        """
        if self._div==2:
            if fi < si[0]:
                result = 0
            else:
                result = 1
        elif self._div==3:
            if fi < si[0]:
                result = 0
            elif fi >= si[0] and fi < si[1]:
                result = 1
            else:
                result = 2
        elif self._div==4:
            if fi < si[0]:
                result = 0
            elif fi >= si[0] and fi < si[1]:
                result = 1
            elif fi >= si[1] and fi < si[2]:
                result = 2
            else:
                result = 3
        elif self._div==5:
            if fi < si[0]:
                result = 0
            elif fi >= si[0] and fi < si[1]:
                result = 1
            elif fi >= si[1] and fi < si[2]:
                result = 2
            elif fi >= si[2] and fi < si[3]:
                result = 3
            else:
                result = 4
        return result                
"""PFH"""                
    # def calc_alpha_phi_theta(self, normals, idxNN_all_qp):
    #     idx_neighbor = np.delete(idxNN_all_qp, 0, 1) ## delete first column(itself)
    #     # n_neighbors = self.n_neighbors - 1
    #     features_list = []
    #     for i in range(self.n):
    #         p_list = [i] + idx_neighbor[i].tolist()
    #         p_list_copy = [i] + idx_neighbor[i].tolist()
    #         # features = np.zeros((n_neighbors, 3))
    #         features = []
    #         for z in p_list:
    #             p_list_copy.pop(0)
    #             for p in p_list_copy:
      
    #                 pi = self.pc[p]
    #                 pj = self.pc[z]
    #                 if np.arccos(np.dot(normals[p], pj - pi)) <= np.arccos(np.dot(normals[z], pi - pj)):
    #                     ps = pi
    #                     pt = pj
    #                     ns = normals[p]
    #                     nt = normals[z]
    #                 else:
    #                     ps = pj
    #                     pt = pi
    #                     ns = normals[z]
    #                     nt = normals[p]      
    #                 u = ns
    #                 difV = pt - ps
    #                 dist = np.linalg.norm(difV)
    #                 difV = difV/dist
    #                 difV = np.asarray(difV).squeeze()
    #                 v = np.cross(difV, u)
    #                 w = np.cross(u, v)
    
    #                 alpha = np.dot(v, nt)
    #                 phi = np.dot(u, difV)
    #                 theta = np.arctan(np.dot(w, nt) / np.dot(u, nt))
                    
    #                 # features[j, 0] = alpha
    #                 # features[j, 1] = phi
    #                 # features[j, 2] = theta
    #                 features.append(np.array([alpha, phi, theta]))
    #         features = np.asarray(features)  
    #         features_list.append(features)
    #     return features_list                         
                
                               
                
                
                
                
                
                
                
                
            
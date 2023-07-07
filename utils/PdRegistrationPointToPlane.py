# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 11:59:28 2022

@author: IanChen
"""
import open3d as o3d 

def PdRegistrationPointToPlane(source_pd, target_pd, trans_init, _radius, _max_nn, threshold):
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()

        source.points = o3d.utility.Vector3dVector(source_pd.to_numpy())
        target.points = o3d.utility.Vector3dVector(target_pd.to_numpy())
        
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius = _radius, max_nn = _max_nn))
        
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius = _radius, max_nn = _max_nn))
        
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        
        return reg_p2l
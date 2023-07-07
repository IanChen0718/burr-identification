# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 17:49:21 2021

@author: IanChen
"""
#%%
import pandas as pd
import numpy as np 
import open3d as o3d
import copy 

import time

from scipy import spatial
from utils import *
from utils.draw import *

import matplotlib.pyplot as plt
import plotly.express as px
 


if __name__=='__main__': 
    p1 = [663.75, -147.03, 279.01]
    p2 = [667.81, -147.03, 256.54]
    p3 = [767.27, -147.03, 256.96]

    #-----------------------target-----------------------#
    target = o3d.io.read_point_cloud("./Hanger/PCD/pcd2xyz_Hanger_block_1000000.pcd")
    #-----------------------source-----------------------#
    source = o3d.io.read_point_cloud("./Hanger/PCD/Hanger_0310_4_0.05.pcd") 

 
    source_orig_pd = pd.DataFrame(source.points, columns =["X", "Y", "Z"])
#%%
    # threshold = 0.00002
    trans_init = np.asarray([[0.0, 1.0, 0.0, 656.5], [-1.0, 0.0, 0.0, -113],
                              [0.0, 0.0, 1.0, 279.253], [0.0, 0.0, 0.0, 1.0]])

    rot_degree = np.radians(7.6)
    trans_rot = np.asarray([[1.0, 0.0, 0.0, 0.0], [0 , np.cos(rot_degree), -np.sin(rot_degree), 0],
                            [0 , np.sin(rot_degree), np.cos(rot_degree), 0], [0.0, 0.0, 0.0, 1.0]])

    trans_ = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 37.5],
                              [0.0, 0.0, 1.0, 19.5], [0.0, 0.0, 0.0, 1.0]])
    
    trans_init2 = trans_ @ trans_rot @ trans_init 

#%%
    """Statistical oulier removal"""
    print("Statistical oulier removal")
    cl, ind = source.remove_statistical_outlier(nb_neighbors=20,
                                            std_ratio=3.0)
    """Voxel Downsampling"""
    uni_down_cl = cl.uniform_down_sample(every_k_points=10)
    uni_down_cl_pd = pd.DataFrame(uni_down_cl.points, columns =["X", "Y", "Z"])
    # o3d.visualization.draw_geometries([uni_down_cl])
    
    uni_down_target = target.uniform_down_sample(every_k_points=3)
    uni_down_target_pd = pd.DataFrame(uni_down_target.points, columns =["X", "Y", "Z"])
    # o3d.visualization.draw_geometries([uni_down_target])      

#%%
    """Point-to-plane
    registration_icp is called with a different parameter TransformationEstimationPointToPlane.     Internally, this class implements functions to compute the residuals and 
    Jacobian matrices of the point-to-plane ICP objective.
    
    """
    print("Apply point-to-plane ICP")
    # threshold = 200
    threshold = 2
    start = time.time()
    # radius=0.1 -> 10cm 
    # uni_down_cl.estimate_normals(
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=5000))
    
    # target.estimate_normals(
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=5000)) 
    uni_down_cl.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=100, max_nn=5000))
    
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=100, max_nn=5000)) 
    
    reg_p2l = o3d.pipelines.registration.registration_icp(
        uni_down_cl, target, threshold, trans_init2,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())    
    
    print("Point-to-plane ICP registration took %.3f sec.\n" % (time.time() - start))
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)
#%%
    """Point-to-point"""
    threshold = 0.2
    reg_p2p = o3d.pipelines.registration.registration_icp(
        uni_down_cl, target, threshold, trans_init2,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200000))
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation) 
    source = cl.transform(reg_p2p.transformation) 
    draw_registration_result(source, target)
#%%
    source_pd = pd.DataFrame(source.points, columns =["X", "Y", "Z"])
    source_pd.insert(0, "Color", "Source")

    target_pd = pd.DataFrame(target.points, columns =["X", "Y", "Z"])
    target_pd.insert(0, "Color", "Target")
#%%
    """Outlier"""
    target_pd_xyz = target_pd[["X", "Y", "Z"]]
    kdtree = spatial.cKDTree(target_pd_xyz)     
    query_xyz = source_pd[["X", "Y", "Z"]]      
    dd, ii = kdtree.query(query_xyz, k=1, p=2, workers=-1)
    i_list = None
    source_len = len(source_pd)
    dd_threshold = 0.5
    for i in range(source_len):
        if dd[i] > dd_threshold:
            source_pd.loc[i, "Color"] = "Outlier" 
    draw_outlier_result(source_pd[source_pd["Color"] == "Source"], source_pd[source_pd["Color"] == "Outlier"])
    source_pd = source_pd[source_pd["Color"] == "Source"]


#%%
    """Attention of Source and Target"""
    print("Source and Target: Trim through the plane!")
    len_local_space = 5000
    target_pd_trim = PdTrimThroughPlane(target_pd, p1, p2, p3, offset = len_local_space)
    source_pd_trim = PdTrimThroughPlane(source_pd, p1, p2, p3, offset = len_local_space)
    draw_pd_result(source_pd_trim, target_pd_trim) 

#%%
    """ICP again"""
    source_pd_trim = source_pd_trim[["X", "Y", "Z"]]
    target_pd_trim = target_pd_trim[["X", "Y", "Z"]]
    
    trans_init2 = np.asarray([[1.0, 0.0, 0.0, 0], [0.0, 1.0, 0.0, 0],
                              [0.0, 0.0, 1.0, 0], [0.0, 0.0, 0.0, 1.0]])
    # draw_pd_result(source_pd_trim, target_pd_trim, trans_init2)
    source2 = o3d.geometry.PointCloud()
    target2 = o3d.geometry.PointCloud()
    
    source2.points = o3d.utility.Vector3dVector(source_pd_trim.to_numpy())
    target2.points = o3d.utility.Vector3dVector(target_pd_trim.to_numpy())
    
    """Point-to-point"""
    trans_init3 = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    reg_p2p2 = o3d.pipelines.registration.registration_icp(
        source2, target2, threshold, trans_init3,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000))
    print(reg_p2p2)
    print("Transformation is:")
    print(reg_p2p2.transformation)
    # draw_pd_result(source_pd_trim, target_pd_trim, reg_p2p2.transformation)

    row_lables = source_pd_trim.index.values
    source2 = source2.transform(reg_p2p2.transformation)
    source_pd_trim = pd.DataFrame(source2.points, columns =["X", "Y", "Z"])
    source_pd_trim.index = row_lables
    draw_pd_result(source_pd_trim, target_pd_trim)
#%%
    """Direction corresponding to coordinate system of CAD""" 
    scan_dir = "X" 
    norm_dir = "Y"
    high_dir = "Z"
    
#%%
    """Source Profile"""
    print("Source: Get the Profile!")
    source_corr_drop, source_corr_trim = SourceProfile(source_pd_trim, scan_dir, norm_dir, high_dir)
    draw_pd_result(source_corr_drop, source_corr_trim)
    
#%%
    """Target Profile"""
    print("Target: Get the Profile!")
    """Notice: The position of target_corr_drop and target_corr_drop"""
    target_corr_trim, target_corr_drop = TargetProfile(target_pd_trim, p1, p2 ,p3, len_local_space)
    draw_pd_custom_result(target_corr_drop, target_corr_trim, [0, 1, 0], [127/255, 0, 1])
#%%
    """3D Visualiztion of Target """
    target_corr_trim["Color"] = "Target"
    fig = px.scatter_3d(target_corr_trim, x='X', y='Y', z='Z',
                  color='Color', color_discrete_sequence=["greenyellow"])
    fig.write_html('3D.html', auto_open=True)

    draw_one_pd_result(target_corr_trim, [0, 1, 0])
#%%
    """Dimension reduction with hyperplane"""
    print("Source and Target: Projection!")
    PdProject(source_corr_trim, p1, p2, p3, ["X", "Y", "Z"], ["PX", "PY", "PZ"])
    PdProject(target_corr_trim, p1, p2, p3, ["X", "Y", "Z"], ["PX", "PY", "PZ"])

    """Rotate to Z axis"""
    p_a, p_b, p_c ,p_d = GetPlaneEq(p1, p2, p3)
    vec1 = [p_a, p_b, p_c]
    vec2 = [0, 1, 0]
    a,b = (vec1/ np.linalg.norm(vec1)).reshape(3), (vec2/np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a,b)
    c = np.dot(a,b)
    s = np.linalg.norm(v)
    vXStr = '{} {} {}; {} {} {}; {} {} {}'.format(0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0)
    k = np.matrix(vXStr)
    I = np.identity(3)
    r = I + k + k @ k * ((1 -c)/(s**2))
    rm = np.empty((4, 4))
    rm[:3, :3] = r

    """Projection of Source aligns to XZ plane"""
    print("Source: Get 2D coordinates!")
    Pd3DPlaneto2D(source_corr_trim, p1, p2, p3, ["PX", "PY", "PZ"], ["U", "V"])

    """Projection of Target aligns to XZ plane"""
    print("Target: Get 2D coordinates!")
    Pd3DPlaneto2D(target_corr_trim, p1, p2, p3, ["PX", "PY", "PZ"], ["U", "V"])

    """Source PC2 aligns to XZ plane"""
    print("PC2 of Source: Projectin and Get 2D coordinates!")
    PdProject(source_corr_trim, p1, p2, p3, ["PC2X", "PC2Y", "PC2Z"], ["PPC2X", "PPC2Y", "PPC2Z"])
    Pd3DPlaneto2D(source_corr_trim, p1, p2, p3, ["PPC2X", "PPC2Y", "PPC2Z"], ["PC2U", "PC2V"])
    
    """Target PC2 aligns to XZ plane"""
    print("PC2 of Target: Projectin and Get 2D coordinates!")
    PdProject(target_corr_trim, p1, p2, p3, ["PC2X", "PC2Y", "PC2Z"], ["PPC2X", "PPC2Y", "PPC2Z"])
    Pd3DPlaneto2D(target_corr_trim, p1, p2, p3, ["PPC2X", "PPC2Y", "PPC2Z"], ["PC2U", "PC2V"])
    
    """Target Normal aligns to XZ"""
    print("Normal of Target: Projectin and Get 2D coordinates!")
    PdProject(target_corr_trim, p1, p2, p3, ["NormalX", "NormalY", "NormalZ"], ["PNormalX", "PNormalY", "PNormalZ"])
    Pd3DPlaneto2D(target_corr_trim, p1, p2, p3, ["PNormalX", "PNormalY", "PNormalZ"], ["NormalU", "NormalV"])

    u_tan = -target_corr_trim.NormalV
    v_tan =  target_corr_trim.NormalU
    
    target_corr_trim["UTAN"] = np.asarray(u_tan).tolist()
    target_corr_trim["VTAN"] = np.asarray(v_tan).tolist()

#%%
    """Arrow of Vector Plot"""   
    
    all_xyz_burr = pd.concat([source_corr_trim, target_corr_trim]) 

    import plotly.figure_factory as ff
    import plotly.graph_objects as go

    fig = ff.create_quiver(target_corr_trim.U, target_corr_trim.V
                            , target_corr_trim.NormalU, target_corr_trim.NormalV)

    fig.add_trace(go.Scatter(x=target_corr_trim.U, y=target_corr_trim.V,
                        mode='markers',
                        marker_size=5,
                        name='Target'))

    fig.write_html('Arrow of vector.html', auto_open=True)

#%%
    """Plot of Dimension reduction"""
    target_corr_trim["Color"] = "Target"
    source_corr_trim["Color"] = "Source"
    all_points = source_corr_trim.append(target_corr_trim, ignore_index = True) 
  
    fig = px.scatter(all_points[["U", "V", "Color"]], x = "U", y = "V", color='Color',
                      color_discrete_sequence=["blue", "green"])    
    fig.write_html('Dimension reduction with hyperplane.html', auto_open=True)  

#%%
    """Manifold Mapping Algorithm"""
    
    """Manifold function"""
    MF = Manifold(source_corr_trim, target_corr_trim, ["U", "V"], "OFF")
    source_new_np, target_new_np, source_index_list, target_index_list, angle_np, r_angle_np, delta, _ = MF.unfold()
    # # anim.save('C:/Users/IanChen/workspace/Burr_Detection_v2/manifold_random10000.mp4')
    source_proj_r = copy.deepcopy(source_corr_trim)

    neg_index = np.where(source_new_np[:, 1] < 0.0)
    u_new = target_new_np[:,0]
    u_new = np.append(u_new, source_new_np[:,0])
    u_new = np.append(u_new, source_new_np[neg_index, 0])
    
    v_new = target_new_np[:,1]
    v_new = np.append(v_new, source_new_np[:,1])
    v_new = np.append(v_new, source_new_np[neg_index, 1])

    color_new = ["Target"]*len(target_new_np[:,0])
    color_new = color_new + ["Source"]*len(source_new_np[:,0])
    color_new = color_new + ["Bad"]*len(source_new_np[neg_index, 0][0])

    all_points_dict = {"U": u_new,
                       "V": v_new,
                       "Color": color_new
                       }
    all_points = pd.DataFrame(all_points_dict)
    fig = px.scatter(all_points[["U", "V", "Color"]], x = "U", y = "V", color='Color',
                      color_discrete_sequence=["green", "blue", "red"])   
    fig.write_html('new.html', auto_open=True)  

    """Map to Original Space"""
    neg_index = np.where(source_new_np[:, 1] <= 0.0)
    source_index_np = np.hstack(source_index_list)
    neg_index_ = source_index_np[neg_index]
    source_proj_r.loc[neg_index_, 'Color'] = "Bad"
    all_points = source_proj_r.append(target_corr_trim[["U", "V", "Color"]], ignore_index = True)   
    fig = px.scatter(all_points[["U", "V", "Color"]], x = "U", y = "V", color='Color',
                      color_discrete_sequence=["blue", "red", "green"])    
    fig.write_html('Hanger.html', auto_open=True)  

    """matplot """
    # fig, ax = plt.subplots()
    # ax.scatter(all_points[all_points["Color"] == "Target"].U, all_points[all_points["Color"] == "Target"].V, color='g',label="Target")
    # ax.scatter(all_points[all_points["Color"] == "Source"].U, all_points[all_points["Color"] == "Source"].V, color='b',label="Source")
    # ax.scatter(all_points[all_points["Color"] == "Bad"].U, all_points[all_points["Color"] == "Bad"].V, color='r',label="Bad")
    # plt.xlabel("U")
    # plt.ylabel("V")
    # plt.legend()
    # plt.show()

#%%
    """Responding Searching Algorithm"""
    size_indeces = np.where((source_new_np[:, 1] <= 0.2) & (source_new_np[:, 1] > 0.0))[0]
    size_indeces_manifold = np.sort(source_index_np[size_indeces])
    
    """Inspection funciton"""
    source_proj_r_ = source_proj_r[source_proj_r["Color"] == "Source"].copy(deep=True)  
    inspection = Inspection(source_proj_r_, target_corr_trim, 0.2, ["U", "V"], ["UTAN", "VTAN"], 6, "No")
    source_proj_r2, target_proj_r2, source_proj_r2_corresponding, dist_list  = inspection.calculate_distance()

    all_xyz_burr = pd.concat([source_proj_r2, target_proj_r2]) 
    fig = px.scatter(all_xyz_burr, x = "U", y = "V", color='Color', 
                      color_discrete_sequence=["red", "blue", "purple", "green", "yellow"])
    fig.write_html('Burr Inspection.html', auto_open=True) 

    size_indeces_inspect = source_proj_r2[source_proj_r2["Color"]=="Source"].index.values


    

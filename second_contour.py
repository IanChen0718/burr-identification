# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 17:49:21 2021

@author: IanChen
"""

import open3d as o3d
import copy 
import numpy as np 
import time

from scipy import spatial
from utils import *
from utils.draw import *

import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd 
#%%
if __name__=='__main__': 
    p1 = [667.11, -107.69, 260.41]
    p2 = [667.81, -147.03, 256.54]
    p3 = [663.75, -147.03, 279.01]
   
    #-----------------------target-----------------------#
    target = o3d.io.read_point_cloud('./Hanger/PCD/pcd2xyz_Hanger_block_1000000.pcd')
    # #-----------------------source-----------------------#
    source = o3d.io.read_point_cloud('./Hanger/PCD/Hanger_side_0321_0.05.pcd')  
    source_orig_pd = pd.DataFrame(np.asarray(source.points), columns =["X", "Y", "Z"])
    # threshold = 0.00002
    trans_init = np.asarray([[1.0, 0.0, 0.0, 662.6], [0.0, 1.0, 0.0, -171],
                              [0.0, 0.0, 1.0, 279.253], [0.0, 0.0, 0.0, 1.0]])

    rot_degree = np.radians(7.6)
    trans_rot = np.asarray([[1.0, 0.0, 0.0, 0.0], [0 , np.cos(rot_degree), -np.sin(rot_degree), 0],
                            [0 , np.sin(rot_degree), np.cos(rot_degree), 0], [0.0, 0.0, 0.0, 1.0]])

    trans_ = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 36.5],
                              [0.0, 0.0, 1.0, 19.3], [0.0, 0.0, 0.0, 1.0]])
    
    radian_ = np.radians(180)
    trans_2 = np.asarray([[np.cos(radian_), -np.sin(radian_), 0.0, 0.0], [np.sin(radian_), np.cos(radian_), 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    trans_3 = np.asarray([[1.0, 0.0, 0.0, 37.0], [0.0, 1.0, 0.0, 108],
                              [0.0, 0.0, 1.0, 29], [0.0, 0.0, 0.0, 1.0]])
    trans_init2 =  trans_3 @ trans_rot @ trans_init @ trans_2

    """Statistical oulier removal"""
    print("Statistical oulier removal")
    cl, ind = source.remove_statistical_outlier(nb_neighbors=20,
                                            std_ratio=3.0)
    """Voxel Downsampling"""
    uni_down_cl = cl.uniform_down_sample(every_k_points=10)
    uni_down_cl_pd = pd.DataFrame(np.asarray(uni_down_cl.points), columns =["X", "Y", "Z"])
    # o3d.visualization.draw_geometries([uni_down_cl])
    
    uni_down_target = target.uniform_down_sample(every_k_points=3)
    uni_down_target_pd = pd.DataFrame(np.asarray(uni_down_target.points), columns =["X", "Y", "Z"])
    # o3d.visualization.draw_geometries([uni_down_target])      

    """Point-to-plane
    registration_icp is called with a different parameter TransformationEstimationPointToPlane.     Internally, this class implements functions to compute the residuals and 
    Jacobian matrices of the point-to-plane ICP objective.
    
    """
    print("Apply point-to-plane ICP")
    # threshold = 200
    threshold = 2
    start = time.time()
    # radius=0.1 -> 10cm 
    uni_down_cl.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=500))
    
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=500)) 
    
    reg_p2l = o3d.pipelines.registration.registration_icp(
        uni_down_cl, target, threshold, trans_init2,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())    
    
    print("Point-to-plane ICP registration took %.3f sec.\n" % (time.time() - start))
    print(reg_p2l)
    # print("Transformation is:")
    # print(reg_p2l.transformation)
    # draw_registration_result(cl, target, reg_p2l.transformation)

    """Point-to-point"""
    threshold = 0.2
    reg_p2p = o3d.pipelines.registration.registration_icp(
        uni_down_cl, target, threshold, trans_init2,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200000))
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # draw_registration_result(cl, target, reg_p2p.transformation)
    source = cl.transform(reg_p2p.transformation) 
    # draw_registration_result(source, target)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(source)
    # o3d.io.write_point_cloud("./PCD/source.pcd", source)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(target)
    # o3d.io.write_point_cloud("./PCD/target.pcd", target)   

   
    source_pd = pd.DataFrame(np.asarray(source.points), columns =["X", "Y", "Z"])
    source_pd.insert(0, "Color", "Source")

    target_pd = pd.DataFrame(np.asarray(target.points), columns =["X", "Y", "Z"])
    target_pd.insert(0, "Color", "Target")

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
    # draw_outlier_result(source_pd[source_pd["Color"] == "Source"], source_pd[source_pd["Color"] == "Outlier"])

    source_pd = source_pd[source_pd["Color"] == "Source"]
    # draw_pd_result(source_pd, target_pd) 
#%%

    """Attention of Target Point Cloud"""
    target_pd_trim = PdTrimThroughPlane(target_pd, p1, p2, p3, offset = 30000)

    """Visualization of Hyperplane"""
    # vertical_stack = pd.concat([source_pd, target_pd_trim], axis=0)
    # fig = px.scatter_3d(vertical_stack, x='X', y='Y', z='Z', color_discrete_sequence=["blue", "green"],
    #               color='Color')
    # x0, y0, z0 = p1
    # x1, y1, z1 = p2
    # x2, y2, z2 = p3
    
    # ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
    # vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]
    
    # u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]
    
    # point  = np.array(p1)
    # normal = np.array(u_cross_v)
    
    # d = -point.dot(normal)
    # x = np.linspace(640, 680, 10)
    # y = np.linspace(-145, -100, 10)
    # xx, yy = np.meshgrid(x, y)
    
    # z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
   
    # import plotly.graph_objects as go
    # fig.add_trace(go.Surface(z=z, x=xx, y=yy))    
   
    # fig.write_html('Visualization of Hyperplane.html', auto_open=True)


    """Attention of Source and Target"""
    print("Source and Target: Trim through the plane!")
    len_local_space = 1000
    target_pd_trim = PdTrimThroughPlane(target_pd, p1, p2, p3, offset = len_local_space)
    source_pd_trim = PdTrimThroughPlane(source_pd, p1, p2, p3, offset = len_local_space)
    # draw_pd_result(source_pd_trim, target_pd_trim) 


    """ICP again"""
    source_pd_trim = source_pd_trim[["X", "Y", "Z"]]
    target_pd_trim = target_pd_trim[["X", "Y", "Z"]]
    
    trans_init2 = np.asarray([[1.0, 0.0, 0.0, 0], [0.0, 1.0, 0.0, 0],
                              [0.0, 0.0, 1.0, 0], [0.0, 0.0, 0.0, 1.0]])
    
    source2 = o3d.geometry.PointCloud()
    target2 = o3d.geometry.PointCloud()
    
    source2.points = o3d.utility.Vector3dVector(source_pd_trim.to_numpy())
    target2.points = o3d.utility.Vector3dVector(target_pd_trim.to_numpy())
    
    trans_init3 = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    reg_p2p2 = o3d.pipelines.registration.registration_icp(
        source2, target2, threshold, trans_init3,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000))
    # print(reg_p2p2)
    # print("Transformation is:")
    # print(reg_p2p2.transformation)
    # draw_pd_result(source_pd_trim, target_pd_trim, reg_p2p2.transformation)

    row_lables = source_pd_trim.index.values
    source2 = source2.transform(reg_p2p2.transformation)
    source_pd_trim = pd.DataFrame(np.asarray(source2.points), columns =["X", "Y", "Z"])
    source_pd_trim.index = row_lables
    # draw_pd_result(source_pd_trim, target_pd_trim)

    """Corresponding to coordinate system of CAD""" 
    scan_dir = "Y" 
    norm_dir = "X"
    high_dir = "Z"
    

    """Source Profile"""
    print("Source: Get the Profile!")
    source_corr_drop, source_corr_trim = SourceProfile(source_pd_trim, scan_dir, norm_dir, high_dir)
    # draw_pd_result(source_corr_drop, source_corr_trim)
    

    """Target Profile"""
    print("Target: Get the Profile!")
    target_corr_drop, target_corr_trim = TargetProfile(target_pd_trim, p1, p2 ,p3, len_local_space)
    print("*Planarity combined with GMM took %.3f sec.\n" % (time.time() - start))
    # draw_pd_custom_result(target_corr_drop, target_corr_trim, [0, 1, 0], [127/255, 0, 1])


    """3D Visualiztion of Target """
    # print("3D Visualiztion of Target")
    # target_corr_trim["Color"] = "Target"
    # fig = px.scatter_3d(target_corr_trim, x='X', y='Y', z='Z',
    #               color='Color', color_discrete_sequence=["greenyellow"])
    # fig.write_html('3D.html', auto_open=True)

    # draw_one_pd_result(target_corr_trim, [0, 1, 0])
    # draw_one_pd_result(source_corr_trim, [0, 0, 1])
    # draw_pd_result(source_corr_trim, target_corr_trim)

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
    r_pd = pd.DataFrame(r);
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
    print("*Dimension reduction with hyperplane took %.3f sec.\n" % (time.time() - start))


    """Plot of Dimension reduction"""
    target_corr_trim["Color"] = "Target"
    source_corr_trim["Color"] = "Source"
    all_points = source_corr_trim.append(target_corr_trim, ignore_index = True) 
  
    fig = px.scatter(all_points[["U", "V", "Color"]], x = "U", y = "V", color='Color',
                      color_discrete_sequence=["blue", "green"])    
    fig.write_html('Dimension reduction with hyperplane.html', auto_open=True)  


    """Create Poisson Disk Sampling"""
    # x, y = PoissonDiskSampling(39, 18, 30, 0.1, "y")
    # poi_pd = pd.DataFrame(np.dstack((x, y))[0])
    
    # poi_pd[1] = poi_pd[1] - 13
    # poi_pd.columns = ['U', 'V']
    # poi_pd.to_csv('./poisson_disk_sampling/poisson_disk_sampling_0.1_18.csv', index=False)  
    
    """Use Poisson Disk Sampling"""
    # fig, ax = plt.subplots();
    # poisson_pd = pd.read_csv('./poisson_disk_sampling/poisson_disk_sampling_0.1_18.csv')  
    # """plotlt"""
    # target_corr_trim["Color"] = "Target"
    # poisson_pd["Color"] = "Source"
    # all_points = poisson_pd.append(target_corr_trim, ignore_index = True) 
  
    # fig = px.scatter(all_points[["U", "V", "Color"]], x = "U", y = "V", color='Color',
    #                   color_discrete_sequence=["yellow", "green"])    
    # fig.write_html('Poisson-Disk_Sampling.html', auto_open=True)  

    # """Manifold function"""
    # MF = Manifold(poisson_pd, target_corr_trim, ["U", "V"], "OFF")
    # source_new_np, target_new_np, source_index_list, target_index_list, angle_np, r_angle_np, delta, _ = MF.unfold()
    # source_proj_r = copy.deepcopy(poisson_pd)

    """Manifold Mapping Algorithm"""
    
    """Manifold function"""
    MF = Manifold(source_corr_trim, target_corr_trim, ["U", "V"], "OFF")
    source_new_np, target_new_np, source_index_list, target_index_list, angle_np, r_angle_np, delta, _ = MF.unfold()
    source_proj_r = copy.deepcopy(source_corr_trim)
    neg_index = np.where(source_new_np[:, 1] < 0.0)
    u_new = target_new_np[:,0]
    u_new = np.append(u_new, source_new_np[:,0])
    u_new = np.append(u_new, source_new_np[neg_index, 0])
    
    v_new = target_new_np[:,1]
    v_new = np.append(v_new, source_new_np[:,1])
    v_new = np.append(v_new, source_new_np[neg_index, 1])
    
    color_new = ["Target"]*len(target_new_np[:,0])
    color_new = color_new + ["Burrs(source)"]*len(source_new_np[:,0])
    color_new = color_new + ["Not Burrs(source)"]*len(source_new_np[neg_index, 0][0])
    
    all_points_dict = {"U": u_new,
                       "V": v_new,
                       "Color": color_new
                       }
    all_points = pd.DataFrame(all_points_dict)
    fig = px.scatter(all_points[["U", "V", "Color"]], x = "U", y = "V", color='Color',
                      color_discrete_sequence=["green", "blue", "red"])   
    fig.write_html('Manifold Mapping.html', auto_open=True)  
    
    """Map to Original Space"""
    neg_index = np.where(source_new_np[:, 1] <= 0.0)
    source_index_np = np.hstack(source_index_list)
    neg_index_ = source_index_np[neg_index]
    source_proj_r.loc[:, 'Color'] = "Burrs(Source)"
    source_proj_r.loc[neg_index_, 'Color'] = "Not Burrs(Source)"
    all_points = source_proj_r.append(target_corr_trim[["U", "V", "Color"]], ignore_index = True)   
    fig = px.scatter(all_points[["U", "V", "Color"]], x = "U", y = "V", color='Color',
                      color_discrete_sequence=["blue", "red", "green"])   
    print("*Manifold Mapping Algorithm took %.3f sec.\n" % (time.time() - start)) 
    fig.write_html('Segmentation result.html', auto_open=True)  
    source_proj_r.loc[:, 'Color'] = "<0.2mm"
    
    """matplot """
    # fig, ax = plt.subplots()
    # ax.scatter(all_points[all_points["Color"] == "Target"].U, all_points[all_points["Color"] == "Target"].V, color='g',label="Target")
    # ax.scatter(all_points[all_points["Color"] == "Source"].U, all_points[all_points["Color"] == "Source"].V, color='b',label="Source")
    # ax.scatter(all_points[all_points["Color"] == "Bad"].U, all_points[all_points["Color"] == "Bad"].V, color='r',label="Bad")
    # plt.xlabel("U")
    # plt.ylabel("V")
    # plt.legend()
    # plt.show()
    

    """Responding Searching Algorithm"""
    size_indeces = np.where((source_new_np[:, 1] <= 0.2) & (source_new_np[:, 1] > 0.0))[0]
    size_indeces_manifold = np.sort(source_index_np[size_indeces])
    
    """Inspection funciton"""

    inspection = Inspection(source_proj_r, target_corr_trim, 0.2, ["U", "V"], ["UTAN", "VTAN"], 6, "No")
    source_proj_r2, target_proj_r2, source_proj_r2_corresponding, dist_list = inspection.calculate_distance()
    
    all_xyz_burr = pd.concat([source_proj_r2[source_proj_r2["Color"]==">=0.2mm"], source_proj_r2[source_proj_r2["Color"]=="<0.2mm"], target_proj_r2]) 
    # fig = px.scatter(all_xyz_burr, x = "U", y = "V", color='Color', 
    #                   color_discrete_sequence=["red", "blue", "purple", "green", "yellow"])
    fig = px.scatter(all_xyz_burr, x = "U", y = "V", color='Color', 
                        color_discrete_sequence=["red", "blue", "green"])
    print("*Corresponding Searching Algorithm took %.3f sec.\n" % (time.time() - start))
    fig.write_html('Burr Inspection.html', auto_open=True) 

    size_indeces_inspect = source_proj_r2[source_proj_r2["Color"]=="Source"].index.values

    """Plot burrs on object"""
    hanger = o3d.io.read_point_cloud('./Hanger/PCD/2464M34P01_0413_10000.pcd')
    hanger_pd = pd.DataFrame(np.asarray(hanger.points), columns =["X", "Y", "Z"])
    # burr_pd = source_proj_r2[source_proj_r2["Color"] == "Burr"]
    burr_pd = source_proj_r2[source_proj_r2["Color"] == ">=0.2mm"]
    burr_index = burr_pd.index.values
    burr_origin = source_corr_trim.loc[burr_index, ["X", "Y", "Z"]]
    draw_outlier_result(hanger_pd, burr_origin)

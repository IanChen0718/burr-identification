# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 10:33:49 2022

@author: IanChen
"""
import copy 
import open3d as o3d

def custom_draw_geometry_with_rotations(pcd1, pcd2):

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(5.0, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd1, pcd2],
                                                              rotate_view)
def draw_registration_result(source, target, transformation=None):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    source_temp.paint_uniform_color([0, 0, 1])
    target_temp.paint_uniform_color([0, 1, 0])
    if transformation is not None:
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    # custom_draw_geometry_with_rotations(source_temp, target_temp)
    
def draw_np_result(source, target, transformation=None):
    source_o3d = o3d.geometry.PointCloud()
    target_o3d = o3d.geometry.PointCloud()
    
    source_o3d.points = o3d.utility.Vector3dVector(source)
    target_o3d.points = o3d.utility.Vector3dVector(target)
    
    source_temp = copy.deepcopy(source_o3d)
    target_temp = copy.deepcopy(target_o3d)
    
    source_temp.paint_uniform_color([0, 0, 1])
    target_temp.paint_uniform_color([0, 1, 0])
    if transformation is not None:
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def draw_one_np_result(source):
    source_o3d = o3d.geometry.PointCloud()
    source_o3d.points = o3d.utility.Vector3dVector(source)    
    source_temp = copy.deepcopy(source_o3d)
    source_temp.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([source_temp])
    
def draw_np_three_result(source, target, three):
    source_o3d = o3d.geometry.PointCloud()
    target_o3d = o3d.geometry.PointCloud()
    three_o3d = o3d.geometry.PointCloud()
    
    source_o3d.points = o3d.utility.Vector3dVector(source)
    target_o3d.points = o3d.utility.Vector3dVector(target)
    three_o3d.points = o3d.utility.Vector3dVector(three)
    
    source_temp = copy.deepcopy(source_o3d)
    target_temp = copy.deepcopy(target_o3d)
    three_temp = copy.deepcopy(three_o3d)
    
    source_temp.paint_uniform_color([0, 0, 1])
    target_temp.paint_uniform_color([0, 1, 0])
    three_temp.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([source_temp, target_temp, three_temp])
    
def draw_np_custom_result(source, target, color1, color2, transformation=None):
    source_o3d = o3d.geometry.PointCloud()
    target_o3d = o3d.geometry.PointCloud()
    
    source_o3d.points = o3d.utility.Vector3dVector(source)
    target_o3d.points = o3d.utility.Vector3dVector(target)
    
    source_temp = copy.deepcopy(source_o3d)
    target_temp = copy.deepcopy(target_o3d)
    
    source_temp.paint_uniform_color(color1)
    target_temp.paint_uniform_color(color2)
    if transformation is not None:
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    
def draw_pd_result(source, target, transformation=None):
    source = source[["X", "Y", "Z"]].to_numpy()
    target = target[["X", "Y", "Z"]].to_numpy()
    source_o3d = o3d.geometry.PointCloud()
    target_o3d = o3d.geometry.PointCloud()
    
    source_o3d.points = o3d.utility.Vector3dVector(source)
    target_o3d.points = o3d.utility.Vector3dVector(target)
    
    source_temp = copy.deepcopy(source_o3d)
    target_temp = copy.deepcopy(target_o3d)
    
    source_temp.paint_uniform_color([0, 0, 1])
    target_temp.paint_uniform_color([0, 1, 0])
    if transformation is not None:
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    # custom_draw_geometry_with_rotations(source_temp, target_temp)
def draw_pd_custom_result(source, target, color1, color2, transformation=None):
    source = source[["X", "Y", "Z"]].to_numpy()
    target = target[["X", "Y", "Z"]].to_numpy()
    source_o3d = o3d.geometry.PointCloud()
    target_o3d = o3d.geometry.PointCloud()
    
    source_o3d.points = o3d.utility.Vector3dVector(source)
    target_o3d.points = o3d.utility.Vector3dVector(target)
    
    source_temp = copy.deepcopy(source_o3d)
    target_temp = copy.deepcopy(target_o3d)
    
    source_temp.paint_uniform_color(color1)
    target_temp.paint_uniform_color(color2)
    if transformation is not None:
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    # custom_draw_geometry_with_rotations(source_temp, target_temp)
    
def draw_one_pd_result(source, color=[0, 0, 1]):
    source = source[["X", "Y", "Z"]].to_numpy()
    source_o3d = o3d.geometry.PointCloud()
    source_o3d.points = o3d.utility.Vector3dVector(source)    
    source_temp = copy.deepcopy(source_o3d)
    source_temp.paint_uniform_color(color)
    o3d.visualization.draw_geometries([source_temp])
def draw_pd_three_result(source, target, three, color1, color2, color3):
    source = source[["X", "Y", "Z"]].to_numpy()
    target = target[["X", "Y", "Z"]].to_numpy()
    three = three[["X", "Y", "Z"]].to_numpy()
    
    source_o3d = o3d.geometry.PointCloud()
    target_o3d = o3d.geometry.PointCloud()
    three_o3d = o3d.geometry.PointCloud()
    
    source_o3d.points = o3d.utility.Vector3dVector(source)
    target_o3d.points = o3d.utility.Vector3dVector(target)
    three_o3d.points = o3d.utility.Vector3dVector(three)
    
    source_temp = copy.deepcopy(source_o3d)
    target_temp = copy.deepcopy(target_o3d)
    three_temp = copy.deepcopy(three_o3d)
    
    source_temp.paint_uniform_color(color1)
    target_temp.paint_uniform_color(color2)
    three_temp.paint_uniform_color(color3)
    o3d.visualization.draw_geometries([source_temp, target_temp, three_temp])
    # custom_draw_geometry_with_rotations(source_temp, target_temp, three_temp)
def draw_pd_four_result(source, target, three, four, color1, color2, color3, color4):
    source = source[["X", "Y", "Z"]].to_numpy()
    target = target[["X", "Y", "Z"]].to_numpy()
    three = three[["X", "Y", "Z"]].to_numpy()
    four = four[["X", "Y", "Z"]].to_numpy()
    
    source_o3d = o3d.geometry.PointCloud()
    target_o3d = o3d.geometry.PointCloud()
    three_o3d = o3d.geometry.PointCloud()
    four_o3d = o3d.geometry.PointCloud()
    
    source_o3d.points = o3d.utility.Vector3dVector(source)
    target_o3d.points = o3d.utility.Vector3dVector(target)
    three_o3d.points = o3d.utility.Vector3dVector(three)
    four_o3d.points = o3d.utility.Vector3dVector(four)
    
    source_temp = copy.deepcopy(source_o3d)
    target_temp = copy.deepcopy(target_o3d)
    three_temp = copy.deepcopy(three_o3d)
    four_temp = copy.deepcopy(four_o3d)
    
    source_temp.paint_uniform_color(color1)
    target_temp.paint_uniform_color(color2)
    three_temp.paint_uniform_color(color3)
    four_temp.paint_uniform_color(color4)
    
    o3d.visualization.draw_geometries([source_temp, target_temp, three_temp, four_temp])
    # custom_draw_geometry_with_rotations(source_temp, target_temp, three_temp)
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
def draw_outlier_result(source, target, transformation=None):
    source = source[["X", "Y", "Z"]].to_numpy()
    target = target[["X", "Y", "Z"]].to_numpy()
    source_o3d = o3d.geometry.PointCloud()
    target_o3d = o3d.geometry.PointCloud()
    
    source_o3d.points = o3d.utility.Vector3dVector(source)
    target_o3d.points = o3d.utility.Vector3dVector(target)
    
    source_temp = copy.deepcopy(source_o3d)
    target_temp = copy.deepcopy(target_o3d)
    
    source_temp.paint_uniform_color([192/255, 192/255, 192/255])
    target_temp.paint_uniform_color([1, 0, 0])
    if transformation is not None:
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
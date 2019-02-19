#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 16:17:50 2018

@author: karol
"""

import open3d as o3d
from os.path import join
import numpy as np

path = r'/media/karol/SSD/Data/omsws/SEQ_1.B 011/2/results'
pcd = o3d.read_point_cloud(join(path, 'flatten.pcd'))
print(np.asarray(pcd.points).shape)
print(np.asarray(pcd.colors).shape)
# comment this to see the point cloud in rainbow
#pcd.paint_uniform_color([1, 0.706, 0])
mesh_frame = o3d.create_mesh_coordinate_frame(size = 600, origin = [0, 0, 0])
# find normals so it's displayed nicer
#o3d.estimate_normals(pcd, search_param = o3d.KDTreeSearchParamHybrid(
#                                                    radius = 5, max_nn = 30))

o3d.draw_geometries([pcd, mesh_frame])
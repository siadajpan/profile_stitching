#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 15:24:51 2018

@author: karol
"""

import numpy as np
from scipy.interpolate import griddata
import open3d as o3d
import stitch_utils as su
import time
import math
from os.path import join
import json
from scipy.linalg import inv
import oms_math.points_utils as pu
from oms_math.geometric import pol2cart_pc, cart2pol_pc, transform



mesh_frame = o3d.create_mesh_coordinate_frame(size = 200, origin = [0, 0, 0])
r = 584
dir_path = r'/media/karol/SSD/Data/omsws/Long UT Section (9.D)/1'
pcd = o3d.read_point_cloud(join(dir_path, 'flatten_dsmpl.pcd'))
points = np.asarray(pcd.points)

json_path = join(dir_path, 'stitched_data.json')
with open(json_path, mode='r', encoding='utf-8') as feedsjson:
    [stitch_data, postprocessing] = json.load(feedsjson)

paths = stitch_data['paths']
points_numbers = stitch_data['points_no']
transformations = stitch_data['transformations']
pprc_transformation = postprocessing['postprocessing_transformation']

if points.shape[0] != sum(points_numbers):
    print("""amount of points in point cloud is different than sum of point sources
          from tdms file. Please check if correct tdms file is loaded""")
    raise Exception('Incorrect point cloud sources file')

[x, y, z] = points.T

x_min = min(x); x_max = max(x)
y_min = min(y); y_max = max(y)

# create ranges for 1mm grid, we will then extend this grid 10 times
grid_step = 1
x_range = (x_max - x_min) // grid_step
y_range = (y_max - y_min) // grid_step
x_max = x_min + x_range
y_max = y_min + y_range

# create grid
mgrid_x, mgrid_y = np.mgrid[x_min : x_max : grid_step, y_min : y_max : grid_step]
grid_x = np.array(list(set(mgrid_x[:, 0])))
grid_y = np.array(list(set(mgrid_y[0, :])))
grid_x.sort(); grid_y.sort()


# create array that will save which omsws each point comes from
sources = []
for i, points_no in enumerate(points_numbers):
    sources_add = np.ones((points_no), dtype=int) * i
    sources.extend(list(sources_add))
sources = np.array(sources)
scans_i = sorted(list(set(sources)))

# create mid_data array, which contains sine wave of the middle of each scan points
# after stitching. This will help us estimate clear boundaries of each scan
# that after stitching were blurry on overlapping regions
grid_data = (grid_x, grid_y, mgrid_x, mgrid_y)
grid_sources = su.create_grid_sources(points, points_numbers, scans_i, grid_data)

mgrid_z = griddata((x,y), (z), (mgrid_x, mgrid_y), method='nearest')

# decrease grid step by m, thus increasing number of points by m^2
m = 10
grid_step /= m
# find all z values of new points by linear interpolation
mgrid_z = pu.interlieve_2d_arr_lin(mgrid_z, m)
# find all omsws sources by nearest interpolation
grid_sources = pu.interlieve_2d_arr(grid_sources, m)
# create new grids
mgrid_x, mgrid_y = np.mgrid[x_min : x_max : grid_step, y_min : y_max : grid_step]
grid_x = np.array(list(set(mgrid_x[:, 0])))
grid_y = np.array(list(set(mgrid_y[0, :])))
grid_x.sort(); grid_y.sort()

# interpret the points on new grid coordinates
scan_sources = np.reshape(grid_sources, -1)
x = np.reshape(mgrid_x, -1)
y = np.reshape(mgrid_y, -1)
z = np.reshape(mgrid_z, -1)
points = np.array([x, y, z]).T

#pcd = o3d.PointCloud()
##pcd.points = o3d.Vector3dVector(np.array(np.hstack((points[:, :2], scan_sources.reshape(-1, 1)))))
#pcd.points = o3d.Vector3dVector(points)
#o3d.draw_geometries([pcd])


# now that we have new grid of points, we want the raw position of points and 
# camera images to assign them to each other

# wrap the points on cylinder
points = pol2cart_pc(points, r)

#pcd.points = o3d.Vector3dVector(np.array(points))
#pcd.transform(inv(pprc_transformation))
#o3d.draw_geometries([pcd])

# transform back from before postprocessing
points = transform(points, inv(pprc_transformation))
#points = np.asarray(pcd.points)
rotation = 0
scan_id = 1
inverse_x = True
cam_angle_of_view = 0.76
omsws_settings = (scan_id, inverse_x, rotation)

for i, path in enumerate(paths):
    print(i)
    points_scan = points[scan_sources == i]

    # all scans after first one have another transformation (stitching). 
    # we need to go back to raw positions, so inverting transformation
    if i > 0:
        points_scan = transform(points_scan, inv(transformations[i-1]))
        
    curr_time = time.time()
    image_loader = su.StitchImageLoader(path, omsws_settings, points_scan)
    print('loading omsws: ', time.time() - curr_time)
    
    if i > 2:
        image_loader.omsws.rotate_image_angles(math.pi)

    # load the images, and return colors for all the points
    curr_time = time.time()
    if i==0:
        cam_rot = np.array([0.05, -0.01, -0.07])
        cam_pos = np.array([30, -13, -2])
    else:
        cam_rot = np.array([0.05, -0.01, -0.07])
        cam_pos = np.array([30, -7, 4])
    
    points_colors = image_loader.load_images(cam_angle_of_view, cam_rot, cam_pos)
    print('loading images: ', time.time() - curr_time)

    if i > 0:
        points_scan = transform(points_scan, np.array(transformations[i-1]))
    
    points_scan = transform(points_scan, np.array(pprc_transformation))
    
    # points to flat surface
    points_flat = cart2pol_pc(points_scan, r)
    
#    if i == 0:
#        stitch_points = points_flat
#        stitch_colors = points_colors
#    else:
#        stitch_colors = np.vstack((stitch_colors, points_colors))
#        stitch_points = np.vstack((stitch_points, points_flat))

    su.save_as_image(points_flat, points_colors, (grid_x, grid_y), 
                                     join(dir_path, 'out' + str(i) + '.png'))

#pcd = o3d.PointCloud()
#pcd.points = o3d.Vector3dVector(stitch_points)
#pcd.colors = o3d.Vector3dVector(stitch_colors)
#o3d.draw_geometries([pcd])

    
    
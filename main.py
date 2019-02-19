#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 11:41:17 2018

@author: karol
"""

from stitch import stitch_pc, postprocess, save_as_grid
import numpy as np
from os.path import join


files_dir = r'/media/karol/SSD/Data/omsws/SEQ_1.B 011/2'
output_dir = join(files_dir, 'results')
cam_angle_of_view = 0.3036873 # 0.3036873, 0.4031711, 0.5951573

# cam rotation: first is clockwise rotation, second is up,third is left
cam_rot = np.array([0, 0, 0.])

# cam pos: first is radial difference from the laser closer(+)/further away(-), 
# second is left(+)/right(-), third is up(-)/down(+)
cam_pos = np.array([-170., 0., 0.])
image_flip_h = True
image_flip_v = True 

# this is constant, we are always taking the first one for each omsws file
scan_id = 1
# rotation of scan because of imperfect laser mounting
rotation = 0
# inverse the x axis, so the new scans are comming on the positive part
# of x axis
inverse_x = True
r = 584


# stitch scans to create point cloud of the whole pipe
pc_path, json_path = stitch_pc(files_dir, output_dir, r, with_images=True, downsample=1,
                               cam_settings=(cam_angle_of_view, cam_rot, cam_pos, image_flip_h, image_flip_v),
                               omsws_settings=(scan_id, inverse_x, rotation))


# postprocess scans so that pipe is alligned with x axis
pc_path = postprocess(pc_path, json_path, r)


# stitch the pictures together to create image of the inside of pipe
resolution = 5
save_as_grid(pc_path, json_path, r, resolution,
             cam_settings=(cam_angle_of_view, cam_rot, cam_pos, image_flip_h, image_flip_v),
             omsws_settings=(scan_id, inverse_x, rotation))

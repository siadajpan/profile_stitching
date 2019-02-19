#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 17:36:15 2018

@author: karol
"""

from tdms.oms_tdms import OMS_Tdms
import open3d as o3d
import numpy as np


path = r'/media/karol/SSD/Data/omsws/Long UT Section (9.D)/results/stitched_downsampled_colored.tdms'
tdms = OMS_Tdms()
tdms.open_tdms_file(path)

x = tdms.tdms_file.channel_data('Laser Data', 'X')
y = tdms.tdms_file.channel_data('Laser Data', 'Y')
z = tdms.tdms_file.channel_data('Laser Data', 'Z')


r = tdms.tdms_file.channel_data('Colour Data', 'R')
g = tdms.tdms_file.channel_data('Colour Data', 'G')
b = tdms.tdms_file.channel_data('Colour Data', 'B')

p = o3d.PointCloud()
p.points = o3d.Vector3dVector(np.array([x, y, z]).T)
p.colors = o3d.Vector3dVector(np.array([r, g, b]).T/255)

o3d.draw_geometries([p])
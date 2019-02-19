# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 13:28:26 2018

@author: Dell
"""
import numpy as np


def wrap_angles(angles, angle_units=None, angle_range=None):

    if angle_units == 'DEG2RAD' or angle_units == 'DEG2DEG':
        angles = np.multiply(angles, np.pi/180)

    angles = np.mod(np.add(angles, 2 * np.pi), 2 * np.pi)

    if angle_range == '-180-180':
        angles = np.subtract(angles, np.pi)
            
    if angle_units == 'RAD2DEG' or angle_units == 'DEG2DEG':
        angles = np.multiply(angles, 180/np.pi)

    return angles

def sort_scans(angles, scans):    
    # sort everything by angles
    angles_s, scans_s = zip(*sorted(zip(angles, scans), key=lambda i: i[0]))

    return list(angles_s), list(scans_s) 
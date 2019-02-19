#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 17:06:36 2018

@author: karol
"""
import numpy as np
from scipy.interpolate import interp2d, griddata
try:
    from fitting import fit_sine_wave
except ImportError:
    from .fitting import fit_sine_wave

def find_nearest_element(array, value):
    array = np.asarray(array)
    ind = (np.abs(array - value)).argmin()
    return ind, array[ind]

def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]

def interlieve_1d_arr(arr, multiplier):
    
    output = np.empty((arr.size * multiplier), dtype=arr.dtype)
     
    if multiplier == 1:
        return arr
    
    for i in range(multiplier):
        output[i::multiplier] = arr
        
    return output[:-multiplier + 1]
    
def interlieve_2d_arr(arr, multiplier):
    # inputs: 2d numpy array, int multiplier
    # output: 2d numpy array
    # function outputs array that dimensions are multiplied by multiplier
    # and all elements are copied from the original array

    if multiplier == 1:
        return arr
    
    new_shape = tuple((np.array(arr.shape)) * multiplier)
    output = np.empty(new_shape, dtype=arr.dtype)
        
    for i in range(multiplier):
        for j in range(multiplier):
            output[i::multiplier, j::multiplier] = arr
        
    return output[:-multiplier + 1, :-multiplier + 1]
 
    
def interlieve_2d_arr_lin(arr, multiplier):
    
    if multiplier == 1:
        return arr
    
    x = list(range(arr.shape[1]))
    y = list(range(arr.shape[0]))
    
    f = interp2d(x, y, arr)
    
    x_n = np.array(list(range((arr.shape[1]-1) * multiplier))) / multiplier
    y_n = np.array(list(range((arr.shape[0]-1) * multiplier))) / multiplier
    x_n = np.concatenate((x_n, [x[-1]]))
    y_n = np.concatenate((y_n, [y[-1]]))
    
    z_new = f(x_n, y_n)
    
    return z_new


def calc_ranges(points, grid_step):
    x_min = min(points[:, 0]); x_max = max(points[:, 0])
    y_min = min(points[:, 1]); y_max = max(points[:, 1])
    # create ranges for curr grid_step
    x_steps = (x_max - x_min) // grid_step + 1
    y_steps = (y_max - y_min) // grid_step + 1
    m = 0.01
    x_max = x_min + x_steps * grid_step + m
    y_max = y_min + y_steps * grid_step + m
    
    return (x_min, x_max, y_min, y_max)

    
def calc_grid(points, ranges, grid_step):
    [x, y, z] = points.T
    (x_min, x_max, y_min, y_max) = ranges
    
    # create grid
    (grid_x, grid_y, mgrid_x, mgrid_y) = make_grid(ranges, grid_step)

    
    # create mid_data array, which contains sine wave of the middle of each scan points
    # after stitching. This will help us estimate clear boundaries of each scan
    # that after stitching were blurry on overlapping regions
    mgrid_z = griddata((x,y), (z), (mgrid_x, mgrid_y), method='nearest')
    
    return (grid_x, grid_y, mgrid_x, mgrid_y), mgrid_z


def make_grid(ranges, grid_step):
    
    (x_min, x_max, y_min, y_max) = ranges
    mgrid_x, mgrid_y = np.mgrid[x_min : x_max : grid_step, y_min : y_max : grid_step]
    grid_x = np.array(list(set(mgrid_x[:, 0])))
    grid_y = np.array(list(set(mgrid_y[0, :])))
    grid_x.sort(); grid_y.sort()

    return (grid_x, grid_y, mgrid_x, mgrid_y)


def calc_points_on_grid(points, points_numbers, resolution):
        
    print('creating 1mm grid')
    # create 1 mm grid from the points that we read from pcd
    
    grid_step = 1
    ranges = calc_ranges(points, grid_step)
    grid_data, mgrid_z = calc_grid(points, ranges, grid_step)
    (grid_x, grid_y, mgrid_x, mgrid_y) = grid_data
    # create array that will save which omsws each point comes from
    sources = []
    for i, points_no in enumerate(points_numbers):
        sources_add = np.ones((points_no), dtype=int) * i
        sources.extend(list(sources_add))
    sources = np.array(sources)
    scans_i = sorted(list(set(sources)))
    
    # create sources array that uses the same 1mm grid of points
    scan_sources_h = create_grid_sources(points+[0, 10, 0], points_numbers, 
                                         scans_i, grid_data)
    scan_sources_l = create_grid_sources(points+[0, -10, 0], points_numbers, 
                                         scans_i, grid_data)
    
    
    print('creating dense grid')
    # divide grid step by resolution, thus increasing number of points by resolution^2
    grid_step /= resolution
    # find all z values of new points by linear interpolation
    mgrid_z = interlieve_2d_arr_lin(mgrid_z, resolution)
    # find all omsws sources by nearest point
    scan_sources_h = interlieve_2d_arr(scan_sources_h, resolution)
    scan_sources_l = interlieve_2d_arr(scan_sources_l, resolution)
    
    # create new grids
    (grid_x, grid_y, mgrid_x, mgrid_y) = make_grid(ranges, grid_step)
    
    print ('reshaping')
    # interpret the points on new grid coordinates
    scan_sources_h = np.reshape(scan_sources_h, -1)
    scan_sources_l = np.reshape(scan_sources_l, -1)
    x = np.reshape(mgrid_x, -1)
    y = np.reshape(mgrid_y, -1)
    z = np.reshape(mgrid_z, -1)
    
    points = np.array([x, y, z])
    points = points.T
    grid_data = (len(grid_y), len(grid_x), grid_step)
    
    return points, (scan_sources_h, scan_sources_l), grid_data
                    


def create_grid_sources(points, points_numbers, scans_i, grid_data):
    # function ceates clear boundaries between stitched scans, by fitting
    # sine wave on each scan points then estimates source of each point on grid
    # using nearest sine wave
    (grid_x, grid_y, mgrid_x, mgrid_y) = grid_data
    last_scan_ind = 0
    mid_data = np.array([])
    for (points_no, scan_no) in zip(points_numbers, scans_i):
        # get all the points from current scan
        curr_scan = points[last_scan_ind : last_scan_ind + points_no]
        last_scan_ind += points_no
        # find points closest to the grid x
        data_y = griddata(curr_scan[:, 0], curr_scan[:, 1], grid_x, method = 'nearest')
        # fit the sine wave to those points
        sine = fit_sine_wave(grid_x, data_y)
        # save those points with index from current scan 
        curr_sine = np.array([grid_x, sine, np.ones_like(grid_x) * scan_no]).T
        if len(mid_data) == 0:
            mid_data = curr_sine
        else:
            mid_data = np.vstack((mid_data, curr_sine))
            
    # for each point find the closest sine wave and set it's index to that sine
    return griddata(mid_data[:, :2], mid_data[:, 2], (mgrid_x, mgrid_y), 
                                                            method='nearest')
 
#@xl_func("numpy_row v1, numpy_row v2: float")
def vec_ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)

#
#if __name__ == '__main__':
#    a = np.array([1, 2, 3])
#    print(interlieve_1d_arr(a, 4))
#    b = np.array([[2, 3, 2], a])
#    print(interlieve_2d_arr(b, 4))
#    print(interlieve_2d_arr_lin(b, 4))
#    
#    
    
    
#    >>> from scipy import interpolate
#>>> x = np.arange(-5.01, 5.01, 0.25)
#>>> y = np.arange(-5.01, 5.01, 0.25)
#>>> xx, yy = np.meshgrid(x, y)
#>>> z = np.sin(xx**2+yy**2)
#>>> f = interpolate.interp2d(x, y, z, kind='cubic')
#
#Now use the obtained interpolation function and plot the result:
#>>>
#
#>>> xnew = np.arange(-5.01, 5.01, 1e-2)
#>>> ynew = np.arange(-5.01, 5.01, 1e-2)
#>>> znew = f(xnew, ynew)
#>>> plt.plot(x, z[0, :], 'ro-', xnew, znew[0, :], 'b-')
#>>> plt.show()


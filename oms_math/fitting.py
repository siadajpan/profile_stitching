#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 12:15:37 2019

@author: karol
"""

import numpy as np
from scipy import optimize
from scipy.optimize import leastsq
import math


def calc_R(x,y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)


def f(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()


def leastsq_circle(x,y):
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x,y))
    xc, yc = center
    Ri       = calc_R(x, y, *center)
    R        = Ri.mean()
    residu   = np.sum((Ri - R)**2)
    return xc, yc, R, residu


def polar2cart(points2d, angles, id_offset):
    # convert scans from cylinder to cartesian
    x = np.add(points2d[1], id_offset) * np.cos(angles)
    y = np.add(points2d[1], id_offset) * np.sin(angles)
    z = points2d[0]
    return x, y, z


def find_centre_of_data(points3d, mask=None):
    # find centre of circle shaped 3d points
    # input tuple (x, y, z) with arrays x, y, z arrays
    if mask is not None:
        masked_2d = np.array(list(zip(*points3d)))[mask]
        points3d = list(zip(*masked_2d))
#    print(points3d[1])
    x0, y0, r, _ = leastsq_circle(points3d[0], points3d[1])
    z0 = np.mean(points3d[2])
    
    return (x0, y0, z0), r



def fit_sine_wave(x, y):
    # sort arrays by x
    t, data = np.array(list(zip(*sorted(zip(x, y), key=lambda x: x[0]))))
    
    guess_freq = (2 * math.pi)/(t[-1] - t[0])
    guess_mean = np.mean(data)
    guess_std = 3 * np.std(data) / (2**0.5) / (2**0.5)
    guess_phase = 0
    guess_amp = guess_std
    
    # Define the function to optimize, in this case, we want to minimize the difference
    # between the actual data and our "guessed" parameters
    optimize_func = lambda x: x[0] * np.sin(x[1] * t + x[2]) + x[3] - data
    est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]
    
    # recreate the fitted curve using the optimized parameters
    data_fit = est_amp*np.sin(est_freq * t + est_phase) + est_mean
    
    # recreate the fitted curve using the optimized parameters
    data_fit = est_amp * np.sin(est_freq * t + est_phase) + est_mean
    
    return data_fit

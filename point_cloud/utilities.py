# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 07:51:12 2018

@author: Dell
"""

import numpy as np

def find_nearest_value(array, value):
    # search for the value nearset "value input"
    # inputs: np array
    # float value
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


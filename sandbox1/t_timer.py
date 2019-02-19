#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 12:19:21 2019

@author: karol
"""

print('importing from t_timer')
try:
    from . import timer
except ImportError:
    import timer
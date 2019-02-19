#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 09:12:17 2019

@author: karol
"""

import cv2
from os.path import join
from panorama import Stitcher
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image

def apply_transform(image, M):
    height, width = image.shape[:2]
    if M[1, 2] > 0:
        height = int(height + M[1, 2])
    if M[0, 2] > 0:
        width = int(width + M[0, 2])
     
    output_im = cv2.warpAffine(image, M, (width, height))
    
    return output_im


dir_path = r'/home/karol/anaconda3/envs/stitching/projects/stitching'

target = cv2.imread(join(dir_path, 'target.bmp'))
source = cv2.imread(join(dir_path, 'source.bmp'))
target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

target = cv2.GaussianBlur(target, (3,3), 0)
plt.imshow(target)
plt.show()
plt.imshow(source)
plt.show()

M = cv2.estimateRigidTransform(target, source, fullAffine=False)
        
print(M)

#
#target_b = cv2.medianBlur(target, 3)
#source_b = cv2.medianBlur(source, 3)
#
#target[target[:, :, 0] == 0] = target_b[target[:, :, 0] == 0]
#source[source[:, :, 0] == 0] = source_b[source[:, :, 0] == 0]
#
#
#print(source.shape)
##target_s = target[:, int(target.shape[1]/2):]
##source_s = source[:, :int(source.shape[1]/2)]
#(im_height_a, im_width_a) = source.shape[:2]
#(im_height_b, im_width_b) = target.shape[:2]
#
#im_height_c = min(im_height_a, im_height_b)
#im_width_c = min(im_width_a, im_width_b)
#
#stitch_width = int(im_width_c/2) - 1
#target_s = target[:im_height_c, im_width_c-stitch_width:im_width_c]
#source_s = source[:im_height_c, 0:stitch_width]
##source_s = source[:im_height_c, :im_width_c]
##target_s = target[:im_height_c, :im_width_c]
#print(source_s.shape)
#plt.imshow(cv2.cvtColor(target_s, cv2.COLOR_RGB2BGR))
#plt.axis('off')
#plt.show()
#
#plt.imshow(cv2.cvtColor(source_s, cv2.COLOR_RGB2BGR))
#plt.axis('off')
#plt.show()
#
#stitcher = Stitcher()
#
#M = cv2.estimateRigidTransform(source_s, target_s, fullAffine=False)
#print(M)
#
#M[0, 2] += im_width_c-stitch_width
#
#print('before', source.shape)
#source = apply_transform(source, M)
#print('after', source.shape)
#plt.imshow(source)
##target = stitcher.resize_image(target, source.shape)
##
##M = cv2.estimateRigidTransform(source_s, target_s, fullAffine=False)
##print(M)
##print('before', source.shape)
##source = apply_transform(source, M)
##print('after', source.shape)
##plt.imshow(source)
#output_im = stitcher.stitch_images(target, source)
#
##source = cv2.warpAffine(source, M, (source.shape[1] + im_width_c, source.shape[0]))
#
#result = Image.fromarray(cv2.cvtColor(output_im, cv2.COLOR_BGR2RGB), mode='RGB')
#result.save(join(dir_path, 'output_im.png'))
#
##
##H, vis = stitcher.stitch((target_s, source_s), showMatches=True)
##plt.imshow(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
##plt.axis('off')
##plt.show()
##print(H)
##dm = cv2.decomposeHomographyMat(H, np.diag((1, 1, 1)))
##rot = dm[1][2][:2, :2]
##trans = dm[2][2][:2]
##print(trans)
##
##result = stitch_images(source, target, translation)
##
##
##print(rot)
##alpha = np.arccos(rot[0, 0])
##print('a', alpha)
##trans = dm[2][2][:2]
##print('trans', trans)
##rows, cols = target_s.shape[:2]
##half_rot = cv2.getRotationMatrix2D((cols/2, rows/2), -alpha*180/math.pi, 1)
##target_s = cv2.warpAffine(target_s, half_rot, (target_s.shape[1], target_s.shape[0]))
##rows, cols = source_s.shape[:2]
##half_rot = cv2.getRotationMatrix2D((cols/2, rows/2), -alpha*180/math.pi, 1)
##source_s = cv2.warpAffine(source_s, half_rot, (source_s.shape[1], source_s.shape[0]))
##
##rows, cols = source.shape[:2]
##half_rot = cv2.getRotationMatrix2D((cols/2, rows/2), -alpha*180/math.pi, 1)
##source = cv2.warpAffine(source, half_rot, (source.shape[1], source.shape[0]))
##
##result = stitch_images(source, target, translation)
##
##cv2.imwrite(join(dir_path, 'result38a.png'), source)
##
##plt.imshow(cv2.cvtColor(target_s, cv2.COLOR_RGB2BGR))
##plt.axis('off')
##plt.show()
##
##plt.imshow(cv2.cvtColor(source_s, cv2.COLOR_RGB2BGR))
##plt.axis('off')
##plt.show()
##
##H, vis = stitcher.stitch((target_s, source_s), ratio=1, reprojThresh=4, showMatches=True)
##print(H)
##plt.imshow(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
##plt.axis('off')
##plt.show()
##dm = cv2.decomposeHomographyMat(H, np.diag((1, 1, 1)))
##rot = dm[1][2][:2, :2]
##
#print(rot)


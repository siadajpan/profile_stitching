import numpy as np
import math
from numpy.linalg import inv
from numba import vectorize

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

    
def pol2cart_3d(x, y, a):
    z = x
    try:
        x = y * math.cos(a)
        y = y * math.sin(a)
    except TypeError:
        x = [yi * math.cos(ai) for (yi, ai) in zip(y, a)]
        y = [yi * math.sin(ai) for (yi, ai) in zip(y, a)]
    return (x, y, z)
    

def pol2cart_pc(pc, r):
    # pc is a np array of shape (n, 3) where n is number of points, and 
    # columns are x, y, z coord of points
    if len(pc) == 0:
        raise Exception('Empty point cloud')
        
    r1 = pc[:,2] + r
    a = pc[:,0] / r
    
    x = r1 * np.cos(a)
    y = r1 * np.sin(a)
    z = pc[:,1]
    
    return np.vstack((x, y, z)).T
    

def cart2pol_pc(pc, r, x_range='0 - 2*pi'):
    
    if x_range == '0 - 2*pi':
        x = (np.arctan2(pc[:,1], pc[:,0]) % (2*math.pi))*r
    else:
        x = np.arctan2(pc[:,1], pc[:,0]) * r
        
    y = pc[:,2]
    z = np.sqrt(np.power(pc[:,0], 2) + np.power(pc[:,1], 2)) - r

    return np.vstack((x, y, z)).T


#import pycuda.autoinit
#import pycuda.gpuarray as gpuarray
#import numpy as np
#import skcuda.linalg as linalg
#import skcuda.misc as misc
#
##@vectorize(['float32(float32, float32, float32, float32)'], target='cuda')
#def cart2pol_pc_cuda(pc, r):
#    linalg.init()
#    [pcx, pcy, pcz] = pc.T
#    pcx_cuda = gpuarray.to_gpu(pc[:, 0])
#    pcy_cuda = gpuarray.to_gpu(pc[:, 1])
#    pcz_cuda = gpuarray.to_gpu(pc[:, 2])
#    x = linalg.
#    x = (math.atan2(pcy, pcx) % (2*math.pi))*r
#    y = pcz
#    z = math.sqrt(pcx ** 2 + pcy ** 2) - r
#
#    return x


#
#r = np.float32(10)
#pc = np.array([[1, 2, 3], [2, 2, 3]], dtype=np.float32)
#pcx = list(pc[:, 0])

#print(cart2pol_pc_cuda(np.array(pc[:, 0]), np.array(pc[:, 1]), np.array(pc[:, 2]), r))


def rot_array_2d(rotation_ccw):
    c = math.cos(rotation_ccw)
    s = math.sin(rotation_ccw)
    return np.array([[c, s], [-s, c]])


def trans_rot_array_3d(rotation_ccw, translation, rot_sequence='xyz'):
    rot_x, rot_y, rot_z = rotation_ccw
    c = math.cos(rot_x)
    s = math.sin(rot_x)
    rx = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
    c = math.cos(rot_y)
    s = math.sin(rot_y)
    ry = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    c = math.cos(rot_z)
    s = math.sin(rot_z)    
    rz = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    
    rt = np.identity(3)
    if rot_sequence == 'xyz':
        rt = np.dot(np.dot(rz, ry), rx)
        
    elif rot_sequence == 'zyz':
        rt = np.dot(np.dot(inv(rz), ry), rz)
    
    r = np.identity(4)
    r[:3, :3] = rt
    r[:3, 3] = translation
    
    return r


def transform(arr, transfromation):
    arr1 = np.hstack((arr, np.ones((arr.shape[0], 1))))
    return np.dot(arr1, transfromation.T)[:, :3]
    
    
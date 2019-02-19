#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:43:30 2018

@author: karol
"""
import oms_math.geometric as oms_geo
import numpy as np
from numpy.linalg import inv
import math
import open3d as o3d
import copy
from oms_math.points_utils import find_nearest_element
from oms_math.fitting import find_centre_of_data
from tdms.oms_tdms import OMSWS1
from PIL import Image
from scipy.interpolate import griddata
from os.path import join, split
import json
from image_stitch.panorama import Stitcher
import cv2


def read_part_scans_3d(omsws, ang_range):
    (angle_start, angle_end) = ang_range
    angles = omsws.scan_angles.copy()
    # this will create list of tuples (index, angle)
    angles_ind = list(zip(range(len(angles)), angles))
    # sort by angles
    angles_ind.sort(key=lambda x: x[1])
    # we are getting angles sorted and indexes of those angles
    indexes, angles = zip(*angles_ind)
    
    ind_start, _ = find_nearest_element(angles, angle_start)
    ind_end, _ = find_nearest_element(angles, angle_end)
    
    index_list = np.array(indexes[ind_start : ind_end])

    # reading scans with indexes. Scans slices starting from 1
    scans = omsws.read_scans_part(index_list+1)
    angles = list(np.array(angles)[index_list])
    
    return scans, angles


def scans_to_pc(scans, angles):
    # return all the scans as a huge list of points (x, y, z)
    scans3d = []
    
    for (scan, angle) in zip(scans, angles):
        scans3d.extend([*zip(*oms_geo.pol2cart(scan[1, :], angle), scan[0, :])])
    scans3d = np.array(scans3d)
    scans3d = sorted(scans3d, key=lambda x: math.atan2(x[1], x[0]))
    return scans3d
    

def read_scans_pc(omsws, part='full'):
    # read scans and return them as a point cloud
    if part == 'full':
        start = 0
        end = 2*math.pi
        return scans_to_pc(*read_part_scans_3d(omsws, (start, end)))

    elif part == 'lower':
        start = math.pi - 0.5
        end = math.pi + 0.5
        return scans_to_pc(*read_part_scans_3d(omsws, (start, end)))
    
    elif part == 'half':
        start = math.pi / 2
        end = math.pi * 3/2
        return scans_to_pc(*read_part_scans_3d(omsws, (start, end)))
    
    elif part == 'upper':
        end = 0.5
        start = math.pi*2 - 0.5
        scans_start = read_part_scans_3d(omsws, (0, end))
        scans_end, _ = read_part_scans_3d(omsws, (start, 2*math.pi))
        return np.concatenate((scans_to_pc(*scans_start), 
                               scans_to_pc(*scans_end)))
    
    elif part == 'wheels':
        right_start = 3/4 * math.pi - 0.25
        right_end = 3/4 * math.pi + 0.25
        left_start = 5/4 * math.pi - 0.25
        left_end = 5/4 * math.pi + 0.25
        scans_right = read_part_scans_3d(omsws, (right_start, right_end))
        scans_left = read_part_scans_3d(omsws, (left_start, left_end))
        return np.concatenate((scans_to_pc(*scans_right), 
                               scans_to_pc(*scans_left)))
        
    
def draw_registration_result(source, target, transformation):
    # display results having two point clouds and transformation for source
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    mesh_frame = o3d.create_mesh_coordinate_frame(size = 200, origin = [0, 0, 0])
    o3d.draw_geometries([source_temp, target_temp, mesh_frame])
    

def load_point_cloud(omsws, part='full'):
    # create point cloud having point cloud of scan points
    scans = read_scans_pc(omsws, part=part)
#    Pc = OMS_PointCloud()
    Pc = o3d.PointCloud()
    Pc.points = o3d.Vector3dVector(scans)

    return Pc


def o3d_estimate_normals(source):
    # downsample and estimate normals
    source = o3d.voxel_down_sample(source, voxel_size = 2.5)
    o3d.estimate_normals(source, search_param = o3d.KDTreeSearchParamHybrid(
            radius = 5, max_nn = 30))

    return source


def estimate_xy_translation(source, target, rotation=None):
    # get source point clouds of source and target
    # and return vector (x, y) that will be used as initial translation

    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    
    # rotate the source and target if there was a straightening rotation
    if rotation is not None:
        source_copy.transform(inv(rotation))
        target_copy.transform(inv(rotation))

    source_points = np.asarray(source_copy.points)
    target_points = np.asarray(target_copy.points)
    
    # find the beginning and end of data in z axis
    z_range = [min(source_points[:, 2]), max(source_points[:, 2])]
    (z_start, z_end) = np.linspace(*z_range, 5)[[1, -2]]
    
    # filter all the points that are bigger than 80% of the range
    t_points_calc = np.array(list(filter(lambda x: x[2] > z_end, target_points)))
    left_centroid = t_points_calc.mean(axis=0)
    target_copy.points = o3d.Vector3dVector(np.array([left_centroid]))
    
    # filter all the points that are smaller than 20% of the range
    s_points_calc = np.array(list(filter(lambda x: x[2] < z_start, source_points)))
    right_centroid = s_points_calc.mean(axis=0)
    source_copy.points = o3d.Vector3dVector(np.array([right_centroid]))
    
    # rotate the centroids to the straighten scan, if the rotation was present
    if rotation is not None:
        source_copy.transform(rotation)
        target_copy.transform(rotation)

    # create the vector
    vector = target_copy.points[0] - source_copy.points[0]
    return vector

 
def find_rotation_zyz(tx, ty, tz):
    # find rotation matrix based on the vector x, y, z to rotate to vector
    # along z axis
    rot_z = math.atan(ty/tx)
    rot_y = -math.pi/2 + math.atan(tz/math.sqrt(tx**2 + ty**2))
    # create rotation translation matrix with z-y-z rotation and translation 0
    rot_trans = oms_geo.trans_rot_array_3d((0, rot_y, rot_z),
                                                            (0,0,0), 'zyz')
    rotation = rot_trans[:3, :3]
    return rotation

 
def calc_centre_correction(pcd):
    # find the rotation and translation of the point cloud so it's starting
    # from pont (0, 0, 0) and is along z axis
    
    points = np.asarray(pcd.points)
    # sort all the points by z axis
    points = sorted(points, key=lambda x: x[2])
    length = len(points)
    
    circle_centres = []
    # slice the point cloud on 10 pieces along z axis, so they have same amount
    # of points
    for i in range (5, 15):
        p_slice = points[int(length/20) * i : int(length/20) * (i+1)]
        # find centre of each slice (elipse) and append to the list
        centre, radius = find_centre_of_data(list(zip(*p_slice)))
        circle_centres.append(centre)

    circle_centres = np.array(circle_centres)
    
    # Calculate the mean of the points, i.e. the 'center' of the cloud
    datamean = circle_centres.mean(axis=0)

    # Do an SVD on the mean-centered data.
    uu, dd, vv = np.linalg.svd(circle_centres - datamean)
    
    rotation = find_rotation_zyz(*vv[0])
    trans_rot = np.identity(4)
    trans_rot [:3, :3] = rotation
    trans_rot [:3, 3] = -circle_centres[0]

    return trans_rot
    

class StitchPart():
    def __init__(self, path, omsws_settings, cam_settings, r): 
        self.omsws_path = path
        self.r = r
        (scan_id, inverse_x, rotation) = omsws_settings
        
        # load and preprocess the scans (apply id offset and rotation) 
        self.omsws = OMSWS1(path, scan_id, inverse_x)
        self.omsws.preprocess_scans(rotation)
        
        # create point cloud from the lower part of the pipe (to speed up stitching)
        self.lower = load_point_cloud(self.omsws, part='lower')
        
        # create normals of the point cloud, to be able to use point-to-plane
        # icp method which is faster and has less error than point-to-point
        self.lower = o3d_estimate_normals(self.lower)
        
        # load full point cloud and copy it, so we have one for stitching, 
        # and second for saving to file
        self.full = load_point_cloud(self.omsws, part='full')
        self.full_copy = copy.deepcopy(self.full)
    
        # estimate normals only to the first one, this will also downsample it
        self.full = o3d_estimate_normals(self.full)
    
        self.distance_driven = self.omsws.distance_driven
        self.id_offset = self.omsws.id_offset
        
        self.load_camera_settings(cam_settings)
        
        
    def load_camera_settings(self, cam_settings):
        # load camera settings
        (cam_angle_of_view, cam_rot, cam_pos, image_flip_h, image_flip_v) = cam_settings
        self.cam_angle_of_view = cam_angle_of_view
        self.cam_rot = cam_rot
        self.cam_pos = cam_pos
        self.update_cam_offset()
        cam_other_side = True
        
        # preprocess images
        self.omsws.preprocess_images(cam_other_side, image_flip_h, image_flip_v)


    def update_cam_offset(self):
        # update camera offset, after it was calibrated
        self.cam_offset = self.omsws.id_offset + self.cam_pos[0]
        
        
    def update_cam_rot_matrix(self):
        # update camera rotation matrix, after it was calibrated
        self.cam_rot_matrix = oms_geo.trans_rot_array_3d(-self.cam_rot, 
                                                    [0, *-self.cam_pos[1:]])
        print('updating cam rot matrix', self.cam_rot_matrix)
    
    
    def find_nearest_points(self, scans_points, points_angles, real_cam_angles):
        # search for point in the point cloud, that is nearest to the angle
        nearest_points_angles = [find_nearest_element(points_angles, cam_angle)
                                        for cam_angle in real_cam_angles]

        i_nearest_points, nearest_points_angles = zip(*nearest_points_angles)
        nearest_points = scans_points[list(i_nearest_points)]

        return nearest_points
    
    
    def calculate_points_angles(self, scans_points):
        
        points_angles = np.arctan2(scans_points[:, 1], scans_points[:, 0])
        points_angles = (points_angles + 2*math.pi) % (2*math.pi)
        
        return points_angles
    
    
    def init_loading_images(self):
        # update rotation and position of the camera based on calibration
        self.update_cam_rot_matrix()
        
        # copy camera angles
        cam_angles = self.omsws.image_angles
        
        self.im_height, self.im_width = self.omsws.images[0].shape[:2]
        
        self.cam_head_rot = [oms_geo.trans_rot_array_3d((0, 0, -cam_angle), 
                                                   (0,0,0))[:3, :3] 
                                                for cam_angle in cam_angles]
        
        # calculate angles of all the points, make sure they are 0 - 2*pi
        self.points_angles = self.calculate_points_angles(self.points)
        
    
    def calculate_angle_ranges(self, cam_angle_i):
        cam_angle = self.omsws.image_angles[cam_angle_i:cam_angle_i+1]
        nearest_points = self.find_nearest_points(self.points, self.points_angles,
                                                  cam_angle)
        
        cam_positions = oms_geo.pol2cart(self.cam_offset, cam_angle)
        cam_positions = np.array(cam_positions).T
        cam_positions = np.hstack((cam_positions, np.zeros((cam_positions.shape[0], 1))))
        mid_image_vector = nearest_points - cam_positions
        mid_image_vector[:,2] = np.zeros((mid_image_vector.shape[0]))
        
        image_beg_rot = oms_geo.trans_rot_array_3d((0, 0, self.cam_angle_of_view/2), 
                                                   (0, 0, 0))
        image_end_rot = oms_geo.trans_rot_array_3d((0, 0, -self.cam_angle_of_view/2), 
                                                   (0, 0, 0))
        
        points_start = oms_geo.transform(mid_image_vector, image_beg_rot) + cam_positions
        angles_start = np.arctan2(points_start[:, 1], points_start[:, 0])
        angles_start = (angles_start + 2*math.pi) % (2*math.pi)
        
        points_end = oms_geo.transform(mid_image_vector, image_end_rot) + cam_positions
        angles_end = np.arctan2(points_end[:, 1], points_end[:, 0])
        angles_end = (angles_end + 2*math.pi) % (2*math.pi)
        
        return angles_start[0], angles_end[0]
    
    
    def load_image(self, cam_angle_i):
        # get image associated with cam_angle_i, get all points that correspond
        # to this image from self.points. Output both points and colors,
        # and info if the image is split (starts to the right, and ends to the 
        # left) on the output stitched image
                
        angle_start, angle_end = self.calculate_angle_ranges(cam_angle_i)
        
        split = angle_start > angle_end
        if split:
            points_image = self.points[np.logical_or(self.points_angles > angle_start,
                                               self.points_angles < angle_end)]
        else:
            points_image = self.points[np.logical_and(self.points_angles > angle_start,
                                               self.points_angles < angle_end)]
                
        if len(points_image != 0):
            im_dist = self.cam_offset + self.im_width /\
                                    (2 * math.tan(self.cam_angle_of_view / 2))
        
            points_cam = np.dot(points_image, self.cam_head_rot[cam_angle_i])
            points_cam[:, 0] -= self.cam_offset
            points_cam = oms_geo.transform(points_cam, self.cam_rot_matrix)

            d = np.linalg.norm(points_cam, axis=1)
            # angle on hight
            rxy = np.arcsin(points_cam[:, 2] / d)
            # angle on width
            rz = np.arctan2(points_cam[:, 1], points_cam[:, 0])
            
            # calculate the pixels that correspond to that angle
            y_im = np.array(im_dist * np.tan(rxy) + self.im_width / 2, dtype=int)
            x_im = np.array(im_dist * np.tan(rz) + self.im_height / 2, dtype=int)
            f = np.logical_and(np.logical_and(0 < x_im, x_im < self.im_height),
                               np.logical_and(0 < y_im, y_im < self.im_width))
            
            scan_colors = self.omsws.images[cam_angle_i][x_im[f], y_im[f], :]
            scan_colors = scan_colors.astype(np.uint8)
            
        return points_image[f], scan_colors, split
    
    def load_images(self):
        cam_rot_matrix = oms_geo.trans_rot_array_3d(-self.cam_rot, [0, 
                                                               *-self.cam_pos[1:]])
        cam_angles = self.omsws.image_angles

        if not isinstance(self, StitchImageLoader):
            scans_points = np.asarray(self.full_copy.points)
        else:
            scans_points = self.points
    
        # initialize point cloud color array
        scan_colors = np.zeros_like(np.asarray(scans_points))
        self.image_sources = np.zeros(scans_points.shape, dtype=np.uint16)
        
        im_height, im_width = self.omsws.images[0].shape[:2]
        # calculate distance of an image from the camera, so we can find pixels
        # for point cloud assuming pixels are millimiters
        im_dist = self.cam_offset + im_width /\
                    (2 * math.tan(self.cam_angle_of_view / 2))
                
        # create array of rotations back to position 0, where we are loading images
        real_cam_angles = np.array(cam_angles) + math.atan2(self.cam_pos[1], 
                                                          self.cam_offset)
        
        cam_head_rot = [oms_geo.trans_rot_array_3d((0, 0, -cam_angle), 
                                                   (0,0,0))[:3, :3] 
                                                for cam_angle in real_cam_angles]
        
        # calculate angles of all the points, make sure they are 0 - 2*pi
        points_angles = np.arctan2(scans_points[:, 1], scans_points[:, 0])
        points_angles = (points_angles + 2*math.pi) % (2*math.pi)

        indexes = list(range(len(cam_angles)))
        ind_cam_angles = sorted(zip(indexes, real_cam_angles), key=lambda x:x[1])
        
        # add first and last cam angle on the beginning and end of angles list
        ind_cam_angles.insert(0, (ind_cam_angles[-1][0], 
                                  ind_cam_angles[-1][1] - 2*math.pi))
        ind_cam_angles.append((ind_cam_angles[1][0], 
                                  ind_cam_angles[1][1] + 2*math.pi))


        (im_index, sorted_cam_angles) = zip(*ind_cam_angles)
        point_im_indexes = griddata(np.array(sorted_cam_angles), np.array(im_index), 
                                    points_angles, method='nearest')
        
        for image_i in range(len(cam_angles)):
            points_i, = np.where(point_im_indexes == image_i)
            points = scans_points[points_i]
            
            if len(points != 0):
                points = np.dot(points, cam_head_rot[image_i])
                points[:, 0] -= self.cam_offset
                points = oms_geo.transform(points, cam_rot_matrix)
#                points = np.dot(points, cam_rot_matrix[:3, :3])
#                points += cam_rot_matrix[:3, 3] 
                
                x = points[:, 0]; y = points[:, 1]; z = points[:, 2]
                d = np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))
                # angle on hight
                rxy = np.arcsin(z / d)
                # angle on width
                rz = np.arctan2(y, x)
                
                # calculate the pixels that correspond to that angle
                y_im = np.array(im_dist * np.tan(rxy) + im_width / 2, dtype=int)
                x_im = np.array(im_dist * np.tan(rz) + im_height / 2, dtype=int)
                ixy = np.column_stack((points_i, x_im, y_im))
                ixy = ixy[np.logical_and(np.logical_and(0 < x_im, x_im < im_height),
                                         np.logical_and(0 < y_im, y_im < im_width))]
                scan_colors[ixy[:, 0]] =\
                        self.omsws.images[image_i][ixy[:, 1], ixy[:, 2], :]
                    
        return scan_colors
    
    
class StitchImageLoader(StitchPart):
    def __init__(self, path, omsws_settings, cam_settings, r, points): 
        self.omsws_path = path
        self.r = r
        (scan_id, inverse_x, rotation) = omsws_settings
        
        # load and preprocess the scans (apply id offset and rotation) 
        self.omsws = OMSWS1(path, scan_id, inverse_x)
        self.omsws.preprocess_scans(rotation)
        
        # save points from an argument we will process it as numpy array
        self.points = points
        
        self.load_camera_settings(cam_settings)
        self.init_loading_images()
        

class StitchResult():
    def __init__(self):
        self.Pc = o3d.PointCloud()
        self.omsws_paths = []
        self.points_no = []
        self.transformations = []
            
    def get_indexes(self, scan_no):
        start_i = sum(self.points_no[:scan_no])
        end_i = start_i + self.points_no[scan_no] + 1
    
        return list(range(start_i, end_i))


def create_image(stitch_points, stitch_colors, grid_step, blur=False):
#    print('create_image, grid step', grid_step)
    min_x_grid = min(stitch_points[:, 0])
    min_y_grid = min(stitch_points[:, 1])
    points_int = np.asarray(stitch_points, dtype=int)
    m = 0.0001
#    print('create_image, min_x_grid', min_x_grid)
    points_int[:, 0] = ((stitch_points[:, 0] - min_x_grid) / grid_step + m).astype(np.int)
    points_int[:, 1] = ((stitch_points[:, 1] - min_y_grid) / grid_step + m).astype(np.int)
    max_x_grid = max(points_int[:, 0]) + 1
    max_y_grid = max(points_int[:, 1]) + 1
    
    output_im = np.zeros((max_y_grid, max_x_grid, 3), dtype=np.uint8)
    output_im[points_int[:, 1], points_int[:, 0]] = stitch_colors
    
    if blur:
        output_im = cv2.medianBlur(output_im, 3)
        print('made blur')

    return output_im


def stitch_images(im_paths):
    output_im = np.array(Image.open(im_paths[0]))
    for im_path in im_paths[1:]:
        image = np.array(Image.open(im_path))
        output_im += image

    result = Image.fromarray(output_im, mode='RGB')
    output_dir, _ = split(im_paths[0])
    result.save(join(output_dir, 'result.png'))


def read_json_file(json_path, expected_len):
     # read data from json file
    with open(json_path, mode='r', encoding='utf-8') as feedsjson:
        [stitch_data, postprocessing] = json.load(feedsjson)
    paths = stitch_data['paths']
    points_numbers = stitch_data['points_no']
    transformations = np.array(stitch_data['transformations'])
    pprc_transformation = np.array(postprocessing['postprocessing_transformation'])
    
    # check if correct json file is read
    if expected_len != sum(points_numbers):
        print("""amount of points in point cloud is different than sum of point sources
              from tdms file. Please check if correct tdms file is loaded""")
        raise Exception('Incorrect point cloud sources file')
        
    return (paths, points_numbers, transformations, pprc_transformation)



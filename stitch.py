import numpy as np
import stitch_utils as su
import os
import open3d as o3d
import copy
from os.path import join, split
import json
from scipy.linalg import inv
import oms_math.points_utils as pu
from oms_math.geometric import pol2cart_pc, cart2pol_pc, transform
from PIL import Image
from image_stitch.panorama import Stitcher

# init some variables that we use as default values for stitch_pc function
cam_angle_of_view = 0.9
cam_rot = np.array([0, 0, 0])
cam_pos = np.array([20, 0, 0])
image_flip_h = False
image_flip_v = False 

# this is constant, we are always taking the first one for each omsws file
scan_id = 1
# rotation of scan because of imperfect laser mounting
rotation = 0
# inverse the x axis, so the new scans are comming on the positive part
# of x axis
inverse_x = True

def stitch_pc(files_directory, output_dir, r, with_images=False, downsample=2.5,
           cam_settings=(cam_angle_of_view, cam_rot, cam_pos, image_flip_h, image_flip_v),
           omsws_settings=(scan_id, inverse_x, rotation)):
    
    (cam_angle_of_view, cam_rot, cam_pos, image_flip_h, image_flip_v) = cam_settings
    
    # icp setting - how close should the point be to be a correspondence
    threshold = 2.5
    
    file_paths = [join(files_directory, path) for path 
                  in os.listdir(files_directory) if path.endswith('.omsws')]
    file_paths.sort()

    target = su.StitchPart(file_paths[0], omsws_settings, cam_settings, r)    
    crawler_position = target.distance_driven * 1000
    
    # create transformation array which identity matrix for now
    transformation = np.identity(4)
        
    print('base ', split(file_paths[0])[1])
    
    # initialize straightening rotation, which will centrecorrect and rotate
    # the scans, so they are palallel to z axis
    rot_trans_straight = None
    
    # initialize the output point cloud that will save all the measured points
    stitched = su.StitchResult()

    if downsample > 0:
        target.full_copy = o3d.voxel_down_sample(target.full_copy, 
                                                 voxel_size=downsample)
    stitched.Pc = copy.deepcopy(target.full_copy)

    # save no of points and path of current file to the class
    stitched.points_no.append(len(target.full_copy.points))
    stitched.omsws_paths.append(target.omsws.path)
    
    if with_images:
        colors = target.load_images()/255
        stitched.Pc.colors = o3d.Vector3dVector(colors)    

    # break for loop if we have only one file path
    dont_stitch = len(file_paths) == 1
        
    for i, path in enumerate(file_paths[1:]):
        
        if dont_stitch:
            break
        
        print('stitching {}'.format(split(path)[1]))
        
        # initialize source point cloud
        source = su.StitchPart(path, omsws_settings, cam_settings, r)
#        o3d.draw_geometries([source.full_copy, target.lower])
        if downsample > 0:
            source.full_copy = o3d.voxel_down_sample(source.full_copy, 
                                                     voxel_size = downsample)
        if with_images:
            colors = source.load_images()/255           
            stitched.Pc.colors.extend(o3d.Vector3dVector(colors))

        # calculate the centroids of point clouds and find the differrence
        transition = su.estimate_xy_translation(source.lower, target.lower, 
                                                rot_trans_straight)
        
        # update x and y of translation so the centroids are in the same place
        trans_init = np.identity(4)
        trans_init[:2, 3] = transition[:2]
        
        # update z from the cralwer position distance saved in omsws file
        trans_init[2, 3] = source.omsws.distance_driven * 1000 - crawler_position
        print('crawler position: %.2f' % crawler_position)
        crawler_position = source.omsws.distance_driven * 1000
        
        # stitch the point clouds of the lower part using icp
        reg_p2p = o3d.registration_icp(source.lower, target.lower, threshold, 
                        trans_init, o3d.TransformationEstimationPointToPlane())

        # stitch the full point clouds using the registration matrix from
        # previous stitch
        reg_p2p = o3d.registration_icp(source.full, target.full, threshold,
                                   reg_p2p.transformation,
                                   o3d.TransformationEstimationPointToPlane())
        
        print('fitness: {}'.format(reg_p2p.fitness))
        trans_init = reg_p2p.transformation
        
        # multiply transformation matrix of current scan that best fits the
        # target scan that is placed in (0,0) point with the global
        # transformation of the target scan (where is it placed in real live)
        transformation = np.dot(transformation, reg_p2p.transformation)
        
        # transform our copy with this transformation and that will place 
        # the source in correct position
        source.full_copy.transform(transformation)
        
        # add all the points of the scan to the stitched scan
        stitched.Pc.points.extend(source.full_copy.points)
        stitched.points_no.append(len(source.full_copy.points))
        
        # add omsws obj to the list in the stitched
        stitched.omsws_paths.append(source.omsws.path)
        
        stitched.transformations.append(transformation)
        
        # update target and target full
        target = source
    
    # create output folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = join(output_dir, "stitched.pcd")
    # save point cloud, downsample and display
    o3d.write_point_cloud(output_path, stitched.Pc)
    
    stitched.Pc = su.o3d_estimate_normals(stitched.Pc)
    o3d.write_point_cloud(join(files_directory, "stitched_downsampled_colored.pcd"), stitched.Pc)
##    stitched.Pc.paint_uniform_color([1, 0.706, 0])
#    mesh_frame = o3d.create_mesh_coordinate_frame(size = 200, origin = [0, 0, 0])
#    o3d.draw_geometries([stitched.Pc, mesh_frame])
    
    # save transformation of the stitched point clouds to json file
    transformations = [list([list(el) for el in t]) for t in stitched.transformations]
    with open(join(output_dir, 'stitched_data.json'), 'w') as outfile:
        data = {'paths': stitched.omsws_paths, 
                'points_no': stitched.points_no, 
                'transformations': transformations}
        json.dump([data], outfile)

    return (output_path, join(output_dir, 'stitched_data.json'))


def postprocess(pc_path, json_path, r):
    
    # find rotation of the point cloud, so it's alligned with z axis
    pcd = o3d.read_point_cloud(pc_path)
    
    # find transformation of the point cloud, that will allign it with x axis
    pcd1  = copy.deepcopy(pcd)
    rot_trans_straight1 = su.calc_centre_correction(pcd1)
    pcd1.transform(rot_trans_straight1)
    rot_trans_straight2 = su.calc_centre_correction(pcd1)
    pcd1.transform(rot_trans_straight2)
    
    # do this twice and combine results
    rot_trans_straight = np.dot(rot_trans_straight2, rot_trans_straight1)
    
    # make the points start from z=0
    points = np.asarray(pcd.points)
    z_trans = min(points[:, 2])
    rot_trans_straight [2, 3] -= z_trans
    
    pcd.transform(rot_trans_straight)
    
    # transform 2d np array to 2d list
    l_transformation = [list(r) for r in rot_trans_straight]
    print(l_transformation)
    # append transformation to the end of json file
    with open(json_path, mode='r', encoding='utf-8') as feedsjson:
        feeds = json.load(feedsjson)
    with open(json_path, mode='w', encoding='utf-8') as feedsjson:
        entry = {'postprocessing_transformation': l_transformation}
        feeds.append(entry)
        json.dump(feeds, feedsjson)
        
    output_folder, _ = split(json_path)
    o3d.write_point_cloud(join(output_folder, 'stitched_straight.pcd'), pcd)
    
    points = np.asarray(pcd.points)
    
    # points to flat surface
    points = cart2pol_pc(points, r)
    
    
    # save points in the point cloud
    pcd.points = o3d.Vector3dVector(points)
    
    # find normals so it's displayed nicer
#    o3d.estimate_normals(pcd, search_param = o3d.KDTreeSearchParamHybrid(
#                                                        radius = 5, max_nn = 30))
#    o3d.draw_geometries([pcd])
    
    output_path = join(output_folder, 'flatten.pcd')
    o3d.write_point_cloud(output_path, pcd)

    return output_path

    
def save_as_grid(pc_path, json_path, r, resolution, cam_settings, omsws_settings):

    (cam_angle_of_view, cam_rot, cam_pos, image_flip_h, image_flip_v) = cam_settings
    dir_path, _ = split(pc_path)
    
    # read flattened point cloud
    pcd = o3d.read_point_cloud(pc_path)
    points = np.asarray(pcd.points)
    
    # read json file
    (paths, points_numbers, transformations, pprc_transformation) =\
                                    su.read_json_file(json_path, points.shape[0])
    
    # create grid and sources - from which scan each point is
    points, (scan_sources_h, scan_sources_l), grid_data = \
                     pu.calc_points_on_grid(points, points_numbers, resolution)
    
    (height, width, grid_step) = grid_data
    
    # create stitcher class and output image
    stitcher = Stitcher()
    stitcher.update_output_image(np.zeros((height, width, 3), dtype=np.uint8))
    
    # wrap the points on cylinder
    points = pol2cart_pc(points, r)

    # transform back from before postprocessing
    points = transform(points, inv(pprc_transformation))
    
    
    for i, path in enumerate(paths):
        print('loading omsws:', split(path)[1])
        
        points_scan = points[np.logical_or(scan_sources_l == i, 
                                           scan_sources_h == i)]

        
        # all scans after first one have another transformation (stitching). 
        # we need to go back to raw positions, so inverting transformation
        if i > 0:
            points_scan = transform(points_scan, inv(transformations[i-1]))
        
        print('loading image')
        image_loader = su.StitchImageLoader(path, omsws_settings, cam_settings,
                                            r, points_scan)

        points_colors = image_loader.load_images()
    
        # points to flat surface
        points_flat = cart2pol_pc(points_scan, r)
        
        output = su.create_image(points_flat, points_colors, grid_step)
        result = Image.fromarray(output, mode='RGB')
        result.save(join(split(json_path)[0], 'pictures/result%02d.png' %i))

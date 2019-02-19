# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 12:32:40 2018

@author: Dell
"""

from nptdms import TdmsFile, GroupObject, RootObject, ChannelObject, TdmsWriter
import numpy as np
import math

#import os
import io
from PIL import Image
#from array import array




def create_rotation_array_2d(rotation_ccw):
    c = math.cos(rotation_ccw)
    s = math.sin(rotation_ccw)
    return np.array([[c, s], [-s, c]])



class OMS_Tdms:

    def __init__(self):
        pass
        
    
    def open_tdms_file(self, path):
        self.path = path
        self.tdms_file = TdmsFile(self.path)
    
    
    def read_properties(self, property_names, property_location):
        properties = []
        try:
            # check if channel name was passed
            (group_name, channel_name) = property_location
        except ValueError:
            # if no channel was passed, set channel name to None
            group_name = property_location
            channel_name = None

        for property_name in property_names:
            # check if channel name was passed
            if channel_name:
                # read channel object
                channel = self.tdms_file.object(group_name, channel_name)
            else:
                # read group object
                channel = self.tdms_file.object(group_name)  
            
            properties.append(channel.property(property_name))
    
        return properties
    
    
    def read_no_scans(self, scan_id):
        group_name = 'SCAN-{:04d}'.format(scan_id)
        property_name = 'Actual No. Slices'
        return int(self.read_properties([property_name], group_name)[0])
     
        
    def read_no_images(self, scan_id):
        group_name = 'SCAN-{:04d}'.format(scan_id)
        property_name = 'Actual No. Images'
        return int(self.read_properties([property_name], group_name)[0])
    
    
    def read_scan1_property(self, property_name):
        group_name = 'SCAN-0001'
        return self.read_properties([property_name], group_name)[0]
    
    
    def print_scan1_properties(self):
        group_object = self.tdms_file.object('SCAN-0001')
        for name, value in group_object.properties.items():
            print("{0}: {1}".format(name, value))
           
            
    def print_channel_names(self, scan_id):
        channels = self.tdms_file.group_channels('SCAN-{:04d}'.format(scan_id))
        print(channels)


    def print_channel_property(self, scan_id, channel_name):
        group_name = 'SCAN-{:04d}'.format(scan_id)
        channel_object = self.tdms_file.object(group_name, channel_name)
        for name, value in channel_object.properties.items():
            print("{0}: {1}".format(name, value))



class OMSWS1(OMS_Tdms):
    
    def __init__(self, path, scan_id, inverse_x):
        self.inverse_x = inverse_x
        self.version = 1
        self.scan_id = scan_id
        self.open_tdms_file(path)
        self.id_offset = float(OMS_Tdms.read_scan1_property(self, 'ID Offset'))
        self.pipe_id = float(OMS_Tdms.read_scan1_property(self, 'ID Average (mm)'))
        self.init_scan_angles()
        self.init_scans()
        self.init_image_angles()
        self.init_images()
        try:
            self.distance_driven = float(OMS_Tdms.read_scan1_property(self, 'Crawler Position (m)'))
        except KeyError:
            print("no info about crawler position")
            self.distance_driven = -1
    
    def open_tdms_file(self, path):
        OMS_Tdms.open_tdms_file(self, path)
        
        
    def init_scan_angles(self):
        group_name = 'SCAN-{:04d}'.format(self.scan_id)
        actual_no_slices = self.read_no_scans(self.scan_id)
        self.scan_angles = []
        for i in range(1, actual_no_slices + 1):
            channel_name = 'Slice-{:04d}-X'.format(i)
            angle = float(self.read_properties(["Angle (deg)"], 
                                          (group_name, channel_name))[0])
            self.scan_angles.append(angle * np.pi/180)
    

    def init_scans(self):
        group_name = 'SCAN-{:04d}'.format(self.scan_id)
        actual_no_slices = self.read_no_scans(self.scan_id)
        self.scans = []
        for i in range(1, actual_no_slices + 1):
            channel_name = 'Slice-{:04d}'.format(i)
            scan = np.array([self.tdms_file.channel_data(group_name, channel_name+'-X'),
                    self.tdms_file.channel_data(group_name, channel_name+'-Y')])
            if self.inverse_x:
                scan[0] = -scan[0]
            self.scans.append(scan)


    def init_image_angles(self):
        group_name = 'SCAN-{:04d}'.format(self.scan_id)
        actual_no_images = self.read_no_images(self.scan_id)
        self.image_angles = []
        for i in range(1, actual_no_images + 1):
            channel_name = 'Image-{:04d}'.format(i)
            angle = float(self.read_properties(["Angle (deg)"], 
                                          (group_name, channel_name))[0])
            self.image_angles.append(angle * np.pi/180)
        
        
    def init_images(self):
        group_name = 'SCAN-{:04d}'.format(self.scan_id)
        actual_no_images = self.read_no_images(self.scan_id)
        self.images = []
        for i in range(1, actual_no_images + 1):
            channel_name = 'Image-{:04d}'.format(i)
            bytes = np.array(self.tdms_file.channel_data(group_name, channel_name))
            img = Image.open(io.BytesIO(bytes))
            self.images.append(np.asarray(img))
    
    
    def read_scans_part(self, index_list):
        return [scan for (i, scan) in enumerate(self.scans) if i in index_list]        
    
    
    def apply_id_offset(self):
        for i, _ in enumerate(self.scans):
            self.scans[i][1] += self.id_offset
        
        
    def apply_rotation(self, rotation):
        if rotation != 0:
            rot_arr = create_rotation_array_2d(math.radians(rotation))
            self.scans = [np.dot(scan.T, rot_arr) for scan in self.scans]


    def preprocess_scans(self, rotation):
        self.apply_rotation(rotation)
        self.apply_id_offset()
        
        
    def preprocess_images(self, cam_other_side, image_flip_h, image_flip_v):
        if cam_other_side:
            self.rotate_image_angles(math.pi)
        if image_flip_h:
            self.flip_images_horizontally()
        if image_flip_v:
            self.flip_images_vertically()
        
  
    def rotate_image_angles(self, angle):
        angles = np.array(self.image_angles)
        angles += angle
        angles = angles % (2*math.pi)
        self.image_angles = list(angles)


    def flip_images_horizontally(self):
        for i, image in enumerate(self.images):
            self.images[i] = np.flip(image, 0)

            
    def flip_images_vertically(self):
        for i, image in enumerate(self.images):
            self.images[i] = np.flip(image, 1)
            
            
    
def save_as_tdms(file_path, pcd):
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255
    x, y, z = points.T.astype('float32')
    r, g, b = colors.T.astype('uint8')
    
    
    root_obj = RootObject(properties={
            'Version': 1, 
            'Day': 11, 
            'Month': 12, 
            'Year': 2018, 
            'Pipe ID': 584})
    
    group_laser = GroupObject('Laser Data', properties={
            'Fields': 'X Y Z',
            'Type': 'F F F',
            'Points': len(x)
            })
    
    group_color = GroupObject('Colour Data', properties={
            'Fields': 'R G B',
            'Type': 'U8 U8 U8',
            'Points': len(r)
            })
    
    channel_x = ChannelObject('Laser Data', 'X', x)
    channel_y = ChannelObject('Laser Data', 'Y', y)
    channel_z = ChannelObject('Laser Data', 'Z', z)
    
    channel_r = ChannelObject('Colour Data', 'R', r)
    channel_g = ChannelObject('Colour Data', 'G', g)
    channel_b = ChannelObject('Colour Data', 'B', b)
    
    with TdmsWriter(file_path) as tdms_writer:
        tdms_writer.write_segment([
                root_obj,
                group_laser, group_color,
                channel_x, channel_y, channel_z,
                channel_r, channel_g, channel_b])

    
def save_array_as_tdms(file_path, array, group, channels):
    # save vertical 2d array each column to a different channel
    # e.g. rgb point cloud shaped (1000, 3) 
    group_obj = GroupObject(group)
    print(channels)
    print(group)
#    print(array.shape)
    channels_obj = [ChannelObject(group, channel, array[:, i]) 
                                        for i, channel in enumerate(channels)]
    
    with TdmsWriter(file_path) as tdms_writer:
        tdms_writer.write_segment([group_obj, *channels_obj])
    

def read_array_from_tdms(file_path, group, channels):
    tdms = OMS_Tdms()
    tdms.open_tdms_file(file_path)
    
    return [tdms.tdms_file.channel_data(group, channel) for channel in channels]
    


if __name__ == '__main__':
    dir_path = r'/media/karol/SSD/Data/omsws/Long UT Section (9.D)/1/'
    file_path = '9.D 004.2_30.omsws'
    omsws = OMS_Tdms()
    omsws.open_tdms_file(dir_path + file_path)
    omsws.print_scan1_properties()

    
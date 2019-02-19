# import the necessary packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import griddata
import math


class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        (major, _, _) = cv2.__version__.split(".")
        self.isv3 = major == '3'
        self.last_im_features = None
        self.image_stitch = None
        self.mid_point_stitch = None
        self.mid_point_stitch_im = None
        self.image_base = None
        

    def update_output_image(self, image):
        self.output_im = image
        
        
    def stitch(self, images, ratio=0.75, reprojThresh=4.0,
        showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB,
            featuresA, featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None

        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M
        fitness = sum(status)/len(status)
        print('fit', fitness)
        
#        x_shift = H[0,2]
#        print (H)
#        result = np.zeros((imageB.shape[0], imageA.shape[1]+imageB.shape[1], 3), np.uint8)
#        print ('result ', result.shape)
        
#        result = cv2.warpPerspective(imageA, H,
#            (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
#        print ('A: ', imageA.shape)
        
        
#        result[:, x_shift:x_shift+imageA.shape[1], :] = imageA
#        result[0:imageB.shape[0], 0:imageB.shape[1], :] = imageB
#        result = result[:, 0:x_shift+imageA.shape[1], :]
        
#        print (result.shape)
        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                status)

            # return a tuple of the stitched image and the
            # visualization
            return (H, fitness, vis)
        # return the stitched image
        return H, fitness, None
        
    
#    def describe_image(self, imageA):
#
#        (kpsA, featuresA) = self.detectAndDescribe(imageA)
#        self.last_im_features = (kpsA, featuresA)
        
        
#    def stitch_new_image(self, imageB, ratio=0.75, reprojThresh=4.0):
#        if self.last_im_features is None:
#            raise Exception('''There are no features to match with. Please run 
#                            describe_image() first''')
#            
#        (kpsA, featuresA) = self.last_im_features
#        (kpsB, featuresB) = self.detectAndDescribe(imageB)
#        
#        M = self.matchKeypoints(kpsA, kpsB,
#            featuresA, featuresB, ratio, reprojThresh)
#
#        # if the match is None, then there aren't enough matched
#        # keypoints to create a panorama
#        if M is None:
#            return None
#
#        # otherwise, apply a perspective warp to stitch the images
#        # together
#        (matches, H, status) = M
#        
#        return H

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)

        # otherwise, we are using OpenCV 2.4.X
        else:
            # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
        ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None

    
    def find_rotation(self, source, target):
        target_s = target[:, int(target.shape[1]/2):]
        source_s = source[:, :int(source.shape[1]/2)]
        plt.imshow(target_s)
        plt.show()
        plt.imshow(source_s)
        plt.show()
        
        stitch = self.stitch((target_s, source_s), ratio = 1, showMatches=True)
        if stitch is not None:
            H, fitness, vis = stitch
        else:
            return None
        
        if vis is not None:
            plt.imshow(vis)
            plt.show()
        dm = cv2.decomposeHomographyMat(H, np.diag((1, 1, 1)))
        rot = dm[1][2][:2, :2]
        trans = dm[2][2][:2] + [int(source.shape[1]/2), 0]
        print('decomposition;', dm[2][2])
        
        return np.arccos(rot[0, 0])/2, trans, fitness
    
    
    def find_transform(self, source, target, show_images=False):
        # stitch images target is on the left from the source
        
        if show_images:
            print('target')
            plt.imshow(target)
            plt.show()
            print('source')
            plt.imshow(source)
            plt.show()
            
            
        (im_height_a, im_width_a) = source.shape[:2]
        (im_height_b, im_width_b) = target.shape[:2]
        
        im_height_c = min(im_height_a, im_height_b)
        im_width_c = min(im_width_a, im_width_b)
        
        stitch_width = int(im_width_c/2) - 1
        target_s = target[:im_height_c, im_width_c - stitch_width : im_width_c]
        source_s = source[:im_height_c, :stitch_width]
        
        if show_images:
            plt.imshow(target_s)
            plt.show()
            plt.imshow(source_s)
            plt.show()
        
        t = Image.fromarray(target_s, mode='RGB')
        s = Image.fromarray(source_s, mode='RGB')
        
        t.save('target.bmp')
        s.save('source.bmp')
        target_s = cv2.imread('target.bmp')
        source_s = cv2.imread('source.bmp')
        
#        target_s = cv2.GaussianBlur(target_s, (5, 5), 0)
#        source_s = cv2.GaussianBlur(source_s, (5, 5), 0)
        
        
#        source_s = source[:im_height_c, :im_width_c]
#        target_s = target[:im_height_c, :im_width_c]
        
        M = cv2.estimateRigidTransform(source_s, target_s, fullAffine=False)
        print(M)

#        source_out = cv2.warpAffine(source_s, M, (stitch_width, im_height_c))
#        
#        print('trans normal: ', M[:, 2])
#        result = Image.fromarray(source_out, mode='RGB')
#        result.save('source_t_ perfect.png') 

        if M is None:
            print('***********couldnt find M***********')
            return None
            
#        print(M)
        
        # find rotation from M matrix
        rot = M[:, :2]
        # check the scale based on the fact that cos(a)**2 + sin(a)**2 should be 1
        scale = np.sqrt(rot[0, 0] ** 2 + rot[0, 1] ** 2)
        rot = rot/scale
        M[:, 2] += np.array([im_width_c - stitch_width, 0])
        # calculate the amount of pixels that the scale is shifting source image
        scale_trans = np.array([stitch_width, im_height_c]) * (scale - 1) / 2
        trans = M[:, 2] + scale_trans
        
        
#        M = np.column_stack((rot, trans))

#        result = Image.fromarray(target, mode='RGB')
#        result.save('target.png')
#        source_out = cv2.warpAffine(source, M, (source.shape[1] + target.shape[1], source.shape[0]))
#        
#        result = Image.fromarray(source_out, mode = 'RGB')
#        result.save('source.png')

        # output rotation angle and translation without scale
        cos_a = rot[0, 0]
        print('cos_a', cos_a)
        alpha = np.arccos(cos_a)/2
        print('aaaaalpha', alpha)
        return alpha, trans, M

    
    def apply_transform(self, image, M):
        height, width = image.shape[:2]
        
        height = int(height + M[1, 2])
        width = int(width + M[0, 2])
         
        output_im = cv2.warpAffine(image, M, (width, height))
        
        return output_im
    
    
    def resize_images(self, target, source):
        new_width = max(target.shape[1], source.shape[1])
        new_height = max(target.shape[0], source.shape[0])
        
        out_target = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        out_source = out_target.copy()
        
        out_target[:target.shape[0], :target.shape[1]] = target
        out_source[:source.shape[0], :source.shape[1]] = source
        
        return out_target, out_source
    
    
    def check_image_non_black(self, image):
        return np.logical_or(image[:, :, 0] != 0, 
                              image[:, :, 1] != 0, 
                              image[:, :, 2] != 0)
    
    
    def calculate_mid(self, image):
        yi, xi = np.nonzero(self.check_image_non_black(image))
        return np.array([np.mean(xi), np.mean(yi)])
    
    
    def calculate_weighted_mid(self, image):
        yi, xi = np.nonzero(self.check_image_non_black(image))
        rgb = image[yi, xi, :]
        rgbs = np.sum(rgb, axis=1)
        
        x = sum(xi * rgbs) / sum(rgbs)
        y = sum(yi * rgbs) / sum(rgbs)
        
        return np.array([y, x])
    
        
    def stitch_images(self, target, source, blending=1):
        # blending of 1 is gaussian filter of size 1/10 of smaler dim of image
        
        if target.shape != source.shape:
            target, source = self.resize_images(target, source)
        
#        print('images for stitching')
#        plt.imshow(target)
#        plt.show()
#        plt.imshow(source)
#        plt.show()
#        blending = np.clip(blending, 1, 9)
        height, width = target.shape[:2]
#        filter_size = int(min(height, width) * blending / 10)
#        filter_size = np.clip(filter_size, 3, np.inf)

#        add_size = int(filter_size/2)
        add_size = 0
        new_width = width + 2 * add_size
        new_height = height + 2 * add_size
        
#        full_mask = np.logical_or(check_image_black(target), 
#                                  check_image_black(source))
        
#        greater_image = np.zeros((width + 2*add_size, height + 2*add_size, 3), 
#                                 dtype=np.uint8)
        target_mask = np.zeros((new_height, new_width), dtype=bool)
        source_mask = np.zeros((new_height, new_width), dtype=bool)
        target_mask[add_size : height + add_size, add_size : width + add_size] =\
                    self.check_image_non_black(target)
        
        source_mask[add_size : height + add_size, add_size : width + add_size] =\
                    self.check_image_non_black(source)
        
        stitch_mask = np.logical_and(target_mask, source_mask)
        sum_mask = np.logical_or(target_mask, source_mask)

        only_images = np.logical_and(np.logical_not(stitch_mask), sum_mask)

        dependency_mask = only_images * (1 * target_mask + 2 * source_mask)
        dependency_mask = dependency_mask.astype(np.float32)
        mgrid_x, mgrid_y = np.mgrid[0 : height, 0 : width]
        
        x, y = np.nonzero(dependency_mask)
        z = dependency_mask[x, y]
        
        dependency_grid = griddata((x,y), (z), (mgrid_x, mgrid_y), method='nearest')
#        print('len fo grid',  len(dependency_grid))
        dependency_mask[mgrid_x, mgrid_y] = dependency_grid
        dependency_mask *= sum_mask
        
        output_im = np.zeros_like(target, dtype=np.uint8)
        
        output_im[dependency_mask == 2] = source[dependency_mask == 2]
        output_im[dependency_mask == 1] = target[dependency_mask == 1]
        dependency_mask -= 1
        stitch_mask = np.logical_and(dependency_mask != 1, dependency_mask != 0)
        stitch_dep = dependency_mask[stitch_mask]
#        print('shape', stitch_dep.shape)
        stitch_dep3 = np.stack((stitch_dep,)*3, axis=1)
        output_im[stitch_mask] = target[stitch_mask, :] * stitch_dep3 +\
                                 source[stitch_mask, :] * (1 - stitch_dep3)
        
#        plt.imshow(output_im)
#        plt.show()
        result = Image.fromarray(output_im, mode='RGB')
        result.save('stitched image.png')
        
#        output_im
    
        return output_im


#def make_grid(ranges, grid_step):
    
    
    
    
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis
    
    
    def find_first_stitch2(self, base_data, stitch_data, grid_step):
        base_image, base_points_mid = base_data
        stitch_image, stitch_points_mid = stitch_data

        stitch = self.find_transform(stitch_image, base_image)
        
        if stitch is None:
            return None
        else:
        
#            print('-------base image--------')
#            plt.imshow(base_image)
#            plt.show()
#            
#            print('-------stitch image--------')
#            plt.imshow(stitch_image)
#            plt.show()

            alpha, translation, M = stitch
            
            diff_points = stitch_points_mid - base_points_mid
            
            base_image_mid = self.calculate_mid(base_image).astype(int)
            
            stitch_image = self.apply_transform(stitch_image, M)
            stitch_image_mid = self.calculate_mid(stitch_image).astype(int)
            
            image_stitched = self.stitch_images(base_image, stitch_image)
            image_stitched[base_image_mid[1]-5 : base_image_mid[1]+5,
                           base_image_mid[0]-5 : base_image_mid[0]+5,:] =\
                           np.ones((10, 10, 3), dtype=np.uint8)*255
                           
            image_stitched[stitch_image_mid[1]-5 : stitch_image_mid[1]+5,
                           stitch_image_mid[0]-5 : stitch_image_mid[0]+5,:] =\
                           np.ones((10, 10, 3), dtype=np.uint8)*255
            
            print('-------stitched image--------')
            plt.imshow(image_stitched)
            plt.show()
#            result = Image.fromarray(image_stitched, mode='RGB')
#            result.save('stitched.png')
            
            diff_im = stitch_image_mid - base_image_mid
            
            scale = np.linalg.norm(diff_im)/np.linalg.norm(diff_points)
            
#            diff_im = (diff_im.astype(float) / scale)
            angle_rot = math.atan2(diff_im[1], diff_im[0]) -\
                        math.atan2(diff_points[1], diff_points[0])    
            # in order to achieve angle_rot, both images needs to be rotated
            # by angle_rot/2
#            angle_rot = angle_rot/2
            
            return scale, angle_rot
    
            
    def find_first_stitch(self, image, points_mid, grid_step):
    
        # save the first base image
        if self.image_stitch is not None:
            self.image_base = self.image_stitch.copy()
            self.mid_point_base = self.mid_point_stitch.copy()
            self.mid_point_base_im = self.mid_point_stitch_im.copy()

        # save new image as image_stitch
        self.image_stitch = image
        self.mid_point_stitch = points_mid
        self.mid_point_stitch_im = self.calculate_mid(self.image_stitch)
        
        if self.image_base is not None:

            stitch = self.find_transform(self.image_stitch, self.image_base, 
                                             show_images=True)
            print (stitch)
            if stitch is not None:
                
                vector_points = self.mid_point_stitch - self.mid_point_base
                print('we are aiming for', vector_points)
                alpha, translation, M = stitch
                
                image_stitch_t = self.apply_transform(self.image_stitch, M)
                mid_point_stitch_t = self.calculate_mid(image_stitch_t).astype(int)
                print('mid point after transformation:', mid_point_stitch_t)
                print('base point', self.mid_point_stitch_im)
#                print('stitch after transforming')
#                plt.imshow(image_stitch_t)
#                plt.show()
                print('first stitch')
                image_stitched = self.stitch_images(self.image_base, image_stitch_t)
                
                vector_im = mid_point_stitch_t - self.mid_point_base_im
                
                diff = vector_points - vector_im
                angle_rot = np.arctan2(diff[1], diff[0])/2
                scale = np.linalg.norm(vector_points)/np.linalg.norm(vector_im)
                
                rows, cols = image_stitched.shape[:2]
                print('angle rotation', angle_rot)
                self.M = cv2.getRotationMatrix2D(tuple(self.mid_point_base_im),
                                                 np.rad2deg(angle_rot), scale) 
                print('M: ', self.M)
                
                print('image after transformation')
                image_stitched = cv2.warpAffine(image_stitched, self.M, (cols, rows))
                test_mid = np.zeros_like(image_stitched, dtype=np.uint8)
                test_mid[mid_point_stitch_t[0], mid_point_stitch_t[1]] =\
                [255, 255, 255]
                                                                
                print('mid points on black screen:, ',self.calculate_weighted_mid(test_mid))
                plt.imshow(test_mid)
                plt.show()                                                                
                test_mid = cv2.warpAffine(test_mid, self.M, (cols, rows))
                plt.imshow(test_mid)
                plt.show()
                new_mid = self.calculate_weighted_mid(test_mid)
                mid_point_stitch_t = (new_mid + 0.49).astype(int)
                print('mid point after warping:', mid_point_stitch_t)
                plt.imshow(image_stitched)
                plt.show()
                print('diff to stitch', self.mid_point_base_im - mid_point_stitch_t)
                
                print('mid point base', self.mid_point_base)
                print('base stotcj', self.mid_point_base_im)
#                mid_point_out_pos = mid_point_out_pos
                start_row_out, start_col_out =\
                            self.mid_point_base - self.mid_point_base_im
                print('starting pos:', start_row_out, start_col_out)
                start_row_out = int(start_row_out)
                start_col_out = int(start_col_out)
                
                print('shape', self.output_im.shape)
                
                self.output_im[start_row_out : start_row_out + rows,
                               start_col_out : start_col_out + cols] =\
                                                               image_stitched
                       
                return True
        
        return False
#                self.output_im[:rows, :cols] = image_stitched
                    # we need the difference in translation in mm
#                    distance_trans = np.linalg.norm(translation)
#                    
#                    distance_or = np.linalg.norm(translation_or)
#            
    
    
    
#    def stitch_images(source, target, translation):
#        height_s, width_s = source.shape[:2]
#        height_t, width_t = target.shape[:2]
#        output_height = max(height_s, height_t, height_s + translation[1])
#        output_widht = max(width_s, width_t, width_s + translation[0])
#        
#        output = np.zeros((output_height, output_widht, 3), dtype=np.uint8)
#        output[:height_t, width_t] = target
#        output[]
        

        
        
        
        
        
        
        
        
        
        
#!/usr/bin/env python

import os
import sys
import math
import lmdb
import rospy
import random
import numpy as np
import cv2 as cv
import matplotlib.pylab as plt

FORE_PROB_ = 1.0
BACK_PROB_ = 0.0
FLT_EPSILON_ = sys.float_info.epsilon

np.set_printoptions(threshold=np.nan)

"""
Class for computing intersection over union(IOU)
"""
class JaccardCoeff:

    def iou(self, a, b):
        i = self.__intersection(a, b)
        if i == 0:
            return 0
        aub = self.__area(self.__union(a, b))
        anb = self.__area(i)
        area_ratio = self.__area(a)/self.__area(b)        
        score = anb/aub
        score /= area_ratio
        return score
        
    def __intersection(self, a, b):
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0]+a[2], b[0]+b[2]) - x
        h = min(a[1]+a[3], b[1]+b[3]) - y
        if w < 0 or h < 0:
            return 0
        else:
            return (x, y, w, h)
        
    def __union(self, a, b):
        x = min(a[0], b[0])
        y = min(a[1], b[1])
        w = max(a[0]+a[2], b[0]+b[2]) - x
        h = max(a[1]+a[3], b[1]+b[3]) - y
        return (x, y, w, h)

    def __area(self, rect):
        return float(rect[2] * rect[3])


class CreateTrainingLMDB:
    def __init__(self):
        ## contains image path
        self.__data_textfile = '/home/krishneel/Desktop/data/log.txt'  
        
        if not os.path.isfile(str(self.__data_textfile)):
            rospy.logfatal('FILE NOT FOUND')
            sys.exit()


        self.__shuffle = rospy.get_param('~shuffle', True) ## shuffle dataset
        self.__num_random = rospy.get_param('~num_random', 10)  ## number of extra samples
        self.__min_size = rospy.get_param('~min_size', 20)  ## min box size

        ##! network size
        self.__net_size_x = rospy.get_param('~net_size_x', 640) ## network image cols
        self.__net_size_y = rospy.get_param('~net_size_y', 480) ## network image rows
        self.__stride = rospy.get_param('~stride', 16)  ## stride of the grid
        
        self.__num_classes = rospy.get_param('~num_classes', 3)  ## number of classes
        
        
        if (self.__net_size_x < 1) or (self.__stride < 16):
            rospy.logfatal('FILE NOT FOUND')
            sys.exit()
        
        self.process_data(self.__data_textfile)
        

    def process_data(self, path_to_txt):
        
        lines = self.read_textfile(path_to_txt)
        img_path, rects, labels = self.decode_lines(lines)
        
        for index, ipath in enumerate(img_path):
            img = cv.imread(str(ipath))
            rect = rects[index]
            
            ### check only29 x 49 from (309, 268)
            ##--------------------------------
            img = cv.resize(img, (610, 610))
            rect = (309, 268, 29, 49)
            ##--------------------------------

            # self.random_argumentation(img, rect)

            self.bounding_box_parameterized_labels(img, rect, self.__stride)

            return
            


    def bounding_box_parameterized_labels(self, img, rect, stride):
        boxes = self.grid_region(img, self.__stride)
        foreground_labels = self.generate_box_labels(img, boxes, rect, FLT_EPSILON_)
        boxes_labels = []
        size_labels = []
        obj_labels = []
        
        K = self.__num_classes
        W = self.__net_size_x / self.__stride
        H = self.__net_size_y / self.__stride
        inputs = np.zeros((1, K, H, W), dtype=np.float)
        
        print inputs.shape

        for index, b in enumerate(foreground_labels):
            if b == 1.0:
                boxes_labels.append(boxes[index])
                size_labels.append([1.0/rect[2], 1.0/rect[3]])
                diff = float(boxes[index][2] * boxes[index][3]) / float(rect[2] * rect[3])
                obj_labels.append(diff)
                  
                
  
        print boxes_labels
        print size_labels
        print obj_labels
        sys.exit()



    def resize_image_and_labels(self, image, labels):
        resize_flag = (self.__net_size_x, self.__net_size_y)
        img_list = []
        label_resize = []
        if resize_flag:
            img = cv.resize(image, resize_flag)
            img_list.append(img)
            # resize label
            ratio_x = float(image.shape[1]) / float(img.shape[1])
            ratio_y = float(image.shape[0]) / float(img.shape[0])

            x = float(labels[0])
            y = float(labels[1])
            w = float(labels[2])
            h = float(labels[3])
            
            xt = x / ratio_x
            yt = y / ratio_y
            xb = (x + w) / ratio_x
            yb = (y + h) / ratio_y

            rect_resize = (int(xt), int(yt), int(xb - xt), int(yb - yt))
            return img, rect_resize


    def random_argumentation(self, image, rect): 
        
        images = []
        rects = []
        ##! save original first
        images.append(image)
        rects.append(rect)

        scale_x = float(image.shape[1]) / float(rect[2])
        scale_y = float(image.shape[0]) / float(rect[3])
        scale_x = int(math.floor(scale_x))
        scale_y = int(math.floor(scale_y))

        for i in xrange(-1, 2, 1):
            flip_flag = random.randint(-1, 1)
            img_flip, rect_flip = self.flip_image(image.copy(), rect, flip_flag)
            
            ##! save flipped 
            images.append(img_flip)
            rects.append(rect_flip)

            ##! flip and save
            enlarge_factor1 = random.randint(2, scale_x)
            enlarge_factor2 = random.randint(2, scale_y)
            widths = (int(rect_flip[2] * enlarge_factor1), rect_flip[2] * enlarge_factor2)
            heights = (int(rect_flip[3] * enlarge_factor1), rect_flip[3] * enlarge_factor2)
            crop_image, crop_rect = self.crop_image_dimension(img_flip, rect_flip, widths, heights)
            images.append(crop_image)
            rects.append(crop_rect)

            ## apply blur
            kernel_x = random.randint(3, 9)
            kernel_y = random.randint(3, 9)
            kernel_x = kernel_x + 1 if kernel_x % 2 is 0 else kernel_x
            kernel_y = kernel_y + 1 if kernel_y % 2 is 0 else kernel_y

            blur_img = cv.GaussianBlur(crop_image, (kernel_x, kernel_y), 0)
            images.append(blur_img)
            rects.append(crop_rect)
            
            # print blur_img.shape
            
            img_flip = blur_img
            rect_flip = crop_rect

            ### plot only
            # cv.rectangle(img_flip, (rect_flip[0], rect_flip[1]), \
            #              (rect_flip[0] + rect_flip[2], rect_flip[1] + rect_flip[3]), (0, 255, 0))
            # cv.namedWindow("img", cv.WINDOW_NORMAL)
            # cv.imshow("img", img_flip)
            # cv.waitKey(0)
            
        return images, rects

    def crop_image_dimension(self, image, rect, widths, heights):
        x = (rect[0] + rect[2]/2) - widths[0]
        y = (rect[1] + rect[3]/2) - heights[0]
        w = widths[1] + widths[0]
        h = heights[1] + heights[0]

        x = 0 if x < 0 else x
        y = 0 if y < 0 else y
        w = ((w - (w + x) - image.shape[1])) if x > image.shape[1] else w
        h = ((h - (h + y) - image.shape[0])) if y > image.shape[0] else h


        roi = image[y:y+h, x:x+w].copy()
        new_rect = [int(rect[0] - x), int(rect[1] - y), rect[2], rect[3]]

        return roi, new_rect

    def flip_image(self, image, rect, flip_flag = -1):
        pt1 = (rect[0], rect[1])
        pt2 = (rect[0] + rect[2], rect[1] + rect[3])
        im_flip = cv.flip(image, flip_flag)
        if flip_flag is -1:
            pt1 = (image.shape[1] - pt1[0] - 1, image.shape[0] - pt1[1] - 1)
            pt2 = (image.shape[1] - pt2[0] - 1, image.shape[0] - pt2[1] - 1)
        elif flip_flag is 0:
            pt1 = (pt1[0], image.shape[0] - pt1[1] - 1)
            pt2 = (pt2[0], image.shape[0] - pt2[1] - 1)
        elif flip_flag is 1:
            pt1 = (image.shape[1] - pt1[0] - 1, pt1[1])
            pt2 = (image.shape[1] - pt2[0] - 1, pt2[1])
        else:
            print 'ERROR: Invalid flip_flag'
            return

        x = min(pt1[0], pt2[0])
        y = min(pt1[1], pt2[1])
        w = np.abs(pt2[0] - pt1[0])
        h = np.abs(pt2[1] - pt1[1])

        flip_rect = [x, y, w, h]

        return im_flip, flip_rect


    """
    Function to label each grid boxes based on IOU score
    """     
    def generate_box_labels(self, image, boxes, rect, label, iou_thresh = FLT_EPSILON_):
        
        jc = JaccardCoeff()
        box_labels = [
            FORE_PROB_ if jc.iou(box, rect) > iou_thresh
            else BACK_PROB_
            for box in boxes
        ]
        return  np.array(box_labels)


    """
    Function to overlay a grid on the image and return the boxes
    """     
    def grid_region(self, image, stride):
        wsize = (image.shape[1]/stride, image.shape[0]/stride)
        boxes = [
            np.array([i, j, stride, stride])
            for i in xrange(0, image.shape[1], stride)
            for j in xrange(0, image.shape[0], stride)
            if (j + stride <= image.shape[0]) and (i + stride <= image.shape[1])
        ]
        return boxes

    """
    Function to extract path to images, bounding boxes and the labels
    """
    def decode_lines(self, lines):
        img_path = []
        rects = []
        labels = []
        for line in lines:
            line_val = line.split()
            img_path.append(line_val[0])
            rect = []
            for i in xrange(1, len(line_val)-1, 1):
                rect.append(int(line_val[i]))
            rects.append(rect)
            labels.append(int(line_val[-1]))
        
        return np.array(img_path), np.array(rects), np.array(labels)
        
    """
    Function to shuffle the input
    """
    def shuffle_array(self, lines):
        return np.random.shuffle(lines)

    """
    Function read text file 
    """
    def read_textfile(self, path_to_txt):
        lines = [line.rstrip('\n')
                 for line in open(path_to_txt)
        ]
        return lines


def main(argv):
    try:
        rospy.init_node('fcn_object_detector', anonymous = True)
        ctl = CreateTrainingLMDB()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logfatal("ros error")
        pass

if __name__ == '__main__':
    main(sys.argv)

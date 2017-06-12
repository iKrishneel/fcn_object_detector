#!/usr/bin/env python

###########################################################################
## Copyright (C) 2017 by Krishneel Chaudhary @ JSK Lab,
## The University of Tokyo, Japan
###########################################################################

import os
import sys
import math
import lmdb
import random
import caffe
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


class ArgumentationEngine(object):
    def __init__(self, im_width, im_height, stride, num_classes):
        self.__net_size_x = im_width
        self.__net_size_y = im_height
        self.__stride = stride
        self.__num_classes = num_classes

    def pack_data(self, img, rect, label):

        foreground_labels, boxes_labels, size_labels, obj_labels, coverage_label = \
        self.bounding_box_parameterized_labels(img, rect, label, self.__stride)

        ## pack all in one data
        K = foreground_labels.shape[0] + boxes_labels.shape[0] + size_labels.shape[0] + \
            obj_labels.shape[0] + coverage_label.shape[0]
        W = boxes_labels.shape[1]
        H = boxes_labels.shape[2]
        
        data_labels = np.zeros((K, W, H))

        start = 0
        end = start + foreground_labels.shape[0]
        data_labels[start:end] = foreground_labels

        start = end
        end = start + boxes_labels.shape[0]
        data_labels[start:end] = boxes_labels

        start = end
        end = start + size_labels.shape[0]
        data_labels[start:end] = size_labels

        start = end
        end = start + obj_labels.shape[0]
        data_labels[start:end] = obj_labels
        
        start = end
        end = start + coverage_label.shape[0]
        data_labels[start:end] = coverage_label

        return data_labels


    def bounding_box_parameterized_labels(self, img, rect, label, stride):
        boxes = self.grid_region(img, stride)
        region_labels = self.generate_box_labels(img, boxes, rect, FLT_EPSILON_)
        
        channel_stride = 4 
        channels = self.__num_classes * channel_stride
        
        foreground_labels = np.zeros((self.__num_classes, boxes.shape[0], boxes.shape[1])) # 1
        boxes_labels = np.zeros((channels, boxes.shape[0], boxes.shape[1])) # 4
        size_labels = np.zeros((channels, boxes.shape[0], boxes.shape[1])) # 2
        obj_labels = np.zeros((channels, boxes.shape[0], boxes.shape[1])) # 1
        coverage_label = np.zeros((channels, boxes.shape[0], boxes.shape[1])) # 1

        k = int(label * channel_stride)

        for j in xrange(0, region_labels.shape[0], 1):
            for i in xrange(0, region_labels.shape[1], 1):
                if region_labels[j, i] == 1.0:
                    t = boxes[j, i]
                    box = np.array([rect[0] - t[0], rect[1] - t[1], (rect[0] + rect[2]) - t[0], (rect[1] + rect[3]) - t[1]])
                    boxes_labels[k + 0, j, i] =  box[0]
                    boxes_labels[k + 1, j, i] =  box[1]
                    boxes_labels[k + 2, j, i] =  box[2]
                    boxes_labels[k + 3, j, i] =  box[3]

                    size_labels[k + 0, j, i] = 1.0 / rect[2]
                    size_labels[k + 1, j, i] = 1.0 / rect[3]
                    size_labels[k + 2, j, i] = 1.0 / rect[2]
                    size_labels[k + 3, j, i] = 1.0 / rect[3]

                    diff = float(boxes[j, i][2] * boxes[j ,i][3]) / float(rect[2] * rect[3])
                    obj_labels[k:k+channel_stride, j, i] = diff

                    coverage_label[k:k+channel_stride, j, i] = region_labels[j, i]
                    
                    foreground_labels[int(label), j, i] = 1.0

        return (foreground_labels, boxes_labels, size_labels, obj_labels, coverage_label)
        

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
        
        scale_x = float(image.shape[1]) / float(rect[2])
        scale_y = float(image.shape[0]) / float(rect[3])
        scale_x = int(math.floor(scale_x))
        scale_y = int(math.floor(scale_y))
        
        flip_flag = random.randint(-1, 1)
        img_flip, rect_flip = self.flip_image(image.copy(), rect, flip_flag)
        
        return (img_flip, rect_flip)
        
        #!TODO
        ##! flip and save
        if scale_x < 3:
            scale_x = 3
        if scale_y < 3:
            scale_y = 3

        enlarge_factor1 = random.randint(2, scale_x)
        enlarge_factor2 = random.randint(2, scale_y)
        widths = (int(rect_flip[2] * enlarge_factor1), rect_flip[2] * enlarge_factor2)
        heights = (int(rect_flip[3] * enlarge_factor1), rect_flip[3] * enlarge_factor2)
        crop_image, crop_rect = self.crop_image_dimension(img_flip, rect_flip, widths, heights)


        ## apply blur
        kernel_x = random.randint(3, 7)
        kernel_y = random.randint(3, 7)
        kernel_x = kernel_x + 1 if kernel_x % 2 is 0 else kernel_x
        kernel_y = kernel_y + 1 if kernel_y % 2 is 0 else kernel_y

        blur_img = cv.GaussianBlur(crop_image, (kernel_x, kernel_y), 0)


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
            return image, rect

        x = min(pt1[0], pt2[0])
        y = min(pt1[1], pt2[1])
        w = np.abs(pt2[0] - pt1[0])
        h = np.abs(pt2[1] - pt1[1])

        x = 0 if x < 0 else x
        y = 0 if y < 0 else y

        flip_rect = [x, y, w, h]
        return im_flip, flip_rect

    """
    Function to label each grid boxes based on IOU score
    """     
    def generate_box_labels(self, image, boxes, rect, label, iou_thresh = FLT_EPSILON_):
        jc = JaccardCoeff()
        box_labels = np.zeros((boxes.shape[0], boxes.shape[1]))
        for j in xrange(0, boxes.shape[0], 1):
            for i in xrange(0, boxes.shape[1], 1):
                if jc.iou(boxes[j, i], rect) > iou_thresh:
                    box_labels[j, i] = FORE_PROB_
        return  box_labels


    """
    Function to overlay a grid on the image and return the boxes
    """     
    def grid_region(self, image, stride):
        wsize = (image.shape[0]/stride, image.shape[1]/stride)
        boxes = np.zeros((wsize[0], wsize[1], 4))
        for j in xrange(0, boxes.shape[0], 1):
            for i in xrange(0, boxes.shape[1], 1):
                boxes[j][i][0] = (i * stride)
                boxes[j][i][1] = (j * stride)
                boxes[j][i][2] = (stride)
                boxes[j][i][3] = (stride)
        return boxes

    """
    Function to demean rgb image using imagenet mean
    """
    def demean_rgb_image(self, im_rgb):
        im_rgb = im_rgb.astype(float)
        im_rgb[:, :, 0] -= float(104.0069879317889)
        im_rgb[:, :, 1] -= float(116.66876761696767)
        im_rgb[:, :, 2] -= float(122.6789143406786)
        im_rgb = (im_rgb - im_rgb.min())/(im_rgb.max() - im_rgb.min())
        return im_rgb

#!/usr/bin/env python

import caffe
import math
import random as rng
import numpy as np
import cv2 as cv
import argumentation_engine as ae

class DataArgumentationLayer(caffe.Layer):

    def setup(self, bottom, top):
        #! check that inputs are pair of image and labels (bbox and label:int)
        if len(bottom) != 2:
            raise Exception('Need two inputs for Argumentation')
        
        if len(top) < 6:
            raise Exception('Current Implementation needs 6 top blobs')

        try:
            plist = self.param_str.split(',')
            self.image_size_x = int(plist[0])
            self.image_size_y = int(plist[1])
            self.stride = int(plist[2])
            self.num_classes = int(plist[3])

            self.__ae = ae.ArgumentationEngine(self.image_size_x, self.image_size_y, self.stride, self.num_classes)

        except ValueError:
            raise ValueError('Parameter string missing or data type is wrong!')

            
    def reshape(self, bottom, top):
        #! check input dimensions
        if bottom[1].count < 5:
            raise Exception('Labels should be 5 dimensional vector')
                
        n_images = bottom[0].data.shape[0]        
        out_size_x = int(self.image_size_x / self.stride)
        out_size_y = int(self.image_size_y / self.stride)
        
        top[0].reshape(n_images, 3, self.image_size_y, self.image_size_x)
        top[1].reshape(n_images, 1, out_size_x, out_size_y) #! cvg labels  # 1
        top[2].reshape(n_images, 4, out_size_x, out_size_y) #! bbox labels # 4
        top[3].reshape(n_images, 4, out_size_x, out_size_y) #! size labels # 4
        top[4].reshape(n_images, 4, out_size_x, out_size_y) #! obj labels  # 4
        top[5].reshape(n_images, 4, out_size_x, out_size_y) #! cvg block   # 4
        
        
    def forward(self, bottom, top):
        for index, (data, labels) in enumerate(zip(bottom[0].data, bottom[1].data)):
            img = data.copy()
            rect = labels[0, 0, 0:4].copy()
            label = labels[0, 0, 4].copy()

            img, rect = self.__ae.random_argumentation(img, rect)
            img, rect = self.__ae.resize_image_and_labels(img, rect)
            foreground_labels, boxes_labels, size_labels, obj_labels, coverage_label = \
            self.__ae.bounding_box_parameterized_labels(img, rect, label, self.stride)
            
            img = img.swapaxes(2, 0)
            img = img.swapaxes(2, 1)
            
            top[0].data[index] = img
            top[1].data[index] = foreground_labels.copy()
            top[2].data[index] = boxes_labels.copy()
            top[3].data[index] = size_labels.copy()
            top[4].data[index] = obj_labels.copy()
            top[5].data[index] = coverage_label.copy()

    def backward(self, top, propagate_down, bottom):
        pass
        


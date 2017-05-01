#!/usr/bin/env python

###########################################################################
## Copyright (C) 2017 by Krishneel Chaudhary @ JSK Lab,
## The University of Tokyo, Japan
###########################################################################

import os
import sys
import math
import lmdb
import rospy
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


class CreateTrainingLMDB:
    def __init__(self):
        ## contains image path
        self.__data_textfile = '/home/krishneel/Desktop/dataset/train.txt'  
        self.__lmdb_labels = '/home/krishneel/Desktop/lmdb/labels'
        self.__lmdb_images = '/home/krishneel/Desktop/lmdb/features'

        if not os.path.isfile(str(self.__data_textfile)):
            rospy.logfatal('datatext filename not found')
            sys.exit()
            
        if not self.__lmdb_labels or not self.__lmdb_images:
            rospy.logfatal('Provide LMDB filename')
            sys.exit()


        self.__shuffle = rospy.get_param('~shuffle', True) ## shuffle dataset
        self.__num_random = rospy.get_param('~num_random', 10)  ## number of extra samples
        self.__min_size = rospy.get_param('~min_size', 20)  ## min box size

        ##! network size
        self.__net_size_x = rospy.get_param('~net_size_x', 448) ## network image cols
        self.__net_size_y = rospy.get_param('~net_size_y', 448) ## network image rows
        self.__stride = rospy.get_param('~stride', 16)  ## stride of the grid
        
        self.__num_classes = None  ## number of classes
        
        
        if (self.__net_size_x < 1) or (self.__stride < 16):
            rospy.logfatal('Incorrect netsize or stride')
            sys.exit()

        rospy.loginfo("running")
        self.process_data(self.__data_textfile)
        ## self.read_lmdb(self.__lmdb_labels)  ## inspection into data
        

    def process_data(self, path_to_txt):
        
        lines = self.read_textfile(path_to_txt)
        # lines = np.array(lines)
        np.random.shuffle(lines)

        img_path, rects, labels = self.decode_lines(lines)
        
        # get total class count
        label_unique, label_indices = np.unique(labels, return_index=False, return_inverse=True, return_counts=False)
        self.__num_classes = label_unique.shape[0]
        organized_label = label_indices

        if self.__num_classes is None:
            rospy.logfatal("Valid class label not found")
            sys.exit()


        ## write data
        map_size = 1e12
        lmdb_labels = lmdb.open(str(self.__lmdb_labels), map_size=int(map_size))
        lmdb_images = lmdb.open(str(self.__lmdb_images), map_size=int(map_size))
        with lmdb_labels.begin(write=True) as lab_db, lmdb_images.begin(write=True) as img_db:
            for index, ipath in enumerate(img_path):

                img = cv.imread(str(ipath))
                rect = rects[index]
                label = organized_label[index]

                print "processs: ", ipath, " ", img.shape

                widths = ([img.shape[1]/4, img.shape[1]/4])
                heights = ([img.shape[0]/4, img.shape[0]/4])
                img, rect = self.crop_image_dimension(img, rect, widths, heights)

                images, drects = self.random_argumentation(img, rect)
                
                for im, bb in zip(images, drects):
                    im2, bb2 = self.resize_image_and_labels(im, bb)
                    data_labels = self.pack_data(im2, bb2, label)

                    # ##! write labels
                    lab_datum = caffe.io.array_to_datum(data_labels)
                    lab_db.put('{:0>10d}'.format(index), lab_datum.SerializeToString())

                    im3 = im2.copy()
                    im3 = im3.swapaxes(2, 0)
                    im3 = im3.swapaxes(2, 1)
                    
                    im_datum = caffe.io.array_to_datum(im3)
                    img_db.put('{:0>10d}'.format(index), im_datum.SerializeToString())

                    # print "size: ", data_labels.shape, " ", im3.shape
                    

        lmdb_labels.close()
        lmdb_images.close()
        rospy.logwarn("done")
        return
            
    def pack_data(self, img, rect, label):

        foreground_labels, boxes_labels, size_labels, obj_labels, coverage_label = \
        self.bounding_box_parameterized_labels(img, rect, label, self.__stride)

        ## pack all in one data
        K = foreground_labels.shape[0] + boxes_labels.shape[0] + size_labels.shape[0] + obj_labels.shape[0] + coverage_label.shape[0]
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
        boxes = self.grid_region(img, self.__stride)
        region_labels = self.generate_box_labels(img, boxes, rect, FLT_EPSILON_)
        
        channel_stride = 4 
        channels = self.__num_classes * channel_stride
        
        foreground_labels = np.zeros((self.__num_classes, boxes.shape[0], boxes.shape[1])) # 1
        boxes_labels = np.zeros((channels, boxes.shape[0], boxes.shape[1])) # 4
        size_labels = np.zeros((channels, boxes.shape[0], boxes.shape[1])) # 2
        obj_labels = np.zeros((channels, boxes.shape[0], boxes.shape[1])) # 1
        coverage_label = np.zeros((channels, boxes.shape[0], boxes.shape[1])) # 1

        k = label * channel_stride

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
                    # obj_labels[k + 1, j, i] = diff
                    # obj_labels[k + 2, j, i] = diff
                    # obj_labels[k + 3, j, i] = diff

                    coverage_label[k:k+channel_stride, j, i] = region_labels[j, i]
                    # coverage_label[k + 1, j, i] = region_labels[j, i]
                    # coverage_label[k + 2, j, i] = region_labels[j, i]
                    # coverage_label[k + 3, j, i] = region_labels[j, i]
                    
                    foreground_labels[label, j, i] = 1.0

                    # print k, " ", boxes_labels[k:k+channel_stride, j, i]

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
        
        images = []
        rects = []
        ##! save original first
        images.append(image)
        rects.append(rect)

        scale_x = float(image.shape[1]) / float(rect[2])
        scale_y = float(image.shape[0]) / float(rect[3])
        scale_x = int(math.floor(scale_x))
        scale_y = int(math.floor(scale_y))

        # for i in xrange(-1, 2, 1):
        for i in xrange(-1, 0, 1):
            flip_flag = random.randint(-1, 1)
            img_flip, rect_flip = self.flip_image(image.copy(), rect, flip_flag)
            
            ##! save flipped 
            images.append(img_flip)
            rects.append(rect_flip)

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
            

            # img_flip = crop_image.copy()
            # rect_flip = crop_rect

            ## plot only
            # print image.shape
            # cv.rectangle(img_flip, (rect_flip[0], rect_flip[1]), \
            #              (rect_flip[0] + rect_flip[2], rect_flip[1] + rect_flip[3]), (0, 255, 0))
            # cv.namedWindow("img", cv.WINDOW_NORMAL)
            # cv.imshow("img", img_flip)
            # cv.waitKey(3)
            
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
        # box_labels = [
        #     FORE_PROB_ if jc.iou(box, rect) > iou_thresh
        #     else BACK_PROB_
        #     for box in boxes
        # ]
        
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
        
        # boxes = [
        #     np.array([i, j, stride, stride])
        #     for i in xrange(0, image.shape[1], stride)
        #     for j in xrange(0, image.shape[0], stride)
        #     if (j + stride <= image.shape[0]) and (i + stride <= image.shape[1])
        # ]
        
        boxes = np.zeros((wsize[0], wsize[1], 4))
        for j in xrange(0, boxes.shape[0], 1):
            for i in xrange(0, boxes.shape[1], 1):
                boxes[j][i][0] = (i * stride)
                boxes[j][i][1] = (j * stride)
                boxes[j][i][2] = (stride)
                boxes[j][i][3] = (stride)


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

    def read_lmdb(self, lmdb_fn):
        in_db = lmdb.open(str(lmdb_fn), readonly=True)
        with in_db.begin() as txn:
            raw_datum = txn.get(b'0000000000')
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(raw_datum)
            flat_x = np.fromstring(datum.data, dtype=np.float)
            print datum

def main(argv):
    try:
        rospy.init_node('create_training_lmdb', anonymous = True)
        ctl = CreateTrainingLMDB()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logfatal("ros error")
        pass

if __name__ == '__main__':
    main(sys.argv)

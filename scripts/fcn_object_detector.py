#!/usr/bin/env python

###########################################################################
## Copyright (C) 2017 by Krishneel Chaudhary @ JSK Lab,
## The University of Tokyo, Japan
###########################################################################

import rospy
import caffe
import os
import sys
import math
import numpy as np
import random
import cv2 as cv
import matplotlib.pylab as plt
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PolygonStamped as Rect
from geometry_msgs.msg import Polygon, Point32

from region_cnn_detector import RCNNDetector

class FCNObjectDetector():

    def __init__(self):
        self.__net = None
        self.__transformer = None
        self.__im_width = None
        self.__im_height = None
        self.__bridge = CvBridge()

        # self.__rcnn = RCNNDetector()

        self.__prob_thresh = rospy.get_param('~detection_threshold', 0.5)  #! threshold for masking the detection
        self.__min_bbox_thresh = rospy.get_param('~min_boxes', 3) #! minimum bounding box
        self.__group_eps_thresh = rospy.get_param('~nms_eps', 0.2) #! bbox grouping        
        self.__weights = rospy.get_param('~pretrained_weights', None)
        self.__model_proto = rospy.get_param('~deployment_prototxt', None)
        self.__device_id = rospy.get_param('device_id', 0)

        if self.is_file_valid():
            self.load_caffe_model()
            rospy.loginfo('DETECTOR SETUP SUCCESSFUL')

        ##! publisher setup
        self.pub_box = rospy.Publisher('/fcn_object_detector/rects', Rect, queue_size = 1)
            
        self.subscribe()

    def run_detector(self, image_msg):
        cv_img = None
        try:
            cv_img = self.__bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except Exception as e:
            print (e)
            return
        
        if cv_img is None:
            return

        #random.seed()
        input_image = cv_img.copy()
        
        caffe.set_device(self.__device_id)
        caffe.set_mode_gpu()

        # cv_img = self.demean_rgb_image(cv_img)
        # cv_img = cv.resize(cv_img, (self.__im_width, self.__im_height))
        self.__net.blobs['data'].data[...] = self.__transformer.preprocess('data', cv_img)
        output = self.__net.forward()

        probability_map = self.__net.blobs['coverage'].data[0]
        bbox_map = self.__net.blobs['bboxes'].data[0]

        # self.vis_square(probability_map)

        object_boxes = []
        label_color = []
        object_labels = []
        for i in xrange(0, 10, 1):
            r = random.random() * 255
            g = random.random() * 255
            b = random.random() * 255
            label_color.append((b, g, r))
            
        
        for index, p_map in enumerate(probability_map):
            idx = index * 4
            propose_boxes, propose_cvgs, mask = self.gridbox_to_boxes(p_map, bbox_map[idx:idx+4], \
                                                                      self.__prob_thresh)
            obj_boxes = self.vote_boxes(propose_boxes, propose_cvgs, mask)
            if obj_boxes:
                ## create unique color
                r = random.random() * 255
                g = random.random() * 255
                b = random.random() * 255

                for box in obj_boxes:
                    #label_color.append((b, g, r))
                    object_boxes.append(box)
                    object_labels.append(index)

                # print index
            
        label_color = np.asarray(label_color, dtype=np.float)
        object_boxes = np.asarray(object_boxes, dtype=np.int)
        object_labels = np.asarray(object_labels, dtype=np.int)

        print object_labels
        
        if not len(object_boxes):
            rospy.logwarn("not detection")
            return

        object_boxes = self.resize_detection(input_image.shape, object_boxes)

        ## give results to rcnn
        # object_boxes, object_labels = self.__rcnn.run_detector(input_image, object_boxes)
        # self.__rcnn.run_detector(input_image, object_boxes)

        if object_labels.shape[0] != object_boxes.shape[0]:
            rospy.logwarn("incorrect size label and rects")
            return

        cv_img = cv.resize(input_image.copy(), (input_image.shape[1], input_image.shape[0]))
        im_out = cv_img.copy()
        [
            [
                cv.rectangle(cv_img, (box[0], box[1]), (box[2], box[3]), label_color[label-1], -1),
                cv.rectangle(cv_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)
            ] for box, label in zip(object_boxes, object_labels)
        ]

        ##! publish
        is_publish = False
        if is_publish:
            rects = Rect()
            for box in object_boxes:
                pt = Point32()
                pt.x = box[0]
                pt.y = box[1]
                rects.polygon.points.append(pt)
                pt = Point32()
                pt.x = box[2]
                pt.y = box[3]
                rects.polygon.points.append(pt)

            rects.header = image_msg.header
            self.pub_box.publish(rects)

        alpha = 0.3
        cv.addWeighted(im_out, alpha, cv_img, 1.0 - alpha, 0, im_out)
    
        cv.namedWindow('detection', cv.WINDOW_NORMAL)
        cv.imshow('detection', im_out)
        # cv.imshow('detection', cv_img)
        cv.waitKey(3)


    def callback(self, image_msg):
        self.run_detector(image_msg)


    """
    function to load caffe models and pretrained weights
    """
    def load_caffe_model(self):
        rospy.loginfo('LOADING CAFFE MODEL..')
        self.__net = caffe.Net(self.__model_proto, self.__weights, caffe.TEST)
        
        self.__transformer = caffe.io.Transformer({'data': self.__net.blobs['data'].data.shape})
        self.__transformer.set_transpose('data', (2,0,1))
        self.__transformer.set_raw_scale('data', 1)
        self.__transformer.set_channel_swap('data', (2,1,0))

        shape = self.__net.blobs['data'].data.shape
        self.__im_height = shape[2]
        self.__im_width = shape[3]

        self.__net.blobs['data'].reshape(1, 3, self.__im_height, self.__im_width)

    def subscribe(self):
        rospy.Subscriber('image', Image, self.callback, tcp_nodelay=True)


    """
    code copied from nvidia detectnet for NMS
    """
    def vote_boxes(self, propose_boxes, propose_cvgs, mask):
        detections_per_image = []
        if not propose_boxes.any():
            return detections_per_image
        nboxes, weights = cv.groupRectangles(np.array(propose_boxes).tolist(),
                                             self.__min_bbox_thresh, self.__group_eps_thresh)
            
        if len(nboxes):
            for rect, weight in zip(nboxes, weights):
                if (rect[3] - rect[1]) >= 20:
                    confidence = math.log(weight[0])
                    detection = [rect[0], rect[1], rect[2], rect[3], confidence]
                    detections_per_image.append(detection)
                    
        return detections_per_image  


    """
    code copied from nvidia detectnet for transforming the grids to boxes
    """
    def gridbox_to_boxes(self, net_cvg, net_boxes, prob_thresh):
        im_sz_x = self.__im_width
        im_sz_y = self.__im_height
        stride = 16

        grid_sz_x = int(im_sz_x / stride)
        grid_sz_y = int(im_sz_y / stride)

        boxes = []
        cvgs = []

        cell_width = im_sz_x / grid_sz_x
        cell_height = im_sz_y / grid_sz_y

        cvg_val = net_cvg[0:grid_sz_y][0:grid_sz_x]

        mask = (cvg_val >= prob_thresh)
        coord = np.where(mask == 1)

        y = np.asarray(coord[0])
        x = np.asarray(coord[1])
    
        mx = x * cell_width
        my = y * cell_height

        x1 = (np.asarray([net_boxes[0][y[i]][x[i]] for i in xrange(x.size)]) + mx)
        y1 = (np.asarray([net_boxes[1][y[i]][x[i]] for i in xrange(x.size)]) + my)
        x2 = (np.asarray([net_boxes[2][y[i]][x[i]] for i in xrange(x.size)]) + mx)
        y2 = (np.asarray([net_boxes[3][y[i]][x[i]] for i in xrange(x.size)]) + my)
        
        boxes = np.transpose(np.vstack((x1, y1, x2, y2)))
        cvgs = np.transpose(np.vstack((x, y, np.asarray(
            [
                cvg_val[y[i]][x[i]] 
                for i in xrange(x.size)
            ]
        ))))
        return boxes, cvgs, mask

    def resize_detection(self, in_size, bbox):
        diffx = float(in_size[1])/float(self.__im_width)
        diffy = float(in_size[0])/float(self.__im_height)
        resize_bbox = bbox
        for index, box in enumerate(bbox):
            resize_bbox[index, 0] = box[0] * diffx
            resize_bbox[index, 1] = box[1] * diffy
            resize_bbox[index, 2] = box[2] * diffx
            resize_bbox[index, 3] = box[3] * diffy 
        return resize_bbox

    def demean_rgb_image(self, im_rgb):
        im_rgb = im_rgb.astype(float)
        im_rgb[:, :, 0] -= float(104.0069879317889)
        im_rgb[:, :, 1] -= float(116.66876761696767)
        im_rgb[:, :, 2] -= float(122.6789143406786)
        im_rgb = (im_rgb - im_rgb.min())/(im_rgb.max() - im_rgb.min())
        return im_rgb


    def vis_square(self, data):
        data = (data - data.min()) / (data.max() - data.min())
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = (((0, n ** 2 - data.shape[0]),
                    (0, 1), (0, 1)) + ((0, 0),) * (data.ndim - 3))
        data = np.pad(data, padding, mode='constant', constant_values=1)
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) +
                                                               tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
        plt.imshow(data); plt.axis('off')
        plt.pause(0.09)

    def is_file_valid(self):
        if self.__model_proto is None or \
           self.__weights is None:
            rospy.logfatal('PROVIDE PRETRAINED MODEL! KILLING NODE...')
            return False
        
        is_file = lambda path : os.path.isfile(str(path))
        if  (not is_file(self.__model_proto)) or (not is_file(self.__weights)):
            rospy.logfatal('NOT SUCH FILES')
            return False

        return True


def main(argv):
    try:
        rospy.init_node('fcn_object_detector', anonymous = True)
        fod = FCNObjectDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logfatal("ros error")
        pass

    
if __name__ == "__main__":
    main(sys.argv)

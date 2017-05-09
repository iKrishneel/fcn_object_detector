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
import message_filters
import matplotlib.pylab as plt
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PolygonStamped as Rect

class RCNNDetector:
    
    def __init__(self):
        print "done"
        self.__net = None
        self.__transformer = None
        self.__im_width = None
        self.__im_height = None
        self.__bridge = CvBridge()

        self.__prob_thresh = rospy.get_param('~detection_threshold', 0.5)  #! threshold for masking the detection
        self.__weights = rospy.get_param('~pretrained_weights', None)
        self.__model_proto = rospy.get_param('~deployment_prototxt', None)
        self.__device_id = rospy.get_param('device_id', 0)

        # ## temp
        dire = "/home/krishneel/nvcaffe/jobs/region_cnn11/"
        self.__weights= dire + "snapshots/snapshot_iter_500.caffemodel"
        self.__model_proto = dire + "deploy.prototxt"
        
        # ## NMS
        # self.__min_bbox_thresh = rospy.get_param('~min_boxes', 3) #! minimum bounding box
        # self.__group_eps_thresh = rospy.get_param('~nms_eps', 0.2) #! bbox grouping        
        
        if self.is_file_valid():
            self.load_caffe_model()
            rospy.loginfo('DETECTOR SETUP SUCCESSFUL')
        else:
            rospy.logfatal("[RCNN DETECTOR:] ERROR SETUP FAILED")
            sys.exit()

        # self.subscribe()


    def run_detector(self, image, rects):
        if len(image.shape) < 3 or len(rects) < 1:
            rospy.loginfo("EMPTY INPUT DATA")
            return
        labels = []
        bboxes = []
        print 
        for index, rect in enumerate(rects):
            x1 = rect[0]
            y1 = rect[1]
            x1 = 0 if x1 < 0 else x1
            y1 = 0 if y1 < 0 else y1
            if rect[2] > 16 and rect[3] > 16:
                im_roi = image[y1:rect[3], x1:rect[2]].copy()
                im_roi = self.demean_rgb_image(im_roi)
                self.__net.blobs['data'].data[...] = self.__transformer.preprocess('data', im_roi)
                output = self.__net.forward()
                output_prob = output['prob']
                
                if (output_prob.argmax() > 0):
                    print "label: ", output_prob.max(), " ", output_prob.argmax()
                    bboxes.append(rect)
                    labels.append(output_prob.argmax())
                

        object_boxes = np.asarray(bboxes, dtype=np.int)
        labels = np.array(labels)
        return  (object_boxes, labels)

    def callback(self, image_msg, rect_msg):
        cv_img = None
        try:
            cv_img = self.__bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except Exception as e:
            print (e)
            return
        
        if cv_img is None:
            return
        
        caffe.set_device(self.__device_id)
        caffe.set_mode_gpu()
        
        if len(rect_msg.polygon.points) < 1:
            rospy.logwarn("No object proposal received")
            return


    def subscribe(self):
        image_sub = message_filters.Subscriber('image', Image)
        rect_sub = message_filters.Subscriber('rects', Rect)
        ts = message_filters.TimeSynchronizer([image_sub, rect_sub], 10)
        ts.registerCallback(self.callback)

    def demean_rgb_image(self, im_rgb):
        im_rgb = im_rgb.astype(float)
        im_rgb[:, :, 0] -= float(104.0069879317889)
        im_rgb[:, :, 1] -= float(116.66876761696767)
        im_rgb[:, :, 2] -= float(122.6789143406786)
        im_rgb = (im_rgb - im_rgb.min())/(im_rgb.max() - im_rgb.min())
        return im_rgb


    """
    function to load caffe models and pretrained __weights
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


    def is_file_valid(self):
        if self.__model_proto is None or self.__weights is None:
            rospy.logfatal('PROVIDE PRETRAINED MODEL! KILLING NODE...')
            return False
        
        is_file = lambda path : os.path.isfile(str(path))
        if  (not is_file(self.__model_proto)) or (not is_file(self.__weights)):
            rospy.logfatal('NOT SUCH FILES')
            return False

        return True


# def main(argv):
#     try:
#         rospy.init_node('region_cnn_detector', anonymous = False)
#         rd = RCNNDetector()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         rospy.logfatal("ros error")
#         pass

    
# if __name__ == "__main__":
#     main(sys.argv)

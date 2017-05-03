#!/usr/bin/env python

import rospy
import caffe
import os
import sys
import math
import numpy as np
import cv2 as cv
import random
import matplotlib.pylab as plt

class RankObjectProposal:
    def __init__(self):
        caffe_root = os.environ['CAFFE_ROOT'] + '/'
        if caffe_root is None:
            rospy.logfatal('CAFFE_ROOT NOT FOUND')
            sys.exit()
        
        prototxt = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
        caffemodel = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

        ##! image input
        self.__transformer = None
        self.__im_width = None
        self.__im_height = None
        ##! network info
        self.__net = None
        self.__model_proto = rospy.get_param('~deployment_prototxt', prototxt)
        self.__weights = rospy.get_param('~pretrained_weights', caffemodel)
        self.__feature = 'fc7'
        
        ##! similarity index
        self.__similarity_distance = rospy.get_param('~similarity_distance', 0.4)

        self.__distance_metric = None
        (major, minor, _) = cv.__version__.split(".")
        if major < str(3):
            self.__distance_metric = cv.cv.CV_COMP_BHATTACHARYYA
        elif major >= str(3):
            self.__distance_metric = cv.HISTCMP_BHATTACHARYYA
        
        if self.is_file_valid():
            self.load_caffe_model()
            rospy.loginfo('DETECTOR SETUP SUCCESSFUL')
        else:
            rospy.logfatal('CAFFE NETWORK MODELS NOT FOUND!')
            sys.exit()

        self.__dataset_labels = rospy.get_param('~dataset_labels', None)
        self.__dataset_lists = rospy.get_param('~dataset_labels', None)
        
        self.__dataset_labels = "/home/krishneel/Desktop/dataset/oreo/"
        self.__dataset_lists = "/home/krishneel/Desktop/dataset/oreo/train.txt"

        if self.__dataset_labels is None or self.__dataset_lists is None:
            rospy.logfatal('PROVIDE DATASET LABELS AND LIST!')
            sys.exit()

        ### label_folder/ -> image.jpg + list.txt

        self.rank_proposals()

    def rank_proposals(self):
        
        caffe.set_device(0)
        caffe.set_mode_gpu()
        
        # labels = self.read_textfile(self.__dataset_labels)
        lists = self.read_textfile(self.__dataset_lists)
        
        labels = []
        labels.append('juice')
        
        img_path, rects, labels = self.decode_lines(lists)
        
        ##! first one is the known good template
        im_templ = cv.imread(str(img_path[0]))
        rect = rects[0]
        im_templ = im_templ[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

        feature_map = self.get_cnn_codes(im_templ)
        update_rate = 0.2
        count = 0

        for i in xrange(1, len(img_path) - 1, 1):
            print i
            im_roi1 = cv.imread(str(img_path[i]))
            rect = rects[i]
            im_roi1 = im_roi1[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
            
            im_roi2 = cv.imread(str(img_path[i+1]))
            rect = rects[i+1]
            im_roi2 = im_roi2[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

            ## relation templ->i,templ->(i+1), i->i+1
            feature_map1 = self.get_cnn_codes(im_roi1)
            feature_map2 = self.get_cnn_codes(im_roi2)

            
            sa = cv.compareHist(feature_map, feature_map1, self.__distance_metric)
            sb = cv.compareHist(feature_map, feature_map2, self.__distance_metric)
            sc = cv.compareHist(feature_map1, feature_map2, self.__distance_metric)
        
            if sa < self.__similarity_distance and \
               sb < self.__similarity_distance and \
               sc < self.__similarity_distance:
                feature_map = feature_map * (1.0 - update_rate) + feature_map1 * update_rate

            else:
                count += 1
                rospy.logwarn('dissimilar: %s', img_path[i])
                print "dist: ", sa, " ", sb, " ", sc

            # plt.plot(feature_map)
            # plt.pause(0.09)

            cv.imshow("img", im_roi1)
            cv.imshow("img2", im_roi2)
            cv.waitKey(0)
        print "TOTAL REJECTED: ", count

    def get_cnn_codes(self, image):
        self.__net.blobs['data'].data[...] = self.__transformer.preprocess('data', image)
        output = self.__net.forward()
        feature_map = self.__net.blobs[self.__feature].data[0]
        return feature_map.copy()

    def read_textfile(self, path_to_txt):
        lines = [line.rstrip('\n')
                 for line in open(path_to_txt)
        ]
        return lines
    
    
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
        plt.show()

    
    def load_caffe_model(self):
        rospy.loginfo('LOADING CAFFE MODEL..')
        self.__net = caffe.Net(self.__model_proto, self.__weights, caffe.TEST)
        
        self.__transformer = caffe.io.Transformer({'data': self.__net.blobs['data'].data.shape})
        self.__transformer.set_transpose('data', (2,0,1))
        self.__transformer.set_raw_scale('data', 255)
        self.__transformer.set_channel_swap('data', (2,1,0))

        shape = self.__net.blobs['data'].data.shape
        self.__im_height = shape[2]
        self.__im_width = shape[3]

        self.__net.blobs['data'].reshape(1, 3, self.__im_height, self.__im_width)


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
        rospy.init_node('rank_object_models', anonymous = False)
        rom = RankObjectProposal()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logfatal("ros error")
        pass

if __name__ == '__main__':
    main(sys.argv)
        

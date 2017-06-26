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
import cv2 as cv
import random
import matplotlib.pylab as plt

from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors as kNN

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
            #self.__distance_metric = cv.HISTCMP_BHATTACHARYYA
            self.__distance_metric = cv.HISTCMP_CHISQR
        
        if self.is_file_valid():
            self.load_caffe_model()
            rospy.loginfo('SETUP SUCCESSFUL')
        else:
            rospy.logfatal('CAFFE NETWORK MODELS NOT FOUND!')
            sys.exit()

        
        self.__dataset_dir = rospy.get_param('~dataset_labels', None)
        self.__dataset_lists = rospy.get_param('~dataset_labels', None)

        self.__object_name = "cheezit/"
        self.__dataset_dir = "/media/volume/prev_data_backup/20170503/" + str(self.__object_name)
        self.__dataset_lists = self.__dataset_dir +  "train.txt"

        if self.__dataset_dir is None or self.__dataset_lists is None:
            rospy.logfatal('PROVIDE DATASET LABELS AND LIST!')
            sys.exit()

        ### label_folder/ -> image.jpg + list.txt

        ##! knn
        self.__is_normalize = rospy.get_param('normalize', True) ##! normalize feature space
        self.__knn = kNN(n_neighbors=2, algorithm='kd_tree', metric='minkowski', n_jobs = 8)
        self.__cnn_codes = None

        self.rank_proposals()

    def rank_proposals(self):
        
        caffe.set_device(0)
        caffe.set_mode_gpu()
        
        lists = self.read_textfile(self.__dataset_lists)
        
        labels = []
        labels.append(self.__object_name)
        
        img_path, rects, labels = self.decode_lines(lists)

        
        ##! clutering
        if not self.cluster_data(img_path, rects):
            rospy.logfatal('FAILED TO CREATE KNN... killing... ')
            sys.exit()
        
        ##! first one is the known good template
        im_templ = cv.imread(str(img_path[0]))
        rect = rects[0]
        im_templ = im_templ[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

        feature_map = self.get_cnn_codes(im_templ)
        update_rate = 0.1
        count = 0

        print self.__knn.kneighbors(feature_map.reshape(1, -1))


        ##! write data
        write_data = False
        if write_data:
            text_file = open(str(self.__dataset_dir + "train2.txt"), "w")
        
        for i in xrange(1, len(img_path), 1):
            print i
            im_roi1 = cv.imread(str(img_path[i]))
            rect = rects[i]
            if rect[2] < 0 or rect[3] < 0 or len(im_roi1.shape) < 3:
                break
            im_roi1 = im_roi1[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
            
            im_roi2 = cv.imread(str(img_path[i-1]))
            rect = rects[i-1]

            if rect[2] < 0 or rect[3] < 0 or len(im_roi2.shape) < 3:
                break
            
            im_roi2 = im_roi2[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

            ## relation templ->i,templ->(i+1), i->i+1
            feature_map1 = self.get_cnn_codes(im_roi1)
            feature_map2 = self.get_cnn_codes(im_roi2)


            ##! get the best match
            distances, indices = self.__knn.kneighbors(feature_map1.reshape(1, -1))
            feature_map = self.__cnn_codes[indices[0][0]].copy()

            sa = cv.compareHist(feature_map.reshape(1, -1), feature_map1, self.__distance_metric)
            print indices, '\t', distances, "\t", sa

            
            print feature_map.sum(), " ", feature_map1.sum(), " ", feature_map2.sum()


            sb = cv.compareHist(feature_map.reshape(1, -1), feature_map2, self.__distance_metric)
            sc = cv.compareHist(feature_map1, feature_map2, self.__distance_metric)
        
            if sa < self.__similarity_distance and \
               sb < self.__similarity_distance and \
               sc < self.__similarity_distance:
                feature_map = feature_map * (1.0 - update_rate) + feature_map1 * update_rate

                if write_data:
                    text_file.write(img_path[i] + " " + str(rect[0]) + " " + str(rect[1]) +" "\
                                    + str(rect[2]) +" " + str(rect[3]) + " "+ str(labels[i]) + "\n")
            else:
                count += 1
                rospy.logwarn('dissimilar: %s', img_path[i])
                print "dist: ", sa, " ", sb, " ", sc

            # plt.plot(feature_map)
            # plt.pause(0.09)

            cv.imshow("img", im_roi1)
            cv.imshow("img2", im_roi2)
            
            key = cv.waitKey(3)                                                                    
            if key == ord('q'):                                                                   
                break


        print "TOTAL REJECTED: ", count
        
        if write_data:
            text_file.close()



    """
    Function to cluster all the pseudo-ground truth
    """
    def cluster_data(self, im_path, rects):

        rospy.loginfo('Extracting features')
        features = []
        for ipath, rect in zip(im_path, rects):
            im_rgb = cv.imread(str(ipath), cv.IMREAD_COLOR)
            x, y, w, h = rect
            f = self.get_cnn_codes(im_rgb[y:y+h, x:x+w].copy())
            if not math.isnan(f.sum()):
                features.append(f[0])
        features = np.array(features)

        rospy.loginfo('Feature extraction successfully completed')

        ##! clustering hyperparams
        epsilon = rospy.get_param('~eps', 0.250)
        min_samples = rospy.get_param('min_samples', 10)
        dbscan = DBSCAN(eps=epsilon, min_samples=10, n_jobs=8, algorithm='kd_tree')
        cluster_per_label = 2
        kmeans = KMeans(n_clusters=cluster_per_label, init='k-means++', n_init=10, \
                        max_iter=300, tol=0.0001,n_jobs=8)

        rospy.loginfo('Clustering feature space')
        dbscan.fit(features)
        labels = dbscan.labels_

        core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
        core_samples_mask[dbscan.core_sample_indices_] = True
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        unique_labels = set(labels)


        print unique_labels
        im_plot = None
        for index, l in enumerate(labels):
            if l > -1:
                im = cv.imread(im_path[index])
                im = cv.resize(im, (256/2, 256/2))
                if im_plot is None:
                    im_plot = im.copy()
                else:
                    w = im_plot.shape[1] + im.shape[1]
                    h = im_plot.shape[0] # + im.shape[0]

                    im_ = np.zeros((h, w, 3), np.uint8)
                    im_[0:im_plot.shape[0], 0:im_plot.shape[1], :] = im_plot
                    im_[0:im_.shape[0], im_plot.shape[1]:w, :] = im
                    im_plot = im_.copy()

        print im_plot.shape

        #! cut and reshape
        h, w, _ = im_plot.shape
        max_w = 768
        
        

        cv.imshow("plot", im_plot)
        cv.waitKey(0)
        sys.exit()

        ##! improve intra-cluster using kmeans
        train_features = []
        for label in unique_labels:
            if label > -1:
                group = features[(labels==label) & core_samples_mask]
                if group.shape[0] > 2:
                    kmeans.fit(group)
                    [
                        train_features.append(centroid)
                        for centroid in kmeans.cluster_centers_
                    ]
        train_features = np.array(train_features)
        train_features = train_features.astype(np.float32)
        if train_features.shape[0] == 0:
            return False

        self.__cnn_codes = train_features.copy()
        self.__knn.fit(train_features)

        rospy.loginfo('KNN SUCCESSFULLY CONSTRUCTURED')
        return True


    def get_cnn_codes(self, image):
        self.__net.blobs['data'].data[...] = self.__transformer.preprocess('data', image)
        output = self.__net.forward()
        feature_map = self.__net.blobs[self.__feature].data[0]
        feature_map = normalize(feature_map.reshape(1, -1), norm='l2', axis=1, copy=True, return_norm=False) \
                      if self.__is_normalize else feature_map
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
        

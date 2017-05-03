#!/usr/bin/env python

import sys
import os
import rospy
import rosbag
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped as Rect
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge

class ImageRectWriter:
    def __init__(self):
        self.__write_path = '/home/krishneel/Desktop/dataset/'
        self.__obj_name = 'tissue/'
        self.__text_filename = 'train.txt'
        self.__label = 4
        self.__bridge = CvBridge()
        self.__counter = 0

        if not os.path.exists(str(self.__write_path + self.__obj_name)):
            os.makedirs(str(self.__write_path + self.__obj_name))
        
        self.subscribe()

        

    def callback(self, image_msg, rect_msg):
        cv_img = None
        try:
            cv_img = self.__bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except Exception as e:
            print (e)
            return
        
        if cv_img is None:
            return

        if not len(rect_msg.polygon.points):
            return

        if len(rect_msg.polygon.points) < 2:
            rospy.loginfo("incorrect rect")
            return
        
        x1 = rect_msg.polygon.points[0].x
        y1 = rect_msg.polygon.points[0].y
        x2 = rect_msg.polygon.points[1].x
        y2 = rect_msg.polygon.points[1].y
        
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > cv_img.shape[1]:
            x2 = cv_img.shape[1]
        if y2 > cv_img.shape[0]:
            y2 = cv_img.shape[0]
        b = np.asarray([x1, y1, x2 - x1, y2 - y1], dtype=np.int)
    
        ## cv.rectangle(cv_img, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (0, 255, 0), 3)

        p =  self.__write_path + self.__obj_name + str(self.__counter).zfill(8) + '.jpg'
        cv.imwrite(p, cv_img)
        text_file = open(str(self.__write_path + self.__text_filename), "a")
        text_file2 = open(str(self.__write_path + self.__obj_name + self.__text_filename), "a")
        text_file.write(p + " " + str(b[0]) + " " + str(b[1]) +" "\
                        + str(b[2]) +" " + str(b[3]) + " "+ str(self.__label) + "\n")
        text_file2.write(p + " " + str(b[0]) + " " + str(b[1]) +" "\
                        + str(b[2]) +" " + str(b[3]) + " "+ str(self.__label) + "\n")
        text_file.close()
        text_file2.close()

        self.__counter += 1

        print p, " ", b

    def subscribe(self):
        image_sub = message_filters.Subscriber('/camera/rgb/image_rect_color', Image)
        rect_sub = message_filters.Subscriber('/object_rect', Rect)
        ts = message_filters.TimeSynchronizer([image_sub, rect_sub], 10)
        ts.registerCallback(self.callback)

def main(argv):
    try:
        rospy.init_node('image_rect_writer', anonymous = False)
        irw = ImageRectWriter()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logfatal("ros error")
        pass

if __name__ == "__main__":
    main(sys.argv)

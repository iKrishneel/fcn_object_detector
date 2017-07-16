
#pragma once
#ifndef _FCN_POINT_MAP_HPP_
#define _FCN_POINT_MAP_HPP_

#include <ros/ros.h>
#include <ros/console.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <cv_bridge/cv_bridge.h>

#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <jsk_recognition_utils/pcl_conversion_util.h>
#include <jsk_recognition_msgs/ModelCoefficientsArray.h>

#include <opencv2/opencv.hpp>

namespace jsk_msgs = jsk_recognition_msgs;

class FCNPointMap {
   
 private:
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::Normal NormalT;
    typedef pcl::PointXYZRGBNormal PointNormalT;
    typedef pcl::PointCloud<PointT> PointCloud;

    boost::mutex mutex_;
    boost::mutex lock_;
    ros::NodeHandle pnh_;

    typedef  message_filters::sync_policies::ApproximateTime<
       sensor_msgs::PointCloud2, sensor_msgs::Image,
       sensor_msgs::Image, jsk_msgs::ModelCoefficientsArray> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<sensor_msgs::Image> sub_pmap_;
    message_filters::Subscriber<sensor_msgs::Image> sub_mask_;
    message_filters::Subscriber<
      jsk_msgs::ModelCoefficientsArray> sub_mca_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    float rect_thresh_;
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();

    ros::Publisher pub_cloud_;
    ros::Publisher pub_indices_;
    ros::Publisher pub_app_grasp_;
    ros::Publisher pub_bbox_;
   
 public:
    FCNPointMap();
    void callback(const sensor_msgs::PointCloud2::ConstPtr &,
                  const sensor_msgs::Image::ConstPtr &,
                  const sensor_msgs::Image::ConstPtr &,
                  const jsk_msgs::ModelCoefficientsArrayConstPtr &);
    cv::Mat imageMsgToCvImage(const sensor_msgs::Image::ConstPtr &,
                              const std::string);
    void regionMask(std::vector<cv::Rect> &, const cv::Mat);
    float jaccardScore(cv::Rect, cv::Rect);
    std::vector<pcl_msgs::PointIndices> convertToROSPointIndices(
      const std::vector<pcl::PointIndices>,
      const std_msgs::Header&);
    void cluster(PointCloud::Ptr, std::vector<pcl::PointIndices> &);
};



#endif /* _FCN_POINT_MAP_HPP_ */

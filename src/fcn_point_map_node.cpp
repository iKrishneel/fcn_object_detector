
#include <fcn_object_detector/fcn_point_map.hpp>


FCNPointMap::FCNPointMap() : rect_thresh_(400) {
    ROS_INFO("RUNNING");
    this->onInit();
}

void FCNPointMap::onInit() {
    this->subscribe();
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/output/points", 1);
    this->pub_indices_ = this->pnh_.advertise<
       jsk_msgs::ClusterPointIndices>("/output/indices", 1);
}

void FCNPointMap::subscribe() {
    this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
    this->sub_pmap_.subscribe(this->pnh_, "input_pmap", 1);
    this->sub_mask_.subscribe(this->pnh_, "input_mask", 1);
    this->sub_mca_.subscribe(this->pnh_, "input_model", 1);
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
       SyncPolicy> >(100);
    this->sync_->connectInput(this->sub_cloud_, this->sub_mask_,
                              this->sub_pmap_, this->sub_mca_);
    this->sync_->registerCallback(
       boost::bind(&FCNPointMap::callback, this, _1, _2, _3, _4));
}

void FCNPointMap::unsubscribe() {
   
}

void FCNPointMap::callback(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const sensor_msgs::Image::ConstPtr &mask_msg,
    const sensor_msgs::Image::ConstPtr &prob_msg,
    const jsk_msgs::ModelCoefficientsArrayConstPtr &model_msg) {
    PointCloud::Ptr cloud(new PointCloud);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    cv::Mat pmap = this->imageMsgToCvImage(prob_msg, "mono8");
    cv::Mat obj_mask = this->imageMsgToCvImage(mask_msg, "mono8");

    
    
    // cv::cvtColor(obj_mask, obj_mask, cv::COLOR_BGR2GRAY);
    std::vector<cv::Rect> prects;
    this->regionMask(prects, pmap);

    
    std::vector<cv::Rect> orects;
    this->regionMask(orects, obj_mask);

    cv::Mat im_mask = cv::Mat::zeros(pmap.size(), CV_8UC1);
    for (int j = 0; j < orects.size(); j++) {
       for (int i = 0; i < prects.size(); i++) {
          float iou = this->jaccardScore(orects[j], prects[i]);
          if (iou > 0.0f) {
             cv::Rect rect = orects[j];
             for (int y = 0; y < rect.y + rect.height; y++) {
                for (int x = 0; x < rect.x + rect.width; x++) {
                   im_mask.at<uchar>(y, x) = obj_mask.at<uchar>(y, x);
                }
             }
          }
       }
    }
    
    cv::bitwise_xor(im_mask, obj_mask, im_mask);

    //! get region point cloud
    PointCloud::Ptr object_cloud(new PointCloud);
    std::vector<pcl::PointIndices> all_indices;
    int icounter = 0;
    for (int k = 0; k < orects.size(); k++) {
       cv::Rect rect = orects[k];
       pcl::PointIndices indices;
       for (int y = 0; y < rect.y + rect.height; y++) {
          for (int x = 0; x < rect.x + rect.width; x++) {
             if (static_cast<int>(im_mask.at<uchar>(y, x)) != 0) {
                int index = x + y * pmap.cols;
                object_cloud->push_back(cloud->points[index]);
                indices.indices.push_back(icounter++);
             }
          }
       }
       if (indices.indices.size() > 0) {
          all_indices.push_back(indices);
       }
    }

    this->cluster(object_cloud, all_indices);

    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*object_cloud, ros_cloud);
    ros_cloud.header = model_msg->header;
    jsk_msgs::ClusterPointIndices ros_indices;
    ros_indices.cluster_indices =
       this->convertToROSPointIndices(all_indices,
                                      model_msg->header);
    ros_indices.header = model_msg->header;
    pub_cloud_.publish(ros_cloud);
    pub_indices_.publish(ros_indices);
    
    cv::imshow("objects", im_mask);
    cv::waitKey(3);
}

void FCNPointMap::cluster(
    PointCloud::Ptr in_cloud,
    std::vector<pcl::PointIndices> & cluster_indices) {
    pcl::EuclideanClusterExtraction<PointT> ec;
    pcl::search::KdTree<PointT>::Ptr tree(
       new pcl::search::KdTree<PointT>);
    ec.setClusterTolerance(0.02);
    ec.setMinClusterSize(100);
    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(in_cloud);
    cluster_indices.clear();
    ec.extract(cluster_indices);
}


float FCNPointMap::jaccardScore(
    cv::Rect rect1, cv::Rect rect2) {
    cv::Rect in = rect1 & rect2;
    cv::Rect un = rect1 | rect2;
    return static_cast<float>(in.area()) / static_cast<float>(un.area());
}

void FCNPointMap::regionMask(
    std::vector<cv::Rect> &rects, const cv::Mat im_gray) {
    if (im_gray.empty()) {
      return;
    }
    cv::Mat im_thres;
    cv::threshold(im_gray, im_thres, 0, 255,
                  CV_THRESH_BINARY | CV_THRESH_OTSU);
    
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(im_thres, contours, hierarchy,
                     CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE,
                     cv::Point(0, 0));    
    rects.clear();
    for (int i = 0; i < contours.size(); i++) {
       cv::Rect r = cv::boundingRect(contours[i]);
       if (r.area() > this->rect_thresh_) {
          rects.push_back(r);
       }
    }
}


cv::Mat FCNPointMap::imageMsgToCvImage(
    const sensor_msgs::Image::ConstPtr &image_msg,
    const std::string encoding) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
       cv_ptr = cv_bridge::toCvCopy(image_msg, encoding);
       
    } catch (cv_bridge::Exception& e) {
       ROS_ERROR("cv_bridge exception: %s", e.what());
       return cv::Mat();
    }
    cv::Mat image = cv_ptr->image.clone();
    return image;
}

std::vector<pcl_msgs::PointIndices>
FCNPointMap::convertToROSPointIndices(
    const std::vector<pcl::PointIndices> cluster_indices,
    const std_msgs::Header& header) {
    std::vector<pcl_msgs::PointIndices> ret;
    for (size_t i = 0; i < cluster_indices.size(); i++) {
       pcl_msgs::PointIndices ros_msg;
       ros_msg.header = header;
       ros_msg.indices = cluster_indices[i].indices;
       ret.push_back(ros_msg);
    }
    return ret;
}

int main(int argc, char *argv[]) {

    ros::init(argc, argv, "fcn_point_map");
    FCNPointMap fpm;
    ros::spin();
    return 0;
}



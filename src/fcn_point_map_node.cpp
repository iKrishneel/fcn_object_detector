
#include <fcn_object_detector/fcn_point_map.hpp>


FCNPointMap::FCNPointMap() {
    ROS_INFO("RUNNING");
    this->onInit();
}

void FCNPointMap::onInit() {
    this->subscribe();
}

void FCNPointMap::subscribe() {
    this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
    this->sub_pmap_.subscribe(this->pnh_, "input_pmap", 1);
    this->sub_mask_.subscribe(this->pnh_, "input_mask", 1);
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
       SyncPolicy> >(100);
    this->sync_->connectInput(this->sub_cloud_,
                              this->sub_mask_, this->sub_pmap_);
    this->sync_->registerCallback(
       boost::bind(&FCNPointMap::callback, this, _1, _2, _3));
}

void FCNPointMap::unsubscribe() {
   
}

void FCNPointMap::callback(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const sensor_msgs::Image::ConstPtr &mask_msg,
    const sensor_msgs::Image::ConstPtr &prob_msg) {
    PointCloud::Ptr cloud(new PointCloud);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    cv::Mat pmap = this->imageMsgToCvImage(prob_msg, "mono8");
    cv::Mat obj_mask = this->imageMsgToCvImage(mask_msg, "mono8");

    // cv::cvtColor(obj_mask, obj_mask, cv::COLOR_BGR2GRAY);
    cv::threshold(obj_mask, obj_mask, 0, 255,
                  CV_THRESH_BINARY | CV_THRESH_OTSU);

    cv::cvtColor(obj_mask, obj_mask, cv::COLOR_GRAY2BGR);
    cv::Mat nobj_mask;
    cv::bitwise_xor(pmap, obj_mask, nobj_mask);

    
    
    cv::imshow("objects", nobj_mask);
    cv::waitKey(3);
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


int main(int argc, char *argv[]) {

    ros::init(argc, argv, "fcn_point_map");
    FCNPointMap fpm;
    ros::spin();
    return 0;
}



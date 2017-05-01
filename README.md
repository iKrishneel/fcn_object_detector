
# FCN_OBJECT_DETECTOR

#1. Info
ROS package for multiclass fully convolutional object detector using pretrained network.

#2. Building the System
###2.1 Requirements
 - [ROS Indigo or Kinetic](http://wiki.ros.org/kinetic)
 - [CUDA (8.0 or 7.5)](https://developer.nvidia.com/cuda-downloads)
 - [OpenCV](https://github.com/opencv/opencv)
 - [image_view] (https://github.com/ros-perception/image_pipeline)
 - [Caffe] (https://github.com/BVLC/caffe) 
 
###2.1 Downloading
Use standard git tools
```
  $ git clone https://github.com/iKrishneel/fcn_object_detector.git

```


###2.3 Build Process
To compile the system, use the standard catkin build on ros environment:
```
  $ catkin build fcn_object_detector
```

#3. Running Nodes
###3.1 Running Detector Node
 ```
 $ roslaunch fcn_object_detector fcn_object_deteor.launch image:=/camera/rgb/image_rect_color
```

# 4 Test data

Download the weights and rosbag file for testing
https://drive.google.com/a/jsk.imi.i.u-tokyo.ac.jp/file/d/0B5hRAGKTOm_KcEZ1Q0U1S011U3c/view?usp=sharing
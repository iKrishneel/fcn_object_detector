
# fcn_object_detector

# 1. Info
ROS package for multiclass fully convolutional object detector using pretrained network.

# 2. Building the System
### 2.1 Requirements
 - [ROS Indigo or Kinetic](http://wiki.ros.org/kinetic)
 - [CUDA (8.0 or 7.5)](https://developer.nvidia.com/cuda-downloads)
 - [OpenCV](https://github.com/opencv/opencv)
 - [image_view](https://github.com/ros-perception/image_pipeline)
 - [Caffe](https://github.com/BVLC/caffe) 
 
### 2.1 Downloading
Use standard git tools
```
  $ git clone https://github.com/iKrishneel/fcn_object_detector.git

```


### 2.3 Build Process
To compile the system, use the standard catkin build on ros environment:
```
  $ catkin build fcn_object_detector
```

# 3. Running Nodes
### 3.1 Running Detector Node
 ```
 $ roslaunch fcn_object_detector fcn_object_deteor.launch image:=/camera/rgb/image_rect_color
```

# 4 Test data

[Download the rosbag file for testing](https://drive.google.com/drive/folders/0B5hRAGKTOm_KQ0lLWmNaSjBwV2s?usp=sharing)


# Training
 - Add the data argumentation layer to the system wide environmental variable
```
  $ export PYTHONPATH=./fcn_object_detector/scripts/data_argumentation_layer:$PYTHONPATH
```
- Create LMDB of the input datasets. The dataset should contain images, bounding boxes and labels. The bounding boxes and labels has to be in .txt file in following format. Note x and y are top left hand corner coordinate and currently it only supports one bounding box per image

```
  /path/to/images x, y, width, height, label 
```

- To create the LMDB run the following command. Make sure to set path to the dataset folder containing the training set
```
  $ roslaunch fcn_object_detector create_training_lmdb.launch
```
This command will create LMDB folder with two file `features` and `labels` where features contains images and labels contains the bounding box coordinates and class label of the object in the image

- In the `train_val.prototxt` file add the data argumentation layer for end-to-end training

```
layer {
  type: 'Python'
  name: 'Argumentation'
  top: "data"
  top: "coverage-label"
  top: "bbox-label"
  top: "size-block"
  top: "obj-block"
  top: "coverage-block"
  bottom: 'data_in'
  bottom: 'label'
  python_param {
      module: 'data_argumentation_layer'
    layer: 'DataArgumentationLayer'
    param_str : '448, 448,16, 1'
    }
 }
```
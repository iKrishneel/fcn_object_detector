#!/usr/bin/env bash

dir=snapshots
if [[ ! -e $dir ]]; then
    mkdir $dir
fi

export CAFFE_ROOT=/home/krishneel/nvcaffe
export PYTHONPATH=$CAFFE_ROOT/python:$PYTHONPATH
pkg_directory=$PWD/..
export PYTHONPATH=$pkg_directory/scripts/data_argumentation_layer:$PYTHONPATH
$CAFFE_ROOT/build/tools/caffe train --solver=solver.prototxt \
    --gpu=0 \
    --weights=/home/krishneel/caffe/models/ilsvrc14/vgg16/VGG_ILSVRC_16_layers.caffemodel \
    2>&1 | tee -a $HOME/.ros/fcn_object_detector_voc.log

#!/usr/bin/env bash

dir=snapshots
if [[ ! -e $dir ]]; then
    mkdir $dir
fi

export CAFFE_ROOT=/home/krishneel/caffe
export PYTHONPATH=$CAFFE_ROOT/python:$PYTHONPATH
pkg_directory=$PWD/..
export PYTHONPATH=$pkg_directory/scripts/data_argumentation_layer:$PYTHONPATH
$CAFFE_ROOT/build/tools/caffe train --solver=semantic_segmentation/solver.prototxt \
    --gpu=0 \
    --weights=/home/krishneel/Documents/programs/fcn.berkeleyvision.org/voc-fcn8s/fcn8s-heavy-pascal.caffemodel \
    2>&1 | tee -a $HOME/.ros/fcn_object_detector.log

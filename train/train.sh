#!/usr/bin/env bash

pkg=$HOME/.ros/fcn_object_detector_log

if [[ ! -e $pkg ]]; then
    mkdir $pkg
fi

dir=snapshots
if [[ ! -e $dir ]]; then
    mkdir $dir
fi

dir=snapshots/labels
if [[ ! -e $dir ]]; then
    mkdir $dir
fi

export CAFFE_ROOT=/home/krishneel/nvcaffe
export PYTHONPATH=$CAFFE_ROOT/python:$PYTHONPATH
pkg_directory=$PWD/..
export PYTHONPATH=$pkg_directory/scripts/data_argumentation_layer:$PYTHONPATH
fname=$(date "+%Y-%m-%d-%H.%M-%S")

$CAFFE_ROOT/build/tools/caffe train --solver=semantic_segmentation/solver.prototxt \
    --gpu=0 \
    --snapshot=snapshots/snapshot_iter_17093.solverstate
    2>&1 | tee -a $pkg/fcn_object_detector_$fname.log

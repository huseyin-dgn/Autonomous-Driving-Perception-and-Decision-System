#!/usr/bin/env bash
set -e

cd "/home/huseyindgn/Masaüstü/Autonomous-Driving-Perception-and-Decision-System/autonomous_driving"

source /opt/ros/humble/setup.bash
source install/setup.bash

env -u WAYLAND_DISPLAY \
QT_QPA_PLATFORM=xcb \
GDK_BACKEND=x11 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
CUDA_VISIBLE_DEVICES=0 \
RAW_CONF_THRESHOLD=0.05 \
PERSON_CONF_THRESHOLD=0.25 \
VEHICLE_CONF_THRESHOLD=0.65 \
MOTORCYCLE_CONF_THRESHOLD=0.45 \
TRAFFIC_LIGHT_CONF_THRESHOLD=0.25 \
TRAFFIC_SIGN_CONF_THRESHOLD=0.40 \
YOLO_IMGSZ=960 \
YOLO_IOU=0.50 \
YOLO_MAX_DET=50 \
bash scripts/run_perception_carla.sh

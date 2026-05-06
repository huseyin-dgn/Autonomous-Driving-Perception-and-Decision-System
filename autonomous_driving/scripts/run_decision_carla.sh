#!/usr/bin/env bash

cd "/home/huseyindgn/Masaüstü/Autonomous-Driving-Perception-and-Decision-System/autonomous_driving"

source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run autonomous_driving decision_node

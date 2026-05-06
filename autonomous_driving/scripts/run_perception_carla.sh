#!/usr/bin/env bash
set -e

MODEL_PATH="${MODEL_PATH:-/home/huseyindgn/Masaüstü/Autonomous-Driving-Perception-and-Decision-System/autonomous_driving/outputs/models/adas5_targeted_aug_finetune_from_old_img1024_b8_ep50/weights/best.pt}"

IMAGE_TOPIC="${IMAGE_TOPIC:-/adas/camera/front/image_raw}"
DETECTIONS_TOPIC="${DETECTIONS_TOPIC:-/adas/perception/detections_json}"
ANNOTATED_TOPIC="${ANNOTATED_TOPIC:-/adas/perception/annotated_image}"

YOLO_IMGSZ="${YOLO_IMGSZ:-640}"
YOLO_IOU="${YOLO_IOU:-0.50}"
YOLO_MAX_DET="${YOLO_MAX_DET:-100}"

RAW_CONF_THRESHOLD="${RAW_CONF_THRESHOLD:-0.005}"
PERSON_CONF_THRESHOLD="${PERSON_CONF_THRESHOLD:-0.02}"
VEHICLE_CONF_THRESHOLD="${VEHICLE_CONF_THRESHOLD:-0.25}"
MOTORCYCLE_CONF_THRESHOLD="${MOTORCYCLE_CONF_THRESHOLD:-0.05}"
TRAFFIC_LIGHT_CONF_THRESHOLD="${TRAFFIC_LIGHT_CONF_THRESHOLD:-0.10}"
TRAFFIC_SIGN_CONF_THRESHOLD="${TRAFFIC_SIGN_CONF_THRESHOLD:-0.20}"

SHOW_DEBUG="${SHOW_DEBUG:-true}"

echo "=========================================="
echo "ADAS CARLA PERCEPTION - FULL/LITE TEST"
echo "MODEL: $MODEL_PATH"
echo "IMAGE_TOPIC: $IMAGE_TOPIC"
echo "YOLO_IMGSZ: $YOLO_IMGSZ"
echo "YOLO_IOU: $YOLO_IOU"
echo "YOLO_MAX_DET: $YOLO_MAX_DET"
echo "RAW_CONF_THRESHOLD: $RAW_CONF_THRESHOLD"
echo "PERSON_CONF_THRESHOLD: $PERSON_CONF_THRESHOLD"
echo "VEHICLE_CONF_THRESHOLD: $VEHICLE_CONF_THRESHOLD"
echo "MOTORCYCLE_CONF_THRESHOLD: $MOTORCYCLE_CONF_THRESHOLD"
echo "TRAFFIC_LIGHT_CONF_THRESHOLD: $TRAFFIC_LIGHT_CONF_THRESHOLD"
echo "TRAFFIC_SIGN_CONF_THRESHOLD: $TRAFFIC_SIGN_CONF_THRESHOLD"
echo "=========================================="

ros2 run autonomous_driving perception_node \
  --ros-args \
  -p image_topic:="$IMAGE_TOPIC" \
  -p detections_topic:="$DETECTIONS_TOPIC" \
  -p annotated_topic:="$ANNOTATED_TOPIC" \
  -p model_path:="$MODEL_PATH" \
  -p raw_conf_threshold:="$RAW_CONF_THRESHOLD" \
  -p person_conf_threshold:="$PERSON_CONF_THRESHOLD" \
  -p vehicle_conf_threshold:="$VEHICLE_CONF_THRESHOLD" \
  -p motorcycle_conf_threshold:="$MOTORCYCLE_CONF_THRESHOLD" \
  -p traffic_light_conf_threshold:="$TRAFFIC_LIGHT_CONF_THRESHOLD" \
  -p traffic_sign_conf_threshold:="$TRAFFIC_SIGN_CONF_THRESHOLD" \
  -p iou_threshold:="$YOLO_IOU" \
  -p imgsz:="$YOLO_IMGSZ" \
  -p max_det:="$YOLO_MAX_DET" \
  -p show_debug:="$SHOW_DEBUG"

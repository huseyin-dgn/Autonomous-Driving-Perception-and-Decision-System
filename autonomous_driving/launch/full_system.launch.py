from pathlib import Path
import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    world_path = Path(
        "/home/huseyindgn/Masaüstü/Autonomous-Driving-Perception-and-Decision-System/autonomous_driving/gazebo/worlds/adas_test_world.sdf"
    )

    model_path = Path(
        "/home/huseyindgn/Autonomous-Driving-Perception-and-Decision-System/ros2_ws/src/autonomous_driving/gazebo/models"
    )

    resource_paths = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    merged_resource_path = f"{model_path}:{resource_paths}" if resource_paths else str(model_path)

    yolo_model_path = "yolov8n.pt"

    return LaunchDescription([
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", merged_resource_path),

        ExecuteProcess(
            cmd=["gz", "sim", "-r", str(world_path)],
            output="screen"
        ),

        ExecuteProcess(
            cmd=[
                "ros2",
                "run",
                "ros_gz_bridge",
                "parameter_bridge",
                "/front_camera@sensor_msgs/msg/Image@gz.msgs.Image",
            ],
            output="screen",
        ),

        Node(
            package="autonomous_driving",
            executable="camera_node",
            name="camera_node",
            output="screen",
            parameters=[
                {
                    "input_topic": "/front_camera",
                    "output_topic": "/adas/camera/front/image_raw",
                }
            ],
        ),

        Node(
            package="autonomous_driving",
            executable="perception_node",
            name="perception_node",
            output="screen",
            parameters=[
                {
                    "image_topic": "/adas/camera/front/image_raw",
                    "detections_topic": "/adas/perception/detections_json",
                    "model_path": yolo_model_path,
                    "conf_threshold": 0.01,
                    "show_debug": True,
                }
            ],
        ),

        Node(
            package="autonomous_driving",
            executable="decision_node",
            name="decision_node",
            output="screen",
            parameters=[
                {
                    "detections_topic": "/adas/perception/detections_json",
                    "decision_topic": "/adas/decision",
                    "distance_k": 1200.0,
                    "lane_center_tolerance_ratio": 0.22,
                }
            ],
        ),
    ])
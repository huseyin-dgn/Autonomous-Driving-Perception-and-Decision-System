from pathlib import Path
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    repo_root = Path(
        "/home/huseyindgn/Masaüstü/Autonomous-Driving-Perception-and-Decision-System"
    )

    package_root = repo_root / "autonomous_driving"

    default_world_path = (
        package_root
        / "gazebo"
        / "worlds"
        / "scenario_01_red_light_only.sdf"
    )

    models_path = package_root / "gazebo" / "models"

    default_model_path = (
        package_root
        / "outputs"
        / "models"
        / "bdd_yolo_v14"
        / "weights"
        / "best.pt"
    )

    old_resource_path = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    merged_resource_path = (
        f"{models_path}:{old_resource_path}"
        if old_resource_path
        else str(models_path)
    )

    world_path_arg = DeclareLaunchArgument(
        "world_path",
        default_value=str(default_world_path),
        description="Gazebo SDF world path",
    )

    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value=str(default_model_path),
        description="YOLO model path",
    )

    return LaunchDescription(
        [
            world_path_arg,
            model_path_arg,

            SetEnvironmentVariable(
                name="GZ_SIM_RESOURCE_PATH",
                value=merged_resource_path,
            ),

            ExecuteProcess(
                cmd=[
                    "gz",
                    "sim",
                    "-r",
                    LaunchConfiguration("world_path"),
                ],
                output="screen",
            ),

            ExecuteProcess(
                cmd=[
                    "ros2",
                    "run",
                    "ros_gz_bridge",
                    "parameter_bridge",
                    "/front_camera@sensor_msgs/msg/Image[gz.msgs.Image"
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
                        "annotated_topic": "/adas/perception/annotated_image",
                        "model_path": LaunchConfiguration("model_path"),

                        "conf_threshold": 0.20,
                        "iou_threshold": 0.45,
                        "max_det": 20,
                        "imgsz": 640,

                        "vehicle_min_conf": 0.20,
                        "person_min_conf": 0.25,
                        "traffic_min_conf": 0.40,

                        "vehicle_min_area_ratio": 0.015,
                        "vehicle_min_width_ratio": 0.08,
                        "vehicle_min_height_ratio": 0.08,
                        "vehicle_min_bottom_ratio": 0.30,
                        "vehicle_min_aspect": 0.8,
                        "vehicle_max_aspect": 4.5,

                        "person_hold_frames": 90,

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
                        "lane_center_tolerance_ratio": 0.60,
                        "stop_distance": 5.0,
                        "slow_distance": 12.0,

                        "vehicle_conf_threshold": 0.20,
                        "min_bbox_height_ratio": 0.08,
                        "min_bbox_area_ratio": 0.015,
                        "min_aspect_ratio": 0.8,
                        "max_aspect_ratio": 4.5,
                        "min_bottom_y_ratio": 0.30,
                    }
                ],
            ),

            Node(
                package="autonomous_driving",
                executable="control_node",
                name="control_node",
                output="screen",
                parameters=[
                    {
                        "decision_topic": "/adas/decision",
                        "cmd_topic": "/cmd_vel",
                    }
                ],
            ),
        ]
    )
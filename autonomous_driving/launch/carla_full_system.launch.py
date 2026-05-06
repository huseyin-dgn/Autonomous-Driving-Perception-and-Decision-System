from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    repo_root = Path(
        "/home/huseyindgn/Masaüstü/Autonomous-Driving-Perception-and-Decision-System"
    )

    package_root = repo_root / "autonomous_driving"

    default_model_path = (
        package_root
        / "outputs"
        / "models"
        / "bdd_yolo_v14"
        / "weights"
        / "best.pt"
    )

    carla_root_arg = DeclareLaunchArgument(
        "carla_root",
        default_value="/mnt/carla/CARLA_0.9.15",
    )

    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value=str(default_model_path),
    )

    town_arg = DeclareLaunchArgument(
        "town",
        default_value="Town03",
    )

    scenario_arg = DeclareLaunchArgument(
        "scenario",
        default_value="basic_static",
    )

    return LaunchDescription(
        [
            carla_root_arg,
            model_path_arg,
            town_arg,
            scenario_arg,

            Node(
                package="autonomous_driving",
                executable="carla_world_manager_node",
                name="carla_world_manager_node",
                output="screen",
                parameters=[
                    {
                        "carla_root": LaunchConfiguration("carla_root"),
                        "town": LaunchConfiguration("town"),
                        "ego_role_name": "ego_vehicle",
                        "ego_blueprint": "vehicle.tesla.model3",
                        "spawn_index": 0,
                        "destroy_on_shutdown": False,
                        "set_spectator": True,
                        "enable_sync_mode": False,
                    }
                ],
            ),

            TimerAction(
                period=3.0,
                actions=[
                    Node(
                        package="autonomous_driving",
                        executable="carla_sensor_bridge_node",
                        name="carla_sensor_bridge_node",
                        output="screen",
                        parameters=[
                            {
                                "carla_root": LaunchConfiguration("carla_root"),
                                "ego_role_name": "ego_vehicle",
                                "image_topic": "/adas/camera/front/image_raw",
                                "camera_width": 800,
                                "camera_height": 600,
                                "camera_fov": 90.0,
                                "camera_x": 1.6,
                                "camera_y": 0.0,
                                "camera_z": 2.2,
                                "camera_pitch": 0.0,
                                "camera_yaw": 0.0,
                                "camera_roll": 0.0,
                            }
                        ],
                    )
                ],
            ),

            TimerAction(
                period=5.0,
                actions=[
                    Node(
                        package="autonomous_driving",
                        executable="carla_scenario_manager_node",
                        name="carla_scenario_manager_node",
                        output="screen",
                        parameters=[
                            {
                                "carla_root": LaunchConfiguration("carla_root"),
                                "ego_role_name": "ego_vehicle",
                                "scenario": LaunchConfiguration("scenario"),
                                "spawn_npc_vehicles": True,
                                "npc_vehicle_count": 8,
                                "spawn_walkers": True,
                                "walker_count": 4,
                                "npc_autopilot": True,
                                "destroy_on_shutdown": True,
                            }
                        ],
                    )
                ],
            ),

            TimerAction(
                period=6.0,
                actions=[
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
                                "decision_topic": "/adas/decision",
                                "model_path": LaunchConfiguration("model_path"),

                                "conf_threshold": 0.20,
                                "iou_threshold": 0.45,
                                "max_det": 20,
                                "imgsz": 640,

                                "vehicle_min_conf": 0.55,
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

                                "sim_red_light_fallback": False,
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
                                "min_bbox_width_ratio": 0.08,
                                "min_bbox_area_ratio": 0.015,
                                "min_aspect_ratio": 0.8,
                                "max_aspect_ratio": 4.5,
                                "min_bottom_y_ratio": 0.30,

                                "person_conf_threshold": 0.08,
                                "traffic_light_conf_threshold": 0.40,
                                "traffic_sign_conf_threshold": 0.20,
                                "sign_classifier_conf_threshold": 0.25,

                                "default_go_speed": 1.5,
                                "slow_speed": 0.8,
                                "stop_speed": 0.0,
                            }
                        ],
                    ),

                    Node(
                        package="autonomous_driving",
                        executable="carla_control_adapter_node",
                        name="carla_control_adapter_node",
                        output="screen",
                        parameters=[
                            {
                                "carla_root": LaunchConfiguration("carla_root"),
                                "decision_topic": "/adas/decision",
                                "ego_role_name": "ego_vehicle",

                                "control_rate_hz": 20.0,
                                "max_throttle": 0.55,
                                "max_brake": 1.0,
                                "speed_kp": 0.45,
                                "speed_ki": 0.02,
                                "speed_kd": 0.03,

                                "default_go_speed": 1.5,
                                "default_slow_speed": 0.8,

                                "enable_lane_keep": True,
                                "lookahead_distance": 6.0,
                                "steer_kp": 0.025,
                                "max_steer": 0.45,
                            }
                        ],
                    ),

                    Node(
                        package="autonomous_driving",
                        executable="carla_logger_node",
                        name="carla_logger_node",
                        output="screen",
                    ),
                ],
            ),
        ]
    )
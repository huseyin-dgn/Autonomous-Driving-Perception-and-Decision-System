import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    carla_root = LaunchConfiguration("carla_root")
    host = LaunchConfiguration("host")
    port = LaunchConfiguration("port")
    town = LaunchConfiguration("town")
    mission_geojson = LaunchConfiguration("mission_geojson")
    round_name = LaunchConfiguration("round_name")
    model_path = LaunchConfiguration("model_path")
    tl_model_path = LaunchConfiguration("tl_model_path")
    log_dir = LaunchConfiguration("log_dir")

    return LaunchDescription([
        DeclareLaunchArgument("carla_root", default_value="/mnt/carla/CARLA_0.9.15"),
        DeclareLaunchArgument("host", default_value="127.0.0.1"),
        DeclareLaunchArgument("port", default_value="2000"),
        DeclareLaunchArgument("town", default_value="Town03"),
        DeclareLaunchArgument("round_name", default_value="round_3"),
        DeclareLaunchArgument("mission_geojson", default_value="missions/teknofest_round3.geojson"),
        DeclareLaunchArgument(
            "model_path",
            default_value="outputs/models/adas5_targeted_aug_finetune_from_old_img1024_b8_ep50/weights/best.pt",
        ),
        DeclareLaunchArgument(
            "tl_model_path",
            default_value="outputs/models/traffic_light_state_resnet18_carla/best.pt",
        ),
        DeclareLaunchArgument("log_dir", default_value="outputs/teknofest_sim_logs"),

        SetEnvironmentVariable("ADAS_HEADLESS", "0"),
        SetEnvironmentVariable("SHOW_DEBUG", "1"),
        SetEnvironmentVariable("MODEL_PATH", model_path),
        SetEnvironmentVariable("TRAFFIC_LIGHT_STATE_MODEL_PATH", tl_model_path),
        SetEnvironmentVariable("TRAFFIC_LIGHT_STATE_CLASSIFIER_ENABLED", "1"),
        SetEnvironmentVariable("TRAFFIC_LIGHT_STATE_USE_HSV_FALLBACK", "1"),

        Node(
            package="autonomous_driving",
            executable="carla_world_manager_node",
            name="carla_world_manager_node",
            output="screen",
            parameters=[{
                "carla_root": carla_root,
                "host": host,
                "port": port,
                "town": town,
                "ego_role_name": "ego_vehicle",
                "ego_blueprint": "vehicle.tesla.model3",
                "spawn_index": 0,
                "destroy_on_shutdown": True,
                "set_spectator": True,
                "enable_sync_mode": False,
            }],
        ),

        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="carla_sensor_bridge_node",
                    name="carla_sensor_bridge_node",
                    output="screen",
                    parameters=[{
                        "carla_root": carla_root,
                        "host": host,
                        "port": port,
                        "ego_role_name": "ego_vehicle",
                        "camera_width": 960,
                        "camera_height": 540,
                        "camera_fov": 90.0,
                        "camera_x": 1.6,
                        "camera_y": 0.0,
                        "camera_z": 2.0,
                        "camera_pitch": -2.0,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="teknofest_scenario_node",
                    name="teknofest_scenario_node",
                    output="screen",
                    parameters=[{
                        "carla_root": carla_root,
                        "host": host,
                        "port": port,
                        "ego_role_name": "ego_vehicle",
                        "scenario_round": round_name,
                        "npc_vehicle_count": 6,
                        "walker_count": 4,
                        "static_obstacle_count": 4,
                        "dynamic_crossing_enabled": True,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=3.5,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="perception_node",
                    name="perception_node",
                    output="screen",
                    parameters=[{
                        "image_topic": "/adas/camera/front/image_raw",
                        "detections_topic": "/adas/perception/detections_json",
                        "annotated_topic": "/adas/perception/annotated_image",
                        "model_path": model_path,
                        "traffic_light_state_model_path": tl_model_path,
                        "traffic_light_state_classifier_enabled": True,
                        "traffic_light_state_use_hsv_fallback": True,
                        "show_debug": True,
                        "imgsz": 960,
                        "raw_conf_threshold": 0.05,
                        "person_conf_threshold": 0.35,
                        "vehicle_conf_threshold": 0.25,
                        "traffic_light_conf_threshold": 0.40,
                        "traffic_sign_conf_threshold": 0.20,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=4.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="decision_node",
                    name="decision_node",
                    output="screen",
                    parameters=[{
                        "detections_topic": "/adas/perception/detections_json",
                        "decision_topic": "/adas/decision",
                        "default_go_speed": 2.0,
                        "slow_speed": 0.8,
                        "stop_speed": 0.0,
                        "stop_distance": 5.0,
                        "slow_distance": 12.0,
                        "person_conf_threshold": 0.08,
                        "traffic_light_conf_threshold": 0.40,
                        "traffic_sign_conf_threshold": 0.20,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=4.5,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="teknofest_mission_node",
                    name="teknofest_mission_node",
                    output="screen",
                    parameters=[{
                        "mission_geojson": mission_geojson,
                        "round_name": round_name,
                        "point_pass_tolerance_m": 1.0,
                        "passenger_stop_min_s": 15.0,
                        "passenger_stop_max_s": 20.0,
                        "park_time_limit_s": 180.0,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="teknofest_route_agent_node",
                    name="teknofest_route_agent_node",
                    output="screen",
                    parameters=[{
                        "carla_root": carla_root,
                        "host": host,
                        "port": port,
                        "ego_role_name": "ego_vehicle",
                        "decision_topic": "/adas/decision",
                        "mission_topic": "/adas/teknofest/mission",
                        "max_speed_mps": 3.0,
                        "go_speed_mps": 2.0,
                        "slow_speed_mps": 0.8,
                        "parking_speed_mps": 0.45,
                        "lookahead_m": 6.0,
                        "max_steer": 0.55,
                        "mission_stop_override": True,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=5.5,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="teknofest_evaluator_node",
                    name="teknofest_evaluator_node",
                    output="screen",
                    parameters=[{
                        "log_dir": log_dir,
                    }],
                ),
            ],
        ),
    ])
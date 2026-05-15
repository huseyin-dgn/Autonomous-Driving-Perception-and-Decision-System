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
        DeclareLaunchArgument("carla_root", default_value="/home/ilker/simulators/CARLA_0.9.15"),
        DeclareLaunchArgument("host", default_value="127.0.0.1"),
        DeclareLaunchArgument("port", default_value="2000"),
        DeclareLaunchArgument("town", default_value=""),
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

        SetEnvironmentVariable("ADAS_HEADLESS", "1"),
        SetEnvironmentVariable("SHOW_DEBUG", "0"),
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
                "timeout": 120.0,
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
            period=8.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="carla_spectator_follow_node",
                    name="carla_spectator_follow_node",
                    output="screen",
                    parameters=[{
                        "carla_root": carla_root,
                        "host": host,
                        "port": port,
                        "timeout": 120.0,
                        "ego_role_name": "ego_vehicle",
                        "follow_distance": 9.0,
                        "follow_height": 5.0,
                        "side_offset": 0.0,
                        "look_at_height": 1.2,
                        "tick_s": 0.05,
                        "status_topic": "/adas/carla/spectator_follow_status",
                    }],
                ),
            ],
        ),

        TimerAction(
            period=20.0,
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
                        "timeout": 120.0,
                        "ego_role_name": "ego_vehicle",
                        "camera_width": 960,
                        "camera_height": 540,
                        "camera_fov": 90.0,
                        "camera_x": 1.6,
                        "camera_y": 0.0,
                        "camera_z": 2.25,
                        "camera_pitch": -1.0,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=25.0,
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
                        "timeout": 120.0,
                        "ego_role_name": "ego_vehicle",
                        "scenario_round": round_name,
                        "npc_vehicle_count": 0,
                        "walker_count": 0,
                        "static_obstacle_count": 0,
                        "dynamic_crossing_enabled": False,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=28.0,
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
                        "traffic_light_state_device": "cuda",
                        "show_debug": False,
                        "imgsz": 640,
                        "raw_conf_threshold": 0.05,
                        "person_conf_threshold": 0.60,
                        "vehicle_conf_threshold": 0.55,
                        "traffic_light_conf_threshold": 0.45,
                        "traffic_sign_conf_threshold": 0.20,
                    }],
                ),
            ],
        ),


        TimerAction(
            period=29.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="lane_assist_node",
                    name="lane_assist_node",
                    output="screen",
                    parameters=[{
                        "image_topic": "/adas/camera/front/image_raw",
                        "lane_topic": "/adas/lane/assist",
                        "annotated_topic": "/adas/lane/annotated_image",
                        "publish_annotated": True,
                        "roi_top_ratio": 0.45,
                        "lane_steer_gain": 0.30,
                        "max_lane_steer": 0.18,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=30.0,
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
                        "stop_distance": 3.0,
                        "slow_distance": 8.0,
                        "vehicle_conf_threshold": 0.60,
                        "max_missing_front": 1,
                        "min_bbox_area_ratio": 0.045,
                        "min_bottom_y_ratio": 0.52,
                        "person_conf_threshold": 0.60,
                        "traffic_light_conf_threshold": 0.45,
                        "traffic_sign_conf_threshold": 0.20,
                        "lane_center_tolerance_ratio": 0.18,
                    }],
                ),
            ],
        ),


        TimerAction(
            period=31.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="traffic_light_decision_gate_node",
                    name="traffic_light_decision_gate_node",
                    output="screen",
                    parameters=[{
                        "enabled": True,
                        "detections_topic": "/adas/perception/detections_json",
                        "base_decision_topic": "/adas/decision",
                        "safe_decision_topic": "/adas/decision_safe",
                        "green_release_ignore_red_s": 6.0,
                        "green_release_speed": 2.0,
                        "green_release_override_enabled": True,
                        "debug_topic": "/adas/traffic_light/gate_debug",
                        "tl_fresh_timeout_s": 0.70,
                        "base_decision_timeout_s": 1.50,
                        "min_det_conf": 0.30,
                        "min_state_conf": 0.50,
                        "red_stop_hold_s": 0.45,
                        "yellow_slow_speed": 1.25,
                        "offlane_person_filter_enabled": True,
                        "person_image_width": 960.0,
                        "person_stop_min_center_x_ratio": 0.30,
                        "person_stop_max_center_x_ratio": 0.70,
                        "person_stop_min_bottom_ratio": 0.45,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=32.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="teknofest_mission_node",
                    name="teknofest_mission_node",
                    output="screen",
                    parameters=[{
                        "mission_geojson": mission_geojson,
                        "round_name": round_name,
                        "point_pass_tolerance_m": 3.0,
                        "passenger_stop_min_s": 5.0,
                        "passenger_stop_max_s": 7.0,
                        "park_time_limit_s": 180.0,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=34.0,
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
                        "timeout": 120.0,
                        "ego_role_name": "ego_vehicle",
                        "decision_topic": "/adas/decision_safe",
                        "green_release_boost_s": 2.5,
                        "green_release_speed_mps": 2.0,
                        "post_tl_ignore_s": 7.0,
                        "ignore_vision_red_when_carla_not_red": True,
                        "carla_tl_override_enabled": True,
                        "mission_topic": "/adas/teknofest/mission",
                        "max_speed_mps": 2.0,
                        "go_speed_mps": 1.8,
                        "slow_speed_mps": 1.10,
                        "parking_speed_mps": 0.55,
                        "lookahead_m": 10.0,
                        "max_steer": 0.42,
                        "mission_stop_override": True,
                        "ignore_decision_for_mission_test": False,
                        "lane_assist_enabled": True,
                        "lane_topic": "/adas/lane/assist",
                        "lane_min_confidence": 0.35,
                        "lane_fresh_timeout_s": 0.50,
                        "lane_blend_straight": 0.28,
                        "lane_blend_turn": 0.0,
                        "lane_turn_steer_threshold": 0.16,
                        "lane_allowed_stages": "GO_TO_TASK,GO_TO_PARK",
                    }],
                ),
            ],
        ),

        TimerAction(
            period=36.0,
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

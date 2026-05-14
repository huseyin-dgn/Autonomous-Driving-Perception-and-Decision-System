from setuptools import find_packages, setup

package_name = "autonomous_driving"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/autonomous_driving"]),
        ("share/autonomous_driving", ["package.xml"]),
        (
            "share/autonomous_driving/launch",
            [
                "launch/perception.launch.py",
                "launch/carla_full_system.launch.py",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ilker",
    maintainer_email="ilker.akbal4822@gop.edu.tr",
    description="Autonomous driving perception, decision and CARLA ROS2 integration package",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_node = ros2_nodes.camera_node:main",
            "lidar_node = ros2_nodes.lidar_node:main",
            "perception_node = ros2_nodes.perception_node:main",
            "decision_node = ros2_nodes.decision_node:main",
            "control_node = ros2_nodes.control_node:main",

            "carla_world_manager_node = ros2_nodes.carla_world_manager_node:main",
            "carla_sensor_bridge_node = ros2_nodes.carla_sensor_bridge_node:main",
            "carla_control_adapter_node = ros2_nodes.carla_control_adapter_node:main",
            "carla_scenario_manager_node = ros2_nodes.carla_scenario_manager_node:main",
            "carla_logger_node = ros2_nodes.carla_logger_node:main",
        ],
    },
)

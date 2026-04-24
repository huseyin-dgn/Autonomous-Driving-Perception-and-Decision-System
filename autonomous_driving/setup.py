from setuptools import find_packages, setup

package_name = "autonomous_driving"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/autonomous_driving"]),
        ("share/autonomous_driving", ["package.xml"]),
        ("share/autonomous_driving/launch", [
            "launch/gazebo_sim.launch.py",
            "launch/perception.launch.py",
            "launch/full_system.launch.py",
        ]),
        ("share/autonomous_driving/gazebo/worlds", [
            "gazebo/worlds/adas_test_world.sdf",
        ]),
        ("share/autonomous_driving/gazebo/models/vehicle_model", [
            "gazebo/models/vehicle_model/model.sdf",
            "gazebo/models/vehicle_model/model.config",
        ]),
        ("share/autonomous_driving/gazebo/config", [
            "gazebo/config/bridge.yaml",
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="you",
    maintainer_email="you@example.com",
    description="Autonomous driving Gazebo + ROS2 integration package",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_node = ros2_nodes.camera_node:main",
            "lidar_node = ros2_nodes.lidar_node:main",
            "perception_node = ros2_nodes.perception_node:main",
            "decision_node = ros2_nodes.decision_node:main",
            "control_node = ros2_nodes.control_node:main",
        ],
    },
)
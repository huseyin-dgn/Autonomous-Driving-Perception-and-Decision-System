from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = Path(get_package_share_directory("autonomous_driving"))
    bridge_config = pkg_share / "gazebo" / "config" / "bridge.yaml"

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["--ros-args", "-p", f"config_file:={bridge_config}"],
        output="screen",
    )

    perception = Node(
        package="autonomous_driving",
        executable="perception_node",
        output="screen",
    )

    return LaunchDescription([
        bridge,
        perception,
    ])
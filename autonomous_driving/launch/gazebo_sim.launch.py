from pathlib import Path
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable


def generate_launch_description():
    pkg_share = Path(get_package_share_directory("autonomous_driving"))
    world_path = pkg_share / "gazebo" / "worlds" / "test_world.sdf"
    model_path = pkg_share / "gazebo" / "models"

    resource_paths = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    merged_resource_path = f"{model_path}:{resource_paths}" if resource_paths else str(model_path)

    return LaunchDescription([
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", merged_resource_path),

        ExecuteProcess(
            cmd=["gz", "sim", "-r", str(world_path)],
            output="screen"
        ),
    ])
import glob
import os
import sys
import time


def load_carla(carla_root: str):
    egg_pattern = os.path.join(
        carla_root,
        "PythonAPI",
        "carla",
        "dist",
        "carla-*%d.%d-%s.egg" % (
            sys.version_info.major,
            sys.version_info.minor,
            "linux-x86_64",
        ),
    )

    eggs = glob.glob(egg_pattern)
    if eggs and eggs[0] not in sys.path:
        sys.path.append(eggs[0])

    import carla
    return carla


import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, NavSatFix
from std_msgs.msg import String


class CarlaSensorBridgeNode(Node):
    def __init__(self):
        super().__init__("carla_sensor_bridge_node")

        self.declare_parameter("carla_root", "/mnt/carla/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 20.0)
        self.declare_parameter("ego_role_name", "ego_vehicle")

        self.declare_parameter("image_topic", "/adas/camera/front/image_raw")
        self.declare_parameter("gnss_topic", "/adas/localization/gnss")
        self.declare_parameter("imu_topic", "/adas/localization/imu")
        self.declare_parameter("collision_topic", "/adas/events/collision")

        self.declare_parameter("camera_width", 800)
        self.declare_parameter("camera_height", 600)
        self.declare_parameter("camera_fov", 90.0)
        self.declare_parameter("camera_x", 1.6)
        self.declare_parameter("camera_y", 0.0)
        self.declare_parameter("camera_z", 2.2)
        self.declare_parameter("camera_pitch", 0.0)
        self.declare_parameter("camera_yaw", 0.0)
        self.declare_parameter("camera_roll", 0.0)

        self.carla_root = self.get_parameter("carla_root").value
        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.ego_role_name = self.get_parameter("ego_role_name").value

        self.image_topic = self.get_parameter("image_topic").value
        self.gnss_topic = self.get_parameter("gnss_topic").value
        self.imu_topic = self.get_parameter("imu_topic").value
        self.collision_topic = self.get_parameter("collision_topic").value

        self.bridge = CvBridge()
        self.carla = load_carla(self.carla_root)

        self.client = self.carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()

        self.image_pub = self.create_publisher(Image, self.image_topic, 10)
        self.gnss_pub = self.create_publisher(NavSatFix, self.gnss_topic, 10)
        self.imu_pub = self.create_publisher(Imu, self.imu_topic, 10)
        self.collision_pub = self.create_publisher(String, self.collision_topic, 10)

        self.ego_vehicle = self.wait_for_ego_vehicle()
        self.sensors = []

        self.spawn_camera()
        self.spawn_gnss()
        self.spawn_imu()
        self.spawn_collision_sensor()

        self.get_logger().info("CARLA sensor bridge hazır")
        self.get_logger().info(f"RGB camera -> {self.image_topic}")

    def wait_for_ego_vehicle(self):
        for _ in range(100):
            vehicles = self.world.get_actors().filter("vehicle.*")
            for vehicle in vehicles:
                if vehicle.attributes.get("role_name", "") == self.ego_role_name:
                    return vehicle
            time.sleep(0.2)

        raise RuntimeError("Ego vehicle bulunamadı. Önce carla_world_manager_node çalışmalı.")

    def spawn_camera(self):
        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.find("sensor.camera.rgb")

        width = int(self.get_parameter("camera_width").value)
        height = int(self.get_parameter("camera_height").value)
        fov = float(self.get_parameter("camera_fov").value)

        camera_bp.set_attribute("image_size_x", str(width))
        camera_bp.set_attribute("image_size_y", str(height))
        camera_bp.set_attribute("fov", str(fov))

        transform = self.carla.Transform(
            self.carla.Location(
                x=float(self.get_parameter("camera_x").value),
                y=float(self.get_parameter("camera_y").value),
                z=float(self.get_parameter("camera_z").value),
            ),
            self.carla.Rotation(
                pitch=float(self.get_parameter("camera_pitch").value),
                yaw=float(self.get_parameter("camera_yaw").value),
                roll=float(self.get_parameter("camera_roll").value),
            ),
        )

        camera = self.world.spawn_actor(
            camera_bp,
            transform,
            attach_to=self.ego_vehicle,
        )

        camera.listen(self.camera_callback)
        self.sensors.append(camera)

    def spawn_gnss(self):
        bp_lib = self.world.get_blueprint_library()
        gnss_bp = bp_lib.find("sensor.other.gnss")

        transform = self.carla.Transform(self.carla.Location(x=0.0, z=2.0))
        gnss = self.world.spawn_actor(gnss_bp, transform, attach_to=self.ego_vehicle)
        gnss.listen(self.gnss_callback)
        self.sensors.append(gnss)

    def spawn_imu(self):
        bp_lib = self.world.get_blueprint_library()
        imu_bp = bp_lib.find("sensor.other.imu")

        transform = self.carla.Transform(self.carla.Location(x=0.0, z=2.0))
        imu = self.world.spawn_actor(imu_bp, transform, attach_to=self.ego_vehicle)
        imu.listen(self.imu_callback)
        self.sensors.append(imu)

    def spawn_collision_sensor(self):
        bp_lib = self.world.get_blueprint_library()
        collision_bp = bp_lib.find("sensor.other.collision")

        transform = self.carla.Transform(self.carla.Location(x=0.0, z=1.0))
        collision = self.world.spawn_actor(
            collision_bp,
            transform,
            attach_to=self.ego_vehicle,
        )
        collision.listen(self.collision_callback)
        self.sensors.append(collision)

    def camera_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))

        bgr = array[:, :, :3]

        msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "carla_front_camera"
        self.image_pub.publish(msg)

    def gnss_callback(self, data):
        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "carla_gnss"
        msg.latitude = float(data.latitude)
        msg.longitude = float(data.longitude)
        msg.altitude = float(data.altitude)
        self.gnss_pub.publish(msg)

    def imu_callback(self, data):
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "carla_imu"

        msg.linear_acceleration.x = float(data.accelerometer.x)
        msg.linear_acceleration.y = float(data.accelerometer.y)
        msg.linear_acceleration.z = float(data.accelerometer.z)

        msg.angular_velocity.x = float(data.gyroscope.x)
        msg.angular_velocity.y = float(data.gyroscope.y)
        msg.angular_velocity.z = float(data.gyroscope.z)

        self.imu_pub.publish(msg)

    def collision_callback(self, event):
        other = event.other_actor

        payload = {
            "stamp": time.time(),
            "ego_id": event.actor.id,
            "other_id": other.id if other else None,
            "other_type": other.type_id if other else None,
            "impulse": {
                "x": event.normal_impulse.x,
                "y": event.normal_impulse.y,
                "z": event.normal_impulse.z,
            },
        }

        msg = String()
        msg.data = str(payload)
        self.collision_pub.publish(msg)
        self.get_logger().warn(f"[COLLISION] {payload}")

    def destroy_node(self):
        for sensor in getattr(self, "sensors", []):
            try:
                sensor.stop()
                sensor.destroy()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CarlaSensorBridgeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
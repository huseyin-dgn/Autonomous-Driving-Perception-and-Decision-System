#!/usr/bin/env python3
import math
import time
import argparse
import numpy as np
import carla

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def yaw_to_target(src, dst):
    dx = dst.x - src.x
    dy = dst.y - src.y
    return math.degrees(math.atan2(dy, dx))


def forward_vec(yaw_deg):
    yaw = math.radians(yaw_deg)
    return carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)


class CarlaRealTLObjectPublisher(Node):
    def __init__(self, args):
        super().__init__("carla_real_tl_object_publisher")

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, "/adas/camera/front/image_raw", 10)

        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library()

        self.ego = None
        self.camera = None
        self.latest = None

        self.setup_scene(args)

        self.timer = self.create_timer(0.05, self.publish_image)

    def setup_scene(self, args):
        tls = list(self.world.get_actors().filter("traffic.traffic_light*"))

        if not tls:
            raise RuntimeError("Bu haritada gerçek traffic light actor yok.")

        target_tl = tls[args.tl_index % len(tls)]
        tl_tf = target_tl.get_transform()
        tl_loc = tl_tf.location

        if args.light == "red":
            target_tl.set_state(carla.TrafficLightState.Red)
        elif args.light == "yellow":
            target_tl.set_state(carla.TrafficLightState.Yellow)
        elif args.light == "green":
            target_tl.set_state(carla.TrafficLightState.Green)
        else:
            raise RuntimeError("light red/yellow/green olmalı")

        target_tl.freeze(True)

        vehicle_bp = self.bp_lib.filter("vehicle.tesla.model3")[0]
        vehicle_bp.set_attribute("role_name", "ego_vehicle")

        # Işığın karşısına ego koy
        tl_yaw = tl_tf.rotation.yaw
        back_yaw = tl_yaw + 180.0
        fwd = forward_vec(back_yaw)

        ego_loc = carla.Location(
            x=tl_loc.x + fwd.x * args.distance,
            y=tl_loc.y + fwd.y * args.distance,
            z=tl_loc.z + 0.2
        )

        ego_yaw = yaw_to_target(ego_loc, tl_loc)

        ego_tf = carla.Transform(
            ego_loc,
            carla.Rotation(pitch=0.0, yaw=ego_yaw, roll=0.0)
        )

        self.ego = self.world.try_spawn_actor(vehicle_bp, ego_tf)

        if self.ego is None:
            # Çakışma varsa biraz yukarı al
            ego_tf.location.z += 1.0
            self.ego = self.world.spawn_actor(vehicle_bp, ego_tf)

        self.ego.set_autopilot(False)

        cam_bp = self.bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(args.width))
        cam_bp.set_attribute("image_size_y", str(args.height))
        cam_bp.set_attribute("fov", str(args.fov))

        cam_tf = carla.Transform(
            carla.Location(x=1.5, y=0.0, z=1.7),
            carla.Rotation(pitch=args.pitch, yaw=0.0, roll=0.0)
        )

        self.camera = self.world.spawn_actor(cam_bp, cam_tf, attach_to=self.ego)
        self.camera.listen(self.camera_callback)

        self.get_logger().info("REAL CARLA TRAFFIC LIGHT OBJECT SCENE READY")
        self.get_logger().info(f"Traffic light id: {target_tl.id}")
        self.get_logger().info(f"Light state: {args.light}")
        self.get_logger().info(f"Topic: /adas/camera/front/image_raw")

    def camera_callback(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        rgb = arr[:, :, :3][:, :, ::-1].copy()
        self.latest = rgb

    def publish_image(self):
        if self.latest is None:
            return

        msg = self.bridge.cv2_to_imgmsg(self.latest, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "front_camera"
        self.pub.publish(msg)

    def destroy_node(self):
        if self.camera is not None:
            self.camera.stop()
            self.camera.destroy()
        if self.ego is not None:
            self.ego.destroy()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--light", default="green", choices=["red", "yellow", "green"])
    parser.add_argument("--tl-index", type=int, default=0)
    parser.add_argument("--distance", type=float, default=16.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    args = parser.parse_args()

    rclpy.init()
    node = CarlaRealTLObjectPublisher(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

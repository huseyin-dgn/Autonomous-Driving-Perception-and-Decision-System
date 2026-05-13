#!/usr/bin/env python3
import argparse
import time
import numpy as np
import carla

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CarlaTLStoplinePublisher(Node):
    def __init__(self, args):
        super().__init__("carla_tl_stopline_publisher")

        self.args = args
        self.bridge = CvBridge()
        self.latest = None

        self.pub = self.create_publisher(Image, "/adas/camera/front/image_raw", 10)

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library()

        self.ego = None
        self.camera = None

        self.clear_old_actors()
        self.setup_scene()
        self.timer = self.create_timer(0.05, self.publish)

    def clear_old_actors(self):
        for actor in self.world.get_actors():
            try:
                if actor.type_id.startswith("vehicle.") or actor.type_id.startswith("walker."):
                    actor.destroy()
            except Exception:
                pass

    def setup_scene(self):
        tls = list(self.world.get_actors().filter("traffic.traffic_light*"))

        if not tls:
            raise RuntimeError("Haritada traffic light actor yok.")

        tl = tls[self.args.tl_index % len(tls)]

        if self.args.light == "red":
            tl.set_state(carla.TrafficLightState.Red)
        elif self.args.light == "yellow":
            tl.set_state(carla.TrafficLightState.Yellow)
        elif self.args.light == "green":
            tl.set_state(carla.TrafficLightState.Green)

        tl.freeze(True)

        stop_wps = tl.get_stop_waypoints()

        if not stop_wps:
            raise RuntimeError("Bu traffic light için stop waypoint bulunamadı. Başka --tl-index dene.")

        stop_wp = stop_wps[0]

        # Stop çizgisinden geriye doğru aracı koy
        prev_wps = stop_wp.previous(self.args.distance)

        if not prev_wps:
            raise RuntimeError("Stop waypoint gerisine gidilemedi. Başka --tl-index dene.")

        ego_wp = prev_wps[0]
        ego_tf = ego_wp.transform
        ego_tf.location.z += 0.3

        vehicle_bp = self.bp_lib.filter("vehicle.tesla.model3")[0]
        vehicle_bp.set_attribute("role_name", "ego_vehicle")

        self.ego = self.world.try_spawn_actor(vehicle_bp, ego_tf)

        if self.ego is None:
            ego_tf.location.z += 1.0
            self.ego = self.world.spawn_actor(vehicle_bp, ego_tf)

        self.ego.set_autopilot(False)

        cam_bp = self.bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.args.width))
        cam_bp.set_attribute("image_size_y", str(self.args.height))
        cam_bp.set_attribute("fov", str(self.args.fov))

        cam_tf = carla.Transform(
            carla.Location(x=1.5, y=0.0, z=1.7),
            carla.Rotation(pitch=self.args.pitch, yaw=0.0, roll=0.0)
        )

        self.camera = self.world.spawn_actor(cam_bp, cam_tf, attach_to=self.ego)
        self.camera.listen(self.camera_cb)

        self.get_logger().info("REAL CARLA TL STOPLINE SCENE READY")
        self.get_logger().info(f"TL index     : {self.args.tl_index}")
        self.get_logger().info(f"TL actor id  : {tl.id}")
        self.get_logger().info(f"Light state  : {self.args.light}")
        self.get_logger().info(f"Distance     : {self.args.distance}")
        self.get_logger().info("Publishing   : /adas/camera/front/image_raw")

    def camera_cb(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        bgr = arr[:, :, :3][:, :, ::-1].copy()
        self.latest = bgr

    def publish(self):
        if self.latest is None:
            return

        msg = self.bridge.cv2_to_imgmsg(self.latest, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "front_camera"
        self.pub.publish(msg)

    def destroy_node(self):
        if self.camera:
            self.camera.stop()
            self.camera.destroy()
        if self.ego:
            self.ego.destroy()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--light", choices=["red", "yellow", "green"], default="green")
    parser.add_argument("--tl-index", type=int, default=0)
    parser.add_argument("--distance", type=float, default=12.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=55.0)
    parser.add_argument("--pitch", type=float, default=-4.0)
    args = parser.parse_args()

    rclpy.init()
    node = CarlaTLStoplinePublisher(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import math
import queue
import threading
import time

import carla
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


STATE_ORDER = ["red", "yellow", "green"]

STATE_MAP = {
    "red": carla.TrafficLightState.Red,
    "yellow": carla.TrafficLightState.Yellow,
    "green": carla.TrafficLightState.Green,
}


def look_at_rotation(src, dst):
    dx = dst.x - src.x
    dy = dst.y - src.y
    dz = dst.z - src.z

    yaw = math.degrees(math.atan2(dy, dx))
    dist_xy = math.sqrt(dx * dx + dy * dy)
    pitch = math.degrees(math.atan2(dz, max(0.001, dist_xy)))

    return carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)


def build_projection_matrix(width, height, fov):
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))

    K = np.identity(3)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0

    return K


def get_image_point(world_point, K, world_to_camera):
    p = np.array([world_point.x, world_point.y, world_point.z, 1.0])
    p_camera = np.dot(world_to_camera, p)

    # CARLA coordinate -> camera coordinate
    # x forward, y right, z up  =>  image: x=y, y=-z, z=x
    p_img = np.array([p_camera[1], -p_camera[2], p_camera[0]])

    if p_img[2] <= 0.05:
        return None

    p_2d = np.dot(K, p_img)
    u = p_2d[0] / p_2d[2]
    v = p_2d[1] / p_2d[2]

    return float(u), float(v)


def get_light_world_vertices(light):
    vertices = []
    tf = light.get_transform()

    try:
        boxes = light.get_light_boxes()
    except Exception:
        boxes = []

    for box in boxes:
        try:
            for v in box.get_world_vertices(tf):
                vertices.append(v)
        except Exception:
            pass

    if vertices:
        return vertices

    # Fallback: actor location çevresinde tahmini kafa hacmi
    base = tf.location
    cx = base.x
    cy = base.y
    cz = base.z + 5.0

    sx = 0.8
    sy = 0.8
    sz = 1.8

    for dx in [-sx, sx]:
        for dy in [-sy, sy]:
            for dz in [-sz, sz]:
                vertices.append(carla.Location(cx + dx, cy + dy, cz + dz))

    return vertices


def get_light_target(light):
    verts = get_light_world_vertices(light)

    if verts:
        xs = [v.x for v in verts]
        ys = [v.y for v in verts]
        zs = [v.z for v in verts]

        return carla.Location(
            x=sum(xs) / len(xs),
            y=sum(ys) / len(ys),
            z=sum(zs) / len(zs),
        )

    tf = light.get_transform()
    return carla.Location(tf.location.x, tf.location.y, tf.location.z + 5.0)


def project_light_bbox(light, camera_tf, width, height, fov, pad_ratio=1.30):
    K = build_projection_matrix(width, height, fov)
    world_to_camera = np.array(camera_tf.get_inverse_matrix())

    points = []
    for v in get_light_world_vertices(light):
        p = get_image_point(v, K, world_to_camera)
        if p is not None:
            u, vv = p
            if -width * 2 <= u <= width * 3 and -height * 2 <= vv <= height * 3:
                points.append((u, vv))

    if len(points) < 2:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)

    bw = x2 - x1
    bh = y2 - y1

    if bw < 2 or bh < 2:
        return None

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    bw *= pad_ratio
    bh *= pad_ratio

    x1 = int(max(0, cx - bw / 2.0))
    y1 = int(max(0, cy - bh / 2.0))
    x2 = int(min(width - 1, cx + bw / 2.0))
    y2 = int(min(height - 1, cy + bh / 2.0))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def score_crop(crop):
    if crop is None or crop.size == 0:
        return 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    red = (((h <= 10) | (h >= 170)) & (s >= 70) & (v >= 90))
    yellow = ((h >= 16) & (h <= 42) & (s >= 60) & (v >= 90))
    green = ((h >= 42) & (h <= 100) & (s >= 45) & (v >= 70))
    dark = ((v <= 100) & (s <= 210))

    color_score = float(red.sum() + yellow.sum() + green.sum())
    dark_score = float(dark.sum()) * 0.10

    h_crop, w_crop = crop.shape[:2]
    size_score = min(5000.0, float(h_crop * w_crop) * 0.01)

    return color_score + dark_score + size_score


def image_to_bgr(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr[:, :, :3].copy()


class TrafficLightCleanPublisher(Node):
    def __init__(self, args):
        super().__init__("carla_tl_clean_scene_publisher")

        self.args = args
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, args.topic, 10)

        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(30.0)

        if args.reload_map:
            self.get_logger().info(f"Loading CARLA map: {args.map}")
            self.world = self.client.load_world(args.map)
            time.sleep(3.0)
        else:
            self.world = self.client.get_world()

        self.setup_world()

        self.lights = list(self.world.get_actors().filter("traffic.traffic_light*"))
        self.lights.sort(key=lambda a: (a.get_transform().location.x, a.get_transform().location.y))

        if not self.lights:
            raise RuntimeError("Bu CARLA map içinde trafik ışığı bulunamadı. Town10HD veya Town03 dene.")

        if args.light_index < 0 or args.light_index >= len(self.lights):
            raise RuntimeError(f"Geçersiz --light-index. Bulunan trafik ışığı sayısı: {len(self.lights)}")

        self.light = self.lights[args.light_index]
        self.target = get_light_target(self.light)

        self.get_logger().info(f"Traffic light count: {len(self.lights)}")
        self.get_logger().info(f"Selected traffic light: index={args.light_index}, id={self.light.id}")
        self.get_logger().info(
            f"Target: x={self.target.x:.2f}, y={self.target.y:.2f}, z={self.target.z:.2f}"
        )

        self.destroy_old_sensors()

        self.current_state_index = 0
        self.current_state = args.state
        self.set_all_lights(self.current_state)

        self.camera_tf = self.find_best_camera_transform()
        self.camera = self.spawn_camera(self.camera_tf)

        self.frame_count = 0
        self.last_log = time.time()

        if args.cycle:
            self.timer = self.create_timer(args.seconds, self.cycle_state)
        else:
            self.timer = None

        self.get_logger().info("===================================================")
        self.get_logger().info("CARLA CLEAN TRAFFIC LIGHT PUBLISHER READY")
        self.get_logger().info(f"ROS topic        : {args.topic}")
        self.get_logger().info(f"Output image     : {args.width}x{args.height}")
        self.get_logger().info(f"Camera FOV       : {args.fov}")
        self.get_logger().info(f"Cycle            : {args.cycle}")
        self.get_logger().info("Publisher preview: OFF")
        self.get_logger().info("===================================================")

    def setup_world(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

        self.world.set_weather(
            carla.WeatherParameters(
                cloudiness=0.0,
                precipitation=0.0,
                sun_altitude_angle=55.0,
                sun_azimuth_angle=20.0,
                fog_density=0.0,
                wetness=0.0,
            )
        )

        if self.args.destroy_dynamic:
            count = 0
            for a in self.world.get_actors():
                if (
                    a.type_id.startswith("vehicle.")
                    or a.type_id.startswith("walker.")
                    or a.type_id.startswith("sensor.")
                    or a.type_id.startswith("controller.ai.walker")
                ):
                    try:
                        a.destroy()
                        count += 1
                    except Exception:
                        pass

            self.get_logger().info(f"Destroyed dynamic actors: {count}")

    def destroy_old_sensors(self):
        count = 0

        for a in self.world.get_actors():
            if a.type_id.startswith("sensor.camera"):
                role_name = a.attributes.get("role_name", "")
                if role_name == "adas_clean_tl_camera":
                    try:
                        a.destroy()
                        count += 1
                    except Exception:
                        pass

        if count:
            self.get_logger().info(f"Destroyed old test cameras: {count}")

    def set_all_lights(self, state_name):
        state = STATE_MAP[state_name]

        for l in self.lights:
            try:
                l.freeze(False)
                l.set_state(state)
                l.set_red_time(9999.0)
                l.set_yellow_time(9999.0)
                l.set_green_time(9999.0)
                l.freeze(True)
            except Exception:
                pass

        self.current_state = state_name
        self.get_logger().info(f"ALL_LIGHTS_STATE={state_name.upper()}")

    def cycle_state(self):
        self.current_state_index = (self.current_state_index + 1) % len(STATE_ORDER)
        self.set_all_lights(STATE_ORDER[self.current_state_index])

    def make_camera_transform(self, angle_deg, distance, z_offset):
        angle = math.radians(angle_deg)

        cam_loc = carla.Location(
            x=self.target.x + math.cos(angle) * distance,
            y=self.target.y + math.sin(angle) * distance,
            z=self.target.z + z_offset,
        )

        cam_rot = look_at_rotation(cam_loc, self.target)

        return carla.Transform(cam_loc, cam_rot)

    def spawn_temp_camera_and_capture(self, tf, timeout=3.0):
        q = queue.Queue(maxsize=1)

        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.args.width))
        bp.set_attribute("image_size_y", str(self.args.height))
        bp.set_attribute("fov", str(self.args.fov))
        bp.set_attribute("sensor_tick", "0.05")

        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "adas_clean_tl_camera_temp")

        cam = self.world.spawn_actor(bp, tf)

        def callback(image):
            try:
                q.put_nowait(image)
            except queue.Full:
                pass

        cam.listen(callback)

        image = None
        try:
            image = q.get(timeout=timeout)
        except Exception:
            image = None

        try:
            cam.stop()
            cam.destroy()
        except Exception:
            pass

        return image

    def find_best_camera_transform(self):
        self.get_logger().info("Searching best traffic-light camera angle...")

        # Seçim için kırmızı yapıyoruz; kırmızı piksel görünüyorsa kamera gerçekten ışığın önünü görüyor.
        self.set_all_lights("red")
        time.sleep(0.5)

        candidates = []

        angles = list(range(0, 360, 20))
        distances = [4.0, 5.5, 7.0, 9.0]
        z_offsets = [-0.5, 0.0, 0.5, 1.0]

        best = None

        for distance in distances:
            for z_offset in z_offsets:
                for angle in angles:
                    tf = self.make_camera_transform(angle, distance, z_offset)

                    image = self.spawn_temp_camera_and_capture(tf)
                    if image is None:
                        continue

                    bgr = image_to_bgr(image)
                    bbox = project_light_bbox(
                        self.light,
                        tf,
                        self.args.width,
                        self.args.height,
                        self.args.fov,
                        pad_ratio=self.args.crop_pad,
                    )

                    if bbox is None:
                        continue

                    x1, y1, x2, y2 = bbox
                    crop = bgr[y1:y2, x1:x2]
                    score = score_crop(crop)

                    # Merkeze yakınlık bonusu
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    center_bonus = 1000.0 * (
                        1.0
                        - min(
                            1.0,
                            abs(cx - self.args.width / 2.0) / max(1.0, self.args.width / 2.0),
                        )
                    )

                    score += center_bonus

                    candidates.append((score, angle, distance, z_offset, tf, bbox))

                    if best is None or score > best[0]:
                        best = (score, angle, distance, z_offset, tf, bbox)

        if best is None:
            self.get_logger().warning("Best camera bulunamadı. Fallback angle=0 distance=6 z=0 kullanılacak.")
            return self.make_camera_transform(0.0, 6.0, 0.0)

        score, angle, distance, z_offset, tf, bbox = best

        self.get_logger().info(
            f"BEST_CAMERA score={score:.1f}, angle={angle}, distance={distance}, z_offset={z_offset}, bbox={bbox}"
        )

        self.world.get_spectator().set_transform(tf)

        return tf

    def spawn_camera(self, tf):
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.args.width))
        bp.set_attribute("image_size_y", str(self.args.height))
        bp.set_attribute("fov", str(self.args.fov))
        bp.set_attribute("sensor_tick", str(1.0 / self.args.fps))

        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "adas_clean_tl_camera")

        cam = self.world.spawn_actor(bp, tf)
        cam.listen(self.on_image)

        self.world.get_spectator().set_transform(tf)

        return cam

    def make_clean_frame(self, bgr):
        bbox = project_light_bbox(
            self.light,
            self.camera_tf,
            self.args.width,
            self.args.height,
            self.args.fov,
            pad_ratio=self.args.crop_pad,
        )

        canvas = np.zeros((self.args.height, self.args.width, 3), dtype=np.uint8)

        if bbox is None:
            cv2.putText(
                canvas,
                "TRAFFIC LIGHT CROP NOT FOUND",
                (80, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            return canvas

        x1, y1, x2, y2 = bbox
        crop = bgr[y1:y2, x1:x2]

        if crop.size == 0:
            return canvas

        ch, cw = crop.shape[:2]

        target_h = int(self.args.height * self.args.object_scale)
        scale = target_h / max(1, ch)

        new_w = int(cw * scale)
        new_h = int(ch * scale)

        if new_w > int(self.args.width * 0.85):
            scale = int(self.args.width * 0.85) / max(1, cw)
            new_w = int(cw * scale)
            new_h = int(ch * scale)

        crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        ox = (self.args.width - new_w) // 2
        oy = (self.args.height - new_h) // 2

        canvas[oy:oy + new_h, ox:ox + new_w] = crop_resized

        if self.args.draw_state_text:
            cv2.putText(
                canvas,
                f"CARLA TL STATE: {self.current_state.upper()}",
                (40, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return canvas

    def on_image(self, image):
        bgr = image_to_bgr(image)
        clean = self.make_clean_frame(bgr)

        msg = self.bridge.cv2_to_imgmsg(clean, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "carla_clean_traffic_light"

        self.publisher.publish(msg)

        self.frame_count += 1
        now = time.time()

        if now - self.last_log >= 2.0:
            self.get_logger().info(
                f"Published clean TL frames: {self.frame_count}, state={self.current_state.upper()}"
            )
            self.last_log = now

    def destroy_node(self):
        try:
            if self.camera is not None:
                self.camera.stop()
                self.camera.destroy()
        except Exception:
            pass

        super().destroy_node()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)

    parser.add_argument("--map", default="Town10HD")
    parser.add_argument("--reload-map", action="store_true")
    parser.add_argument("--destroy-dynamic", action="store_true")

    parser.add_argument("--light-index", type=int, default=0)
    parser.add_argument("--state", choices=["red", "yellow", "green"], default="red")
    parser.add_argument("--cycle", action="store_true")
    parser.add_argument("--seconds", type=float, default=6.0)

    parser.add_argument("--topic", default="/adas/camera/front/image_raw")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=35.0)
    parser.add_argument("--fps", type=float, default=20.0)

    parser.add_argument("--crop-pad", type=float, default=2.2)
    parser.add_argument("--object-scale", type=float, default=0.78)
    parser.add_argument("--draw-state-text", action="store_true")

    args = parser.parse_args()

    rclpy.init()
    node = TrafficLightCleanPublisher(args)

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

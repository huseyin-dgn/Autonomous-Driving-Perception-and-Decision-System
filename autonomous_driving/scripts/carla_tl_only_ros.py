#!/usr/bin/env python3
import argparse
import math
import time
import signal
import sys

import carla
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--timeout", type=float, default=20.0)

    p.add_argument("--topic", default="/adas/camera/front/image_raw")

    # Görüntüde sadece merkez ROI kalsın, diğer yerler siyaha boyansın.
    # -1 verirsen kapalı olur.
    p.add_argument("--roi-left", type=int, default=-1)
    p.add_argument("--roi-right", type=int, default=-1)
    p.add_argument("--roi-top", type=int, default=-1)
    p.add_argument("--roi-bottom", type=int, default=-1)

    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--fov", type=float, default=35.0)

    p.add_argument("--state", choices=["red", "yellow", "green"], default="green")

    # Hangi trafik ışığı seçilecek?
    p.add_argument("--tl-index", type=int, default=0)

    # Kamera yerleşimi:
    # lane: trafik ışığının kontrol ettiği şeritten geriye gidip ışığa bakar.
    # orbit: ışığın etrafında yaw-offset ile manuel döndürür.
    p.add_argument("--placement", choices=["lane", "orbit"], default="lane")

    # lane placement ayarları
    p.add_argument("--distance", type=float, default=18.0)
    p.add_argument("--camera-z", type=float, default=1.70)
    p.add_argument("--target-z", type=float, default=4.60)

    # orbit fallback ayarları
    p.add_argument("--yaw-offset", type=float, default=0.0)

    p.add_argument("--destroy-vehicles", action="store_true", default=True)
    p.add_argument("--keep-vehicles", dest="destroy_vehicles", action="store_false")

    return p.parse_args()


def carla_state(name):
    if name == "red":
        return carla.TrafficLightState.Red
    if name == "yellow":
        return carla.TrafficLightState.Yellow
    if name == "green":
        return carla.TrafficLightState.Green
    raise ValueError(name)


def look_at(src, dst):
    dx = dst.x - src.x
    dy = dst.y - src.y
    dz = dst.z - src.z

    yaw = math.degrees(math.atan2(dy, dx))
    dist_xy = math.sqrt(dx * dx + dy * dy)
    pitch = math.degrees(math.atan2(dz, dist_xy))

    return carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)


def destroy_dynamic_actors(client, world):
    actors = world.get_actors()
    kill = []

    for a in actors:
        tid = a.type_id
        if tid.startswith("vehicle.") or tid.startswith("walker.") or tid.startswith("sensor."):
            kill.append(a)

    if not kill:
        print("[SIM] temizlenecek vehicle/walker/sensor yok")
        return

    print(f"[SIM] temizleniyor: {len(kill)} actor")
    cmds = [carla.command.DestroyActor(a.id) for a in kill]
    client.apply_batch_sync(cmds, True)
    time.sleep(0.5)


def get_traffic_lights(world):
    tls = []
    for a in world.get_actors():
        if "traffic_light" in a.type_id:
            tls.append(a)

    tls = sorted(tls, key=lambda x: x.id)
    return tls


def freeze_all_lights(tls, selected_tl, selected_state):
    for tl in tls:
        try:
            tl.freeze(True)
        except Exception:
            pass

        try:
            tl.set_green_time(9999.0)
            tl.set_yellow_time(9999.0)
            tl.set_red_time(9999.0)
        except Exception:
            pass

        try:
            if tl.id == selected_tl.id:
                tl.set_state(selected_state)
            else:
                tl.set_state(carla.TrafficLightState.Off)
        except Exception:
            pass


def build_camera_transform_lane(tl, distance, camera_z, target_z):
    """
    Trafik ışığının kontrol ettiği şeridi bulur.
    Kamerayı o şeritte stop line'ın gerisine koyar.
    Bu, ışığa sürücü bakışı verir.
    """
    target = carla.Location(
        x=tl.get_transform().location.x,
        y=tl.get_transform().location.y,
        z=tl.get_transform().location.z + target_z,
    )

    affected = []
    try:
        affected = list(tl.get_affected_lane_waypoints())
    except Exception:
        affected = []

    if not affected:
        raise RuntimeError("Bu trafik ışığında affected lane waypoint bulunamadı")

    wp = affected[0]

    prev = []
    try:
        prev = wp.previous(distance)
    except Exception:
        prev = []

    cam_wp = prev[0] if prev else wp

    cam_loc = carla.Location(
        x=cam_wp.transform.location.x,
        y=cam_wp.transform.location.y,
        z=cam_wp.transform.location.z + camera_z,
    )

    rot = look_at(cam_loc, target)
    return carla.Transform(cam_loc, rot)


def build_camera_transform_orbit(tl, distance, camera_z, target_z, yaw_offset):
    tf = tl.get_transform()

    target = carla.Location(
        x=tf.location.x,
        y=tf.location.y,
        z=tf.location.z + target_z,
    )

    base_yaw = tf.rotation.yaw + yaw_offset
    rad = math.radians(base_yaw)

    cam_loc = carla.Location(
        x=target.x + math.cos(rad) * distance,
        y=target.y + math.sin(rad) * distance,
        z=target.z + camera_z,
    )

    rot = look_at(cam_loc, target)
    return carla.Transform(cam_loc, rot)


class CarlaImagePublisher(Node):
    def __init__(self, topic, roi_left=-1, roi_right=-1, roi_top=-1, roi_bottom=-1):
        super().__init__("carla_tl_only_ros_publisher")
        self.pub = self.create_publisher(Image, topic, qos_profile_sensor_data)
        self.topic = topic
        self.roi_left = roi_left
        self.roi_right = roi_right
        self.roi_top = roi_top
        self.roi_bottom = roi_bottom
        self.frame_id = "carla_front_camera"
        self.count = 0
        self.last_log_t = time.time()
        self.last_count = 0

    def on_image(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))

        # CARLA raw = BGRA. ROS'a rgb8 basıyoruz.
        rgb = arr[:, :, :3][:, :, ::-1].copy()

        # ROI maskesi: ROI dışını siyaha boya.
        # Amaç: kenarda yarım görünen trafik ışıklarını modele hiç vermemek.
        h, w = rgb.shape[:2]
        if self.roi_left >= 0 and self.roi_right > self.roi_left and self.roi_top >= 0 and self.roi_bottom > self.roi_top:
            x1 = max(0, min(w, self.roi_left))
            x2 = max(0, min(w, self.roi_right))
            y1 = max(0, min(h, self.roi_top))
            y2 = max(0, min(h, self.roi_bottom))

            masked = np.zeros_like(rgb)
            masked[y1:y2, x1:x2] = rgb[y1:y2, x1:x2]
            rgb = masked

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = image.height
        msg.width = image.width
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = image.width * 3
        msg.data = rgb.tobytes()

        self.pub.publish(msg)

        self.count += 1
        now = time.time()
        if now - self.last_log_t >= 2.0:
            fps = (self.count - self.last_count) / (now - self.last_log_t)
            self.get_logger().info(f"publishing {self.topic} fps={fps:.1f}")
            self.last_log_t = now
            self.last_count = self.count


def main():
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    world = client.get_world()

    if args.destroy_vehicles:
        destroy_dynamic_actors(client, world)

    tls = get_traffic_lights(world)
    if not tls:
        print("[ERROR] CARLA haritasında traffic_light actor yok")
        sys.exit(1)

    if args.tl_index < 0 or args.tl_index >= len(tls):
        print(f"[ERROR] --tl-index geçersiz. Toplam trafik ışığı: {len(tls)}")
        for i, tl in enumerate(tls[:30]):
            loc = tl.get_transform().location
            print(f"  index={i:02d} id={tl.id} loc=({loc.x:.1f},{loc.y:.1f},{loc.z:.1f})")
        sys.exit(1)

    selected = tls[args.tl_index]
    selected_state = carla_state(args.state)

    freeze_all_lights(tls, selected, selected_state)

    loc = selected.get_transform().location
    print("==========================================")
    print("CARLA TL ONLY ROS PUBLISHER")
    print(f"topic       : {args.topic}")
    print(f"state       : {args.state}")
    print(f"traffic id  : {selected.id}")
    print(f"tl-index    : {args.tl_index} / total={len(tls)}")
    print(f"tl location : x={loc.x:.2f} y={loc.y:.2f} z={loc.z:.2f}")
    print(f"placement   : {args.placement}")
    print(f"resolution  : {args.width}x{args.height}")
    print(f"fov/fps     : {args.fov}/{args.fps}")
    print("==========================================")

    if args.placement == "lane":
        try:
            cam_tf = build_camera_transform_lane(
                selected,
                distance=args.distance,
                camera_z=args.camera_z,
                target_z=args.target_z,
            )
        except Exception as e:
            print(f"[WARN] lane placement olmadı: {e}")
            print("[WARN] orbit placement fallback kullanılacak")
            cam_tf = build_camera_transform_orbit(
                selected,
                distance=args.distance,
                camera_z=args.camera_z,
                target_z=args.target_z,
                yaw_offset=args.yaw_offset,
            )
    else:
        cam_tf = build_camera_transform_orbit(
            selected,
            distance=args.distance,
            camera_z=args.camera_z,
            target_z=args.target_z,
            yaw_offset=args.yaw_offset,
        )

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(args.width))
    cam_bp.set_attribute("image_size_y", str(args.height))
    cam_bp.set_attribute("fov", str(args.fov))
    cam_bp.set_attribute("sensor_tick", str(1.0 / max(args.fps, 0.1)))
    cam_bp.set_attribute("gamma", "2.2")

    try:
        cam_bp.set_attribute("enable_postprocess_effects", "False")
    except Exception:
        pass

    camera = world.spawn_actor(cam_bp, cam_tf)
    world.get_spectator().set_transform(cam_tf)

    rclpy.init()
    node = CarlaImagePublisher(args.topic, args.roi_left, args.roi_right, args.roi_top, args.roi_bottom)

    camera.listen(lambda img: node.on_image(img))

    stop = {"flag": False}

    def handle_sigint(sig, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while rclpy.ok() and not stop["flag"]:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        print("[SIM] kapatılıyor")
        try:
            camera.stop()
        except Exception:
            pass
        try:
            camera.destroy()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

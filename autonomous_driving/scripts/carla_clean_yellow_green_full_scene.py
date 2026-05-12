#!/usr/bin/env python3
import argparse
import math
import random
import time
import weakref

import carla

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
except Exception:
    rclpy = None
    Node = object
    Image = None


ACTORS = []


def destroy_all():
    global ACTORS
    for a in ACTORS:
        try:
            if a is not None and a.is_alive:
                a.destroy()
        except Exception:
            pass
    ACTORS = []


def find_bp(world, filters):
    bps = world.get_blueprint_library()
    for f in filters:
        xs = bps.filter(f)
        if xs:
            return xs[0]
    return None


def set_attr(bp, key, value):
    if bp and bp.has_attribute(key):
        try:
            bp.set_attribute(key, str(value))
        except Exception:
            pass


def spawn_actor(world, bp, transform, name):
    actor = world.try_spawn_actor(bp, transform)
    if actor is None:
        raise RuntimeError(f"Spawn failed: {name}")
    ACTORS.append(actor)
    print(f"[SPAWN] {name}: id={actor.id}, type={actor.type_id}")
    return actor


def yaw_to_forward(yaw_deg):
    r = math.radians(yaw_deg)
    return carla.Vector3D(math.cos(r), math.sin(r), 0.0)


def yaw_to_right(yaw_deg):
    r = math.radians(yaw_deg + 90.0)
    return carla.Vector3D(math.cos(r), math.sin(r), 0.0)


def loc(base, yaw, forward=0.0, right=0.0, z=0.0):
    f = yaw_to_forward(yaw)
    s = yaw_to_right(yaw)
    return carla.Location(
        x=base.x + f.x * forward + s.x * right,
        y=base.y + f.y * forward + s.y * right,
        z=base.z + z,
    )


def spawn_vehicle(world, bp_filter, transform, name, color=None):
    bp = find_bp(world, bp_filter)
    if bp is None:
        raise RuntimeError(f"Blueprint not found: {name} {bp_filter}")

    set_attr(bp, "role_name", name)
    set_attr(bp, "is_invincible", "true")

    if color and bp.has_attribute("color"):
        bp.set_attribute("color", color)

    actor = spawn_actor(world, bp, transform, name)
    try:
        actor.set_autopilot(False)
        actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
    except Exception:
        pass
    return actor


def spawn_walker(world, transform, name):
    bp = find_bp(world, [
        "walker.pedestrian.0001",
        "walker.pedestrian.0002",
        "walker.pedestrian.0003",
        "walker.pedestrian.*",
    ])
    if bp is None:
        raise RuntimeError("Walker blueprint not found")

    set_attr(bp, "role_name", name)
    set_attr(bp, "is_invincible", "true")

    actor = spawn_actor(world, bp, transform, name)

    try:
        actor.set_simulate_physics(True)
    except Exception:
        pass

    return actor


def draw_clean_traffic_light(world, center, yaw, state, life_time=0.20):
    """
    CARLA debug geometry ile sahne içine 3 lensli trafik ışığı çizer.
    Üstünde yazı yok. Sadece panel + üç lens.
    Kamera görüntüsünde görünür, ama gereksiz RED/YELLOW/GREEN yazısı basmaz.
    """

    # Renkler
    black = carla.Color(3, 3, 3)
    dark = carla.Color(18, 18, 18)
    pole_color = carla.Color(45, 45, 45)

    if state == "yellow":
        active = carla.Color(255, 235, 0)
    elif state == "green":
        active = carla.Color(0, 255, 35)
    elif state == "red":
        active = carla.Color(255, 0, 0)
    else:
        active = carla.Color(80, 80, 80)

    red_c = active if state == "red" else dark
    yellow_c = active if state == "yellow" else dark
    green_c = active if state == "green" else dark

    # Panel ölçüsü
    # Büyük çiziyoruz ki YOLO küçük/uzak ışık diye kaçırmasın.
    panel_h = 2.25
    panel_w = 0.82
    panel_t = 0.12

    # Panel
    panel_tf = carla.Transform(
        center,
        carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0)
    )
    panel_box = carla.BoundingBox(
        carla.Location(0.0, 0.0, 0.0),
        carla.Vector3D(panel_t, panel_w / 2.0, panel_h / 2.0)
    )
    world.debug.draw_box(
        panel_box,
        panel_tf.rotation,
        thickness=0.08,
        color=black,
        life_time=life_time
    )

    # Direk
    pole_center = carla.Location(center.x, center.y, center.z - 1.85)
    pole_box = carla.BoundingBox(
        pole_center,
        carla.Vector3D(0.06, 0.06, 1.25)
    )
    world.debug.draw_box(
        pole_box,
        carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0),
        thickness=0.06,
        color=pole_color,
        life_time=life_time
    )

    # Lensleri panel yüzeyine yerleştir
    f = yaw_to_forward(yaw)
    lens_x_offset = 0.12
    lens_base = carla.Location(
        x=center.x + f.x * lens_x_offset,
        y=center.y + f.y * lens_x_offset,
        z=center.z
    )

    lens_positions = [
        ("red", 0.62, red_c),
        ("yellow", 0.0, yellow_c),
        ("green", -0.62, green_c),
    ]

    for _, dz, color in lens_positions:
        p = carla.Location(lens_base.x, lens_base.y, lens_base.z + dz)

        # Daireyi kutucuk/cross çizgilerle kalınlaştırıyoruz.
        world.debug.draw_point(
            p,
            size=0.22,
            color=color,
            life_time=life_time
        )

        # Lens çerçevesi
        world.debug.draw_point(
            carla.Location(p.x, p.y, p.z),
            size=0.30,
            color=dark if color != active else color,
            life_time=life_time
        )


class CarlaImagePublisher(Node):
    def __init__(self, world, camera, width, height, fps, draw_lights, lights):
        super().__init__("carla_clean_full_scene_publisher")
        self.world = world
        self.camera = camera
        self.width = width
        self.height = height
        self.fps = fps
        self.draw_lights = draw_lights
        self.lights = lights
        self.pub = self.create_publisher(Image, "/adas/camera/front/image_raw", 10)
        self.last_image = None
        self.timer = self.create_timer(1.0 / float(fps), self.publish_latest)

    def publish_latest(self):
        if self.draw_lights:
            for light in self.lights:
                draw_clean_traffic_light(
                    self.world,
                    light["center"],
                    light["yaw"],
                    light["state"],
                    life_time=0.20
                )

        if self.last_image is None:
            return

        img = self.last_image
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "carla_front_camera"
        msg.height = self.height
        msg.width = self.width
        msg.encoding = "bgra8"
        msg.is_bigendian = 0
        msg.step = self.width * 4
        msg.data = img.raw_data
        self.pub.publish(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Town10HD_Opt")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--yellow-green", action="store_true", default=True)
    args = parser.parse_args()

    if rclpy is None:
        raise RuntimeError("ROS2 Python modülleri yok. source /opt/ros/humble/setup.bash yap.")

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    world = client.load_world(args.map)
    time.sleep(2.0)

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    world.set_weather(carla.WeatherParameters(
        cloudiness=5.0,
        precipitation=0.0,
        sun_altitude_angle=55.0,
        sun_azimuth_angle=35.0,
        fog_density=0.0,
        wetness=0.0
    ))

    destroy_all()

    # Var olan CARLA map trafik ışıklarını kapatıyoruz.
    # Böylece sahnede yanlışlıkla kırmızı/başka ışık baskın gelmez.
    for tl in world.get_actors().filter("traffic.traffic_light*"):
        try:
            tl.set_state(carla.TrafficLightState.Off)
            tl.freeze(True)
        except Exception:
            pass

    # Stabil Town10HD yolu.
    # Kamera karşıdan bakacak; nesneler geniş ve net görünecek.
    base = carla.Location(x=-45.0, y=16.0, z=0.40)
    yaw = 0.0

    # Ego sadece kamera taşıyıcı.
    ego_bp = find_bp(world, ["vehicle.tesla.model3", "vehicle.audi.a2", "vehicle.*"])
    set_attr(ego_bp, "role_name", "ego_vehicle")
    ego = spawn_actor(
        world,
        ego_bp,
        carla.Transform(
            carla.Location(x=base.x - 14.0, y=base.y, z=0.45),
            carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0)
        ),
        "ego_vehicle"
    )
    try:
        ego.set_autopilot(False)
        ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
    except Exception:
        pass

    # Araçlar: 2 adet
    spawn_vehicle(
        world,
        ["vehicle.tesla.model3", "vehicle.audi.a2", "vehicle.dodge.charger_police"],
        carla.Transform(loc(base, yaw, forward=28.0, right=-1.8, z=0.35), carla.Rotation(yaw=yaw)),
        "adas_test_vehicle_1",
        color="255,0,0"
    )

    spawn_vehicle(
        world,
        ["vehicle.audi.a2", "vehicle.tesla.model3", "vehicle.dodge.charger_2020"],
        carla.Transform(loc(base, yaw, forward=43.0, right=1.8, z=0.35), carla.Rotation(yaw=yaw)),
        "adas_test_vehicle_2",
        color="0,0,255"
    )

    # Motosikletler: 2 adet, arka arkaya
    spawn_vehicle(
        world,
        ["vehicle.kawasaki.ninja", "vehicle.yamaha.yzf", "vehicle.harley-davidson.low_rider"],
        carla.Transform(loc(base, yaw, forward=20.0, right=-5.2, z=0.45), carla.Rotation(yaw=yaw)),
        "adas_test_motorcycle_1"
    )

    spawn_vehicle(
        world,
        ["vehicle.yamaha.yzf", "vehicle.kawasaki.ninja", "vehicle.harley-davidson.low_rider"],
        carla.Transform(loc(base, yaw, forward=28.0, right=-5.2, z=0.45), carla.Rotation(yaw=yaw)),
        "adas_test_motorcycle_2"
    )

    # İnsanlar: 2 adet, arka arkaya ve yakın/büyük
    spawn_walker(
        world,
        carla.Transform(loc(base, yaw, forward=18.0, right=5.0, z=0.55), carla.Rotation(yaw=180.0)),
        "adas_test_person_1"
    )

    spawn_walker(
        world,
        carla.Transform(loc(base, yaw, forward=25.0, right=5.0, z=0.55), carla.Rotation(yaw=180.0)),
        "adas_test_person_2"
    )

    # Temiz trafik ışıkları: sadece sarı ve yeşil.
    # Bunlar sahne içine çizilir, üstünde yazı yok.
    lights = [
        {
            "state": "yellow",
            "center": loc(base, yaw, forward=36.0, right=-0.7, z=3.10),
            "yaw": yaw,
        },
        {
            "state": "green",
            "center": loc(base, yaw, forward=42.0, right=0.9, z=3.10),
            "yaw": yaw,
        },
    ]

    # Kamera
    cam_bp = find_bp(world, ["sensor.camera.rgb"])
    cam_bp.set_attribute("image_size_x", str(args.width))
    cam_bp.set_attribute("image_size_y", str(args.height))
    cam_bp.set_attribute("fov", str(args.fov))
    cam_bp.set_attribute("sensor_tick", str(1.0 / float(args.fps)))

    cam_tf = carla.Transform(
        carla.Location(x=1.70, y=0.0, z=1.55),
        carla.Rotation(pitch=-2.0, yaw=0.0, roll=0.0)
    )

    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
    ACTORS.append(camera)
    print(f"[SPAWN] camera: id={camera.id}, topic=/adas/camera/front/image_raw")

    rclpy.init()
    node = CarlaImagePublisher(
        world=world,
        camera=camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        draw_lights=True,
        lights=lights
    )

    weak_node = weakref.ref(node)

    def on_image(image):
        n = weak_node()
        if n is not None:
            n.last_image = image

    camera.listen(on_image)

    print("========================================")
    print("CLEAN CARLA FULL ADAS SCENE")
    print("Objects:")
    print("  vehicles     : 2")
    print("  motorcycles  : 2, arka arkaya")
    print("  pedestrians  : 2, arka arkaya")
    print("  traffic light: yellow + green only")
    print("ROS topic:")
    print("  /adas/camera/front/image_raw")
    print("========================================")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            camera.stop()
        except Exception:
            pass
        destroy_all()
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()

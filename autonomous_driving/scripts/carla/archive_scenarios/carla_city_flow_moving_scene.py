#!/usr/bin/env python3
import os
import sys
import glob
import time
import math
import random
import argparse


def add_carla_api():
    try:
        import carla
        return
    except ImportError:
        pass

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    roots = [
        os.environ.get("CARLA_ROOT"),
        os.path.expanduser("~/CARLA_DISK"),
        os.path.expanduser("~/CARLA"),
        "/opt/carla",
    ]

    for root in roots:
        if not root:
            continue

        pattern = os.path.join(
            root,
            "PythonAPI",
            "carla",
            "dist",
            f"carla-*{py_ver}-linux-x86_64.egg",
        )

        eggs = glob.glob(pattern)

        if eggs:
            sys.path.append(eggs[0])
            return


add_carla_api()

import carla




# ADAS_CAMERA_STABLE_SETTINGS
def safe_set_camera_attribute(camera_bp, key, value):
    try:
        if camera_bp.has_attribute(key):
            camera_bp.set_attribute(key, str(value))
    except Exception:
        pass


def apply_stable_rgb_camera_settings(camera_bp):
    safe_set_camera_attribute(camera_bp, "enable_postprocess_effects", "false")
    safe_set_camera_attribute(camera_bp, "gamma", "2.2")
    safe_set_camera_attribute(camera_bp, "exposure_mode", "manual")
    safe_set_camera_attribute(camera_bp, "exposure_compensation", "0.0")
    safe_set_camera_attribute(camera_bp, "shutter_speed", "200.0")
    safe_set_camera_attribute(camera_bp, "iso", "100.0")
    safe_set_camera_attribute(camera_bp, "fstop", "8.0")
    safe_set_camera_attribute(camera_bp, "bloom_intensity", "0.0")
    safe_set_camera_attribute(camera_bp, "lens_flare_intensity", "0.0")
    safe_set_camera_attribute(camera_bp, "motion_blur_intensity", "0.0")

# ADAS_FORCE_CLEAR_DAY_WEATHER
def force_clear_day_weather(world):
    try:
        weather = carla.WeatherParameters(
            cloudiness=5.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=0.0,
            sun_azimuth_angle=45.0,
            sun_altitude_angle=75.0,
            fog_density=0.0,
            fog_distance=100000.0,
            fog_falloff=0.0,
            wetness=0.0,
        )
        world.set_weather(weather)
        print("[ADAS] Weather forced: CLEAR DAY / NO FOG / NO RAIN / NO NIGHT")
    except Exception as exc:
        print(f"[WARN] force_clear_day_weather failed: {exc}")

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalize_angle(angle):
    while angle > 180:
        angle -= 360

    while angle < -180:
        angle += 360

    return angle


def forward_vector(transform):
    yaw = math.radians(transform.rotation.yaw)
    return carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)


def right_vector(transform):
    yaw = math.radians(transform.rotation.yaw + 90.0)
    return carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)


def offset_transform(transform, forward=0.0, lateral=0.0, z=0.0, yaw=0.0):
    fv = forward_vector(transform)
    rv = right_vector(transform)

    loc = carla.Location(
        x=transform.location.x + fv.x * forward + rv.x * lateral,
        y=transform.location.y + fv.y * forward + rv.y * lateral,
        z=transform.location.z + z,
    )

    rot = carla.Rotation(
        pitch=transform.rotation.pitch,
        yaw=transform.rotation.yaw + yaw,
        roll=transform.rotation.roll,
    )

    return carla.Transform(loc, rot)


def get_speed_kmh(actor):
    v = actor.get_velocity()
    return 3.6 * math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def pick_bp(world, patterns):
    bp_lib = world.get_blueprint_library()

    for pattern in patterns:
        bps = bp_lib.filter(pattern)
        if bps:
            return random.choice(bps)

    return None


def set_vehicle_attrs(bp, role_name):
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role_name)

    if bp.has_attribute("color"):
        colors = bp.get_attribute("color").recommended_values
        if colors:
            bp.set_attribute("color", random.choice(colors))

    if bp.has_attribute("driver_id"):
        drivers = bp.get_attribute("driver_id").recommended_values
        if drivers:
            bp.set_attribute("driver_id", random.choice(drivers))

    return bp


def get_straight_spawn(world):
    carla_map = world.get_map()
    spawns = carla_map.get_spawn_points()

    best = None
    best_score = -999999

    for sp in spawns:
        wp = carla_map.get_waypoint(
            sp.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if wp is None:
            continue

        if wp.is_junction:
            continue

        score = 0
        current = wp
        base_yaw = wp.transform.rotation.yaw

        for _ in range(12):
            nxts = current.next(8.0)

            if not nxts:
                score -= 100
                break

            nxt = min(
                nxts,
                key=lambda w: abs(normalize_angle(w.transform.rotation.yaw - current.transform.rotation.yaw)),
            )

            yaw_diff = abs(normalize_angle(nxt.transform.rotation.yaw - base_yaw))

            if yaw_diff < 8:
                score += 5
            elif yaw_diff < 18:
                score += 2
            else:
                score -= 8

            if nxt.is_junction:
                score -= 2
            else:
                score += 1

            current = nxt

        if score > best_score:
            best_score = score
            best = sp

    if best is None:
        best = random.choice(spawns)

    best.location.z += 0.6
    return best


def advance_waypoint(wp, distance):
    current = wp
    left = distance

    while left > 0:
        step = min(5.0, left)
        nxts = current.next(step)

        if not nxts:
            return current

        current = min(
            nxts,
            key=lambda w: abs(normalize_angle(w.transform.rotation.yaw - current.transform.rotation.yaw)),
        )

        left -= step

    return current


def clear_scene(world):
    actors = world.get_actors()
    targets = []

    for a in actors:
        tid = a.type_id

        if tid.startswith("sensor."):
            targets.append(a)
        elif tid.startswith("vehicle."):
            targets.append(a)
        elif tid.startswith("walker."):
            targets.append(a)
        elif tid.startswith("controller.ai.walker"):
            targets.append(a)
        elif tid.startswith("static.prop.trafficcone"):
            targets.append(a)
        elif tid.startswith("static.prop.trafficwarning"):
            targets.append(a)
        elif tid.startswith("static.prop.streetsign"):
            targets.append(a)
        elif tid.startswith("static.prop.busstop"):
            targets.append(a)

    for a in targets:
        try:
            if a.is_alive:
                a.destroy()
        except Exception:
            pass

    print(f"[CLEAR] Temizlenen actor sayısı: {len(targets)}")


def setup_world(world):
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    world.set_weather(
        carla.WeatherParameters(
            cloudiness=25.0,
            precipitation=0.0,
            sun_altitude_angle=45.0,
            sun_azimuth_angle=35.0,
            fog_density=0.0,
            wetness=0.0,
        )
    )


def spawn_ego(world, spawn_tf):
    bp = pick_bp(
        world,
        [
            "vehicle.tesla.model3",
            "vehicle.lincoln.mkz_2020",
            "vehicle.audi.tt",
            "vehicle.*",
        ],
    )

    if bp is None:
        raise RuntimeError("Ego araç blueprint bulunamadı.")

    set_vehicle_attrs(bp, "ego_vehicle")

    ego = world.try_spawn_actor(bp, spawn_tf)

    if ego is None:
        spawn_tf.location.z += 1.0
        ego = world.try_spawn_actor(bp, spawn_tf)

    if ego is None:
        raise RuntimeError("Ego araç spawn edilemedi.")

    ego.set_autopilot(False)

    print(f"[EGO] Spawn edildi: {ego.type_id}")
    return ego


def spawn_rgb_front_camera(world, ego, width, height, fov):
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    apply_stable_rgb_camera_settings(cam_bp)  # ADAS_CAMERA_STABLE_CALL

    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov))
    cam_bp.set_attribute("sensor_tick", "0.033")

    if cam_bp.has_attribute("role_name"):
        cam_bp.set_attribute("role_name", "rgb_front")

    cam_tf = carla.Transform(
        carla.Location(x=1.85, y=0.0, z=1.55),
        carla.Rotation(pitch=-3.0, yaw=0.0, roll=0.0),
    )

    cam = world.spawn_actor(
        cam_bp,
        cam_tf,
        attach_to=ego,
        attachment_type=carla.AttachmentType.Rigid,
    )

    print("[CAMERA] Ego üzerine rgb_front kamera bağlandı.")
    print("[CAMERA] role_name = rgb_front")
    return cam


def spawn_front_vehicles(world, tm, tm_port, ego_wp):
    actors = []

    distances = [24, 45]

    for i, d in enumerate(distances):
        wp = advance_waypoint(ego_wp, d)

        tf = carla.Transform(
            carla.Location(
                wp.transform.location.x,
                wp.transform.location.y,
                wp.transform.location.z + 0.6,
            ),
            wp.transform.rotation,
        )

        bp = pick_bp(
            world,
            [
                "vehicle.audi.*",
                "vehicle.bmw.*",
                "vehicle.mercedes.*",
                "vehicle.lincoln.*",
                "vehicle.tesla.*",
                "vehicle.toyota.*",
                "vehicle.*",
            ],
        )

        if bp is None:
            continue

        set_vehicle_attrs(bp, f"front_vehicle_{i}")

        v = world.try_spawn_actor(bp, tf)

        if v:
            v.set_autopilot(True, tm_port)
            tm.distance_to_leading_vehicle(v, 8.0)
            tm.vehicle_percentage_speed_difference(v, random.uniform(15, 35))
            actors.append(v)
            print(f"[VEHICLE] Öne araç eklendi: {v.type_id} / mesafe={d}m")

    return actors


def spawn_side_vehicles(world, ego_wp):
    actors = []

    specs = [
        (32, 4.3),
    ]

    for i, (d, lateral) in enumerate(specs):
        wp = advance_waypoint(ego_wp, d)

        base_tf = carla.Transform(
            carla.Location(
                wp.transform.location.x,
                wp.transform.location.y,
                wp.transform.location.z + 0.6,
            ),
            wp.transform.rotation,
        )

        tf = offset_transform(
            base_tf,
            lateral=lateral,
            yaw=random.choice([0.0, 180.0]),
        )

        bp = pick_bp(
            world,
            [
                "vehicle.audi.*",
                "vehicle.bmw.*",
                "vehicle.mercedes.*",
                "vehicle.lincoln.*",
                "vehicle.tesla.*",
                "vehicle.toyota.*",
                "vehicle.*",
            ],
        )

        if bp is None:
            continue

        set_vehicle_attrs(bp, f"side_parked_vehicle_{i}")

        v = world.try_spawn_actor(bp, tf)

        if v:
            v.set_autopilot(False)
            actors.append(v)
            print(f"[VEHICLE] Yan/park araç eklendi: {v.type_id}")

    return actors


def spawn_city_traffic(world, tm, tm_port, ego_location, count=10):
    actors = []
    spawns = world.get_map().get_spawn_points()
    random.shuffle(spawns)

    made = 0

    for sp in spawns:
        if made >= count:
            break

        dist = sp.location.distance(ego_location)

        if dist < 45 or dist > 220:
            continue

        sp.location.z += 0.5

        bp = pick_bp(
            world,
            [
                "vehicle.audi.*",
                "vehicle.bmw.*",
                "vehicle.mercedes.*",
                "vehicle.lincoln.*",
                "vehicle.tesla.*",
                "vehicle.toyota.*",
                "vehicle.*",
            ],
        )

        if bp is None:
            continue

        set_vehicle_attrs(bp, f"city_flow_vehicle_{made}")

        v = world.try_spawn_actor(bp, sp)

        if v:
            v.set_autopilot(True, tm_port)
            tm.distance_to_leading_vehicle(v, 7.0)
            tm.vehicle_percentage_speed_difference(v, random.uniform(0, 45))
            actors.append(v)
            made += 1
            print(f"[TRAFFIC] Şehir trafiği aracı eklendi: {v.type_id}")

    return actors


def spawn_walkers(world, ego_wp, enable=True):
    actors = []

    if not enable:
        print("[WALKER] Yayalar kapalı.")
        return actors

    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter("walker.pedestrian.*")

    if not walker_bps:
        print("[WALKER] Pedestrian blueprint bulunamadı.")
        return actors

    # Demo için yayalar hareket etmiyor.
    # Amaç: kırmızı bariyer/bina/direk arkasına girmeden net insan bbox göstermek.
    specs = [
        (22, 1.8),
        (34, -1.8),
        (48, 1.8),
    ]

    for i, (d, lateral) in enumerate(specs):
        wp = advance_waypoint(ego_wp, d)

        base_tf = carla.Transform(
            carla.Location(
                wp.transform.location.x,
                wp.transform.location.y,
                wp.transform.location.z + 0.75,
            ),
            wp.transform.rotation,
        )

        # Yaya yolun açık kısmında dursun, kameraya doğru baksın.
        tf = offset_transform(
            base_tf,
            lateral=lateral,
            yaw=180.0,
        )

        walker_bp = walker_bps[i % len(walker_bps)]

        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")

        w = world.try_spawn_actor(walker_bp, tf)

        if not w:
            print(f"[WALKER] Yaya spawn edilemedi: d={d}, lateral={lateral}")
            continue

        try:
            w.set_simulate_physics(False)
        except Exception:
            pass

        actors.append(w)
        print(f"[WALKER] Statik demo yayası eklendi: {w.type_id} d={d} lateral={lateral}")

    return actors

def spawn_props(world, ego_wp):
    print("[PROP] Temiz demo: prop/levha/yol çalışma objeleri kapalı.")
    return []

def drive_ego_lane_follow(ego, carla_map, target_speed):
    tf = ego.get_transform()

    wp = carla_map.get_waypoint(
        tf.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if wp is None:
        ego.apply_control(carla.VehicleControl(throttle=0.25, steer=0.0, brake=0.0))
        return

    nxts = wp.next(10.0)

    if not nxts:
        ego.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.4))
        return

    nxt = min(
        nxts,
        key=lambda w: abs(normalize_angle(w.transform.rotation.yaw - tf.rotation.yaw)),
    )

    target = nxt.transform.location

    dx = target.x - tf.location.x
    dy = target.y - tf.location.y

    target_yaw = math.degrees(math.atan2(dy, dx))
    yaw_err = normalize_angle(target_yaw - tf.rotation.yaw)

    steer = clamp(yaw_err / 45.0, -0.38, 0.38)

    speed = get_speed_kmh(ego)

    if speed < target_speed - 3:
        throttle = 0.42
        brake = 0.0
    elif speed > target_speed + 5:
        throttle = 0.0
        brake = 0.18
    else:
        throttle = 0.22
        brake = 0.0

    ego.apply_control(
        carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
        )
    )


def follow_spectator(world, ego):
    spectator = world.get_spectator()
    tf = ego.get_transform()

    cam_tf = offset_transform(
        tf,
        forward=-8.0,
        lateral=0.0,
        z=4.2,
        yaw=0.0,
    )

    cam_tf.rotation.pitch = -18.0
    cam_tf.rotation.yaw = tf.rotation.yaw

    spectator.set_transform(cam_tf)

def force_move_ego_on_lane(ego, carla_map, speed_kmh, dt=0.05):
    speed_mps = speed_kmh / 3.6
    step_distance = max(0.15, speed_mps * dt)

    tf = ego.get_transform()

    wp = carla_map.get_waypoint(
        tf.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if wp is None:
        fv = forward_vector(tf)
        new_tf = carla.Transform(
            carla.Location(
                x=tf.location.x + fv.x * step_distance,
                y=tf.location.y + fv.y * step_distance,
                z=tf.location.z,
            ),
            tf.rotation,
        )
        ego.set_transform(new_tf)
        return

    next_points = wp.next(step_distance)

    if not next_points:
        next_points = wp.next(1.0)

    if not next_points:
        return

    next_wp = min(
        next_points,
        key=lambda w: abs(normalize_angle(w.transform.rotation.yaw - tf.rotation.yaw)),
    )

    new_tf = carla.Transform(
        carla.Location(
            x=next_wp.transform.location.x,
            y=next_wp.transform.location.y,
            z=next_wp.transform.location.z + 0.60,
        ),
        next_wp.transform.rotation,
    )

    ego.set_transform(new_tf)

    fv = forward_vector(new_tf)
    ego.set_target_velocity(
        carla.Vector3D(
            fv.x * speed_mps,
            fv.y * speed_mps,
            0.0,
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Town04")
    parser.add_argument("--no-load-map", action="store_true")
    parser.add_argument("--clear", action="store_true")

    parser.add_argument("--ego-speed", type=float, default=28.0)
    parser.add_argument("--tm-port", type=int, default=8000)

    parser.add_argument("--width", type=int, default=1240)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=85.0)

    parser.add_argument("--no-walkers", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)

    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    if args.no_load_map:
        world = client.get_world()
        force_clear_day_weather(world)  # ADAS_FORCE_CLEAR_DAY_CALL
        print(f"[WORLD] Mevcut map kullanılıyor: {world.get_map().name}")
    else:
        print(f"[WORLD] Map yükleniyor: {args.map}")
        world = client.load_world(args.map)
        force_clear_day_weather(world)  # ADAS_FORCE_CLEAR_DAY_CALL
        time.sleep(2.0)

    setup_world(world)

    if args.clear:
        clear_scene(world)
        time.sleep(1.0)

    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(False)
    tm.set_global_distance_to_leading_vehicle(7.0)
    tm.global_percentage_speed_difference(20.0)

    carla_map = world.get_map()

    ego_spawn = get_straight_spawn(world)
    ego = spawn_ego(world, ego_spawn)

    ego.set_autopilot(False)
    try:
        ego.set_simulate_physics(False)
    except Exception:
        pass
    print("[EGO] Kinematik hareket aktif. Ego araç waypoint üzerinden zorla ilerleyecek.")

    ego_wp = carla_map.get_waypoint(
        ego.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    actors = [ego]

    rgb_front = spawn_rgb_front_camera(
        world,
        ego,
        width=args.width,
        height=args.height,
        fov=args.fov,
    )

    actors.append(rgb_front)

    actors.extend(spawn_front_vehicles(world, tm, args.tm_port, ego_wp))
    actors.extend(spawn_side_vehicles(world, ego_wp))
    actors.extend(spawn_city_traffic(world, tm, args.tm_port, ego.get_location(), count=1))
    actors.extend(spawn_walkers(world, ego_wp, enable=not args.no_walkers))
    actors.extend(spawn_props(world, ego_wp))

    print("")
    print("====================================================")
    print(" CARLA CITY FLOW MOVING ADAS SCENE")
    print("====================================================")
    print(f"Map              : {world.get_map().name}")
    print(f"Ego speed target : {args.ego_speed} km/h")
    print(f"Camera role_name : rgb_front")
    print(f"Camera size      : {args.width}x{args.height}")
    print("")
    print("Bundan sonra Terminal 3:")
    print("  python3 scripts/carla_rgb_front_ros_only.py")
    print("")
    print("Bundan sonra Terminal 4:")
    print("  bash scripts/run_perception_carla.sh")
    print("")
    print("Çıkmak için CTRL+C")
    print("====================================================")
    print("")

    start = time.time()

    try:
        while True:
            force_move_ego_on_lane(ego, carla_map, args.ego_speed, dt=0.05)
            follow_spectator(world, ego)

            try:
                world.wait_for_tick(1.0)
            except Exception:
                pass

            if args.duration > 0:
                if time.time() - start >= args.duration:
                    break

            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n[EXIT] Sahne durduruldu.")

    finally:
        print("[CLEANUP] Actorlar temizleniyor...")

        for a in reversed(actors):
            try:
                if a.is_alive:
                    if a.type_id.startswith("controller.ai.walker"):
                        a.stop()
                    a.destroy()
            except Exception:
                pass

        print("[DONE]")


if __name__ == "__main__":
    main()

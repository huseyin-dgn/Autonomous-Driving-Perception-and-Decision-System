#!/usr/bin/env python3
import argparse
import math
import random
import time

import carla


ROLE_PREFIX = "adas_big_showcase"
CAMERA_ROLE = "rgb_front"


def set_attr_if(bp, name, value):
    if bp.has_attribute(name):
        bp.set_attribute(name, str(value))


def destroy_actor_safe(actor):
    try:
        actor.destroy()
    except Exception:
        pass


def clear_scene(world):
    actors = world.get_actors()
    targets = []

    for a in actors:
        tid = a.type_id
        role = a.attributes.get("role_name", "")

        if tid.startswith("vehicle."):
            targets.append(a)
        elif tid.startswith("walker.pedestrian."):
            targets.append(a)
        elif tid.startswith("controller.ai.walker"):
            targets.append(a)
        elif tid.startswith("sensor.camera."):
            targets.append(a)
        elif role.startswith(ROLE_PREFIX) or role == CAMERA_ROLE:
            targets.append(a)

    for a in targets:
        destroy_actor_safe(a)

    print(f"[CLEAR] Destroyed actors: {len(targets)}")


def local_to_world(base_tf, x, y, z=0.0):
    yaw = math.radians(base_tf.rotation.yaw)
    bx = base_tf.location.x
    by = base_tf.location.y
    bz = base_tf.location.z

    wx = bx + x * math.cos(yaw) - y * math.sin(yaw)
    wy = by + x * math.sin(yaw) + y * math.cos(yaw)
    wz = bz + z

    return carla.Location(wx, wy, wz)


def yaw_to_location(src, dst):
    dx = dst.x - src.x
    dy = dst.y - src.y
    return math.degrees(math.atan2(dy, dx))


def choose_ego_transform_near_traffic_light(world, distance=30.0):
    carla_map = world.get_map()
    traffic_lights = list(world.get_actors().filter("*traffic_light*"))

    random.shuffle(traffic_lights)

    for tl in traffic_lights:
        try:
            tl_loc = tl.get_location()
            wp = carla_map.get_waypoint(
                tl_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            if wp is None:
                continue

            prev_wps = wp.previous(distance)
            if prev_wps:
                ego_wp = prev_wps[0]
                ego_tf = ego_wp.transform
                ego_tf.location.z += 0.35

                yaw = yaw_to_location(ego_tf.location, tl_loc)
                ego_tf.rotation = carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0)

                print("[SCENE] Selected traffic light:", tl.id)
                print("[SCENE] Ego placed near traffic light.")
                return ego_tf, tl
        except Exception:
            continue

    spawns = carla_map.get_spawn_points()
    if not spawns:
        raise RuntimeError("No spawn points found.")

    tf = spawns[0]
    tf.location.z += 0.35
    print("[SCENE] No usable traffic light found. Using spawn point 0.")
    return tf, None


def set_traffic_lights(world, state_name="Red"):
    state_map = {
        "red": carla.TrafficLightState.Red,
        "yellow": carla.TrafficLightState.Yellow,
        "green": carla.TrafficLightState.Green,
    }

    state = state_map.get(state_name.lower(), carla.TrafficLightState.Red)
    tls = list(world.get_actors().filter("*traffic_light*"))

    for tl in tls:
        try:
            tl.set_state(state)
            tl.set_red_time(999.0)
            tl.set_yellow_time(999.0)
            tl.set_green_time(999.0)
            try:
                tl.freeze(True)
            except Exception:
                pass
        except Exception:
            pass

    print(f"[TRAFFIC_LIGHT] Count={len(tls)} State={state_name}")


def pick_bp(blueprints, preferred):
    for name in preferred:
        bp = blueprints.find(name) if blueprints.find(name) else None
        if bp:
            return bp
    return None


def spawn_vehicle(world, bp_lib, tf, role_name, color=None):
    vehicle_candidates = [
        "vehicle.tesla.model3",
        "vehicle.lincoln.mkz_2020",
        "vehicle.audi.tt",
        "vehicle.mercedes.coupe",
        "vehicle.nissan.patrol",
        "vehicle.dodge.charger_2020",
        "vehicle.bmw.grandtourer",
    ]

    available = []
    for name in vehicle_candidates:
        try:
            available.append(bp_lib.find(name))
        except Exception:
            pass

    if not available:
        available = list(bp_lib.filter("vehicle.*"))

    bp = random.choice(available)

    set_attr_if(bp, "role_name", role_name)

    if color and bp.has_attribute("color"):
        bp.set_attribute("color", color)
    elif bp.has_attribute("color"):
        colors = bp.get_attribute("color").recommended_values
        if colors:
            bp.set_attribute("color", random.choice(colors))

    actor = world.try_spawn_actor(bp, tf)

    if actor is None:
        tf2 = carla.Transform(
            carla.Location(tf.location.x, tf.location.y, tf.location.z + 0.5),
            tf.rotation
        )
        actor = world.try_spawn_actor(bp, tf2)

    if actor:
        try:
            actor.set_autopilot(False)
        except Exception:
            pass
        try:
            actor.set_simulate_physics(False)
        except Exception:
            pass

    return actor


def spawn_walker(world, bp_lib, tf, role_name):
    walkers = list(bp_lib.filter("walker.pedestrian.*"))
    if not walkers:
        return None

    bp = random.choice(walkers)
    set_attr_if(bp, "role_name", role_name)

    actor = world.try_spawn_actor(bp, tf)

    if actor is None:
        tf2 = carla.Transform(
            carla.Location(tf.location.x, tf.location.y, tf.location.z + 0.5),
            tf.rotation
        )
        actor = world.try_spawn_actor(bp, tf2)

    if actor:
        try:
            actor.set_simulate_physics(False)
        except Exception:
            pass

    return actor


def spawn_front_camera(world, bp_lib, ego, width=1240, height=720, fov=90):
    cam_bp = bp_lib.find("sensor.camera.rgb")

    cam_bp.set_attribute("role_name", CAMERA_ROLE)
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov))
    cam_bp.set_attribute("sensor_tick", "0.05")

    cam_tf = carla.Transform(
        carla.Location(x=1.65, y=0.0, z=1.65),
        carla.Rotation(pitch=-2.0, yaw=0.0, roll=0.0)
    )

    cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
    print(f"[CAMERA] Spawned {CAMERA_ROLE}, id={cam.id}")
    return cam


def set_spectator(world, ego_tf):
    spectator = world.get_spectator()

    loc = local_to_world(ego_tf, -8.0, 0.0, 5.0)
    rot = carla.Rotation(
        pitch=-18.0,
        yaw=ego_tf.rotation.yaw,
        roll=0.0
    )

    spectator.set_transform(carla.Transform(loc, rot))


def spawn_dense_scene(world, args):
    bp_lib = world.get_blueprint_library()

    weather = carla.WeatherParameters(
        cloudiness=20.0,
        precipitation=0.0,
        sun_altitude_angle=45.0,
        fog_density=0.0,
        wetness=0.0
    )
    world.set_weather(weather)

    set_traffic_lights(world, args.light_state)

    ego_tf, selected_tl = choose_ego_transform_near_traffic_light(
        world,
        distance=args.light_distance
    )

    ego_bp = bp_lib.find("vehicle.tesla.model3")
    set_attr_if(ego_bp, "role_name", f"{ROLE_PREFIX}_ego")
    if ego_bp.has_attribute("color"):
        ego_bp.set_attribute("color", "0,0,255")

    ego = world.try_spawn_actor(ego_bp, ego_tf)
    if ego is None:
        ego_tf.location.z += 1.0
        ego = world.try_spawn_actor(ego_bp, ego_tf)

    if ego is None:
        raise RuntimeError("Ego vehicle spawn failed.")

    try:
        ego.set_simulate_physics(False)
    except Exception:
        pass

    print(f"[EGO] Spawned ego id={ego.id}")
    cam = spawn_front_camera(world, bp_lib, ego, args.width, args.height, args.fov)

    set_spectator(world, ego_tf)

    actors = [ego, cam]

    vehicle_layout = [
        (12, -3.2, 0.0, 0),
        (16,  0.0, 0.0, 0),
        (20,  3.2, 0.0, 0),
        (26, -2.5, 0.0, 7),
        (31,  1.8, 0.0, -6),
        (38, -4.0, 0.0, 12),
        (45,  3.8, 0.0, -10),
        (55,  0.0, 0.0, 0),
        (65, -2.7, 0.0, 5),
        (75,  2.9, 0.0, -5),
    ]

    spawned_vehicle_count = 0

    for i, (x, y, z, yaw_delta) in enumerate(vehicle_layout[:args.vehicles]):
        loc = local_to_world(ego_tf, x, y, 0.25 + z)
        yaw = ego_tf.rotation.yaw + yaw_delta
        tf = carla.Transform(loc, carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0))

        actor = spawn_vehicle(
            world,
            bp_lib,
            tf,
            f"{ROLE_PREFIX}_vehicle_{i}"
        )

        if actor:
            actors.append(actor)
            spawned_vehicle_count += 1
            print(f"[VEHICLE] {i} id={actor.id} x={x} y={y}")

    walker_layout = [
        (9,   4.5, 0.8, -90),
        (13, -4.2, 0.8, 90),
        (18,  5.2, 0.8, -90),
        (24, -5.1, 0.8, 90),
        (30,  3.8, 0.8, -75),
        (36, -3.9, 0.8, 75),
        (44,  5.4, 0.8, -90),
        (52, -5.3, 0.8, 90),
        (60,  4.8, 0.8, -90),
        (70, -4.7, 0.8, 90),
    ]

    spawned_walker_count = 0

    for i, (x, y, z, yaw_delta) in enumerate(walker_layout[:args.walkers]):
        loc = local_to_world(ego_tf, x, y, z)
        yaw = ego_tf.rotation.yaw + yaw_delta
        tf = carla.Transform(loc, carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0))

        actor = spawn_walker(
            world,
            bp_lib,
            tf,
            f"{ROLE_PREFIX}_walker_{i}"
        )

        if actor:
            actors.append(actor)
            spawned_walker_count += 1
            print(f"[WALKER] {i} id={actor.id} x={x} y={y}")

    print("")
    print("========== BIG DETECTION SHOWCASE READY ==========")
    print(f"Ego id: {ego.id}")
    print(f"Camera role: {CAMERA_ROLE}")
    print(f"Spawned vehicles: {spawned_vehicle_count}")
    print(f"Spawned walkers: {spawned_walker_count}")
    print(f"Traffic lights state: {args.light_state}")
    print("ROS publisher should listen to: rgb_front")
    print("ROS topic expected: /adas/camera/front/image_raw")
    print("==================================================")
    print("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Town04")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--vehicles", type=int, default=10)
    parser.add_argument("--walkers", type=int, default=8)
    parser.add_argument("--light-state", default="Red", choices=["Red", "Yellow", "Green", "red", "yellow", "green"])
    parser.add_argument("--light-distance", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1240)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--no-load-map", action="store_true")
    args = parser.parse_args()

    random.seed(42)

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    if args.clear:
        world = client.get_world()
        clear_scene(world)
        return

    world = client.get_world()

    if not args.no_load_map:
        current_map = world.get_map().name
        if args.map not in current_map:
            print(f"[MAP] Loading {args.map} ...")
            world = client.load_world(args.map)
            time.sleep(2.0)
        else:
            print(f"[MAP] Already on {current_map}")

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    clear_scene(world)
    time.sleep(0.5)

    spawn_dense_scene(world, args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import math
import random
import time
import carla

ROLE_PREFIX = "adas_clear_straight"
CAMERA_ROLE = "rgb_front"


def clear_scene(world):
    targets = []

    for a in world.get_actors():
        tid = a.type_id
        role = a.attributes.get("role_name", "")

        if tid.startswith("sensor.camera."):
            targets.append(a)
        elif tid.startswith("vehicle."):
            targets.append(a)
        elif tid.startswith("walker.pedestrian."):
            targets.append(a)
        elif tid.startswith("controller.ai.walker"):
            targets.append(a)
        elif role.startswith(ROLE_PREFIX) or role == CAMERA_ROLE:
            targets.append(a)

    for a in targets:
        try:
            a.destroy()
        except Exception:
            pass

    print(f"[CLEAR] destroyed actors: {len(targets)}")


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


def yaw_diff(a, b):
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    return d


def is_straight_segment(wp, distance=45.0):
    try:
        yaw0 = wp.transform.rotation.yaw

        wps1 = wp.next(distance * 0.33)
        wps2 = wp.next(distance * 0.66)
        wps3 = wp.next(distance)

        if not wps1 or not wps2 or not wps3:
            return False

        yaws = [
            wps1[0].transform.rotation.yaw,
            wps2[0].transform.rotation.yaw,
            wps3[0].transform.rotation.yaw,
        ]

        return all(yaw_diff(yaw0, y) < 12.0 for y in yaws)
    except Exception:
        return False


def choose_ego_transform(world, light_distance=28.0):
    carla_map = world.get_map()
    traffic_lights = list(world.get_actors().filter("*traffic_light*"))
    random.shuffle(traffic_lights)

    for tl in traffic_lights:
        try:
            tl_loc = tl.get_location()
            light_wp = carla_map.get_waypoint(
                tl_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )

            if light_wp is None:
                continue

            prev_wps = light_wp.previous(light_distance)

            if not prev_wps:
                continue

            ego_wp = prev_wps[0]

            if not is_straight_segment(ego_wp, 45.0):
                continue

            ego_tf = ego_wp.transform
            ego_tf.location.z += 0.35
            ego_tf.rotation = carla.Rotation(
                pitch=0.0,
                yaw=yaw_to_location(ego_tf.location, tl_loc),
                roll=0.0,
            )

            print(f"[SELECT] straight road near traffic light id={tl.id}")
            return ego_tf
        except Exception:
            continue

    spawn_points = carla_map.get_spawn_points()
    random.shuffle(spawn_points)

    for sp in spawn_points:
        wp = carla_map.get_waypoint(
            sp.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if wp and is_straight_segment(wp, 45.0):
            sp.location.z += 0.35
            print("[SELECT] fallback straight spawn point")
            return sp

    if not spawn_points:
        raise RuntimeError("Spawn point bulunamadı.")

    sp = spawn_points[0]
    sp.location.z += 0.35
    print("[SELECT] fallback spawn point 0")
    return sp


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

    print(f"[TRAFFIC_LIGHT] count={len(tls)} state={state_name}")


def get_vehicle_bp(bp_lib):
    names = [
        "vehicle.tesla.model3",
        "vehicle.lincoln.mkz_2020",
        "vehicle.audi.tt",
        "vehicle.nissan.patrol",
        "vehicle.dodge.charger_2020",
        "vehicle.bmw.grandtourer",
    ]

    available = []

    for n in names:
        try:
            available.append(bp_lib.find(n))
        except Exception:
            pass

    if available:
        return random.choice(available)

    all_vehicles = list(bp_lib.filter("vehicle.*"))
    return random.choice(all_vehicles)


def spawn_vehicle(world, bp_lib, ego_tf, idx, x, y, yaw_delta=0.0):
    bp = get_vehicle_bp(bp_lib)
    bp.set_attribute("role_name", f"{ROLE_PREFIX}_vehicle_{idx}")

    if bp.has_attribute("color"):
        colors = bp.get_attribute("color").recommended_values
        if colors:
            bp.set_attribute("color", random.choice(colors))

    loc = local_to_world(ego_tf, x, y, 0.25)
    tf = carla.Transform(
        loc,
        carla.Rotation(
            pitch=0.0,
            yaw=ego_tf.rotation.yaw + yaw_delta,
            roll=0.0,
        ),
    )

    actor = world.try_spawn_actor(bp, tf)

    if actor is None:
        tf.location.z += 0.5
        actor = world.try_spawn_actor(bp, tf)

    if actor:
        try:
            actor.set_autopilot(False)
            actor.set_simulate_physics(False)
        except Exception:
            pass

        print(f"[VEHICLE] {idx} id={actor.id} x={x} y={y}")

    return actor


def spawn_walker(world, bp_lib, ego_tf, idx, x, y, yaw_delta=180.0):
    walkers = list(bp_lib.filter("walker.pedestrian.*"))

    if not walkers:
        return None

    bp = random.choice(walkers)
    bp.set_attribute("role_name", f"{ROLE_PREFIX}_walker_{idx}")

    loc = local_to_world(ego_tf, x, y, 0.85)
    tf = carla.Transform(
        loc,
        carla.Rotation(
            pitch=0.0,
            yaw=ego_tf.rotation.yaw + yaw_delta,
            roll=0.0,
        ),
    )

    actor = world.try_spawn_actor(bp, tf)

    if actor is None:
        tf.location.z += 0.5
        actor = world.try_spawn_actor(bp, tf)

    if actor:
        try:
            actor.set_simulate_physics(False)
        except Exception:
            pass

        print(f"[WALKER] {idx} id={actor.id} x={x} y={y}")

    return actor


def spawn_camera(world, bp_lib, ego, width, height, fov):
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("role_name", CAMERA_ROLE)
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov))
    cam_bp.set_attribute("sensor_tick", "0.05")

    cam_tf = carla.Transform(
        carla.Location(x=1.65, y=0.0, z=1.65),
        carla.Rotation(pitch=-2.0, yaw=0.0, roll=0.0),
    )

    cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    print(f"[CAMERA] id={cam.id} role={CAMERA_ROLE} size={width}x{height} fov={fov}")
    return cam


def set_spectator(world, ego_tf):
    spectator = world.get_spectator()
    loc = local_to_world(ego_tf, -8.0, 0.0, 5.0)
    rot = carla.Rotation(pitch=-18.0, yaw=ego_tf.rotation.yaw, roll=0.0)
    spectator.set_transform(carla.Transform(loc, rot))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Town04")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--vehicles", type=int, default=5)
    parser.add_argument("--walkers", type=int, default=3)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=80.0)
    parser.add_argument("--light-state", default="Red")
    parser.add_argument("--light-distance", type=float, default=28.0)
    args = parser.parse_args()

    random.seed(10)

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    world = client.get_world()

    if args.clear:
        clear_scene(world)
        return

    current_map = world.get_map().name

    if args.map not in current_map:
        print(f"[MAP] loading {args.map}")
        world = client.load_world(args.map)
        time.sleep(2.0)

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    world.set_weather(
        carla.WeatherParameters(
            cloudiness=5.0,
            precipitation=0.0,
            sun_altitude_angle=55.0,
            fog_density=0.0,
            wetness=0.0,
        )
    )

    clear_scene(world)
    time.sleep(0.5)

    set_traffic_lights(world, args.light_state)

    bp_lib = world.get_blueprint_library()

    ego_tf = choose_ego_transform(world, args.light_distance)

    ego_bp = bp_lib.find("vehicle.tesla.model3")
    ego_bp.set_attribute("role_name", f"{ROLE_PREFIX}_ego")

    if ego_bp.has_attribute("color"):
        ego_bp.set_attribute("color", "0,0,255")

    ego = world.try_spawn_actor(ego_bp, ego_tf)

    if ego is None:
        ego_tf.location.z += 1.0
        ego = world.try_spawn_actor(ego_bp, ego_tf)

    if ego is None:
        raise RuntimeError("Ego spawn başarısız.")

    try:
        ego.set_simulate_physics(False)
    except Exception:
        pass

    print(f"[EGO] id={ego.id}")

    camera = spawn_camera(world, bp_lib, ego, args.width, args.height, args.fov)
    set_spectator(world, ego_tf)

    vehicle_layout = [
        (14.0, -2.7, 0.0),
        (20.0,  0.0, 0.0),
        (27.0,  2.7, 0.0),
        (36.0, -2.7, 0.0),
        (46.0,  0.0, 0.0),
        (58.0,  2.7, 0.0),
    ]

    walker_layout = [
        (12.0,  3.8, 180.0),
        (18.0, -3.8, 180.0),
        (26.0,  4.2, 180.0),
        (34.0, -4.2, 180.0),
    ]

    spawned_vehicles = 0
    spawned_walkers = 0

    for i, (x, y, yaw_delta) in enumerate(vehicle_layout[:args.vehicles]):
        if spawn_vehicle(world, bp_lib, ego_tf, i, x, y, yaw_delta):
            spawned_vehicles += 1

    for i, (x, y, yaw_delta) in enumerate(walker_layout[:args.walkers]):
        if spawn_walker(world, bp_lib, ego_tf, i, x, y, yaw_delta):
            spawned_walkers += 1

    print("")
    print("========== CLEAR STRAIGHT SHOWCASE READY ==========")
    print(f"Camera role: {CAMERA_ROLE}")
    print(f"Spawned vehicles: {spawned_vehicles}")
    print(f"Spawned walkers: {spawned_walkers}")
    print(f"Camera: {args.width}x{args.height}, FOV={args.fov}")
    print("ROS topic expected: /adas/camera/front/image_raw")
    print("===================================================")
    print("")


if __name__ == "__main__":
    main()

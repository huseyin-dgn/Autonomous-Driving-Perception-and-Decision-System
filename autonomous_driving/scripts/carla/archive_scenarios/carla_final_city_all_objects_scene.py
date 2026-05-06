#!/usr/bin/env python3
import argparse
import math
import random
import time
import carla

ROLE_PREFIX = "adas_final_city_all"
CAMERA_ROLE = "rgb_front"


def set_attr_if(bp, name, value):
    try:
        if bp.has_attribute(name):
            bp.set_attribute(name, str(value))
    except Exception:
        pass


def clear_scene(world):
    killed = 0

    for actor in list(world.get_actors()):
        tid = actor.type_id
        role = actor.attributes.get("role_name", "")

        if (
            tid.startswith("sensor.camera.")
            or tid.startswith("vehicle.")
            or tid.startswith("walker.pedestrian.")
            or tid.startswith("controller.ai.walker")
            or role.startswith(ROLE_PREFIX)
            or role == CAMERA_ROLE
        ):
            try:
                actor.destroy()
                killed += 1
            except Exception:
                pass

    print(f"[CLEAR] destroyed actors: {killed}")


def local_to_world(base_tf, x, y, z=0.0):
    yaw = math.radians(base_tf.rotation.yaw)

    bx = base_tf.location.x
    by = base_tf.location.y
    bz = base_tf.location.z

    wx = bx + x * math.cos(yaw) - y * math.sin(yaw)
    wy = by + x * math.sin(yaw) + y * math.cos(yaw)
    wz = bz + z

    return carla.Location(wx, wy, wz)


def world_to_local(base_tf, loc):
    yaw = math.radians(base_tf.rotation.yaw)

    dx = loc.x - base_tf.location.x
    dy = loc.y - base_tf.location.y

    x = dx * math.cos(yaw) + dy * math.sin(yaw)
    y = -dx * math.sin(yaw) + dy * math.cos(yaw)

    return x, y


def yaw_to_location(src, dst):
    dx = dst.x - src.x
    dy = dst.y - src.y
    return math.degrees(math.atan2(dy, dx))


def yaw_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def is_straight(wp, distance=35.0):
    try:
        yaw0 = wp.transform.rotation.yaw

        for d in [distance * 0.33, distance * 0.66, distance]:
            nxt = wp.next(d)
            if not nxt:
                return False

            y = nxt[0].transform.rotation.yaw

            if yaw_diff(yaw0, y) > 15.0:
                return False

        return True
    except Exception:
        return False


def choose_ego_transform(world, light_distance=24.0):
    carla_map = world.get_map()

    tls = list(world.get_actors().filter("*traffic_light*"))
    random.shuffle(tls)

    for tl in tls:
        try:
            tl_loc = tl.get_location()

            wp = carla_map.get_waypoint(
                tl_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )

            if wp is None:
                continue

            prevs = wp.previous(light_distance)

            if not prevs:
                continue

            ego_wp = prevs[0]

            if not is_straight(ego_wp, 32.0):
                continue

            tf = ego_wp.transform
            tf.location.z += 0.35
            tf.rotation = carla.Rotation(
                pitch=0.0,
                yaw=yaw_to_location(tf.location, tl_loc),
                roll=0.0,
            )

            print(f"[SELECT] ego near traffic light id={tl.id}")
            return tf

        except Exception:
            continue

    spawns = carla_map.get_spawn_points()
    random.shuffle(spawns)

    for sp in spawns:
        try:
            wp = carla_map.get_waypoint(
                sp.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )

            if wp and is_straight(wp, 32.0):
                sp.location.z += 0.35
                print("[SELECT] fallback straight spawn")
                return sp
        except Exception:
            pass

    if not spawns:
        raise RuntimeError("Spawn point bulunamadı.")

    spawns[0].location.z += 0.35
    print("[SELECT] fallback spawn 0")
    return spawns[0]


def set_traffic_lights(world, ego_tf):
    tls = list(world.get_actors().filter("*traffic_light*"))

    front_lights = []

    for tl in tls:
        try:
            loc = tl.get_location()
            lx, ly = world_to_local(ego_tf, loc)

            if 0.0 < lx < 90.0 and abs(ly) < 45.0:
                front_lights.append((lx, abs(ly), tl))
        except Exception:
            pass

    front_lights.sort(key=lambda item: (item[0], item[1]))

    states = [
        ("Red", carla.TrafficLightState.Red),
        ("Yellow", carla.TrafficLightState.Yellow),
        ("Green", carla.TrafficLightState.Green),
    ]

    used = front_lights[:6] if front_lights else [(999, 999, tl) for tl in tls[:6]]

    count = 0

    for i, (_, _, tl) in enumerate(used):
        name, state = states[i % len(states)]

        try:
            tl.set_state(state)
            tl.set_red_time(999.0)
            tl.set_yellow_time(999.0)
            tl.set_green_time(999.0)

            try:
                tl.freeze(True)
            except Exception:
                pass

            print(f"[LIGHT] id={tl.id} state={name}")
            count += 1
        except Exception as exc:
            print(f"[WARN] traffic light set failed id={tl.id}: {exc}")

    print(f"[LIGHT] configured count={count}")
    return count


def get_vehicle_bp(bp_lib, idx):
    names = [
        "vehicle.tesla.model3",
        "vehicle.lincoln.mkz_2020",
        "vehicle.audi.tt",
        "vehicle.nissan.patrol",
        "vehicle.bmw.grandtourer",
        "vehicle.dodge.charger_2020",
    ]

    ordered = names[idx % len(names):] + names[:idx % len(names)]

    for name in ordered:
        try:
            return bp_lib.find(name)
        except Exception:
            pass

    all_vehicles = list(bp_lib.filter("vehicle.*"))

    if not all_vehicles:
        return None

    return random.choice(all_vehicles)


def spawn_ego(world, bp_lib, ego_tf):
    bp = bp_lib.find("vehicle.tesla.model3")
    set_attr_if(bp, "role_name", f"{ROLE_PREFIX}_ego")

    if bp.has_attribute("color"):
        bp.set_attribute("color", "0,0,255")

    ego = world.try_spawn_actor(bp, ego_tf)

    if ego is None:
        ego_tf.location.z += 1.0
        ego = world.try_spawn_actor(bp, ego_tf)

    if ego is None:
        raise RuntimeError("Ego spawn başarısız.")

    try:
        ego.set_simulate_physics(False)
    except Exception:
        pass

    print(f"[EGO] id={ego.id}")
    return ego


def spawn_vehicle(world, bp_lib, ego_tf, idx, x, y, yaw_delta=0.0):
    bp = get_vehicle_bp(bp_lib, idx)

    if bp is None:
        print("[WARN] vehicle blueprint bulunamadı")
        return None

    set_attr_if(bp, "role_name", f"{ROLE_PREFIX}_vehicle_{idx}")

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

        print(f"[VEHICLE] id={actor.id} idx={idx} x={x} y={y}")

    return actor


def spawn_walker(world, bp_lib, ego_tf, idx, x, y, yaw_delta=180.0):
    walkers = list(bp_lib.filter("walker.pedestrian.*"))

    if not walkers:
        print("[WARN] walker blueprint bulunamadı")
        return None

    bp = random.choice(walkers)
    set_attr_if(bp, "role_name", f"{ROLE_PREFIX}_walker_{idx}")

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

        print(f"[WALKER] id={actor.id} idx={idx} x={x} y={y}")

    return actor


def get_first_bp(bp_lib, candidates):
    for name in candidates:
        try:
            return bp_lib.find(name)
        except Exception:
            pass

    return None


def spawn_sign(world, bp_lib, ego_tf, idx, sign_name, candidates, x, y, z=0.25):
    bp = get_first_bp(bp_lib, candidates)

    if bp is None:
        print(f"[WARN] sign blueprint yok: {sign_name}")
        return None

    set_attr_if(bp, "role_name", f"{ROLE_PREFIX}_sign_{sign_name}")

    loc = local_to_world(ego_tf, x, y, z)

    tf = carla.Transform(
        loc,
        carla.Rotation(
            pitch=0.0,
            yaw=ego_tf.rotation.yaw + 180.0,
            roll=0.0,
        ),
    )

    actor = world.try_spawn_actor(bp, tf)

    if actor is None:
        tf.location.z += 0.5
        actor = world.try_spawn_actor(bp, tf)

    if actor:
        print(f"[SIGN] id={actor.id} type={sign_name} bp={bp.id} x={x} y={y}")

    return actor


def spawn_signs(world, bp_lib, ego_tf):
    sign_specs = [
        (
            "stop",
            [
                "traffic.stop",
                "traffic.stop_sign",
                "static.prop.stop",
                "static.prop.stop_sign",
            ],
            12.0,
            5.0,
        ),
        (
            "yield",
            [
                "traffic.yield",
                "traffic.yield_sign",
                "static.prop.yield",
                "static.prop.yield_sign",
            ],
            17.0,
            5.0,
        ),
        (
            "speed_30",
            [
                "traffic.speed_limit.30",
                "traffic.speed_limit_30",
                "traffic.speed_limit.40",
                "traffic.speed_limit.50",
            ],
            22.0,
            5.0,
        ),
        (
            "speed_50",
            [
                "traffic.speed_limit.50",
                "traffic.speed_limit.60",
                "traffic.speed_limit_50",
                "traffic.speed_limit_60",
            ],
            27.0,
            5.0,
        ),
    ]

    count = 0

    for idx, (name, candidates, x, y) in enumerate(sign_specs):
        actor = spawn_sign(world, bp_lib, ego_tf, idx, name, candidates, x, y)

        if actor:
            count += 1

    print(f"[SIGN] spawned count={count}")
    return count


def spawn_camera(world, bp_lib, ego, width, height, fov):
    bp = bp_lib.find("sensor.camera.rgb")
    bp.set_attribute("role_name", CAMERA_ROLE)
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", str(fov))
    bp.set_attribute("sensor_tick", "0.05")

    tf = carla.Transform(
        carla.Location(x=1.65, y=0.0, z=1.65),
        carla.Rotation(pitch=-2.0, yaw=0.0, roll=0.0),
    )

    cam = world.spawn_actor(bp, tf, attach_to=ego)

    print(f"[CAMERA] id={cam.id} role={CAMERA_ROLE} size={width}x{height} fov={fov}")
    return cam


def set_spectator(world, ego_tf):
    spectator = world.get_spectator()

    loc = local_to_world(ego_tf, -7.5, 0.0, 4.8)

    rot = carla.Rotation(
        pitch=-18.0,
        yaw=ego_tf.rotation.yaw,
        roll=0.0,
    )

    spectator.set_transform(carla.Transform(loc, rot))


def list_related_bps(bp_lib):
    names = []

    for bp in bp_lib:
        bid = bp.id.lower()

        if (
            "traffic" in bid
            or "sign" in bid
            or "speed" in bid
            or "stop" in bid
            or "yield" in bid
        ):
            names.append(bp.id)

    print("[INFO] related blueprints:")
    for n in sorted(set(names))[:120]:
        print("  ", n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Town04")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--list-bps", action="store_true")
    parser.add_argument("--vehicles", type=int, default=4)
    parser.add_argument("--walkers", type=int, default=3)
    parser.add_argument("--ped-side", default="right", choices=["right", "left"])
    parser.add_argument("--light-distance", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=76.0)
    args = parser.parse_args()

    random.seed(65)

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    if args.list_bps:
        list_related_bps(bp_lib)
        return

    if args.clear:
        clear_scene(world)
        return

    current_map = world.get_map().name

    if args.map not in current_map:
        print(f"[MAP] loading {args.map}")
        world = client.load_world(args.map)
        time.sleep(2.0)
        bp_lib = world.get_blueprint_library()

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

    ego_tf = choose_ego_transform(world, args.light_distance)
    ego = spawn_ego(world, bp_lib, ego_tf)
    spawn_camera(world, bp_lib, ego, args.width, args.height, args.fov)
    set_spectator(world, ego_tf)

    set_traffic_lights(world, ego_tf)

    vehicle_layout = [
        (13.0, 0.0, 0.0),
        (20.0, 2.8, 0.0),
        (25.0, -2.8, 0.0),
        (34.0, 0.0, 0.0),
        (43.0, 2.8, 0.0),
    ]

    side_y = 5.2 if args.ped_side == "right" else -5.2

    walker_layout = [
        (10.0, side_y, 180.0),
        (15.5, side_y + (0.5 if side_y > 0 else -0.5), 180.0),
        (21.0, side_y, 180.0),
        (26.0, side_y + (0.5 if side_y > 0 else -0.5), 180.0),
    ]

    v_count = 0
    w_count = 0

    for i, (x, y, yaw) in enumerate(vehicle_layout[:args.vehicles]):
        if spawn_vehicle(world, bp_lib, ego_tf, i, x, y, yaw):
            v_count += 1

    for i, (x, y, yaw) in enumerate(walker_layout[:args.walkers]):
        if spawn_walker(world, bp_lib, ego_tf, i, x, y, yaw):
            w_count += 1

    s_count = spawn_signs(world, bp_lib, ego_tf)

    print("")
    print("========== FINAL CITY ALL OBJECTS READY ==========")
    print(f"Map: {args.map}")
    print(f"Camera role: {CAMERA_ROLE}")
    print(f"Spawned vehicles: {v_count}")
    print(f"Spawned walkers: {w_count}")
    print(f"Spawned signs: {s_count}")
    print(f"Camera: {args.width}x{args.height}, FOV={args.fov}")
    print("Objects: vehicles + pedestrians + traffic lights + signs")
    print("ROS topic expected: /adas/camera/front/image_raw")
    print("=================================================")
    print("")


if __name__ == "__main__":
    main()

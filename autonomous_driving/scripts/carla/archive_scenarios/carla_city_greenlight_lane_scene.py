#!/usr/bin/env python3
import argparse
import math
import random
import time
import carla

ROLE_PREFIX = "adas_city_green_lane"
CAMERA_ROLE = "rgb_front"


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


def set_attr_if(bp, key, value):
    try:
        if bp.has_attribute(key):
            bp.set_attribute(key, str(value))
    except Exception:
        pass


def yaw_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def yaw_to_location(src, dst):
    dx = dst.x - src.x
    dy = dst.y - src.y
    return math.degrees(math.atan2(dy, dx))


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


def choose_ego_transform_near_light(world, distance=28.0):
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

            prev_wps = light_wp.previous(distance)

            if not prev_wps:
                continue

            ego_wp = prev_wps[0]

            if not is_straight(ego_wp, 35.0):
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

    spawn_points = carla_map.get_spawn_points()
    random.shuffle(spawn_points)

    for sp in spawn_points:
        try:
            wp = carla_map.get_waypoint(
                sp.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )

            if wp and is_straight(wp, 35.0):
                sp.location.z += 0.35
                print("[SELECT] fallback straight spawn")
                return sp
        except Exception:
            pass

    if not spawn_points:
        raise RuntimeError("Spawn point bulunamadı.")

    sp = spawn_points[0]
    sp.location.z += 0.35
    print("[SELECT] fallback spawn 0")
    return sp


def set_all_lights_green(world, ego_tf):
    lights = list(world.get_actors().filter("*traffic_light*"))
    front_lights = []

    for tl in lights:
        try:
            lx, ly = world_to_local(ego_tf, tl.get_location())

            if 0.0 < lx < 90.0 and abs(ly) < 50.0:
                front_lights.append((lx, abs(ly), tl))
        except Exception:
            pass

    front_lights.sort(key=lambda x: (x[0], x[1]))

    count = 0

    for tl in lights:
        try:
            tl.set_state(carla.TrafficLightState.Green)
            tl.set_green_time(999.0)
            tl.set_yellow_time(999.0)
            tl.set_red_time(999.0)

            try:
                tl.freeze(True)
            except Exception:
                pass

            count += 1
        except Exception:
            pass

    print(f"[LIGHT] all map lights forced GREEN: {count}")

    for i, (_, _, tl) in enumerate(front_lights[:6]):
        print(f"[FRONT_LIGHT] id={tl.id} forced GREEN index={i}")


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

    vehicles = list(bp_lib.filter("vehicle.*"))

    if not vehicles:
        return None

    return random.choice(vehicles)


def spawn_vehicle_at_transform(world, bp_lib, idx, tf):
    bp = get_vehicle_bp(bp_lib, idx)

    if bp is None:
        print("[WARN] no vehicle bp")
        return None

    set_attr_if(bp, "role_name", f"{ROLE_PREFIX}_vehicle_{idx}")

    if bp.has_attribute("color"):
        colors = bp.get_attribute("color").recommended_values
        if colors:
            bp.set_attribute("color", random.choice(colors))

    actor = world.try_spawn_actor(bp, tf)

    if actor is None:
        tf2 = carla.Transform(
            carla.Location(tf.location.x, tf.location.y, tf.location.z + 0.5),
            tf.rotation,
        )
        actor = world.try_spawn_actor(bp, tf2)

    if actor:
        try:
            actor.set_autopilot(False)
            actor.set_simulate_physics(False)
        except Exception:
            pass

        print(f"[VEHICLE] id={actor.id} idx={idx} loc=({tf.location.x:.1f},{tf.location.y:.1f})")

    return actor


def compatible_adjacent(base_wp, candidate_wp):
    if candidate_wp is None:
        return None

    try:
        if candidate_wp.lane_type != carla.LaneType.Driving:
            return None

        if yaw_diff(base_wp.transform.rotation.yaw, candidate_wp.transform.rotation.yaw) > 35.0:
            return None

        return candidate_wp
    except Exception:
        return None


def spawn_lane_vehicles(world, bp_lib, ego_tf):
    carla_map = world.get_map()

    ego_wp = carla_map.get_waypoint(
        ego_tf.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if ego_wp is None:
        print("[WARN] ego waypoint yok, local vehicle spawn kullanılacak")
        return spawn_local_fallback_vehicles(world, bp_lib, ego_tf)

    lead_wp_list = ego_wp.next(15.0)

    if not lead_wp_list:
        print("[WARN] lead wp yok, local fallback")
        return spawn_local_fallback_vehicles(world, bp_lib, ego_tf)

    lead_wp = lead_wp_list[0]

    left_wp = compatible_adjacent(lead_wp, lead_wp.get_left_lane())
    right_wp = compatible_adjacent(lead_wp, lead_wp.get_right_lane())

    adjacent_wp = left_wp or right_wp

    spawned = 0

    # Önde 1. araç: ego şeridinde
    tf1 = lead_wp.transform
    tf1.location.z += 0.35
    if spawn_vehicle_at_transform(world, bp_lib, 0, tf1):
        spawned += 1

    # Önde 2. araç: yan şeritte
    if adjacent_wp is not None:
        tf2 = adjacent_wp.transform
        tf2.location.z += 0.35
        if spawn_vehicle_at_transform(world, bp_lib, 1, tf2):
            spawned += 1
    else:
        loc = local_to_world(ego_tf, 15.0, 3.2, 0.35)
        tf2 = carla.Transform(loc, carla.Rotation(pitch=0.0, yaw=ego_tf.rotation.yaw, roll=0.0))
        if spawn_vehicle_at_transform(world, bp_lib, 1, tf2):
            spawned += 1

    # Arkadan/ileriden birkaç araç, şerit içinde
    for idx, d in enumerate([23.0, 32.0], start=2):
        nxt = ego_wp.next(d)

        if nxt:
            tf = nxt[0].transform
            tf.location.z += 0.35
        else:
            loc = local_to_world(ego_tf, d, 0.0, 0.35)
            tf = carla.Transform(loc, carla.Rotation(pitch=0.0, yaw=ego_tf.rotation.yaw, roll=0.0))

        if spawn_vehicle_at_transform(world, bp_lib, idx, tf):
            spawned += 1

    print(f"[VEHICLE] spawned lane vehicles: {spawned}")
    return spawned


def spawn_local_fallback_vehicles(world, bp_lib, ego_tf):
    layout = [
        (14.0, 0.0),
        (14.0, 3.2),
        (23.0, 0.0),
        (32.0, -3.2),
    ]

    count = 0

    for idx, (x, y) in enumerate(layout):
        loc = local_to_world(ego_tf, x, y, 0.35)
        tf = carla.Transform(loc, carla.Rotation(pitch=0.0, yaw=ego_tf.rotation.yaw, roll=0.0))

        if spawn_vehicle_at_transform(world, bp_lib, idx, tf):
            count += 1

    print(f"[VEHICLE] fallback spawned: {count}")
    return count


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
        raise RuntimeError("Ego spawn failed")

    try:
        ego.set_simulate_physics(False)
    except Exception:
        pass

    print(f"[EGO] id={ego.id}")
    return ego


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


def spawn_walker(world, bp_lib, ego_tf, idx, x, y, yaw_delta=180.0):
    walkers = list(bp_lib.filter("walker.pedestrian.*"))

    if not walkers:
        print("[WARN] no walker bp")
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


def spawn_sidewalk_walkers(world, bp_lib, ego_tf, side="right"):
    # Küçük insan yok. Kaldırım/kenar hattında 3 büyük ve net insan.
    side_y = 5.8 if side == "right" else -5.8

    layout = [
        (9.0, side_y, 180.0),
        (15.0, side_y + (0.4 if side_y > 0 else -0.4), 180.0),
        (21.0, side_y, 180.0),
    ]

    count = 0

    for idx, (x, y, yaw) in enumerate(layout):
        if spawn_walker(world, bp_lib, ego_tf, idx, x, y, yaw):
            count += 1

    print(f"[WALKER] spawned sidewalk walkers: {count}")
    return count


def set_spectator(world, ego_tf):
    spectator = world.get_spectator()
    loc = local_to_world(ego_tf, -7.5, 0.0, 4.8)
    rot = carla.Rotation(pitch=-18.0, yaw=ego_tf.rotation.yaw, roll=0.0)
    spectator.set_transform(carla.Transform(loc, rot))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Town03")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--ped-side", default="right", choices=["right", "left"])
    parser.add_argument("--light-distance", type=float, default=28.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=76.0)
    args = parser.parse_args()

    random.seed(101)

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

    bp_lib = world.get_blueprint_library()

    ego_tf = choose_ego_transform_near_light(world, args.light_distance)
    ego = spawn_ego(world, bp_lib, ego_tf)

    spawn_camera(world, bp_lib, ego, args.width, args.height, args.fov)
    set_spectator(world, ego_tf)

    set_all_lights_green(world, ego_tf)

    v_count = spawn_lane_vehicles(world, bp_lib, ego_tf)
    w_count = spawn_sidewalk_walkers(world, bp_lib, ego_tf, args.ped_side)

    print("")
    print("========== CITY GREEN LIGHT LANE SCENE READY ==========")
    print(f"Map: {args.map}")
    print(f"Camera role: {CAMERA_ROLE}")
    print(f"Vehicles: {v_count}")
    print(f"Sidewalk walkers: {w_count}")
    print("Scene: GREEN light + two front lane vehicles + sidewalk pedestrians")
    print("ROS topic expected: /adas/camera/front/image_raw")
    print("=======================================================")
    print("")


if __name__ == "__main__":
    main()

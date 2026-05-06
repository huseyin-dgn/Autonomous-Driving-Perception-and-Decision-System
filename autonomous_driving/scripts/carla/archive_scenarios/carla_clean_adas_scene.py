#!/usr/bin/env python3
import argparse
import math
import random
import time
import sys

try:
    import carla
except Exception as exc:
    print(f"[ERROR] carla import edilemedi: {exc}")
    sys.exit(1)


def set_attr(bp, key, value):
    try:
        if bp.has_attribute(key):
            bp.set_attribute(key, str(value))
    except Exception:
        pass


def get_blueprint(world, preferred_ids, fallback_filter):
    bps = world.get_blueprint_library()

    for bp_id in preferred_ids:
        try:
            bp = bps.find(bp_id)
            if bp is not None:
                return bp
        except Exception:
            pass

    candidates = list(bps.filter(fallback_filter))
    if not candidates:
        raise RuntimeError(f"Blueprint bulunamadı: {preferred_ids} / {fallback_filter}")

    return random.choice(candidates)


def apply_clean_world_settings(world):
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    weather = carla.WeatherParameters(
        cloudiness=10.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=0.0,
        sun_azimuth_angle=45.0,
        sun_altitude_angle=60.0,
        fog_density=0.0,
        fog_distance=100000.0,
        fog_falloff=0.0,
        wetness=0.0,
    )
    world.set_weather(weather)

    print("[ADAS] Weather: CLEAR DAY / stable daylight / no rain / no fog")


def clear_dynamic_actors(world):
    actors = world.get_actors()

    filters = [
        "vehicle.*",
        "walker.pedestrian.*",
        "controller.ai.walker",
        "sensor.*",
    ]

    to_destroy = []

    for pattern in filters:
        for actor in actors.filter(pattern):
            to_destroy.append(actor)

    for actor in to_destroy:
        try:
            actor.destroy()
        except Exception:
            pass

    print(f"[ADAS] Cleared actors: {len(to_destroy)}")


def choose_ego_waypoint(world):
    cmap = world.get_map()
    spawn_points = cmap.get_spawn_points()

    if not spawn_points:
        raise RuntimeError("Map spawn point yok")

    candidates = []

    for sp in spawn_points:
        wp = cmap.get_waypoint(
            sp.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if wp is None:
            continue

        if wp.is_junction:
            continue

        ok = True
        for dist in [20.0, 40.0, 60.0, 80.0]:
            nxt = wp.next(dist)
            if not nxt:
                ok = False
                break

        if ok:
            candidates.append(wp)

    if candidates:
        return candidates[0]

    return cmap.get_waypoint(
        spawn_points[0].location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )


def spawn_vehicle_at_waypoint(world, wp, role_name, preferred_ids, color=None):
    bp = get_blueprint(
        world,
        preferred_ids,
        "vehicle.*",
    )

    set_attr(bp, "role_name", role_name)

    if color is not None:
        set_attr(bp, "color", color)

    tr = wp.transform
    tr.location.z += 0.35

    actor = world.try_spawn_actor(bp, tr)

    if actor is None:
        raise RuntimeError(f"Vehicle spawn failed: {role_name}")

    return actor


def get_forward_location(wp, dist):
    nxt = wp.next(float(dist))
    if not nxt:
        return None
    return nxt[0]


def set_vehicle_speed(vehicle, speed_mps):
    try:
        fwd = vehicle.get_transform().get_forward_vector()
        vehicle.set_target_velocity(
            carla.Vector3D(
                x=fwd.x * speed_mps,
                y=fwd.y * speed_mps,
                z=0.0,
            )
        )
    except Exception:
        pass


def spawn_front_vehicles(world, ego_wp):
    vehicles = []

    specs = [
        {
            "dist": 28.0,
            "role": "adas_front_vehicle_near",
            "ids": ["vehicle.audi.tt", "vehicle.tesla.model3", "vehicle.lincoln.mkz_2017"],
            "color": "20,20,220",
            "speed": 5.0,
        },
        {
            "dist": 55.0,
            "role": "adas_front_vehicle_far",
            "ids": ["vehicle.lincoln.mkz_2017", "vehicle.audi.etron", "vehicle.tesla.model3"],
            "color": "20,20,20",
            "speed": 5.0,
        },
        {
            "dist": 82.0,
            "role": "adas_front_vehicle_very_far",
            "ids": ["vehicle.mercedes.coupe", "vehicle.audi.tt", "vehicle.lincoln.mkz_2017"],
            "color": "220,220,220",
            "speed": 5.0,
        },
    ]

    for spec in specs:
        wp = get_forward_location(ego_wp, spec["dist"])
        if wp is None:
            continue

        try:
            v = spawn_vehicle_at_waypoint(
                world,
                wp,
                spec["role"],
                spec["ids"],
                spec["color"],
            )
            set_vehicle_speed(v, spec["speed"])
            vehicles.append(v)
            print(f"[ADAS] Spawned vehicle: {spec['role']} dist={spec['dist']}m")
        except Exception as exc:
            print(f"[WARN] Vehicle spawn failed at {spec['dist']}m: {exc}")

    return vehicles


def spawn_pedestrian(world, ego_wp, dist=35.0, side_offset=5.2):
    wp = get_forward_location(ego_wp, dist)
    if wp is None:
        return None

    bps = world.get_blueprint_library().filter("walker.pedestrian.*")
    if not bps:
        print("[WARN] Walker blueprint yok")
        return None

    bp = random.choice(list(bps))
    set_attr(bp, "role_name", "adas_pedestrian")

    base = wp.transform.location
    right = wp.transform.get_right_vector()

    loc = carla.Location(
        x=base.x + right.x * side_offset,
        y=base.y + right.y * side_offset,
        z=base.z + 0.8,
    )

    rot = carla.Rotation(
        pitch=0.0,
        yaw=wp.transform.rotation.yaw + 180.0,
        roll=0.0,
    )

    tr = carla.Transform(loc, rot)

    actor = world.try_spawn_actor(bp, tr)

    if actor is None:
        print("[WARN] Pedestrian spawn failed")
        return None

    try:
        ctrl = carla.WalkerControl()
        ctrl.speed = 0.0
        ctrl.direction = carla.Vector3D(0.0, 0.0, 0.0)
        actor.apply_control(ctrl)
    except Exception:
        pass

    print(f"[ADAS] Spawned pedestrian at {dist}m right sidewalk")
    return actor


def spawn_sign_prop(world, ego_wp, dist=42.0, side_offset=6.5):
    wp = get_forward_location(ego_wp, dist)
    if wp is None:
        return None

    bps = world.get_blueprint_library()

    preferred = [
        "static.prop.trafficwarning",
        "static.prop.streetsign04",
        "static.prop.streetsign01",
        "static.prop.streetsign",
    ]

    bp = None
    for bp_id in preferred:
        try:
            bp = bps.find(bp_id)
            break
        except Exception:
            pass

    if bp is None:
        print("[WARN] Sign prop blueprint bulunamadı")
        return None

    base = wp.transform.location
    right = wp.transform.get_right_vector()

    loc = carla.Location(
        x=base.x + right.x * side_offset,
        y=base.y + right.y * side_offset,
        z=base.z + 1.1,
    )

    rot = carla.Rotation(
        pitch=0.0,
        yaw=wp.transform.rotation.yaw + 180.0,
        roll=0.0,
    )

    try:
        actor = world.try_spawn_actor(bp, carla.Transform(loc, rot))
        if actor:
            print(f"[ADAS] Spawned sign prop: {bp.id}")
        return actor
    except Exception as exc:
        print(f"[WARN] Sign spawn failed: {exc}")
        return None


def force_all_traffic_lights_red(world):
    count = 0

    for tl in world.get_actors().filter("traffic.traffic_light*"):
        try:
            tl.set_state(carla.TrafficLightState.Red)
            tl.set_red_time(999.0)
            tl.freeze(True)
            count += 1
        except Exception:
            pass

    print(f"[ADAS] Frozen red traffic lights: {count}")


def set_spectator(world, ego):
    spectator = world.get_spectator()
    tr = ego.get_transform()
    fwd = tr.get_forward_vector()

    loc = carla.Location(
        x=tr.location.x - fwd.x * 8.0,
        y=tr.location.y - fwd.y * 8.0,
        z=tr.location.z + 5.0,
    )

    rot = carla.Rotation(
        pitch=-18.0,
        yaw=tr.rotation.yaw,
        roll=0.0,
    )

    spectator.set_transform(carla.Transform(loc, rot))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Town03")
    parser.add_argument("--ego-speed", type=float, default=6.0)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    print(f"[ADAS] Loading map: {args.map}")
    world = client.load_world(args.map)
    time.sleep(2.0)

    apply_clean_world_settings(world)

    if args.clear:
        clear_dynamic_actors(world)
        time.sleep(1.0)

    force_all_traffic_lights_red(world)

    ego_wp = choose_ego_waypoint(world)

    ego = spawn_vehicle_at_waypoint(
        world,
        ego_wp,
        "ego",
        ["vehicle.tesla.model3", "vehicle.audi.tt", "vehicle.lincoln.mkz_2017"],
        color="255,0,0",
    )

    set_vehicle_speed(ego, args.ego_speed)

    vehicles = spawn_front_vehicles(world, ego_wp)
    pedestrian = spawn_pedestrian(world, ego_wp)
    sign = spawn_sign_prop(world, ego_wp)

    for _ in range(10):
        world.wait_for_tick()

    set_spectator(world, ego)

    print("")
    print("======================================")
    print(" CLEAN ADAS CARLA SCENE READY")
    print("======================================")
    print(f"Map        : {args.map}")
    print(f"Ego role   : ego")
    print(f"Ego speed  : {args.ego_speed} m/s")
    print(f"Vehicles   : {len(vehicles)}")
    print(f"Pedestrian : {'yes' if pedestrian else 'no'}")
    print(f"Sign prop  : {'yes' if sign else 'no'}")
    print("Weather    : clear day")
    print("Traffic    : red lights frozen")
    print("======================================")
    print("Bu terminal açık kalacak. Kapatırsan sahne gider.")
    print("")

    try:
        while True:
            set_vehicle_speed(ego, args.ego_speed)
            for v in vehicles:
                set_vehicle_speed(v, 5.0)
            set_spectator(world, ego)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[ADAS] Scene stopped by user")


if __name__ == "__main__":
    main()

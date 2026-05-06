#!/usr/bin/env python3
import argparse
import random
import sys
import time

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


def get_bp(world, preferred_ids, fallback_filter):
    bps = world.get_blueprint_library()

    for bp_id in preferred_ids:
        try:
            return bps.find(bp_id)
        except Exception:
            pass

    candidates = list(bps.filter(fallback_filter))
    if not candidates:
        raise RuntimeError(f"Blueprint bulunamadı: {preferred_ids} / {fallback_filter}")

    return random.choice(candidates)


def apply_weather(world):
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
        sun_azimuth_angle=35.0,
        sun_altitude_angle=55.0,
        fog_density=0.0,
        fog_distance=100000.0,
        fog_falloff=0.0,
        wetness=0.0,
    )

    world.set_weather(weather)
    print("[ADAS] Hava: temiz gündüz")


def clear_world(world):
    patterns = [
        "vehicle.*",
        "walker.pedestrian.*",
        "controller.ai.walker",
        "sensor.*",
    ]

    count = 0

    for pattern in patterns:
        for actor in world.get_actors().filter(pattern):
            try:
                actor.destroy()
                count += 1
            except Exception:
                pass

    print(f"[ADAS] Temizlenen actor sayısı: {count}")


def yaw_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def find_straight_waypoint(world):
    cmap = world.get_map()
    spawn_points = cmap.get_spawn_points()

    for sp in spawn_points:
        wp = cmap.get_waypoint(
            sp.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if wp is None or wp.is_junction:
            continue

        yaw0 = wp.transform.rotation.yaw
        ok = True

        for dist in [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0]:
            nxt = wp.next(dist)

            if not nxt:
                ok = False
                break

            wp2 = nxt[0]

            if wp2.is_junction:
                ok = False
                break

            if yaw_diff(yaw0, wp2.transform.rotation.yaw) > 4.0:
                ok = False
                break

        if ok:
            print(
                f"[ADAS] Düz yol seçildi: "
                f"x={wp.transform.location.x:.1f}, "
                f"y={wp.transform.location.y:.1f}, "
                f"yaw={wp.transform.rotation.yaw:.1f}"
            )
            return wp

    sp = spawn_points[0]
    wp = cmap.get_waypoint(
        sp.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    print("[WARN] Tam düz yol bulunamadı, ilk spawn kullanılıyor.")
    return wp


def make_transform(base_loc, yaw, forward, right, forward_offset, side_offset, z_offset):
    loc = carla.Location(
        x=base_loc.x + forward.x * forward_offset + right.x * side_offset,
        y=base_loc.y + forward.y * forward_offset + right.y * side_offset,
        z=base_loc.z + z_offset,
    )

    rot = carla.Rotation(
        pitch=0.0,
        yaw=yaw,
        roll=0.0,
    )

    return carla.Transform(loc, rot)


def spawn_vehicle(world, transform, role_name, preferred_ids, color):
    bp = get_bp(world, preferred_ids, "vehicle.*")

    set_attr(bp, "role_name", role_name)
    set_attr(bp, "color", color)

    actor = world.try_spawn_actor(bp, transform)

    if actor is None:
        raise RuntimeError(f"Araç spawn başarısız: {role_name}")

    try:
        actor.set_simulate_physics(False)
    except Exception:
        pass

    print(f"[ADAS] Araç eklendi: {role_name}")
    return actor


def spawn_pedestrian(world, transform, role_name):
    bps = list(world.get_blueprint_library().filter("walker.pedestrian.*"))

    if not bps:
        print("[WARN] Pedestrian blueprint yok")
        return None

    for _ in range(30):
        bp = random.choice(bps)
        set_attr(bp, "role_name", role_name)

        actor = world.try_spawn_actor(bp, transform)

        if actor is not None:
            try:
                actor.set_simulate_physics(False)
            except Exception:
                pass

            print(f"[ADAS] İnsan eklendi: {role_name}")
            return actor

    print(f"[WARN] İnsan spawn başarısız: {role_name}")
    return None


def set_spectator(world, ego, yaw):
    spectator = world.get_spectator()
    tr = ego.get_transform()
    forward = tr.get_forward_vector()

    loc = carla.Location(
        x=tr.location.x - forward.x * 9.0,
        y=tr.location.y - forward.y * 9.0,
        z=tr.location.z + 5.0,
    )

    rot = carla.Rotation(
        pitch=-18.0,
        yaw=yaw,
        roll=0.0,
    )

    spectator.set_transform(carla.Transform(loc, rot))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Town03")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--ego-speed", type=float, default=4.0)
    parser.add_argument("--loop-distance", type=float, default=150.0)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    print(f"[ADAS] Map yükleniyor: {args.map}")
    world = client.load_world(args.map)
    time.sleep(2.0)

    apply_weather(world)

    if args.clear:
        clear_world(world)
        time.sleep(1.0)

    ego_wp = find_straight_waypoint(world)
    base_tr = ego_wp.transform
    base_loc = base_tr.location
    yaw = base_tr.rotation.yaw

    forward = base_tr.get_forward_vector()
    right = base_tr.get_right_vector()

    actors = []

    ego_tr = make_transform(
        base_loc,
        yaw,
        forward,
        right,
        forward_offset=0.0,
        side_offset=0.0,
        z_offset=0.35,
    )

    ego = spawn_vehicle(
        world,
        ego_tr,
        "ego",
        ["vehicle.tesla.model3", "vehicle.audi.tt", "vehicle.lincoln.mkz_2017"],
        "255,0,0",
    )

    actors.append({
        "actor": ego,
        "forward_offset": 0.0,
        "side_offset": 0.0,
        "z_offset": 0.35,
        "yaw": yaw,
    })

    vehicle_specs = [
        {
            "offset": 28.0,
            "side": 0.0,
            "role": "front_vehicle_28m",
            "ids": ["vehicle.audi.tt", "vehicle.lincoln.mkz_2017", "vehicle.tesla.model3"],
            "color": "0,0,255",
        },
        {
            "offset": 45.0,
            "side": 0.0,
            "role": "front_vehicle_45m",
            "ids": ["vehicle.tesla.model3", "vehicle.audi.etron", "vehicle.lincoln.mkz_2017"],
            "color": "20,20,20",
        },
        {
            "offset": 65.0,
            "side": 0.0,
            "role": "front_vehicle_65m",
            "ids": ["vehicle.lincoln.mkz_2017", "vehicle.audi.tt", "vehicle.tesla.model3"],
            "color": "220,220,220",
        },
        {
            "offset": 85.0,
            "side": 0.0,
            "role": "front_vehicle_85m",
            "ids": ["vehicle.mercedes.coupe", "vehicle.audi.tt", "vehicle.lincoln.mkz_2017"],
            "color": "0,120,255",
        },
        {
            "offset": 110.0,
            "side": 0.0,
            "role": "front_vehicle_110m",
            "ids": ["vehicle.lincoln.mkz_2017", "vehicle.audi.etron", "vehicle.tesla.model3"],
            "color": "255,255,255",
        },
    ]

    for spec in vehicle_specs:
        tr = make_transform(
            base_loc,
            yaw,
            forward,
            right,
            forward_offset=spec["offset"],
            side_offset=spec["side"],
            z_offset=0.35,
        )

        try:
            actor = spawn_vehicle(
                world,
                tr,
                spec["role"],
                spec["ids"],
                spec["color"],
            )

            actors.append({
                "actor": actor,
                "forward_offset": spec["offset"],
                "side_offset": spec["side"],
                "z_offset": 0.35,
                "yaw": yaw,
            })

        except Exception as exc:
            print(f"[WARN] Araç eklenemedi: {exc}")

    pedestrian_specs = [
        {
            "offset": 22.0,
            "side": 3.0,
            "role": "person_right_22m",
        },
        {
            "offset": 38.0,
            "side": -2.8,
            "role": "person_left_38m",
        },
        {
            "offset": 55.0,
            "side": 3.2,
            "role": "person_right_55m",
        },
        {
            "offset": 78.0,
            "side": -3.0,
            "role": "person_left_78m",
        },
        {
            "offset": 100.0,
            "side": 3.4,
            "role": "person_right_100m",
        },
    ]

    for spec in pedestrian_specs:
        tr = make_transform(
            base_loc,
            yaw + 180.0,
            forward,
            right,
            forward_offset=spec["offset"],
            side_offset=spec["side"],
            z_offset=0.80,
        )

        actor = spawn_pedestrian(world, tr, spec["role"])

        if actor:
            actors.append({
                "actor": actor,
                "forward_offset": spec["offset"],
                "side_offset": spec["side"],
                "z_offset": 0.80,
                "yaw": yaw + 180.0,
            })

    for _ in range(10):
        world.wait_for_tick()

    set_spectator(world, ego, yaw)

    print("")
    print("======================================")
    print(" STRAIGHT PEOPLE + CARS SCENE READY")
    print("======================================")
    print(f"Map          : {args.map}")
    print(f"Ego          : düz gidiyor")
    print(f"Ego speed    : {args.ego_speed} m/s")
    print("Front cars   : 5 adet")
    print("Persons      : 5 adet")
    print("Objects      : sadece araba + insan")
    print("Sign         : yok")
    print("Extra traffic: yok")
    print("Steering     : yok")
    print("Side motion  : yok")
    print("======================================")
    print("Bu terminal açık kalacak.")
    print("")

    start_t = time.time()

    try:
        while True:
            elapsed = time.time() - start_t
            travel = (elapsed * args.ego_speed) % args.loop_distance

            for item in actors:
                actor = item["actor"]

                if actor is None:
                    continue

                tr = make_transform(
                    base_loc,
                    item["yaw"],
                    forward,
                    right,
                    forward_offset=item["forward_offset"] + travel,
                    side_offset=item["side_offset"],
                    z_offset=item["z_offset"],
                )

                try:
                    actor.set_transform(tr)
                except Exception:
                    pass

            set_spectator(world, ego, yaw)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[ADAS] Scene stopped")


if __name__ == "__main__":
    main()

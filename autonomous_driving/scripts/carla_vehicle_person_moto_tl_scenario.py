#!/usr/bin/env python3
import argparse
import json
import math
import random
import time
from pathlib import Path

import carla


STATE_MAP = {
    "red": carla.TrafficLightState.Red,
    "yellow": carla.TrafficLightState.Yellow,
    "green": carla.TrafficLightState.Green,
}


def loc_dict(l):
    return {"x": float(l.x), "y": float(l.y), "z": float(l.z)}


def rot_dict(r):
    return {
        "pitch": float(r.pitch),
        "yaw": float(r.yaw),
        "roll": float(r.roll),
    }


def look_at(src, dst):
    dx = dst.x - src.x
    dy = dst.y - src.y
    dz = dst.z - src.z

    yaw = math.degrees(math.atan2(dy, dx))
    dist_xy = math.sqrt(dx * dx + dy * dy)
    pitch = math.degrees(math.atan2(dz, max(0.001, dist_xy)))

    return carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)


def destroy_dynamic(world):
    count = 0

    for a in world.get_actors():
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

    print(f"[SCENE] destroyed dynamic actors: {count}")


def set_weather(world):
    world.set_weather(
        carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            sun_altitude_angle=65.0,
            sun_azimuth_angle=20.0,
            fog_density=0.0,
            wetness=0.0,
        )
    )


def get_light_vertices(light):
    tf = light.get_transform()
    points = []

    try:
        boxes = light.get_light_boxes()
        for box in boxes:
            for v in box.get_world_vertices(tf):
                points.append(v)
    except Exception:
        pass

    if points:
        return points

    loc = tf.location
    z = loc.z + 5.0

    for dx in [-0.4, 0.4]:
        for dy in [-0.4, 0.4]:
            for dz in [-0.9, 0.9]:
                points.append(carla.Location(loc.x + dx, loc.y + dy, z + dz))

    return points


def get_light_target(light):
    pts = get_light_vertices(light)

    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    zs = [p.z for p in pts]

    return carla.Location(
        x=sum(xs) / len(xs),
        y=sum(ys) / len(ys),
        z=sum(zs) / len(zs),
    )


def set_light_state(light, state_name):
    state = STATE_MAP[state_name]

    try:
        light.freeze(False)
        light.set_state(state)
        light.set_red_time(9999.0)
        light.set_yellow_time(9999.0)
        light.set_green_time(9999.0)
        light.freeze(True)
    except Exception as exc:
        print(f"[WARN] traffic light state set failed id={light.id}: {exc}")


def find_blueprint(bp_lib, preferred_names, fallback_filter):
    for name in preferred_names:
        bp = bp_lib.find(name)
        if bp is not None:
            return bp

    candidates = list(bp_lib.filter(fallback_filter))
    if not candidates:
        return None

    return random.choice(candidates)


def find_motorcycle_blueprint(bp_lib):
    # Bisikletleri kesinlikle alma:
    # vehicle.bh.crossbike
    # vehicle.diamondback.century
    # vehicle.gazelle.omafiets
    blocked = [
        "crossbike",
        "diamondback",
        "gazelle",
        "omafiets",
        "century",
        "bicycle",
    ]

    # Öncelik gerçek motor modellerinde.
    preferred = [
        "vehicle.kawasaki.ninja",
        "vehicle.yamaha.yzf",
        "vehicle.harley-davidson.low_rider",
        "vehicle.vespa.zx125",
    ]

    for tid in preferred:
        try:
            bp = bp_lib.find(tid)
            if bp is not None:
                print(f"[SCENE] selected motorcycle blueprint: {bp.id}")
                return bp
        except Exception:
            pass

    candidates = []

    for bp in bp_lib.filter("vehicle.*"):
        tid = bp.id.lower()

        if any(b in tid for b in blocked):
            continue

        if (
            "kawasaki" in tid
            or "yamaha" in tid
            or "harley" in tid
            or "vespa" in tid
            or "ninja" in tid
            or "yzf" in tid
            or "low_rider" in tid
            or "zx125" in tid
        ):
            candidates.append(bp)

    if candidates:
        candidates.sort(key=lambda b: b.id)
        print("[SCENE] motorcycle candidates:")
        for b in candidates:
            print(f"  - {b.id}")

        print(f"[SCENE] selected motorcycle blueprint: {candidates[0].id}")
        return candidates[0]

    print("[SCENE] motorcycle blueprint bulunamadı.")
    return None


def prepare_vehicle_bp(bp, color=None):
    if bp is None:
        return None

    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "adas_test_actor")

    if color and bp.has_attribute("color"):
        colors = bp.get_attribute("color").recommended_values
        if colors:
            bp.set_attribute("color", color if color in colors else colors[0])

    return bp


def ground_transform(world, location, yaw):
    carla_map = world.get_map()

    try:
        wp = carla_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Any,
        )
    except Exception:
        wp = None

    if wp is not None:
        loc = wp.transform.location
        loc.z += 0.20
    else:
        loc = carla.Location(location.x, location.y, location.z + 0.20)

    return carla.Transform(
        loc,
        carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0),
    )


def try_spawn_with_offsets(world, bp, base_loc, yaw, name):
    offsets = [
        (0.0, 0.0),
        (0.8, 0.0),
        (-0.8, 0.0),
        (0.0, 0.8),
        (0.0, -0.8),
        (1.2, 0.6),
        (-1.2, -0.6),
    ]

    yaw_rad = math.radians(yaw)
    fwd = carla.Vector3D(math.cos(yaw_rad), math.sin(yaw_rad), 0.0)
    right = carla.Vector3D(math.cos(yaw_rad + math.pi / 2.0), math.sin(yaw_rad + math.pi / 2.0), 0.0)

    for forward_off, right_off in offsets:
        loc = carla.Location(
            x=base_loc.x + fwd.x * forward_off + right.x * right_off,
            y=base_loc.y + fwd.y * forward_off + right.y * right_off,
            z=base_loc.z,
        )

        tf = ground_transform(world, loc, yaw)

        actor = world.try_spawn_actor(bp, tf)

        if actor is not None:
            print(f"[SCENE] spawned {name}: id={actor.id}, type={actor.type_id}, loc={loc_dict(actor.get_transform().location)}")
            return actor

    raise RuntimeError(f"{name} spawn edilemedi. Konum çakışıyor olabilir.")


def make_camera_transform(aim, angle_deg, distance, camera_z_offset):
    a = math.radians(angle_deg)

    cam_loc = carla.Location(
        x=aim.x + math.cos(a) * distance,
        y=aim.y + math.sin(a) * distance,
        z=aim.z + camera_z_offset,
    )

    cam_rot = look_at(cam_loc, aim)

    return carla.Transform(cam_loc, cam_rot)


def camera_vectors(cam_tf):
    yaw = math.radians(cam_tf.rotation.yaw)

    forward = carla.Vector3D(
        x=math.cos(yaw),
        y=math.sin(yaw),
        z=0.0,
    )

    right = carla.Vector3D(
        x=math.cos(yaw + math.pi / 2.0),
        y=math.sin(yaw + math.pi / 2.0),
        z=0.0,
    )

    return forward, right


def loc_from_camera(cam_tf, forward_dist, right_dist, z=0.0):
    forward, right = camera_vectors(cam_tf)

    return carla.Location(
        x=cam_tf.location.x + forward.x * forward_dist + right.x * right_dist,
        y=cam_tf.location.y + forward.y * forward_dist + right.y * right_dist,
        z=z,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)

    parser.add_argument("--map", default="Town10HD")
    parser.add_argument("--reload-map", action="store_true")
    parser.add_argument("--destroy-dynamic", action="store_true")

    parser.add_argument("--light-index", type=int, default=0)
    parser.add_argument("--light-state", choices=["red", "yellow", "green"], default="red")

    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--distance", type=float, default=16.0)
    parser.add_argument("--aim-z-add", type=float, default=3.0)
    parser.add_argument("--camera-z-offset", type=float, default=0.0)

    parser.add_argument("--car-depth", type=float, default=10.0)
    parser.add_argument("--ped-depth", type=float, default=8.5)
    parser.add_argument("--moto-depth", type=float, default=11.5)

    parser.add_argument("--car-right", type=float, default=-3.5)
    parser.add_argument("--ped-right", type=float, default=0.0)
    parser.add_argument("--moto-right", type=float, default=3.5)

    parser.add_argument("--out", default="/tmp/adas_tl_only_scene.json")

    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    if args.reload_map:
        print(f"[SCENE] loading map: {args.map}")
        world = client.load_world(args.map)
        time.sleep(3.0)
    else:
        world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    set_weather(world)

    if args.destroy_dynamic:
        destroy_dynamic(world)

    bp_lib = world.get_blueprint_library()

    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    lights.sort(key=lambda a: (a.get_transform().location.x, a.get_transform().location.y))

    if not lights:
        raise RuntimeError("Bu map içinde traffic light bulunamadı. Town10HD/Town03 dene.")

    if args.light_index < 0 or args.light_index >= len(lights):
        raise RuntimeError(f"Geçersiz --light-index. Bulunan ışık sayısı: {len(lights)}")

    selected_light = lights[args.light_index]

    # Diğer ışıkları red yap, seçili ışığı istenen state'e sabitle.
    for l in lights:
        set_light_state(l, "red")

    set_light_state(selected_light, args.light_state)

    light_tf = selected_light.get_transform()
    light_target = get_light_target(selected_light)

    aim = carla.Location(
        x=light_target.x,
        y=light_target.y,
        z=light_tf.location.z + args.aim_z_add,
    )

    cam_tf = make_camera_transform(
        aim=aim,
        angle_deg=args.angle,
        distance=args.distance,
        camera_z_offset=args.camera_z_offset,
    )

    world.get_spectator().set_transform(cam_tf)

    # Kamera görüş alanına objeleri koyuyoruz.
    actor_yaw = cam_tf.rotation.yaw + 180.0

    car_loc = loc_from_camera(cam_tf, args.car_depth, args.car_right)
    ped_loc = loc_from_camera(cam_tf, args.ped_depth, args.ped_right)
    moto_loc = loc_from_camera(cam_tf, args.moto_depth, args.moto_right)

    car_bp = find_blueprint(
        bp_lib,
        [
            "vehicle.tesla.model3",
            "vehicle.audi.tt",
            "vehicle.dodge.charger_2020",
            "vehicle.lincoln.mkz_2020",
        ],
        "vehicle.*",
    )

    moto_bp = find_motorcycle_blueprint(bp_lib)

    walker_candidates = list(bp_lib.filter("walker.pedestrian.*"))
    if not walker_candidates:
        raise RuntimeError("walker.pedestrian.* blueprint bulunamadı.")

    walker_bp = random.choice(walker_candidates)

    car_bp = prepare_vehicle_bp(car_bp, color="255,0,0")
    moto_bp = prepare_vehicle_bp(moto_bp, color="0,0,255")

    if car_bp is None:
        raise RuntimeError("Araç blueprint bulunamadı.")

    if moto_bp is None:
        raise RuntimeError("Motorsiklet blueprint bulunamadı. CARLA blueprint listesinde motorcycle yok.")

    spawned = []

    car = try_spawn_with_offsets(world, car_bp, car_loc, actor_yaw, "vehicle")
    spawned.append(car)

    pedestrian = try_spawn_with_offsets(world, walker_bp, ped_loc, actor_yaw, "pedestrian")
    spawned.append(pedestrian)

    motorcycle = try_spawn_with_offsets(world, moto_bp, moto_loc, actor_yaw, "motorcycle")
    spawned.append(motorcycle)

    # Araçlar sabit kalsın.
    for a in spawned:
        try:
            if a.type_id.startswith("vehicle."):
                a.set_autopilot(False)
                a.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                a.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        except Exception:
            pass

    payload = {
        "host": args.host,
        "port": args.port,
        "map": world.get_map().name,
        "selected_light": {
            "index": args.light_index,
            "id": int(selected_light.id),
            "state": args.light_state,
            "target": loc_dict(light_target),
        },
        "actors": [
            {"kind": "vehicle", "id": int(car.id), "type": car.type_id, "location": loc_dict(car.get_transform().location)},
            {"kind": "person", "id": int(pedestrian.id), "type": pedestrian.type_id, "location": loc_dict(pedestrian.get_transform().location)},
            {"kind": "motorcycle", "id": int(motorcycle.id), "type": motorcycle.type_id, "location": loc_dict(motorcycle.get_transform().location)},
        ],
        "camera_transform": {
            "location": loc_dict(cam_tf.location),
            "rotation": rot_dict(cam_tf.rotation),
        },
    }

    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("===================================================")
    print("[SCENE] VEHICLE + PERSON + MOTORCYCLE + TRAFFIC LIGHT READY")
    print(f"[SCENE] map          : {world.get_map().name}")
    print(f"[SCENE] light index  : {args.light_index}")
    print(f"[SCENE] light id     : {selected_light.id}")
    print(f"[SCENE] light state  : {args.light_state}")
    print(f"[SCENE] vehicle      : id={car.id}, type={car.type_id}")
    print(f"[SCENE] pedestrian   : id={pedestrian.id}, type={pedestrian.type_id}")
    print(f"[SCENE] motorcycle   : id={motorcycle.id}, type={motorcycle.type_id}")
    print(f"[SCENE] camera json  : {args.out}")
    print("===================================================")
    print("[SCENE] Bu terminal açık kalacak. Ctrl+C ile kapat.")

    try:
        while True:
            set_light_state(selected_light, args.light_state)

            for a in spawned:
                try:
                    if a.type_id.startswith("vehicle."):
                        a.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                        a.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                except Exception:
                    pass

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("[SCENE] closing...")

    finally:
        # İstersen burada otomatik silme yapabiliriz ama test sırasında sahne kalsın diye silmiyoruz.
        pass


if __name__ == "__main__":
    main()

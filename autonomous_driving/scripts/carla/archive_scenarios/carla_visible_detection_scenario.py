#!/usr/bin/env python3

import argparse
import json
import math
import random
import time
from pathlib import Path

import carla


ACTOR_FILE = Path("/tmp/carla_visible_detection_actor_ids.json")


def connect():
    client = carla.Client("localhost", 2000)
    client.set_timeout(40.0)
    return client, client.get_world()


def load_ids():
    if ACTOR_FILE.exists():
        try:
            return json.loads(ACTOR_FILE.read_text())
        except Exception:
            return []
    return []


def save_ids(ids):
    ACTOR_FILE.write_text(json.dumps(ids, indent=2))


def tick(world, n=5):
    for _ in range(n):
        try:
            world.tick()
        except Exception:
            try:
                world.wait_for_tick(seconds=1.0)
            except Exception:
                pass
        time.sleep(0.05)


def destroy_old(world):
    roles = {
        "ego_vehicle",
        "rgb_front",
        "adas_visible_vehicle",
        "adas_visible_pedestrian",
    }

    destroyed = 0

    for actor_id in load_ids():
        actor = world.get_actor(actor_id)
        if actor is not None:
            try:
                actor.destroy()
                destroyed += 1
            except Exception:
                pass

    for actor in world.get_actors():
        role = actor.attributes.get("role_name", "")
        if role in roles:
            try:
                actor.destroy()
                destroyed += 1
            except Exception:
                pass

    save_ids([])
    print(f"Temizlenen actor sayısı: {destroyed}")


def set_world(world):
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    world.set_weather(
        carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            sun_altitude_angle=70.0,
            sun_azimuth_angle=35.0,
            fog_density=0.0,
            wetness=0.0,
        )
    )


def angle_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def get_wp(world, loc):
    return world.get_map().get_waypoint(
        loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )


def is_clean_straight(wp):
    base_yaw = wp.transform.rotation.yaw

    for d in [8, 15, 22, 30, 40, 50]:
        nxt = wp.next(float(d))
        if not nxt:
            return False

        nwp = nxt[0]

        if nwp.is_junction:
            return False

        if angle_diff(base_yaw, nwp.transform.rotation.yaw) > 12.0:
            return False

    return True


def adjacent_lanes(wp):
    lanes = []

    left = wp.get_left_lane()
    right = wp.get_right_lane()

    if left is not None and left.lane_type == carla.LaneType.Driving:
        if angle_diff(wp.transform.rotation.yaw, left.transform.rotation.yaw) < 25.0:
            lanes.append(("left", left))

    if right is not None and right.lane_type == carla.LaneType.Driving:
        if angle_diff(wp.transform.rotation.yaw, right.transform.rotation.yaw) < 25.0:
            lanes.append(("right", right))

    return lanes


def choose_spawn(world):
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    for sp in spawn_points:
        wp = get_wp(world, sp.location)

        if wp is None:
            continue

        if wp.is_junction:
            continue

        if not is_clean_straight(wp):
            continue

        if len(adjacent_lanes(wp)) == 0:
            continue

        return sp

    raise RuntimeError("Düz ve yan şeritli spawn bulunamadı. Town04/Town05 dene.")


def choose_vehicle_bp(bp_lib):
    preferred = [
        "vehicle.tesla.model3",
        "vehicle.audi.tt",
        "vehicle.lincoln.mkz_2020",
        "vehicle.toyota.prius",
        "vehicle.bmw.grandtourer",
    ]

    for name in preferred:
        xs = bp_lib.filter(name)
        if len(xs) > 0:
            return xs[0]

    xs = bp_lib.filter("vehicle.*")
    if len(xs) == 0:
        raise RuntimeError("Vehicle blueprint yok.")

    return random.choice(xs)


def spawn_ego(world, actor_ids):
    bp_lib = world.get_blueprint_library()

    bp = bp_lib.find("vehicle.tesla.model3")
    bp.set_attribute("role_name", "ego_vehicle")

    if bp.has_attribute("color"):
        bp.set_attribute("color", "0,0,255")

    tf = choose_spawn(world)
    tf.location.z += 0.5

    ego = world.try_spawn_actor(bp, tf)

    if ego is None:
        raise RuntimeError("Ego spawn edilemedi.")

    ego.set_autopilot(False)
    actor_ids.append(ego.id)

    print(f"Ego spawn edildi: id={ego.id}")
    return ego


def spawn_camera(world, ego, actor_ids):
    bp_lib = world.get_blueprint_library()

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("role_name", "rgb_front")
    cam_bp.set_attribute("image_size_x", "800")
    cam_bp.set_attribute("image_size_y", "450")
    cam_bp.set_attribute("fov", "85")
    cam_bp.set_attribute("sensor_tick", "0.15")

    cam_tf = carla.Transform(
        carla.Location(x=1.80, y=0.0, z=1.55),
        carla.Rotation(pitch=-4.0, yaw=0.0, roll=0.0),
    )

    cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
    actor_ids.append(cam.id)

    print(f"RGB front kamera spawn edildi: id={cam.id}, 800x450")
    return cam


def spawn_visible_vehicles(world, ego, actor_ids, count):
    bp_lib = world.get_blueprint_library()
    ego_wp = get_wp(world, ego.get_location())
    lanes = adjacent_lanes(ego_wp)

    if not lanes:
        raise RuntimeError("Yan şerit bulunamadı.")

    distances = [10.0, 16.0, 23.0, 31.0]
    spawned = []

    for i in range(count):
        side, lane_wp = lanes[i % len(lanes)]
        d = distances[i % len(distances)]

        nxt = lane_wp.next(d)
        if not nxt:
            continue

        wp = nxt[0]
        tf = wp.transform
        tf.location.z += 0.5
        tf.rotation.pitch = 0.0
        tf.rotation.roll = 0.0

        bp = choose_vehicle_bp(bp_lib)
        bp.set_attribute("role_name", "adas_visible_vehicle")

        if bp.has_attribute("color"):
            colors = bp.get_attribute("color").recommended_values
            if colors:
                bp.set_attribute("color", random.choice(colors))

        v = world.try_spawn_actor(bp, tf)

        if v is None:
            continue

        v.set_autopilot(False)
        v.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                hand_brake=True,
                steer=0.0,
            )
        )

        actor_ids.append(v.id)
        spawned.append(v)

        print(f"Görünür yan şerit aracı: id={v.id}, side={side}, distance={d}")

    print(f"Toplam görünür araç: {len(spawned)} / {count}")
    return spawned


def spawn_visible_pedestrians(world, ego, actor_ids, count):
    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter("walker.pedestrian.*")

    if len(walker_bps) == 0:
        raise RuntimeError("Yaya blueprint yok.")

    ego_wp = get_wp(world, ego.get_location())
    distances = [8.0, 14.0, 21.0, 29.0]
    base_offset = ego_wp.lane_width / 2.0 + 3.8

    spawned = []

    for i in range(count):
        d = distances[i % len(distances)]

        fwd = ego_wp.next(d)
        if not fwd:
            continue

        wp = fwd[0]
        right = wp.transform.get_right_vector()

        sign = 1.0 if i % 2 == 0 else -1.0
        offset = sign * base_offset

        loc = carla.Location(
            x=wp.transform.location.x + right.x * offset,
            y=wp.transform.location.y + right.y * offset,
            z=wp.transform.location.z + 1.0,
        )

        tf = carla.Transform(
            loc,
            carla.Rotation(
                pitch=0.0,
                yaw=wp.transform.rotation.yaw,
                roll=0.0,
            ),
        )

        bp = random.choice(walker_bps)

        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "adas_visible_pedestrian")

        p = world.try_spawn_actor(bp, tf)

        if p is None:
            continue

        actor_ids.append(p.id)
        spawned.append(p)

        side = "right" if sign > 0 else "left"
        print(f"Görünür yol kenarı yaya: id={p.id}, side={side}, distance={d}, offset={offset:.1f}")

    print(f"Toplam görünür yaya: {len(spawned)} / {count}")
    return spawned


def follow_spectator(world, ego):
    tf = ego.get_transform()
    yaw = math.radians(tf.rotation.yaw)

    world.get_spectator().set_transform(
        carla.Transform(
            carla.Location(
                x=tf.location.x - 8.0 * math.cos(yaw),
                y=tf.location.y - 8.0 * math.sin(yaw),
                z=tf.location.z + 4.0,
            ),
            carla.Rotation(
                pitch=-18.0,
                yaw=tf.rotation.yaw,
                roll=0.0,
            ),
        )
    )


def stop_ego(ego):
    ego.set_autopilot(False)
    ego.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            brake=1.0,
            hand_brake=True,
            steer=0.0,
        )
    )


def run_ego(client, world, ego, seconds):
    tm = client.get_trafficmanager(8000)
    tm.set_global_distance_to_leading_vehicle(10.0)
    tm.set_random_device_seed(42)
    tm.auto_lane_change(ego, False)
    tm.vehicle_percentage_speed_difference(ego, 85.0)

    ego.set_autopilot(True, tm.get_port())

    print(f"Ego {seconds} saniye gidecek.")

    start = time.time()

    while time.time() - start < seconds:
        follow_spectator(world, ego)
        time.sleep(0.05)

    stop_ego(ego)
    print("Süre doldu. Ego durdu.")

    while True:
        follow_spectator(world, ego)
        time.sleep(0.05)


def print_status(world):
    vehicles = world.get_actors().filter("vehicle.*")
    walkers = world.get_actors().filter("walker.pedestrian.*")
    cameras = world.get_actors().filter("sensor.camera.rgb")

    print(f"Vehicle count: {len(vehicles)}")
    for v in vehicles:
        print(f"  vehicle id={v.id}, type={v.type_id}, role_name={v.attributes.get('role_name')}")

    print(f"Walker count: {len(walkers)}")
    for w in walkers:
        print(f"  walker id={w.id}, type={w.type_id}, role_name={w.attributes.get('role_name')}")

    print(f"RGB camera count: {len(cameras)}")
    for c in cameras:
        parent = getattr(c, "parent", None)
        parent_id = parent.id if parent is not None else None
        print(f"  camera id={c.id}, role_name={c.attributes.get('role_name')}, parent={parent_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="Town04")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--vehicles", type=int, default=4)
    parser.add_argument("--peds", type=int, default=4)
    parser.add_argument("--move-seconds", type=float, default=20.0)
    parser.add_argument("--no-load-map", action="store_true")
    args = parser.parse_args()

    client, world = connect()

    if args.clear:
        destroy_old(world)
        print_status(world)
        return

    if not args.no_load_map:
        if args.map not in world.get_map().name:
            print(f"Map yükleniyor: {args.map}")
            world = client.load_world(args.map)
            time.sleep(2.0)

    set_world(world)
    destroy_old(world)
    tick(world, 5)

    actor_ids = []

    ego = spawn_ego(world, actor_ids)
    spawn_camera(world, ego, actor_ids)

    spawn_visible_vehicles(world, ego, actor_ids, args.vehicles)
    spawn_visible_pedestrians(world, ego, actor_ids, args.peds)

    save_ids(actor_ids)
    tick(world, 10)

    print_status(world)

    print("")
    print("GÖRÜNÜR ALGILAMA SENARYOSU HAZIR")
    print("Objeler kameraya yakın.")
    print("Ego ön şeridinde engel yok.")
    print("Araçlar yan şeritlerde düz.")
    print("Yayalar yol kenarında.")
    print("Ego 20 saniye gider, sonra durur.")
    print("")

    run_ego(client, world, ego, args.move_seconds)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import json
import math
import random
import time
from pathlib import Path

import carla


ACTOR_FILE = Path("/tmp/carla_adas_side_traffic_actor_ids.json")


def load_ids():
    if ACTOR_FILE.exists():
        try:
            return json.loads(ACTOR_FILE.read_text())
        except Exception:
            return []
    return []


def save_ids(ids):
    ACTOR_FILE.write_text(json.dumps(ids, indent=2))


def connect():
    client = carla.Client("localhost", 2000)
    client.set_timeout(40.0)
    world = client.get_world()
    return client, world


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
    destroyed = 0

    for actor_id in load_ids():
        actor = world.get_actor(actor_id)
        if actor is not None:
            try:
                actor.destroy()
                destroyed += 1
            except Exception:
                pass

    roles = {
        "ego_vehicle",
        "rgb_front",
        "adas_side_vehicle_left",
        "adas_side_vehicle_right",
        "adas_roadside_pedestrian",
    }

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


def set_world_settings(world):
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    world.set_weather(
        carla.WeatherParameters(
            cloudiness=5.0,
            precipitation=0.0,
            sun_altitude_angle=65.0,
            sun_azimuth_angle=20.0,
            fog_density=0.0,
            wetness=0.0,
        )
    )


def get_driving_waypoint(world, location):
    return world.get_map().get_waypoint(
        location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )


def has_clear_forward(wp, distance=70.0):
    return bool(wp.next(distance))


def choose_spawn_with_side_lanes(world):
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    for sp in spawn_points:
        wp = get_driving_waypoint(world, sp.location)
        if wp is None:
            continue

        if wp.is_junction:
            continue

        if not has_clear_forward(wp, 70.0):
            continue

        left = wp.get_left_lane()
        right = wp.get_right_lane()

        left_ok = left is not None and left.lane_type == carla.LaneType.Driving and not left.is_junction
        right_ok = right is not None and right.lane_type == carla.LaneType.Driving and not right.is_junction

        if left_ok or right_ok:
            return sp

    for sp in spawn_points:
        wp = get_driving_waypoint(world, sp.location)
        if wp is not None and has_clear_forward(wp, 50.0):
            return sp

    if not spawn_points:
        raise RuntimeError("Map spawn point üretmedi.")

    return spawn_points[0]


def choose_bp(bp_lib, patterns):
    for pattern in patterns:
        matches = bp_lib.filter(pattern)
        if len(matches) > 0:
            return matches[0]
    return None


def spawn_ego(world, actor_ids):
    bp_lib = world.get_blueprint_library()
    ego_bp = bp_lib.find("vehicle.tesla.model3")
    ego_bp.set_attribute("role_name", "ego_vehicle")

    if ego_bp.has_attribute("color"):
        ego_bp.set_attribute("color", "0,0,255")

    tf = choose_spawn_with_side_lanes(world)
    tf.location.z += 0.5

    ego = world.try_spawn_actor(ego_bp, tf)

    if ego is None:
        raise RuntimeError("Ego araç spawn edilemedi.")

    ego.set_autopilot(False)
    actor_ids.append(ego.id)

    print(f"Ego araç spawn edildi: id={ego.id}, type={ego.type_id}")
    return ego


def spawn_camera(world, ego, actor_ids):
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")

    cam_bp.set_attribute("role_name", "rgb_front")
    cam_bp.set_attribute("image_size_x", "640")
    cam_bp.set_attribute("image_size_y", "360")
    cam_bp.set_attribute("fov", "90")
    cam_bp.set_attribute("sensor_tick", "0.10")

    cam_tf = carla.Transform(
        carla.Location(x=1.80, y=0.0, z=1.55),
        carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0),
    )

    cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
    actor_ids.append(cam.id)

    print(f"RGB front kamera spawn edildi: id={cam.id}, 640x360, 10 FPS")
    return cam


def get_side_lane_waypoints(ego_wp):
    side_lanes = []

    left = ego_wp.get_left_lane()
    right = ego_wp.get_right_lane()

    if left is not None and left.lane_type == carla.LaneType.Driving and not left.is_junction:
        side_lanes.append(("left", left))

    if right is not None and right.lane_type == carla.LaneType.Driving and not right.is_junction:
        side_lanes.append(("right", right))

    return side_lanes


def spawn_side_vehicles(client, world, ego, actor_ids, count):
    bp_lib = world.get_blueprint_library()

    vehicle_bps = [
        bp for bp in bp_lib.filter("vehicle.*")
        if not bp.id.endswith("microlino")
        and not bp.id.endswith("carlacola")
        and not bp.id.endswith("firetruck")
        and not bp.id.endswith("ambulance")
    ]

    if not vehicle_bps:
        raise RuntimeError("Vehicle blueprint yok.")

    ego_wp = get_driving_waypoint(world, ego.get_location())
    side_lanes = get_side_lane_waypoints(ego_wp)

    if not side_lanes:
        print("Uyarı: Yan şerit bulunamadı. Ego şeridine araç koymayacağım.")
        return []

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(8.0)
    traffic_manager.set_random_device_seed(13)

    spawned = []
    distances = [18.0, 32.0, 48.0, 65.0, 82.0]

    for i in range(count):
        side_name, lane_wp = side_lanes[i % len(side_lanes)]
        d = distances[i % len(distances)]

        next_wps = lane_wp.next(d)
        if not next_wps:
            continue

        tf = next_wps[0].transform
        tf.location.z += 0.5

        bp = random.choice(vehicle_bps)

        role = "adas_side_vehicle_left" if side_name == "left" else "adas_side_vehicle_right"
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", role)

        if bp.has_attribute("color"):
            colors = bp.get_attribute("color").recommended_values
            if colors:
                bp.set_attribute("color", random.choice(colors))

        vehicle = world.try_spawn_actor(bp, tf)

        if vehicle is None:
            continue

        actor_ids.append(vehicle.id)
        spawned.append(vehicle)

        vehicle.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.vehicle_percentage_speed_difference(vehicle, 65.0)

        print(f"Yan şeride araç spawn edildi: id={vehicle.id}, side={side_name}, distance={d}")

    return spawned


def spawn_roadside_pedestrians(world, ego, actor_ids, count):
    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter("walker.pedestrian.*")

    if len(walker_bps) == 0:
        raise RuntimeError("Yaya blueprint yok.")

    ego_wp = get_driving_waypoint(world, ego.get_location())

    spawned = []
    distances = [14.0, 24.0, 36.0, 50.0, 64.0]

    # Şerit merkezinden kaldırım/yol kenarı tarafına itiyoruz.
    # lane_width genelde 3.5 civarı. +2.4 veya -2.4 yolun kenarına yakın tutar.
    base_edge_offset = ego_wp.lane_width / 2.0 + 1.8

    side_offsets = [
        base_edge_offset,
        -base_edge_offset,
        base_edge_offset + 0.8,
        -base_edge_offset - 0.8,
    ]

    for i in range(count):
        d = distances[i % len(distances)]
        next_wps = ego_wp.next(d)

        if not next_wps:
            continue

        wp = next_wps[0]
        tf_base = wp.transform
        right = tf_base.get_right_vector()

        off = side_offsets[i % len(side_offsets)]

        loc = carla.Location(
            x=tf_base.location.x + right.x * off,
            y=tf_base.location.y + right.y * off,
            z=tf_base.location.z + 1.0,
        )

        walker_bp = random.choice(walker_bps)

        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")

        if walker_bp.has_attribute("role_name"):
            walker_bp.set_attribute("role_name", "adas_roadside_pedestrian")

        ped_tf = carla.Transform(
            loc,
            carla.Rotation(
                pitch=0.0,
                yaw=tf_base.rotation.yaw + 90.0,
                roll=0.0,
            ),
        )

        ped = world.try_spawn_actor(walker_bp, ped_tf)

        if ped is None:
            continue

        actor_ids.append(ped.id)
        spawned.append(ped)

        print(f"Yol kenarına yaya spawn edildi: id={ped.id}, distance={d}, edge_offset={off:.2f}")

    return spawned


def apply_ego_motion(ego, speed):
    if speed == "stop":
        control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
    elif speed == "slow":
        control = carla.VehicleControl(throttle=0.20, brake=0.0, steer=0.0)
    else:
        control = carla.VehicleControl(throttle=0.30, brake=0.0, steer=0.0)

    ego.apply_control(control)


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
    parser.add_argument("--map", default="Town03")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--side-vehicles", type=int, default=4)
    parser.add_argument("--roadside-peds", type=int, default=4)
    parser.add_argument("--speed", choices=["stop", "slow", "normal"], default="slow")
    parser.add_argument("--no-load-map", action="store_true")
    args = parser.parse_args()

    client, world = connect()

    if args.clear:
        destroy_old(world)
        print_status(world)
        return

    if not args.no_load_map:
        current_map = world.get_map().name
        if args.map not in current_map:
            print(f"Map yükleniyor: {args.map}")
            world = client.load_world(args.map)
            time.sleep(2.0)

    set_world_settings(world)
    destroy_old(world)
    tick(world, 5)

    actor_ids = []

    ego = spawn_ego(world, actor_ids)
    spawn_camera(world, ego, actor_ids)

    spawn_side_vehicles(client, world, ego, actor_ids, args.side_vehicles)
    spawn_roadside_pedestrians(world, ego, actor_ids, args.roadside_peds)

    save_ids(actor_ids)
    tick(world, 10)

    print_status(world)

    print("")
    print("ALGILAMA SENARYOSU HAZIR")
    print("Ego önünde aynı şeritte araç yok.")
    print("Arabalar yan/farklı şeritlerde.")
    print("Yayalar yol kenarında.")
    print(f"Ego hareket modu: {args.speed}")
    print("Spectator takip ediyor. Çıkmak için CTRL+C.")
    print("")

    while True:
        apply_ego_motion(ego, args.speed)
        follow_spectator(world, ego)
        time.sleep(0.05)


if __name__ == "__main__":
    main()

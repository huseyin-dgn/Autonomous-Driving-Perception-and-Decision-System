#!/usr/bin/env python3

import argparse
import json
import math
import random
import time
from pathlib import Path

import carla


ACTOR_FILE = Path("/tmp/carla_adas_moving_actor_ids.json")


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

    for actor in world.get_actors():
        role = actor.attributes.get("role_name", "")
        if role in [
            "ego_vehicle",
            "adas_lead_vehicle",
            "adas_road_pedestrian",
            "rgb_front",
        ]:
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

    weather = carla.WeatherParameters(
        cloudiness=5.0,
        precipitation=0.0,
        sun_altitude_angle=65.0,
        sun_azimuth_angle=20.0,
        fog_density=0.0,
        wetness=0.0,
    )
    world.set_weather(weather)


def choose_straight_spawn(world, min_forward=35.0):
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    for sp in spawn_points:
        wp = world.get_map().get_waypoint(
            sp.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if wp is None:
            continue

        if wp.is_junction:
            continue

        if wp.next(min_forward):
            return sp

    if not spawn_points:
        raise RuntimeError("Map spawn point üretmedi.")

    return spawn_points[0]


def choose_bp(bp_lib, patterns):
    for p in patterns:
        matches = bp_lib.filter(p)
        if len(matches) > 0:
            return matches[0]
    return None


def spawn_camera(world, ego, actor_ids):
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")

    cam_bp.set_attribute("role_name", "rgb_front")
    cam_bp.set_attribute("image_size_x", "960")
    cam_bp.set_attribute("image_size_y", "540")
    cam_bp.set_attribute("fov", "90")
    cam_bp.set_attribute("sensor_tick", "0.08")

    cam_tf = carla.Transform(
        carla.Location(x=1.80, y=0.0, z=1.55),
        carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0),
    )

    cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
    actor_ids.append(cam.id)

    print(f"RGB front kamera spawn edildi: id={cam.id}, 640x360, 10 FPS")
    return cam


def spawn_ego(world, actor_ids):
    bp_lib = world.get_blueprint_library()

    ego_bp = bp_lib.find("vehicle.tesla.model3")
    ego_bp.set_attribute("role_name", "ego_vehicle")

    if ego_bp.has_attribute("color"):
        ego_bp.set_attribute("color", "0,0,255")

    ego_tf = choose_straight_spawn(world)
    ego_tf.location.z += 0.5

    ego = world.try_spawn_actor(ego_bp, ego_tf)

    if ego is None:
        raise RuntimeError("Ego araç spawn edilemedi.")

    ego.set_autopilot(False)
    actor_ids.append(ego.id)

    print(f"Ego araç spawn edildi: id={ego.id}, type={ego.type_id}")
    return ego


def spawn_front_vehicle(world, ego, actor_ids, distance):
    bp_lib = world.get_blueprint_library()

    vehicle_bp = choose_bp(
        bp_lib,
        [
            "vehicle.audi.tt",
            "vehicle.lincoln.mkz_2020",
            "vehicle.toyota.prius",
            "vehicle.tesla.model3",
        ],
    )

    if vehicle_bp is None:
        vehicles = bp_lib.filter("vehicle.*")
        if len(vehicles) == 0:
            raise RuntimeError("Vehicle blueprint yok.")
        vehicle_bp = random.choice(vehicles)

    vehicle_bp.set_attribute("role_name", "adas_lead_vehicle")

    if vehicle_bp.has_attribute("color"):
        colors = vehicle_bp.get_attribute("color").recommended_values
        if colors:
            vehicle_bp.set_attribute("color", colors[0])

    ego_wp = world.get_map().get_waypoint(
        ego.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    for d in [distance, distance + 4.0, distance + 8.0]:
        wps = ego_wp.next(float(d))

        if not wps:
            continue

        tf = wps[0].transform
        tf.location.z += 0.5

        lead = world.try_spawn_actor(vehicle_bp, tf)

        if lead is not None:
            lead.set_autopilot(False)
            lead.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    hand_brake=True,
                )
            )

            actor_ids.append(lead.id)

            print(f"YOL ÜSTÜNE öndeki araç spawn edildi: id={lead.id}, distance={d}")
            return lead

    raise RuntimeError("Öndeki araç yol üstüne spawn edilemedi.")


def spawn_road_pedestrian(world, ego, actor_ids, distance, lane_offset):
    bp_lib = world.get_blueprint_library()

    walker_bps = bp_lib.filter("walker.pedestrian.*")

    if len(walker_bps) == 0:
        raise RuntimeError("Yaya blueprint yok.")

    walker_bp = random.choice(walker_bps)

    if walker_bp.has_attribute("is_invincible"):
        walker_bp.set_attribute("is_invincible", "false")

    ego_wp = world.get_map().get_waypoint(
        ego.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    candidate_distances = [
        distance,
        distance + 2.0,
        max(6.0, distance - 2.0),
        distance + 4.0,
    ]

    candidate_offsets = [
        lane_offset,
        0.0,
        0.5,
        -0.5,
        1.0,
        -1.0,
    ]

    for d in candidate_distances:
        wps = ego_wp.next(float(d))

        if not wps:
            continue

        wp = wps[0]
        base_tf = wp.transform
        right = base_tf.get_right_vector()

        for off in candidate_offsets:
            loc = carla.Location(
                x=base_tf.location.x + right.x * float(off),
                y=base_tf.location.y + right.y * float(off),
                z=base_tf.location.z + 0.8,
            )

            tf = carla.Transform(
                loc,
                carla.Rotation(
                    pitch=0.0,
                    yaw=base_tf.rotation.yaw + 180.0,
                    roll=0.0,
                ),
            )

            ped = world.try_spawn_actor(walker_bp, tf)

            if ped is not None:
                actor_ids.append(ped.id)

                print(
                    f"YOL ÜSTÜNE yaya spawn edildi: id={ped.id}, distance={d}, lane_offset={off}"
                )

                return ped

    raise RuntimeError("Yaya yol üstüne spawn edilemedi.")


def follow_spectator(world, ego):
    tf = ego.get_transform()
    yaw = math.radians(tf.rotation.yaw)

    spectator_tf = carla.Transform(
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

    world.get_spectator().set_transform(spectator_tf)


def apply_slow_motion_control(ego, speed_level):
    if speed_level == "stop":
        control = carla.VehicleControl(
            throttle=0.0,
            brake=1.0,
            steer=0.0,
            hand_brake=False,
        )
    elif speed_level == "slow":
        control = carla.VehicleControl(
            throttle=0.22,
            brake=0.0,
            steer=0.0,
            hand_brake=False,
        )
    else:
        control = carla.VehicleControl(
            throttle=0.32,
            brake=0.0,
            steer=0.0,
            hand_brake=False,
        )

    ego.apply_control(control)


def print_status(world):
    vehicles = world.get_actors().filter("vehicle.*")
    walkers = world.get_actors().filter("walker.pedestrian.*")
    cameras = world.get_actors().filter("sensor.camera.rgb")

    print(f"Vehicle count: {len(vehicles)}")
    for v in vehicles:
        print(
            f"  vehicle id={v.id}, type={v.type_id}, role_name={v.attributes.get('role_name')}"
        )

    print(f"Walker count: {len(walkers)}")
    for w in walkers:
        print(f"  walker id={w.id}, type={w.type_id}, role_name={w.attributes.get('role_name')}")

    print(f"RGB camera count: {len(cameras)}")
    for c in cameras:
        parent = getattr(c, "parent", None)
        parent_id = parent.id if parent is not None else None
        print(
            f"  camera id={c.id}, role_name={c.attributes.get('role_name')}, parent={parent_id}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="Town03")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--front-vehicle-distance", type=float, default=18.0)
    parser.add_argument("--ped-distance", type=float, default=10.0)
    parser.add_argument("--ped-offset", type=float, default=0.0)
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

    lead = spawn_front_vehicle(
        world,
        ego,
        actor_ids,
        distance=args.front_vehicle_distance,
    )

    ped = spawn_road_pedestrian(
        world,
        ego,
        actor_ids,
        distance=args.ped_distance,
        lane_offset=args.ped_offset,
    )

    save_ids(actor_ids)
    tick(world, 10)

    print_status(world)

    print("")
    print("HAREKETLİ ALGILAMA SENARYOSU HAZIR")
    print(f"Ego hareket modu: {args.speed}")
    print("Yol üstünde: ego araç + önde araç + yaya")
    print("Spectator takip ediyor. Çıkmak için CTRL+C.")
    print("")

    while True:
        apply_slow_motion_control(ego, args.speed)
        follow_spectator(world, ego)
        time.sleep(0.05)


if __name__ == "__main__":
    main()

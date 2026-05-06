#!/usr/bin/env python3

import argparse
import json
import math
import random
import time
from pathlib import Path

import carla


ACTOR_FILE = Path("/tmp/carla_static_roadside_detection_actor_ids.json")


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
        "adas_static_roadside_vehicle",
        "adas_static_roadside_pedestrian",
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


def set_world(world):
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


def get_wp(world, loc):
    return world.get_map().get_waypoint(
        loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )


def choose_ego_spawn(world):
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    for sp in spawn_points:
        wp = get_wp(world, sp.location)

        if wp is None:
            continue

        if wp.is_junction:
            continue

        if not wp.next(95.0):
            continue

        return sp

    if not spawn_points:
        raise RuntimeError("CARLA map spawn point üretmedi.")

    return spawn_points[0]


def spawn_ego(world, actor_ids):
    bp_lib = world.get_blueprint_library()
    bp = bp_lib.find("vehicle.tesla.model3")
    bp.set_attribute("role_name", "ego_vehicle")

    if bp.has_attribute("color"):
        bp.set_attribute("color", "0,0,255")

    tf = choose_ego_spawn(world)
    tf.location.z += 0.5

    ego = world.try_spawn_actor(bp, tf)

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


def choose_vehicle_bp(bp_lib):
    preferred = [
        "vehicle.audi.tt",
        "vehicle.lincoln.mkz_2020",
        "vehicle.toyota.prius",
        "vehicle.tesla.model3",
        "vehicle.mercedes.coupe",
        "vehicle.bmw.grandtourer",
    ]

    for name in preferred:
        matches = bp_lib.filter(name)
        if len(matches) > 0:
            return matches[0]

    vehicles = bp_lib.filter("vehicle.*")
    if len(vehicles) == 0:
        raise RuntimeError("Vehicle blueprint yok.")

    return random.choice(vehicles)


def try_spawn_static_vehicle(world, bp, base_tf, right_vec, offset, yaw_offset):
    loc = carla.Location(
        x=base_tf.location.x + right_vec.x * offset,
        y=base_tf.location.y + right_vec.y * offset,
        z=base_tf.location.z + 0.5,
    )

    tf = carla.Transform(
        loc,
        carla.Rotation(
            pitch=0.0,
            yaw=base_tf.rotation.yaw + yaw_offset,
            roll=0.0,
        ),
    )

    return world.try_spawn_actor(bp, tf)


def spawn_roadside_vehicles(world, ego, actor_ids, count):
    bp_lib = world.get_blueprint_library()
    ego_wp = get_wp(world, ego.get_location())

    if ego_wp is None:
        raise RuntimeError("Ego waypoint alınamadı.")

    spawned = []

    distances = [16.0, 30.0, 46.0, 62.0, 78.0, 92.0]

    # Araçları şerit dışına alıyoruz. Ego şeridinin üstüne koymuyoruz.
    base_offset = ego_wp.lane_width / 2.0 + 3.2

    offset_candidates = [
        base_offset,
        -base_offset,
        base_offset + 1.0,
        -base_offset - 1.0,
        base_offset + 1.8,
        -base_offset - 1.8,
    ]

    yaw_offsets = [0.0, 180.0, 0.0, 180.0]

    for i, d in enumerate(distances):
        if len(spawned) >= count:
            break

        wps = ego_wp.next(d)

        if not wps:
            continue

        wp = wps[0]
        base_tf = wp.transform
        right_vec = base_tf.get_right_vector()

        bp = choose_vehicle_bp(bp_lib)
        bp.set_attribute("role_name", "adas_static_roadside_vehicle")

        if bp.has_attribute("color"):
            colors = bp.get_attribute("color").recommended_values
            if colors:
                bp.set_attribute("color", random.choice(colors))

        for off in offset_candidates:
            actor = try_spawn_static_vehicle(
                world=world,
                bp=bp,
                base_tf=base_tf,
                right_vec=right_vec,
                offset=off,
                yaw_offset=yaw_offsets[i % len(yaw_offsets)],
            )

            if actor is not None:
                actor.set_autopilot(False)
                actor.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        brake=1.0,
                        hand_brake=True,
                        steer=0.0,
                    )
                )

                actor_ids.append(actor.id)
                spawned.append(actor)

                print(
                    f"Yol kenarı STATİK araç spawn edildi: id={actor.id}, distance={d}, offset={off:.2f}"
                )
                break

    print(f"Toplam statik yol kenarı araç: {len(spawned)} / {count}")
    return spawned


def try_spawn_static_pedestrian(world, bp, base_tf, right_vec, offset):
    loc = carla.Location(
        x=base_tf.location.x + right_vec.x * offset,
        y=base_tf.location.y + right_vec.y * offset,
        z=base_tf.location.z + 1.0,
    )

    tf = carla.Transform(
        loc,
        carla.Rotation(
            pitch=0.0,
            yaw=base_tf.rotation.yaw + 90.0,
            roll=0.0,
        ),
    )

    return world.try_spawn_actor(bp, tf)


def spawn_roadside_pedestrians(world, ego, actor_ids, count):
    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter("walker.pedestrian.*")

    if len(walker_bps) == 0:
        raise RuntimeError("Yaya blueprint yok.")

    ego_wp = get_wp(world, ego.get_location())

    if ego_wp is None:
        raise RuntimeError("Ego waypoint alınamadı.")

    spawned = []

    distances = [22.0, 38.0, 54.0, 70.0, 86.0, 100.0]

    # Yayaları araçlardan daha dışarı alıyoruz.
    base_offset = ego_wp.lane_width / 2.0 + 5.0

    offset_candidates = [
        base_offset,
        -base_offset,
        base_offset + 0.8,
        -base_offset - 0.8,
        base_offset + 1.5,
        -base_offset - 1.5,
    ]

    for i, d in enumerate(distances):
        if len(spawned) >= count:
            break

        wps = ego_wp.next(d)

        if not wps:
            continue

        wp = wps[0]
        base_tf = wp.transform
        right_vec = base_tf.get_right_vector()

        bp = random.choice(walker_bps)

        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "adas_static_roadside_pedestrian")

        for off in offset_candidates:
            ped = try_spawn_static_pedestrian(
                world=world,
                bp=bp,
                base_tf=base_tf,
                right_vec=right_vec,
                offset=off,
            )

            if ped is not None:
                actor_ids.append(ped.id)
                spawned.append(ped)

                print(
                    f"Yol kenarı STATİK yaya spawn edildi: id={ped.id}, distance={d}, offset={off:.2f}"
                )
                break

    print(f"Toplam statik yol kenarı yaya: {len(spawned)} / {count}")
    return spawned


def apply_ego_motion(ego, speed):
    if speed == "stop":
        control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
    elif speed == "slow":
        control = carla.VehicleControl(throttle=0.18, brake=0.0, steer=0.0)
    else:
        control = carla.VehicleControl(throttle=0.28, brake=0.0, steer=0.0)

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
    parser.add_argument("--vehicles", type=int, default=4)
    parser.add_argument("--peds", type=int, default=4)
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

    set_world(world)
    destroy_old(world)
    tick(world, 5)

    actor_ids = []

    ego = spawn_ego(world, actor_ids)
    spawn_camera(world, ego, actor_ids)

    spawn_roadside_vehicles(world, ego, actor_ids, args.vehicles)
    spawn_roadside_pedestrians(world, ego, actor_ids, args.peds)

    save_ids(actor_ids)
    tick(world, 10)

    print_status(world)

    print("")
    print("STATİK YOL BOYU ALGILAMA SENARYOSU HAZIR")
    print("Ego şeridinin önünde engel yok.")
    print("Yol boyunca 4 araba + 4 yaya var.")
    print("Araçlar ve yayalar SABİT.")
    print("Ego yavaş ilerliyor.")
    print("Amaç: perception ekranında person / vehicle algısını görmek.")
    print("Çıkmak için CTRL+C.")
    print("")

    while True:
        apply_ego_motion(ego, args.speed)
        follow_spectator(world, ego)
        time.sleep(0.05)


if __name__ == "__main__":
    main()

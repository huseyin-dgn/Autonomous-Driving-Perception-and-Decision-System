#!/usr/bin/env python3

import argparse
import json
import math
import random
import time
from pathlib import Path

import carla


ACTOR_FILE = Path("/tmp/carla_dense_static_detection_actor_ids.json")


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
        "adas_dense_vehicle",
        "adas_dense_pedestrian",
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
        if actor.attributes.get("role_name", "") in roles:
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


def get_wp(world, loc):
    return world.get_map().get_waypoint(
        loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )


def angle_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def clean_spawn(world):
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    for sp in spawn_points:
        wp = get_wp(world, sp.location)
        if wp is None or wp.is_junction:
            continue

        base_yaw = wp.transform.rotation.yaw
        ok = True

        for d in [8, 14, 20, 28, 36]:
            nxt = wp.next(float(d))
            if not nxt:
                ok = False
                break
            nwp = nxt[0]
            if nwp.is_junction:
                ok = False
                break
            if angle_diff(base_yaw, nwp.transform.rotation.yaw) > 12.0:
                ok = False
                break

        if ok:
            return sp

    if not spawn_points:
        raise RuntimeError("Spawn point yok.")

    return spawn_points[0]


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

    tf = clean_spawn(world)
    tf.location.z += 0.5

    ego = world.try_spawn_actor(bp, tf)
    if ego is None:
        raise RuntimeError("Ego spawn edilemedi.")

    ego.set_autopilot(False)
    ego.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            brake=1.0,
            hand_brake=True,
            steer=0.0,
        )
    )

    actor_ids.append(ego.id)
    print(f"Ego sabit spawn edildi: id={ego.id}")

    return ego


def spawn_camera(world, ego, actor_ids):
    bp_lib = world.get_blueprint_library()

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("role_name", "rgb_front")
    cam_bp.set_attribute("image_size_x", "800")
    cam_bp.set_attribute("image_size_y", "450")
    cam_bp.set_attribute("fov", "65")
    cam_bp.set_attribute("sensor_tick", "0.15")

    cam_tf = carla.Transform(
        carla.Location(x=1.80, y=0.0, z=1.55),
        carla.Rotation(pitch=-4.0, yaw=0.0, roll=0.0),
    )

    cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
    actor_ids.append(cam.id)

    print(f"RGB front kamera spawn edildi: id={cam.id}, 800x450, FOV=65")

    return cam


def relative_transform(ego, x, y, z, yaw_offset=0.0):
    ego_tf = ego.get_transform()
    fwd = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()

    loc = carla.Location(
        x=ego_tf.location.x + fwd.x * x + right.x * y,
        y=ego_tf.location.y + fwd.y * x + right.y * y,
        z=ego_tf.location.z + z,
    )

    return carla.Transform(
        loc,
        carla.Rotation(
            pitch=0.0,
            yaw=ego_tf.rotation.yaw + yaw_offset,
            roll=0.0,
        ),
    )


def spawn_dense_vehicles(world, ego, actor_ids):
    bp_lib = world.get_blueprint_library()

    # Kamerada büyük görünsün diye yakın/orta mesafe.
    # y değerleri sağ/sola dağıtılmış; ego tam önüne engel koymuyoruz.
    placements = [
        (7.0,  2.7),
        (9.0, -2.7),
        (13.0,  3.1),
        (16.0, -3.1),
    ]

    spawned = []

    for x, y in placements:
        bp = choose_vehicle_bp(bp_lib)
        bp.set_attribute("role_name", "adas_dense_vehicle")

        if bp.has_attribute("color"):
            colors = bp.get_attribute("color").recommended_values
            if colors:
                bp.set_attribute("color", random.choice(colors))

        tf = relative_transform(ego, x=x, y=y, z=0.5, yaw_offset=0.0)

        v = world.try_spawn_actor(bp, tf)
        if v is None:
            print(f"Araç spawn edilemedi: x={x}, y={y}")
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

        side = "sağ" if y > 0 else "sol"
        print(f"Kamera içi araç: id={v.id}, side={side}, x={x}, y={y}")

    print(f"Toplam kamera içi araç: {len(spawned)} / 4")
    return spawned


def spawn_dense_pedestrians(world, ego, actor_ids):
    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter("walker.pedestrian.*")

    if len(walker_bps) == 0:
        raise RuntimeError("Yaya blueprint yok.")

    # Yayaları kenara değil, kamera içinde kalacak şekilde dağıtıyoruz.
    # Ego sabit olduğu için çarpma yok.
    placements = [
        (5.0,  0.2),
        (7.0, -0.2),
        (9.0,  0.4),
        (11.0, -0.4),
    ]

    spawned = []

    for x, y in placements:
        bp = random.choice(walker_bps)

        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "adas_dense_pedestrian")

        tf = relative_transform(ego, x=x, y=y, z=1.0, yaw_offset=0.0)

        p = world.try_spawn_actor(bp, tf)
        if p is None:
            print(f"Yaya spawn edilemedi: x={x}, y={y}")
            continue

        actor_ids.append(p.id)
        spawned.append(p)

        side = "sağ" if y > 0 else "sol"
        print(f"Kamera içi yaya: id={p.id}, side={side}, x={x}, y={y}")

    print(f"Toplam kamera içi yaya: {len(spawned)} / 4")
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
    spawn_dense_vehicles(world, ego, actor_ids)
    # spawn_dense_pedestrians(world, ego, actor_ids)

    save_ids(actor_ids)
    tick(world, 10)
    print_status(world)

    print("")
    print("YOĞUN SABİT ALGILAMA SAHNESİ HAZIR")
    print("Ego sabit.")
    print("4 araç kamera içine yakın yerleştirildi. Yayalar kapalı.")
    print("Amaç: YOLO vehicle/person algısını net test etmek.")
    print("")

    while True:
        follow_spectator(world, ego)
        time.sleep(0.05)


if __name__ == "__main__":
    main()

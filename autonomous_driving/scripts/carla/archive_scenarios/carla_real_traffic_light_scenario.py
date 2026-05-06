#!/usr/bin/env python3

import argparse
import json
import math
import random
import time
from pathlib import Path

import carla


ACTOR_FILE = Path("/tmp/carla_real_traffic_light_actor_ids.json")


def load_ids():
    if not ACTOR_FILE.exists():
        return []
    try:
        return json.loads(ACTOR_FILE.read_text())
    except Exception:
        return []


def save_ids(ids):
    ACTOR_FILE.write_text(json.dumps(list(dict.fromkeys(ids)), indent=2))


def connect():
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    return client, world


def tick(world, n=5):
    for _ in range(n):
        try:
            world.tick()
        except Exception:
            world.wait_for_tick(seconds=2.0)
        time.sleep(0.05)


def destroy_old(world):
    ids = load_ids()
    destroyed = 0

    for actor_id in ids:
        actor = world.get_actor(actor_id)
        if actor is not None:
            try:
                actor.destroy()
                destroyed += 1
            except Exception:
                pass

    for actor in world.get_actors():
        role = actor.attributes.get("role_name", "")
        if role in ["ego_vehicle", "rgb_front", "adas_test_front_vehicle"]:
            try:
                actor.destroy()
                destroyed += 1
            except Exception:
                pass

    save_ids([])
    print(f"Temizlenen actor sayısı: {destroyed}")


def vec_dot(ax, ay, bx, by):
    return ax * bx + ay * by


def norm(x, y):
    return math.sqrt(x * x + y * y)


def find_best_light_and_spawn(world):
    lights = list(world.get_actors().filter("traffic.traffic_light*"))

    if not lights:
        raise RuntimeError("Bu haritada traffic light actor bulunamadı.")

    carla_map = world.get_map()
    waypoints = carla_map.generate_waypoints(2.0)

    candidates = []

    for light in lights:
        l_loc = light.get_location()

        for wp in waypoints:
            w_loc = wp.transform.location
            dist = w_loc.distance(l_loc)

            if dist < 6.0 or dist > 18.0:
                continue

            fwd = wp.transform.get_forward_vector()
            tx = l_loc.x - w_loc.x
            ty = l_loc.y - w_loc.y
            tnorm = norm(tx, ty)

            if tnorm < 0.001:
                continue

            dot = vec_dot(fwd.x, fwd.y, tx / tnorm, ty / tnorm)

            if dot < 0.30:
                continue

            score = abs(dist - 10.0) - dot * 8.0
            candidates.append((score, light, wp, dist, dot))

    if not candidates:
        raise RuntimeError("Trafik ışığına bakan uygun waypoint bulunamadı.")

    candidates.sort(key=lambda x: x[0])
    return candidates[0]


def set_near_lights(world, target_light):
    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    target_loc = target_light.get_location()

    near = sorted(
        lights,
        key=lambda l: l.get_location().distance(target_loc)
    )[:3]

    states = [
        ("red", carla.TrafficLightState.Red),
        ("yellow", carla.TrafficLightState.Yellow),
        ("green", carla.TrafficLightState.Green),
    ]

    for i, light in enumerate(near):
        name, state = states[i % len(states)]
        try:
            light.set_state(state)
            light.freeze(True)
            print(f"Işık ayarlandı: id={light.id}, state={name}, loc={light.get_location()}")
        except Exception as exc:
            print(f"Işık ayarlanamadı: id={light.id}, hata={exc}")

    return near


def spawn_ego_and_camera(world, spawn_wp, actor_ids):
    bp_lib = world.get_blueprint_library()

    ego_bp = bp_lib.find("vehicle.tesla.model3")
    ego_bp.set_attribute("role_name", "ego_vehicle")

    if ego_bp.has_attribute("color"):
        colors = ego_bp.get_attribute("color").recommended_values
        if colors:
            ego_bp.set_attribute("color", random.choice(colors))

    ego_tf = spawn_wp.transform
    ego_tf.location.z += 0.5

    ego = None

    for offset in [0.0, -2.0, 2.0, -4.0, 4.0]:
        tf = carla.Transform(
            carla.Location(
                x=ego_tf.location.x,
                y=ego_tf.location.y + offset,
                z=ego_tf.location.z
            ),
            ego_tf.rotation
        )

        ego = world.try_spawn_actor(ego_bp, tf)

        if ego is not None:
            break

    if ego is None:
        raise RuntimeError("Ego araç spawn edilemedi.")

    ego.set_autopilot(False)
    actor_ids.append(ego.id)

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("role_name", "rgb_front")
    cam_bp.set_attribute("image_size_x", "1280")
    cam_bp.set_attribute("image_size_y", "720")
    cam_bp.set_attribute("fov", "55")
    cam_bp.set_attribute("sensor_tick", "0.15")

    cam_tf = carla.Transform(
        carla.Location(x=1.80, y=0.0, z=1.55),
        carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0),
    )

    cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
    actor_ids.append(cam.id)

    print(f"Ego spawn edildi: id={ego.id}")
    print(f"RGB kamera spawn edildi: id={cam.id}, role_name=rgb_front")

    return ego, cam


def set_spectator(world, ego):
    tf = ego.get_transform()
    yaw = math.radians(tf.rotation.yaw)

    spec_tf = carla.Transform(
        carla.Location(
            x=tf.location.x - 8.0 * math.cos(yaw),
            y=tf.location.y - 8.0 * math.sin(yaw),
            z=tf.location.z + 4.0,
        ),
        carla.Rotation(
            pitch=-18.0,
            yaw=tf.rotation.yaw,
            roll=0.0,
        )
    )

    world.get_spectator().set_transform(spec_tf)


def print_status(world):
    vehicles = world.get_actors().filter("vehicle.*")
    cameras = world.get_actors().filter("sensor.camera.rgb")
    lights = world.get_actors().filter("traffic.traffic_light*")

    print(f"Vehicle count: {len(vehicles)}")
    for v in vehicles:
        print(f"  vehicle id={v.id}, role_name={v.attributes.get('role_name')}, type={v.type_id}")

    print(f"Camera count: {len(cameras)}")
    for c in cameras:
        parent = getattr(c, "parent", None)
        pid = parent.id if parent is not None else None
        print(f"  camera id={c.id}, role_name={c.attributes.get('role_name')}, parent={pid}")

    print(f"Traffic light count: {len(lights)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    _, world = connect()

    if args.clear:
        destroy_old(world)
        tick(world, 5)
        print_status(world)
        return

    if args.status:
        print_status(world)
        return

    destroy_old(world)
    tick(world, 5)

    score, target_light, spawn_wp, dist, dot = find_best_light_and_spawn(world)

    print(f"Seçilen trafik ışığı: id={target_light.id}, dist={dist:.2f}, dot={dot:.2f}, loc={target_light.get_location()}")

    near_lights = set_near_lights(world, target_light)

    actor_ids = []
    ego, cam = spawn_ego_and_camera(world, spawn_wp, actor_ids)
    save_ids(actor_ids)

    tick(world, 10)
    set_spectator(world, ego)

    print("Gerçek trafik ışığı senaryosu hazır.")
    print_status(world)


if __name__ == "__main__":
    main()

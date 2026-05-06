#!/usr/bin/env python3
import argparse
import math
import random
import time
import carla

ROLE_PREFIX = "adas_extra_visible_ped"


def destroy_old(world):
    targets = []
    for actor in world.get_actors():
        role = actor.attributes.get("role_name", "")
        if role.startswith(ROLE_PREFIX):
            targets.append(actor)

    for actor in targets:
        try:
            actor.destroy()
        except Exception:
            pass

    print(f"[CLEAR] old visible pedestrians destroyed: {len(targets)}")


def find_ego(world):
    vehicles = world.get_actors().filter("vehicle.*")

    for v in vehicles:
        role = v.attributes.get("role_name", "")
        if "adas_big_showcase_ego" in role:
            return v

    for v in vehicles:
        role = v.attributes.get("role_name", "")
        if "ego" in role.lower():
            return v

    if len(vehicles) > 0:
        return vehicles[0]

    return None


def local_to_world(base_tf, x, y, z=0.0):
    yaw = math.radians(base_tf.rotation.yaw)

    bx = base_tf.location.x
    by = base_tf.location.y
    bz = base_tf.location.z

    wx = bx + x * math.cos(yaw) - y * math.sin(yaw)
    wy = by + x * math.sin(yaw) + y * math.cos(yaw)
    wz = bz + z

    return carla.Location(wx, wy, wz)


def spawn_walker(world, bp_lib, ego_tf, idx, x, y, yaw_delta):
    walkers = list(bp_lib.filter("walker.pedestrian.*"))

    if not walkers:
        print("[ERROR] pedestrian blueprint yok")
        return None

    bp = random.choice(walkers)
    bp.set_attribute("role_name", f"{ROLE_PREFIX}_{idx}")

    loc = local_to_world(ego_tf, x, y, 0.85)
    yaw = ego_tf.rotation.yaw + yaw_delta

    tf = carla.Transform(
        loc,
        carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0)
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

        print(f"[PED] id={actor.id} x={x} y={y} yaw_delta={yaw_delta}")

    return actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()

    random.seed(7)

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    world = client.get_world()

    destroy_old(world)

    if args.clear:
        return

    ego = find_ego(world)

    if ego is None:
        raise RuntimeError("Ego vehicle bulunamadı. Önce carla_big_detection_showcase.py çalışmalı.")

    ego_tf = ego.get_transform()
    bp_lib = world.get_blueprint_library()

    layout = [
        # x ileri mesafe, y sağ/sol kayma, yaw_delta
        # Kamerada net ve dik görünmeleri için ego'ya doğru baktırıyoruz.
        (11.5,  2.7, 180),
        (15.5,  3.2, 180),
        (20.0, -2.8, 180),
        (24.0,  3.8, 180),
    ]

    spawned = 0

    for idx, (x, y, yaw_delta) in enumerate(layout[:args.count]):
        actor = spawn_walker(world, bp_lib, ego_tf, idx, x, y, yaw_delta)
        if actor:
            spawned += 1

    print("")
    print("========== VISIBLE PEDESTRIANS READY ==========")
    print(f"Spawned visible pedestrians: {spawned}")
    print("Mevcut kamera/publisher aynı kalabilir.")
    print("===============================================")
    print("")


if __name__ == "__main__":
    main()

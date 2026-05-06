#!/usr/bin/env python3
import math
import random
import carla

ROLE_PREFIX = "adas_force_visible_ped"

def local_to_world(base_tf, x, y, z=0.0):
    yaw = math.radians(base_tf.rotation.yaw)
    bx, by, bz = base_tf.location.x, base_tf.location.y, base_tf.location.z
    wx = bx + x * math.cos(yaw) - y * math.sin(yaw)
    wy = by + x * math.sin(yaw) + y * math.cos(yaw)
    return carla.Location(wx, wy, bz + z)

def clear_old(world):
    killed = 0
    for a in world.get_actors():
        role = a.attributes.get("role_name", "")
        if role.startswith(ROLE_PREFIX):
            try:
                a.destroy()
                killed += 1
            except Exception:
                pass
    print(f"[CLEAR] old forced pedestrians: {killed}")

def find_ego(world):
    vehicles = world.get_actors().filter("vehicle.*")

    for v in vehicles:
        role = v.attributes.get("role_name", "")
        if "adas_clear_straight_ego" in role:
            return v
        if "adas_big_showcase_ego" in role:
            return v
        if "ego" in role.lower():
            return v

    if len(vehicles) > 0:
        return vehicles[0]

    return None

def spawn_ped(world, bp_lib, ego_tf, idx, x, y, yaw_delta=180.0):
    walkers = list(bp_lib.filter("walker.pedestrian.*"))
    if not walkers:
        print("[ERROR] walker blueprint yok")
        return None

    bp = random.choice(walkers)
    bp.set_attribute("role_name", f"{ROLE_PREFIX}_{idx}")

    loc = local_to_world(ego_tf, x, y, 0.85)
    tf = carla.Transform(
        loc,
        carla.Rotation(
            pitch=0.0,
            yaw=ego_tf.rotation.yaw + yaw_delta,
            roll=0.0,
        ),
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
        print(f"[PED] id={actor.id} x={x} y={y}")

    return actor

def main():
    random.seed(44)

    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()

    clear_old(world)

    ego = find_ego(world)
    if ego is None:
        raise RuntimeError("Ego bulunamadı. Önce sahne scriptini çalıştır.")

    ego_tf = ego.get_transform()
    bp_lib = world.get_blueprint_library()

    # Kamerada net görünsün diye yolun sağ/sol önüne yakın koyuyoruz.
    layout = [
        (10.0,  2.2, 180.0),
        (13.0, -2.2, 180.0),
        (17.0,  2.7, 180.0),
    ]

    count = 0
    for i, (x, y, yaw) in enumerate(layout):
        if spawn_ped(world, bp_lib, ego_tf, i, x, y, yaw):
            count += 1

    print("")
    print("========== FORCED VISIBLE PEDESTRIANS READY ==========")
    print(f"Spawned pedestrians: {count}")
    print("======================================================")
    print("")

if __name__ == "__main__":
    main()

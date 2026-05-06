#!/usr/bin/env python3
import argparse
import random
import time
import carla


def get_ego(world):
    actors = world.get_actors()

    vehicles = [a for a in actors if a.type_id.startswith("vehicle.")]

    for a in vehicles:
        role = a.attributes.get("role_name", "")
        if role in ["ego", "hero", "rgb_front_ego", "player"]:
            return a

    if vehicles:
        # En büyük araç/ilk spawn edilen araç ego kabul edilir.
        return sorted(vehicles, key=lambda x: x.id)[0]

    return None


def destroy_old(world):
    for a in world.get_actors():
        if a.type_id.startswith("walker.") or a.type_id.startswith("controller.ai.walker"):
            try:
                a.destroy()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--distance", type=float, default=18.0)
    parser.add_argument("--side-step", type=float, default=2.2)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()

    if args.clear:
        destroy_old(world)
        time.sleep(0.5)

    ego = get_ego(world)
    if ego is None:
        raise RuntimeError("Ego araç bulunamadı. Önce carla_dense_static_detection_scene.py çalışmalı.")

    bp_lib = world.get_blueprint_library()
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))

    if not walker_bps:
        raise RuntimeError("walker.pedestrian blueprint bulunamadı.")

    ego_tf = ego.get_transform()
    forward = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()
    base = ego_tf.location

    spawn_points = []

    # Kameranın tam önünde ve sağ/sol şeritlerde görünür olacak şekilde.
    layout = [
        (-1.5, args.distance),
        (0.0, args.distance + 5.0),
        (1.5, args.distance + 10.0),
        (-2.8, args.distance + 15.0),
        (2.8, args.distance + 20.0),
    ]

    for side, dist in layout[:args.count]:
        loc = base + forward * dist + right * (side * args.side_step)
        loc.z += 0.5

        # Zemine oturt
        wp = world.get_map().get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Any
        )
        if wp is not None:
            loc = wp.transform.location + right * (side * args.side_step)
            loc.z += 0.3

        yaw = ego_tf.rotation.yaw + 180.0
        spawn_points.append(carla.Transform(loc, carla.Rotation(yaw=yaw)))

    spawned = []

    for i, sp in enumerate(spawn_points):
        bp = random.choice(walker_bps)

        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        if bp.has_attribute("speed"):
            bp.set_attribute("speed", "0.0")

        actor = world.try_spawn_actor(bp, sp)

        if actor is None:
            print(f"[WARN] pedestrian spawn olmadı: {i} {sp.location}")
            continue

        spawned.append(actor)
        print(f"[OK] pedestrian id={actor.id} loc={sp.location}")

    world.tick() if world.get_settings().synchronous_mode else time.sleep(0.5)

    print(f"DONE spawned={len(spawned)}")


if __name__ == "__main__":
    main()

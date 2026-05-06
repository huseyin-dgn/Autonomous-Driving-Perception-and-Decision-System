#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path

import carla


ACTOR_FILE = Path("/tmp/carla_extra_person_ids.json")


def load_ids():
    if not ACTOR_FILE.exists():
        return []
    try:
        return json.loads(ACTOR_FILE.read_text())
    except Exception:
        return []


def save_ids(ids):
    ACTOR_FILE.write_text(json.dumps(list(dict.fromkeys(ids)), indent=2))


def find_ego(world):
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.attributes.get("role_name") == "ego_vehicle":
            return actor
    return None


def clear(world):
    destroyed = 0
    for actor_id in load_ids():
        actor = world.get_actor(actor_id)
        if actor is not None:
            try:
                actor.destroy()
                destroyed += 1
            except Exception:
                pass

    save_ids([])
    print(f"Temizlenen ekstra insan sayısı: {destroyed}")


def spawn_person(world, distance, side):
    ego = find_ego(world)

    if ego is None:
        raise RuntimeError("ego_vehicle bulunamadı.")

    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter("walker.pedestrian.*")

    if not walker_bps:
        raise RuntimeError("walker.pedestrian blueprint bulunamadı.")

    bp = random.choice(walker_bps)

    if bp.has_attribute("is_invincible"):
        bp.set_attribute("is_invincible", "false")

    ego_tf = ego.get_transform()
    fwd = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()

    offsets = [side, side - 0.5, side + 0.5, side - 1.0, side + 1.0]
    distances = [distance, distance + 1.0, max(5.0, distance - 1.0)]

    for dist in distances:
        for off in offsets:
            loc = ego_tf.location + fwd * float(dist) + right * float(off)
            loc.z += 1.0

            tf = carla.Transform(
                loc,
                carla.Rotation(
                    pitch=0.0,
                    yaw=ego_tf.rotation.yaw + 180.0,
                    roll=0.0,
                ),
            )

            actor = world.try_spawn_actor(bp, tf)

            if actor is not None:
                ids = load_ids()
                ids.append(actor.id)
                save_ids(ids)

                print(f"Ekstra insan spawn edildi: id={actor.id}, distance={dist}, side={off}")
                return

    raise RuntimeError("Ekstra insan spawn edilemedi. side/distance değiştir.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--distance", type=float, default=9.0)
    parser.add_argument("--side", type=float, default=-2.8)
    args = parser.parse_args()

    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()

    if args.clear:
        clear(world)
        return

    spawn_person(world, args.distance, args.side)


if __name__ == "__main__":
    main()

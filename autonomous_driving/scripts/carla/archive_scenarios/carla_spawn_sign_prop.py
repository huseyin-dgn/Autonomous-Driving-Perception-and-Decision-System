#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import carla


ACTOR_FILE = Path("/tmp/carla_sign_prop_actor_ids.json")


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
    client.set_timeout(20.0)
    world = client.get_world()
    return client, world


def find_ego(world):
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.attributes.get("role_name") == "ego_vehicle":
            return actor
    return None


def clear_props(world):
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
    print(f"Temizlenen sign prop sayısı: {destroyed}")


def spawn_prop(world, bp_id, distance, side, z, yaw_offset):
    ego = find_ego(world)

    if ego is None:
        raise RuntimeError("ego_vehicle bulunamadı. Önce CARLA senaryosunu kur.")

    bp_lib = world.get_blueprint_library()
    bp = bp_lib.find(bp_id)

    ego_tf = ego.get_transform()
    fwd = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()

    loc = ego_tf.location + fwd * float(distance) + right * float(side)
    loc.z += float(z)

    tf = carla.Transform(
        loc,
        carla.Rotation(
            pitch=0.0,
            yaw=ego_tf.rotation.yaw + float(yaw_offset),
            roll=0.0,
        )
    )

    actor = world.try_spawn_actor(bp, tf)

    if actor is None:
        raise RuntimeError("Prop spawn edilemedi. Mesafe/side/z değiştir.")

    ids = load_ids()
    ids.append(actor.id)
    save_ids(ids)

    print(f"Prop spawn edildi: id={actor.id}, bp={bp_id}")
    print(f"distance={distance}, side={side}, z={z}, yaw_offset={yaw_offset}")


def status(world):
    print("Kayıtlı prop actorları:")
    for actor_id in load_ids():
        actor = world.get_actor(actor_id)
        if actor is not None:
            print(f"id={actor.id}, type={actor.type_id}, loc={actor.get_location()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--bp", default="static.prop.trafficwarning")
    parser.add_argument("--distance", type=float, default=10.0)
    parser.add_argument("--side", type=float, default=2.5)
    parser.add_argument("--z", type=float, default=0.0)
    parser.add_argument("--yaw-offset", type=float, default=180.0)
    args = parser.parse_args()

    _, world = connect()

    if args.clear:
        clear_props(world)
        return

    if args.status:
        status(world)
        return

    spawn_prop(
        world=world,
        bp_id=args.bp,
        distance=args.distance,
        side=args.side,
        z=args.z,
        yaw_offset=args.yaw_offset,
    )


if __name__ == "__main__":
    main()

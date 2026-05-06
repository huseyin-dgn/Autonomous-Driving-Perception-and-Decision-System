import argparse
import json
import random
import time
from pathlib import Path

import carla


ACTOR_FILE = Path("/tmp/carla_adas_test_actor_ids.json")


def load_actor_ids():
    if not ACTOR_FILE.exists():
        return []

    try:
        return json.loads(ACTOR_FILE.read_text())
    except Exception:
        return []


def save_actor_ids(ids):
    ACTOR_FILE.write_text(json.dumps(ids, indent=2))


def get_client():
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    return client


def find_ego_vehicle(world):
    vehicles = world.get_actors().filter("vehicle.*")

    for actor in vehicles:
        if actor.attributes.get("role_name") == "ego_vehicle":
            return actor

    for actor in vehicles:
        role_name = actor.attributes.get("role_name", "")
        if "ego" in role_name:
            return actor

    return None


def destroy_previous_test_actors(world):
    ids = load_actor_ids()
    destroyed = 0

    for actor_id in ids:
        actor = world.get_actor(actor_id)

        if actor is not None:
            actor.destroy()
            destroyed += 1

    save_actor_ids([])
    print(f"Temizlenen test actor sayısı: {destroyed}")


def choose_blueprint(bp_lib, patterns):
    for pattern in patterns:
        matches = bp_lib.filter(pattern)

        if len(matches) > 0:
            return matches[0]

    return None


def spawn_front_vehicle(world, ego, distance):
    bp_lib = world.get_blueprint_library()

    vehicle_bp = choose_blueprint(
        bp_lib,
        [
            "vehicle.tesla.model3",
            "vehicle.audi.tt",
            "vehicle.lincoln.mkz_2020",
            "vehicle.toyota.prius",
        ],
    )

    if vehicle_bp is None:
        vehicles = bp_lib.filter("vehicle.*")

        if len(vehicles) == 0:
            print("Vehicle blueprint bulunamadı.")
            return None

        vehicle_bp = random.choice(vehicles)

    if vehicle_bp.has_attribute("role_name"):
        vehicle_bp.set_attribute("role_name", "adas_test_front_vehicle")

    if vehicle_bp.has_attribute("color"):
        colors = vehicle_bp.get_attribute("color").recommended_values

        if len(colors) > 0:
            vehicle_bp.set_attribute("color", random.choice(colors))

    ego_wp = world.get_map().get_waypoint(
        ego.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    candidate_distances = [
        distance,
        distance + 4.0,
        distance + 8.0,
        max(8.0, distance - 4.0),
    ]

    for dist in candidate_distances:
        next_wps = ego_wp.next(dist)

        if not next_wps:
            continue

        spawn_tf = next_wps[0].transform
        spawn_tf.location.z += 0.5

        actor = world.try_spawn_actor(vehicle_bp, spawn_tf)

        if actor is not None:
            actor.set_autopilot(False)
            actor.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    hand_brake=True,
                )
            )

            print(f"Öne araç spawn edildi: id={actor.id}, type={actor.type_id}, distance={dist}")
            return actor

    print("Öne araç spawn edilemedi. Konum dolu olabilir.")
    return None


def spawn_pedestrian(world, ego, distance, lateral_offset):
    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter("walker.pedestrian.*")

    if len(walker_bps) == 0:
        print("Walker blueprint bulunamadı.")
        return None

    walker_bp = random.choice(walker_bps)

    if walker_bp.has_attribute("is_invincible"):
        walker_bp.set_attribute("is_invincible", "false")

    ego_tf = ego.get_transform()
    fwd = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()

    location = ego_tf.location + fwd * distance + right * lateral_offset
    location.z += 1.0

    spawn_tf = carla.Transform(
        location,
        carla.Rotation(
            pitch=0.0,
            yaw=ego_tf.rotation.yaw + 180.0,
            roll=0.0,
        ),
    )

    actor = world.try_spawn_actor(walker_bp, spawn_tf)

    if actor is not None:
        print(
            f"İnsan spawn edildi: id={actor.id}, type={actor.type_id}, "
            f"distance={distance}, lateral_offset={lateral_offset}"
        )
        return actor

    alternative_offsets = [
        lateral_offset + 1.5,
        lateral_offset - 1.5,
        0.0,
        2.5,
        -2.5,
        4.0,
        -4.0,
    ]

    for offset in alternative_offsets:
        location = ego_tf.location + fwd * distance + right * offset
        location.z += 1.0

        spawn_tf = carla.Transform(
            location,
            carla.Rotation(
                pitch=0.0,
                yaw=ego_tf.rotation.yaw + 180.0,
                roll=0.0,
            ),
        )

        actor = world.try_spawn_actor(walker_bp, spawn_tf)

        if actor is not None:
            print(
                f"İnsan spawn edildi: id={actor.id}, type={actor.type_id}, "
                f"distance={distance}, lateral_offset={offset}"
            )
            return actor

    print("İnsan spawn edilemedi.")
    return None


def tick_world(world):
    try:
        world.tick()
    except Exception:
        try:
            world.wait_for_tick(seconds=2.0)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--front-distance", type=float, default=16.0)
    parser.add_argument("--ped-distance", type=float, default=10.0)
    parser.add_argument("--ped-offset", type=float, default=1.5)
    parser.add_argument("--only-vehicle", action="store_true")
    parser.add_argument("--only-person", action="store_true")
    args = parser.parse_args()

    client = get_client()
    world = client.get_world()

    if args.clear:
        destroy_previous_test_actors(world)
        tick_world(world)
        return

    ego = find_ego_vehicle(world)

    if ego is None:
        print("Ego vehicle bulunamadı. Önce ego bridge launch çalışmalı.")
        print("Mevcut araçlar:")

        for actor in world.get_actors().filter("vehicle.*"):
            print(actor.id, actor.type_id, actor.attributes.get("role_name"))

        return

    print(f"Ego bulundu: id={ego.id}, type={ego.type_id}")
    print(f"Ego location: {ego.get_location()}")

    destroy_previous_test_actors(world)

    spawned_ids = []

    if not args.only_person:
        front_vehicle = spawn_front_vehicle(world, ego, args.front_distance)

        if front_vehicle is not None:
            spawned_ids.append(front_vehicle.id)

    if not args.only_vehicle:
        pedestrian = spawn_pedestrian(world, ego, args.ped_distance, args.ped_offset)

        if pedestrian is not None:
            spawned_ids.append(pedestrian.id)

    save_actor_ids(spawned_ids)

    for _ in range(10):
        tick_world(world)
        time.sleep(0.1)

    print("Senaryo hazır.")
    print("Spawn edilen actor idleri:", spawned_ids)


if __name__ == "__main__":
    main()

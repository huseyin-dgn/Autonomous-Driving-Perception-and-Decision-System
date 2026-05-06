#!/usr/bin/env python3
import argparse
import random
import time
import math
import carla


PED_ROLE_NAME = "adas_test_pedestrian"


def wait_for_world(client, timeout=30.0):
    client.set_timeout(timeout)
    world = client.get_world()
    return world


def find_ego_vehicle(world):
    actors = world.get_actors()

    cameras = actors.filter("sensor.camera.rgb")
    for cam in cameras:
        role = cam.attributes.get("role_name", "")
        parent = cam.parent
        if role in ["rgb_front", "front", "adas_front_camera"] and parent is not None:
            print(f"[OK] Kamera parent ego bulundu: actor_id={parent.id}, type={parent.type_id}")
            return parent

    vehicles = actors.filter("vehicle.*")

    for vehicle in vehicles:
        role = vehicle.attributes.get("role_name", "")
        if role in ["ego", "hero", "adas_ego"]:
            print(f"[OK] role_name ile ego bulundu: actor_id={vehicle.id}, type={vehicle.type_id}")
            return vehicle

    if len(vehicles) > 0:
        vehicle = vehicles[0]
        print(f"[WARN] Ego role bulunamadı. İlk vehicle ego kabul edildi: actor_id={vehicle.id}, type={vehicle.type_id}")
        return vehicle

    return None


def destroy_old_pedestrians(world):
    destroyed = 0

    for actor in world.get_actors().filter("walker.pedestrian.*"):
        role = actor.attributes.get("role_name", "")
        if role == PED_ROLE_NAME:
            actor.destroy()
            destroyed += 1

    print(f"[OK] Eski ADAS pedestrian silindi: {destroyed}")


def get_vectors(transform):
    yaw = math.radians(transform.rotation.yaw)

    forward = carla.Vector3D(
        x=math.cos(yaw),
        y=math.sin(yaw),
        z=0.0,
    )

    right = carla.Vector3D(
        x=math.cos(yaw + math.pi / 2.0),
        y=math.sin(yaw + math.pi / 2.0),
        z=0.0,
    )

    return forward, right


def make_relative_location(base_location, forward, right, forward_dist, lateral_dist, z_offset=0.8):
    return carla.Location(
        x=base_location.x + forward.x * forward_dist + right.x * lateral_dist,
        y=base_location.y + forward.y * forward_dist + right.y * lateral_dist,
        z=base_location.z + z_offset,
    )


def get_grounded_transform(world, location, yaw):
    carla_map = world.get_map()

    waypoint = carla_map.get_waypoint(
        location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if waypoint is not None:
        loc = waypoint.transform.location
        loc.z += 1.0
        return carla.Transform(
            loc,
            carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0),
        )

    location.z += 1.0
    return carla.Transform(
        location,
        carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0),
    )


def choose_walker_blueprints(world, count):
    library = world.get_blueprint_library()
    walkers = list(library.filter("walker.pedestrian.*"))

    if not walkers:
        raise RuntimeError("walker.pedestrian.* blueprint bulunamadı")

    random.shuffle(walkers)

    selected = []
    for i in range(count):
        bp = walkers[i % len(walkers)]

        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", PED_ROLE_NAME)

        selected.append(bp)

    return selected


def spawn_pedestrians(world, ego, count):
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    ego_yaw = ego_tf.rotation.yaw

    forward, right = get_vectors(ego_tf)

    spawn_layout = [
        # Kamera görüntüsünde yolun ortasına yakın, net görünmesi için.
        (12.0, 0.0),
        (16.0, -2.2),
        (18.0, 2.2),
        (24.0, 0.8),
        (28.0, -1.4),
        (32.0, 1.6),
    ]

    blueprints = choose_walker_blueprints(world, count)

    spawned = []

    for i in range(count):
        forward_dist, lateral_dist = spawn_layout[i % len(spawn_layout)]

        raw_loc = make_relative_location(
            ego_loc,
            forward,
            right,
            forward_dist,
            lateral_dist,
            z_offset=0.5,
        )

        # İnsanlar kameraya doğru baksın.
        ped_yaw = ego_yaw + 180.0

        tf = get_grounded_transform(world, raw_loc, ped_yaw)

        bp = blueprints[i]

        actor = world.try_spawn_actor(bp, tf)

        if actor is None:
            print(f"[WARN] Pedestrian spawn başarısız: index={i}, forward={forward_dist}, lateral={lateral_dist}")
            continue

        actor.set_simulate_physics(True)

        spawned.append(actor)

        print(
            f"[OK] Pedestrian eklendi: "
            f"id={actor.id}, type={actor.type_id}, "
            f"x={tf.location.x:.2f}, y={tf.location.y:.2f}, z={tf.location.z:.2f}"
        )

    return spawned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    world = wait_for_world(client)

    print("[OK] CARLA bağlantısı var")
    print("[INFO] Map:", world.get_map().name)
    print("[INFO] Actors:", len(world.get_actors()))

    if args.clear:
        destroy_old_pedestrians(world)
        time.sleep(0.5)

    ego = find_ego_vehicle(world)

    if ego is None:
        raise RuntimeError("Ego araç bulunamadı. Önce carla_dense_static_detection_scene.py çalıştır.")

    print("[INFO] Ego:", ego.id, ego.type_id, ego.get_transform())

    spawned = spawn_pedestrians(world, ego, args.count)

    print("==========================================")
    print(f"[DONE] Eklenen pedestrian sayısı: {len(spawned)}")
    print("Perception tarafında beklenen:")
    print("original_label=pedestrian")
    print("label=person")
    print("==========================================")


if __name__ == "__main__":
    main()

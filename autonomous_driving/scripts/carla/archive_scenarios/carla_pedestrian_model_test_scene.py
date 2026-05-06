#!/usr/bin/env python3
import argparse
import math
import time
import carla


ROLE_NAME = "adas_pedestrian_test"


def clear_old_test_actors(world):
    removed = 0

    for actor in world.get_actors():
        role = actor.attributes.get("role_name", "")

        if role == ROLE_NAME:
            actor.destroy()
            removed += 1

    print(f"[OK] Eski test pedestrian aktörleri silindi: {removed}")


def find_ego(world):
    cameras = world.get_actors().filter("sensor.camera.rgb")

    for cam in cameras:
        role = cam.attributes.get("role_name", "")
        if role == "rgb_front" and cam.parent is not None:
            print(f"[OK] rgb_front parent ego bulundu: {cam.parent.id}")
            return cam.parent

    vehicles = world.get_actors().filter("vehicle.*")

    for vehicle in vehicles:
        role = vehicle.attributes.get("role_name", "")
        if role in ["ego", "hero", "adas_ego"]:
            print(f"[OK] ego role ile bulundu: {vehicle.id}")
            return vehicle

    if len(vehicles) > 0:
        print(f"[WARN] Ego role yok. İlk araç ego kabul edildi: {vehicles[0].id}")
        return vehicles[0]

    raise RuntimeError("Ego araç bulunamadı. Önce ana CARLA sahnesini başlat.")


def get_forward_right(transform):
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


def relative_location(base, forward, right, fwd, lat, z=1.0):
    return carla.Location(
        x=base.x + forward.x * fwd + right.x * lat,
        y=base.y + forward.y * fwd + right.y * lat,
        z=base.z + z,
    )


def get_walker_blueprints(world):
    bp_lib = world.get_blueprint_library()
    walkers = list(bp_lib.filter("walker.pedestrian.*"))

    if not walkers:
        raise RuntimeError("walker.pedestrian.* blueprint bulunamadı")

    preferred_ids = [
        "walker.pedestrian.0001",
        "walker.pedestrian.0002",
        "walker.pedestrian.0003",
        "walker.pedestrian.0004",
        "walker.pedestrian.0005",
        "walker.pedestrian.0006",
        "walker.pedestrian.0007",
        "walker.pedestrian.0008",
        "walker.pedestrian.0009",
        "walker.pedestrian.0010",
    ]

    preferred = [bp for bp in walkers if bp.id in preferred_ids]

    if not preferred:
        preferred = walkers

    return preferred


def spawn_walkers(world, ego, count):
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    ego_yaw = ego_tf.rotation.yaw

    forward, right = get_forward_right(ego_tf)

    layouts = [
        (8.0, 0.0),
        (11.0, -1.2),
        (11.0, 1.2),
        (15.0, 0.0),
        (18.0, -1.8),
        (18.0, 1.8),
    ]

    blueprints = get_walker_blueprints(world)
    spawned = []

    for i in range(count):
        fwd, lat = layouts[i % len(layouts)]

        loc = relative_location(
            ego_loc,
            forward,
            right,
            fwd,
            lat,
            z=1.0,
        )

        # Yol yüzeyine projekte et.
        waypoint = world.get_map().get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if waypoint is not None:
            loc = waypoint.transform.location
            loc.z += 1.0

        # Kameraya doğru baksın.
        rot = carla.Rotation(
            pitch=0.0,
            yaw=ego_yaw + 180.0,
            roll=0.0,
        )

        tf = carla.Transform(loc, rot)

        bp = blueprints[i % len(blueprints)]

        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", ROLE_NAME)

        actor = world.try_spawn_actor(bp, tf)

        if actor is None:
            print(f"[WARN] Spawn başarısız: index={i}, fwd={fwd}, lat={lat}")
            continue

        actor.set_simulate_physics(True)
        spawned.append(actor)

        print(
            f"[OK] Pedestrian eklendi: id={actor.id}, "
            f"type={actor.type_id}, fwd={fwd}, lat={lat}, "
            f"x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f}"
        )

    return spawned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    world = client.get_world()

    print("[OK] CARLA bağlantısı var")
    print("[INFO] Map:", world.get_map().name)
    print("[INFO] Actor count:", len(world.get_actors()))

    if args.clear:
        clear_old_test_actors(world)
        time.sleep(0.5)

    ego = find_ego(world)

    print("[INFO] Ego:", ego.id, ego.type_id, ego.get_transform())

    spawned = spawn_walkers(world, ego, args.count)

    print("==========================================")
    print(f"[DONE] Eklenen pedestrian sayısı: {len(spawned)}")
    print("Beklenen YOLO sonucu:")
    print("original_label=pedestrian")
    print("label=person")
    print("Eğer yine motorcycle çıkarsa model pedestrian sınıfını bu CARLA görüntüsünde yanlış öğrenmiş demektir.")
    print("==========================================")


if __name__ == "__main__":
    main()

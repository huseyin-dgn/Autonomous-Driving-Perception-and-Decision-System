#!/usr/bin/env python3
import argparse
import math
import time
import carla


ROLE_NAME = "adas_single_pedestrian_test"


def clear_old(world):
    removed = 0

    for actor in world.get_actors().filter("walker.pedestrian.*"):
        role = actor.attributes.get("role_name", "")
        if role == ROLE_NAME:
            actor.destroy()
            removed += 1

    print(f"[OK] Eski test yayaları silindi: {removed}")


def find_ego(world):
    cameras = world.get_actors().filter("sensor.camera.rgb")

    for cam in cameras:
        role = cam.attributes.get("role_name", "")
        if role == "rgb_front" and cam.parent is not None:
            print(f"[OK] Ego rgb_front parent üzerinden bulundu: {cam.parent.id}")
            return cam.parent

    vehicles = world.get_actors().filter("vehicle.*")

    for vehicle in vehicles:
        role = vehicle.attributes.get("role_name", "")
        if role in ["ego", "hero", "adas_ego"]:
            print(f"[OK] Ego role üzerinden bulundu: {vehicle.id}")
            return vehicle

    if len(vehicles) > 0:
        print(f"[WARN] Ego role yok. İlk araç ego kabul edildi: {vehicles[0].id}")
        return vehicles[0]

    raise RuntimeError("Ego araç bulunamadı.")


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


def rel_loc(base, forward, right, fwd, lat, z=1.0):
    return carla.Location(
        x=base.x + forward.x * fwd + right.x * lat,
        y=base.y + forward.y * fwd + right.y * lat,
        z=base.z + z,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--bp", default="walker.pedestrian.0001")
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()

    print("[OK] CARLA bağlantısı var")
    print("[INFO] Map:", world.get_map().name)

    if args.clear:
        clear_old(world)
        time.sleep(0.5)

    ego = find_ego(world)
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    ego_yaw = ego_tf.rotation.yaw

    forward, right = get_forward_right(ego_tf)

    bp_lib = world.get_blueprint_library()

    try:
        bp = bp_lib.find(args.bp)
    except Exception:
        raise RuntimeError(f"Blueprint bulunamadı: {args.bp}")

    if bp.has_attribute("is_invincible"):
        bp.set_attribute("is_invincible", "false")

    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", ROLE_NAME)

    layouts = [
        (9.0, 0.0),
        (12.0, -1.0),
        (12.0, 1.0),
        (16.0, 0.0),
    ]

    spawned = []

    for i, (fwd, lat) in enumerate(layouts):
        loc = rel_loc(ego_loc, forward, right, fwd, lat, z=1.0)

        wp = world.get_map().get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if wp is not None:
            loc = wp.transform.location
            loc.z += 1.0

        rot = carla.Rotation(
            pitch=0.0,
            yaw=ego_yaw + 180.0,
            roll=0.0,
        )

        tf = carla.Transform(loc, rot)

        actor = world.try_spawn_actor(bp, tf)

        if actor is None:
            print(f"[WARN] Spawn başarısız: fwd={fwd}, lat={lat}")
            continue

        actor.set_simulate_physics(True)
        spawned.append(actor)

        print(
            f"[OK] Pedestrian eklendi: id={actor.id}, type={actor.type_id}, "
            f"fwd={fwd}, lat={lat}, x={loc.x:.2f}, y={loc.y:.2f}"
        )

    print("==========================================")
    print(f"[DONE] Blueprint: {args.bp}")
    print(f"[DONE] Spawn count: {len(spawned)}")
    print("Perception beklenen:")
    print("original_label=pedestrian")
    print("label=person")
    print("==========================================")


if __name__ == "__main__":
    main()


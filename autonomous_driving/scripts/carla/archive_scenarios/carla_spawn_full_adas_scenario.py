import argparse
import json
import random
import time
from pathlib import Path

import carla


ACTOR_FILE = Path("/tmp/carla_adas_full_actor_ids.json")


def load_actor_ids():
    if not ACTOR_FILE.exists():
        return []

    try:
        return json.loads(ACTOR_FILE.read_text())
    except Exception:
        return []


def save_actor_ids(ids):
    ACTOR_FILE.write_text(json.dumps(ids, indent=2))


def connect():
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    return client, world


def tick_world(world, n=3):
    for _ in range(n):
        try:
            world.tick()
        except Exception:
            try:
                world.wait_for_tick(seconds=2.0)
            except Exception:
                pass
        time.sleep(0.1)


def destroy_saved_actors(world):
    ids = load_actor_ids()
    destroyed = 0

    for actor_id in ids:
        actor = world.get_actor(actor_id)
        if actor is not None:
            try:
                actor.destroy()
                destroyed += 1
            except Exception:
                pass

    save_actor_ids([])
    print(f"Temizlenen actor sayısı: {destroyed}")


def find_ego(world):
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.attributes.get("role_name") == "ego_vehicle":
            return actor

    return None


def find_rgb_front_sensor(world, ego):
    for actor in world.get_actors().filter("sensor.camera.rgb"):
        role = actor.attributes.get("role_name", "")
        parent = getattr(actor, "parent", None)

        if role == "rgb_front" and parent is not None and parent.id == ego.id:
            return actor

    return None


def get_spawn_transform(world, index=0):
    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        raise RuntimeError("CARLA map spawn point üretmedi.")

    index = max(0, min(index, len(spawn_points) - 1))
    return spawn_points[index]


def spawn_ego(world, actor_ids):
    ego = find_ego(world)

    if ego is not None:
        print(f"Ego zaten var: id={ego.id}, type={ego.type_id}")
    else:
        bp_lib = world.get_blueprint_library()
        ego_bp = bp_lib.find("vehicle.tesla.model3")

        if ego_bp.has_attribute("role_name"):
            ego_bp.set_attribute("role_name", "ego_vehicle")

        if ego_bp.has_attribute("color"):
            colors = ego_bp.get_attribute("color").recommended_values
            if colors:
                ego_bp.set_attribute("color", "0,0,255" if "0,0,255" in colors else colors[0])

        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        ego = None
        preferred = [0, 1, 2, 3, 4, 5]

        for idx in preferred:
            if idx < len(spawn_points):
                ego = world.try_spawn_actor(ego_bp, spawn_points[idx])
                if ego is not None:
                    break

        if ego is None:
            for sp in spawn_points:
                ego = world.try_spawn_actor(ego_bp, sp)
                if ego is not None:
                    break

        if ego is None:
            raise RuntimeError("Ego vehicle spawn edilemedi.")

        ego.set_autopilot(False)
        actor_ids.append(ego.id)
        print(f"Ego spawn edildi: id={ego.id}, type={ego.type_id}, role_name={ego.attributes.get('role_name')}")

    sensor = find_rgb_front_sensor(world, ego)

    if sensor is not None:
        print(f"RGB front kamera zaten var: id={sensor.id}")
    else:
        bp_lib = world.get_blueprint_library()
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("role_name", "rgb_front")
        cam_bp.set_attribute("image_size_x", "1280")
        cam_bp.set_attribute("image_size_y", "720")
        cam_bp.set_attribute("fov", "70")
        cam_bp.set_attribute("sensor_tick", "0.08")

        cam_tf = carla.Transform(
            carla.Location(x=1.80, y=0.0, z=1.55),
            carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0),
        )

        sensor = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
        actor_ids.append(sensor.id)
        print(f"RGB front kamera spawn edildi: id={sensor.id}, topic beklenen=/carla/ego_vehicle/rgb_front/image")

    tick_world(world, 5)
    return ego


def choose_bp(bp_lib, patterns):
    for pattern in patterns:
        matches = bp_lib.filter(pattern)
        if len(matches) > 0:
            return matches[0]

    return None


def spawn_front_vehicle(world, ego, actor_ids, distance):
    bp_lib = world.get_blueprint_library()

    vehicle_bp = choose_bp(
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
            print("Vehicle blueprint yok.")
            return None
        vehicle_bp = random.choice(vehicles)

    if vehicle_bp.has_attribute("role_name"):
        vehicle_bp.set_attribute("role_name", "adas_test_front_vehicle")

    if vehicle_bp.has_attribute("color"):
        colors = vehicle_bp.get_attribute("color").recommended_values
        if colors:
            vehicle_bp.set_attribute("color", random.choice(colors))

    ego_wp = world.get_map().get_waypoint(
        ego.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    for dist in [distance, distance + 4, distance + 8, max(8, distance - 4)]:
        next_wps = ego_wp.next(float(dist))

        if not next_wps:
            continue

        tf = next_wps[0].transform
        tf.location.z += 0.5

        actor = world.try_spawn_actor(vehicle_bp, tf)

        if actor is not None:
            actor.set_autopilot(False)
            actor.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    hand_brake=True,
                )
            )
            actor_ids.append(actor.id)
            print(f"Öne araç spawn edildi: id={actor.id}, type={actor.type_id}, distance={dist}")
            return actor

    print("Öne araç spawn edilemedi.")
    return None


def spawn_person(world, ego, actor_ids, distance, offset):
    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter("walker.pedestrian.*")

    if len(walker_bps) == 0:
        print("Yaya blueprint yok.")
        return None

    walker_bp = random.choice(walker_bps)

    if walker_bp.has_attribute("is_invincible"):
        walker_bp.set_attribute("is_invincible", "false")

    ego_tf = ego.get_transform()
    fwd = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()

    offsets = [
        offset,
        0.0,
        0.5,
        -0.5,
        1.0,
        -1.0,
        1.5,
        -1.5,
        2.5,
        -2.5,
    ]

    distances = [
        distance,
        distance + 2,
        max(5, distance - 2),
        distance + 4,
    ]

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

            actor = world.try_spawn_actor(walker_bp, tf)

            if actor is not None:
                actor_ids.append(actor.id)
                print(f"İnsan spawn edildi: id={actor.id}, type={actor.type_id}, distance={dist}, offset={off}")
                return actor

    print("İnsan spawn edilemedi.")
    return None


def print_world_state(world):
    vehicles = world.get_actors().filter("vehicle.*")
    walkers = world.get_actors().filter("walker.pedestrian.*")
    cameras = world.get_actors().filter("sensor.camera.rgb")

    print(f"Vehicle count: {len(vehicles)}")
    for v in vehicles:
        print(f"  vehicle id={v.id}, type={v.type_id}, role_name={v.attributes.get('role_name')}")

    print(f"Walker count: {len(walkers)}")
    for w in walkers:
        print(f"  walker id={w.id}, type={w.type_id}")

    print(f"RGB camera count: {len(cameras)}")
    for c in cameras:
        parent = getattr(c, "parent", None)
        parent_id = parent.id if parent is not None else None
        print(f"  camera id={c.id}, role_name={c.attributes.get('role_name')}, parent={parent_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--spawn-ego", action="store_true")
    parser.add_argument("--scenario", action="store_true")
    parser.add_argument("--only-person", action="store_true")
    parser.add_argument("--only-vehicle", action="store_true")
    parser.add_argument("--front-distance", type=float, default=16.0)
    parser.add_argument("--ped-distance", type=float, default=9.0)
    parser.add_argument("--ped-offset", type=float, default=0.5)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    _, world = connect()

    if args.clear:
        destroy_saved_actors(world)
        tick_world(world, 3)
        print_world_state(world)
        return

    if args.status:
        print_world_state(world)
        return

    actor_ids = load_actor_ids()

    ego = find_ego(world)

    if args.spawn_ego or ego is None:
        ego = spawn_ego(world, actor_ids)
        save_actor_ids(actor_ids)

    if ego is None:
        print("Ego bulunamadı. --spawn-ego ile tekrar çalıştır.")
        print_world_state(world)
        return

    print(f"Ego hazır: id={ego.id}, type={ego.type_id}")

    if args.scenario or args.only_person or args.only_vehicle:
        if not args.only_person:
            spawn_front_vehicle(world, ego, actor_ids, args.front_distance)

        if not args.only_vehicle:
            spawn_person(world, ego, actor_ids, args.ped_distance, args.ped_offset)

        save_actor_ids(actor_ids)
        tick_world(world, 10)
        print("Senaryo hazır.")

    print_world_state(world)


if __name__ == "__main__":
    main()

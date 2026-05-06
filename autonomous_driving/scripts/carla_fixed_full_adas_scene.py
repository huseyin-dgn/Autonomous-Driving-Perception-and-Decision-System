#!/usr/bin/env python3

import argparse
import json
import math
import random
import time
from pathlib import Path

import carla


ACTOR_FILE = Path("/tmp/carla_fixed_full_adas_actor_ids.json")


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
    client.set_timeout(60.0)
    return client, client.get_world()


def tick(world, n=5):
    for _ in range(n):
        try:
            world.tick()
        except Exception:
            try:
                world.wait_for_tick(seconds=2.0)
            except Exception:
                pass
        time.sleep(0.05)


def destroy_old(world):
    destroyed = 0

    for actor_id in load_ids():
        actor = world.get_actor(actor_id)
        if actor is not None:
            try:
                actor.destroy()
                destroyed += 1
            except Exception:
                pass

    for actor in world.get_actors():
        role = actor.attributes.get("role_name", "")
        if role in ["ego_vehicle", "rgb_front", "adas_test_front_vehicle"]:
            try:
                actor.destroy()
                destroyed += 1
            except Exception:
                pass

    save_ids([])
    print(f"Temizlenen actor sayısı: {destroyed}")


def choose_spawn_near_lights(world, rank):
    lights = list(world.get_actors().filter("traffic.traffic_light*"))

    if not lights:
        raise RuntimeError("Bu haritada traffic light actor yok.")

    waypoints = world.get_map().generate_waypoints(2.0)
    candidates = []

    for wp in waypoints:
        if wp.lane_type != carla.LaneType.Driving:
            continue

        wp_loc = wp.transform.location
        fwd = wp.transform.get_forward_vector()

        matched = []

        for light in lights:
            l_loc = light.get_location()

            dx = l_loc.x - wp_loc.x
            dy = l_loc.y - wp_loc.y
            dz = l_loc.z - wp_loc.z

            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            if dist < 8.0 or dist > 45.0:
                continue

            flat = math.sqrt(dx * dx + dy * dy)
            if flat < 0.001:
                continue

            dot = (fwd.x * dx + fwd.y * dy) / flat

            if dot < 0.25:
                continue

            matched.append((light, dist, dot))

        if len(matched) < 1:
            continue

        matched.sort(key=lambda x: x[1])

        score = 0.0
        for _, dist, dot in matched[:4]:
            score += (50.0 - dist) + dot * 20.0

        candidates.append((score, wp, matched))

    if not candidates:
        raise RuntimeError("Trafik ışığına bakan uygun spawn noktası bulunamadı.")

    candidates.sort(key=lambda x: x[0], reverse=True)

    rank = max(0, min(rank, len(candidates) - 1))
    score, wp, matched = candidates[rank]

    print(f"Seçilen spawn rank={rank}, score={score:.2f}, ışık sayısı={len(matched)}")

    for i, (light, dist, dot) in enumerate(matched[:5]):
        print(f"  light[{i}] id={light.id}, dist={dist:.2f}, dot={dot:.2f}, loc={light.get_location()}")

    return wp, [x[0] for x in matched[:3]]


def set_lights(lights):
    states = [
        ("red", carla.TrafficLightState.Red),
        ("yellow", carla.TrafficLightState.Yellow),
        ("green", carla.TrafficLightState.Green),
    ]

    for i, light in enumerate(lights):
        name, state = states[i % 3]
        try:
            light.set_state(state)
            light.freeze(True)
            print(f"Işık ayarlandı: id={light.id}, state={name}")
        except Exception as exc:
            print(f"Işık ayarlanamadı: id={light.id}, hata={exc}")


def spawn_ego_and_camera(world, wp, actor_ids, width, height, fov, sensor_tick):
    bp_lib = world.get_blueprint_library()

    ego_bp = bp_lib.find("vehicle.tesla.model3")
    ego_bp.set_attribute("role_name", "ego_vehicle")

    if ego_bp.has_attribute("color"):
        colors = ego_bp.get_attribute("color").recommended_values
        if colors:
            ego_bp.set_attribute("color", random.choice(colors))

    tf = wp.transform
    tf.location.z += 0.5

    ego = world.try_spawn_actor(ego_bp, tf)

    if ego is None:
        raise RuntimeError("Ego araç spawn edilemedi.")

    ego.set_autopilot(False)
    ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    actor_ids.append(ego.id)

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("role_name", "rgb_front")
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov))
    cam_bp.set_attribute("sensor_tick", str(sensor_tick))

    cam_tf = carla.Transform(
        carla.Location(x=1.80, y=0.0, z=1.55),
        carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0),
    )

    cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
    actor_ids.append(cam.id)

    print(f"Ego spawn edildi: id={ego.id}")
    print(f"RGB front kamera spawn edildi: id={cam.id}, {width}x{height}, fov={fov}")

    return ego


def spawn_front_vehicle(world, ego, actor_ids, distance):
    bp_lib = world.get_blueprint_library()

    bp = bp_lib.find("vehicle.tesla.model3")
    bp.set_attribute("role_name", "adas_test_front_vehicle")

    if bp.has_attribute("color"):
        colors = bp.get_attribute("color").recommended_values
        if colors:
            bp.set_attribute("color", random.choice(colors))

    ego_wp = world.get_map().get_waypoint(
        ego.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    for dist in [distance, distance + 3.0, max(5.0, distance - 3.0)]:
        next_wps = ego_wp.next(float(dist))
        if not next_wps:
            continue

        tf = next_wps[0].transform
        tf.location.z += 0.5

        actor = world.try_spawn_actor(bp, tf)

        if actor is not None:
            actor.set_autopilot(False)
            actor.apply_control(
                carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
            )
            actor_ids.append(actor.id)
            print(f"Öndeki araç spawn edildi: id={actor.id}, distance={dist}")
            return actor

    print("Öndeki araç spawn edilemedi.")
    return None


def spawn_person(world, ego, actor_ids, distance, side):
    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter("walker.pedestrian.*")

    if not walker_bps:
        print("Yaya blueprint yok.")
        return None

    bp = random.choice(walker_bps)

    if bp.has_attribute("is_invincible"):
        bp.set_attribute("is_invincible", "false")

    ego_tf = ego.get_transform()
    fwd = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()

    offsets = [side, -side, side + 1.0, -side - 1.0, 1.0, -1.0]

    for off in offsets:
        loc = ego_tf.location + fwd * float(distance) + right * float(off)
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
            actor_ids.append(actor.id)
            print(f"Yaya spawn edildi: id={actor.id}, distance={distance}, side={off}")
            return actor

    print("Yaya spawn edilemedi.")
    return None


def set_spectator(world, ego):
    tf = ego.get_transform()
    yaw = math.radians(tf.rotation.yaw)

    spec_tf = carla.Transform(
        carla.Location(
            x=tf.location.x - 8.0 * math.cos(yaw),
            y=tf.location.y - 8.0 * math.sin(yaw),
            z=tf.location.z + 4.0,
        ),
        carla.Rotation(
            pitch=-18.0,
            yaw=tf.rotation.yaw,
            roll=0.0,
        ),
    )

    world.get_spectator().set_transform(spec_tf)


def status(world):
    vehicles = world.get_actors().filter("vehicle.*")
    walkers = world.get_actors().filter("walker.pedestrian.*")
    cameras = world.get_actors().filter("sensor.camera.rgb")
    lights = world.get_actors().filter("traffic.traffic_light*")

    print(f"Vehicle count: {len(vehicles)}")
    for v in vehicles:
        print(f"  vehicle id={v.id}, role_name={v.attributes.get('role_name')}, type={v.type_id}")

    print(f"Walker count: {len(walkers)}")
    for w in walkers:
        print(f"  walker id={w.id}, type={w.type_id}")

    print(f"RGB camera count: {len(cameras)}")
    for c in cameras:
        parent = getattr(c, "parent", None)
        pid = parent.id if parent is not None else None
        print(f"  camera id={c.id}, role_name={c.attributes.get('role_name')}, parent={pid}")

    print(f"Traffic light count: {len(lights)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--town", default="Town03")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--sensor-tick", type=float, default=0.10)
    parser.add_argument("--front-distance", type=float, default=7.0)
    parser.add_argument("--ped-distance", type=float, default=10.0)
    parser.add_argument("--ped-side", type=float, default=2.5)
    args = parser.parse_args()

    client, world = connect()

    if args.clear:
        destroy_old(world)
        tick(world, 5)
        status(world)
        return

    if args.status:
        status(world)
        return

    if args.town:
        world = client.load_world(args.town)
        tick(world, 10)
        print(f"Map loaded: {world.get_map().name}")

    destroy_old(world)
    tick(world, 5)

    wp, lights = choose_spawn_near_lights(world, args.rank)
    set_lights(lights)

    actor_ids = []

    ego = spawn_ego_and_camera(
        world,
        wp,
        actor_ids,
        args.width,
        args.height,
        args.fov,
        args.sensor_tick,
    )

    tick(world, 5)

    spawn_front_vehicle(world, ego, actor_ids, args.front_distance)
    spawn_person(world, ego, actor_ids, args.ped_distance, args.ped_side)

    save_ids(actor_ids)

    tick(world, 10)
    set_spectator(world, ego)

    print("Sabit ADAS sahnesi hazır.")
    status(world)


if __name__ == "__main__":
    main()

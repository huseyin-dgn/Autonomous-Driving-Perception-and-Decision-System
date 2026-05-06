#!/usr/bin/env python3

import argparse
import json
import math
import random
import time
from pathlib import Path

import carla


ACTOR_FILE = Path("/tmp/carla_perception_drive_scene_actor_ids.json")


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
        if role.startswith("adas_demo_") or role in ["ego_vehicle", "rgb_front"]:
            try:
                actor.destroy()
                destroyed += 1
            except Exception:
                pass

    save_ids([])
    print(f"Temizlenen actor sayısı: {destroyed}")


def choose_spawn_near_lights(world, rank):
    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    waypoints = world.get_map().generate_waypoints(2.0)

    if not lights:
        raise RuntimeError("Bu haritada traffic light actor yok.")

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

            if dist < 10.0 or dist > 65.0:
                continue

            flat = math.sqrt(dx * dx + dy * dy)
            if flat < 0.001:
                continue

            dot = (fwd.x * dx + fwd.y * dy) / flat

            if dot < 0.15:
                continue

            matched.append((light, dist, dot))

        if len(matched) < 2:
            continue

        matched.sort(key=lambda x: x[1])

        score = 0.0
        for _, dist, dot in matched[:6]:
            score += (70.0 - dist) + dot * 20.0

        candidates.append((score, wp, matched))

    if not candidates:
        raise RuntimeError("Trafik ışığına bakan uygun spawn noktası bulunamadı.")

    candidates.sort(key=lambda x: x[0], reverse=True)

    rank = max(0, min(rank, len(candidates) - 1))
    score, wp, matched = candidates[rank]

    print(f"Seçilen spawn rank={rank}, score={score:.2f}, ışık sayısı={len(matched)}")

    for i, (light, dist, dot) in enumerate(matched[:6]):
        print(f"  light[{i}] id={light.id}, dist={dist:.2f}, dot={dot:.2f}")

    return wp


def set_mixed_lights(world):
    lights = list(world.get_actors().filter("traffic.traffic_light*"))

    states = [
        carla.TrafficLightState.Red,
        carla.TrafficLightState.Yellow,
        carla.TrafficLightState.Green,
    ]

    for i, light in enumerate(lights):
        try:
            light.set_state(states[i % 3])
            light.freeze(True)
        except Exception:
            pass

    print(f"Trafik ışıkları red/yellow/green karışık ayarlandı. count={len(lights)}")


def random_vehicle_bp(bp_lib):
    preferred = [
        "vehicle.tesla.model3",
        "vehicle.audi.tt",
        "vehicle.lincoln.mkz_2020",
        "vehicle.toyota.prius",
        "vehicle.mercedes.coupe",
        "vehicle.bmw.grandtourer",
        "vehicle.nissan.patrol",
        "vehicle.mini.cooper_s",
    ]

    candidates = []

    for item in preferred:
        found = bp_lib.filter(item)
        if found:
            candidates.extend(found)

    if not candidates:
        candidates = list(bp_lib.filter("vehicle.*"))

    return random.choice(candidates)


def spawn_ego_and_camera(world, wp, actor_ids, width, height, fov, sensor_tick, tm_port):
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

    ego.set_autopilot(True, tm_port)
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

    print(f"Ego araç spawn edildi: id={ego.id}, autopilot=True")
    print(f"RGB kamera spawn edildi: id={cam.id}, {width}x{height}, fov={fov}")

    return ego


def spawn_side_vehicles(world, ego, actor_ids, count):
    bp_lib = world.get_blueprint_library()
    ego_wp = world.get_map().get_waypoint(
        ego.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    spawned = 0
    distances = [14, 22, 30, 38, 48, 58, 70, 82]
    sides = [-4.2, 4.2, -5.8, 5.8]

    for dist in distances:
        if spawned >= count:
            break

        next_wps = ego_wp.next(float(dist))
        if not next_wps:
            continue

        wp = next_wps[0]
        base_tf = wp.transform
        right = base_tf.get_right_vector()

        for side in sides:
            if spawned >= count:
                break

            bp = random_vehicle_bp(bp_lib)

            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", f"adas_demo_side_vehicle_{spawned}")

            if bp.has_attribute("color"):
                colors = bp.get_attribute("color").recommended_values
                if colors:
                    bp.set_attribute("color", random.choice(colors))

            loc = base_tf.location + right * float(side)
            loc.z += 0.5

            spawn_tf = carla.Transform(
                loc,
                carla.Rotation(
                    pitch=0.0,
                    yaw=base_tf.rotation.yaw + random.choice([0.0, 180.0]),
                    roll=0.0,
                ),
            )

            actor = world.try_spawn_actor(bp, spawn_tf)

            if actor is not None:
                actor.set_autopilot(False)
                actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                actor_ids.append(actor.id)
                spawned += 1
                print(f"Kenar araç spawn edildi: id={actor.id}, type={actor.type_id}")

    print(f"Toplam kenar araç: {spawned}")


def spawn_side_people(world, ego, actor_ids, count):
    bp_lib = world.get_blueprint_library()
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))

    if not walker_bps:
        print("Yaya blueprint yok.")
        return

    ego_wp = world.get_map().get_waypoint(
        ego.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    spawned = 0
    distances = [10, 16, 24, 32, 44, 56, 68, 80]
    sides = [-3.2, 3.2, -4.8, 4.8]

    for dist in distances:
        if spawned >= count:
            break

        next_wps = ego_wp.next(float(dist))
        if not next_wps:
            continue

        wp = next_wps[0]
        base_tf = wp.transform
        right = base_tf.get_right_vector()

        for side in sides:
            if spawned >= count:
                break

            bp = random.choice(walker_bps)

            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")

            loc = base_tf.location + right * float(side)
            loc.z += 1.0

            spawn_tf = carla.Transform(
                loc,
                carla.Rotation(
                    pitch=0.0,
                    yaw=base_tf.rotation.yaw + 180.0,
                    roll=0.0,
                ),
            )

            actor = world.try_spawn_actor(bp, spawn_tf)

            if actor is not None:
                actor_ids.append(actor.id)
                spawned += 1
                print(f"Kenar yaya spawn edildi: id={actor.id}")

    print(f"Toplam yaya: {spawned}")


def follow_spectator(world, ego):
    tf = ego.get_transform()
    yaw = math.radians(tf.rotation.yaw)

    spec_tf = carla.Transform(
        carla.Location(
            x=tf.location.x - 8.0 * math.cos(yaw),
            y=tf.location.y - 8.0 * math.sin(yaw),
            z=tf.location.z + 4.0,
        ),
        carla.Rotation(pitch=-18.0, yaw=tf.rotation.yaw, roll=0.0),
    )

    world.get_spectator().set_transform(spec_tf)


def status(world):
    print(f"Vehicle count: {len(world.get_actors().filter('vehicle.*'))}")
    print(f"Walker count: {len(world.get_actors().filter('walker.pedestrian.*'))}")
    print(f"RGB camera count: {len(world.get_actors().filter('sensor.camera.rgb'))}")
    print(f"Traffic light count: {len(world.get_actors().filter('traffic.traffic_light*'))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--town", default="Town03")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=80.0)
    parser.add_argument("--sensor-tick", type=float, default=0.10)
    parser.add_argument("--vehicle-count", type=int, default=8)
    parser.add_argument("--person-count", type=int, default=8)
    parser.add_argument("--speed-diff", type=float, default=65.0)
    parser.add_argument("--keep-alive", action="store_true")
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

    world = client.load_world(args.town)
    tick(world, 10)
    print(f"Map loaded: {world.get_map().name}")

    destroy_old(world)
    tick(world, 5)

    set_mixed_lights(world)

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(5.0)

    wp = choose_spawn_near_lights(world, args.rank)

    actor_ids = []

    ego = spawn_ego_and_camera(
        world,
        wp,
        actor_ids,
        args.width,
        args.height,
        args.fov,
        args.sensor_tick,
        traffic_manager.get_port(),
    )

    try:
        traffic_manager.vehicle_percentage_speed_difference(ego, args.speed_diff)
        traffic_manager.ignore_lights_percentage(ego, 100.0)
        traffic_manager.ignore_walkers_percentage(ego, 70.0)
    except Exception as exc:
        print(f"TrafficManager ayarı kısmen uygulanamadı: {exc}")

    tick(world, 5)

    spawn_side_vehicles(world, ego, actor_ids, args.vehicle_count)
    spawn_side_people(world, ego, actor_ids, args.person_count)

    save_ids(actor_ids)

    tick(world, 10)
    follow_spectator(world, ego)

    print("Perception-only hareketli CARLA sahnesi hazır.")
    status(world)

    if args.keep_alive:
        print("Sahne aktif. Ego hareket edecek. Çıkmak için CTRL+C.")
        while True:
            try:
                set_mixed_lights(world)
                ego.set_autopilot(True, traffic_manager.get_port())
                follow_spectator(world, ego)
                time.sleep(1.0)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()

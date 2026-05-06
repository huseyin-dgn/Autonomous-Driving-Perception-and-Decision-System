#!/usr/bin/env python3
import argparse
import math
import random
import time
import carla


TAG = "adas_visible_human"


def role(actor):
    try:
        return actor.attributes.get("role_name", "")
    except Exception:
        return ""


def safe_destroy(actor):
    try:
        if actor is not None and actor.is_alive:
            print(f"[CLEAR] destroy id={actor.id} type={actor.type_id} role={role(actor)}")
            actor.destroy()
    except Exception as e:
        print(f"[CLEAR] destroy hata: {e}")


def clear_scene(world):
    actors = world.get_actors()
    targets = []

    for a in actors:
        tid = a.type_id
        r = role(a)

        if r.startswith(TAG) or r in ["rgb_front", "ego_vehicle", "hero", "ego"]:
            targets.append(a)
        elif tid.startswith("sensor.camera"):
            targets.append(a)
        elif tid.startswith("walker.pedestrian"):
            targets.append(a)
        elif tid.startswith("controller.ai.walker"):
            targets.append(a)
        elif tid.startswith("vehicle."):
            targets.append(a)

    print(f"[CLEAR] Silinecek actor sayısı: {len(targets)}")

    controllers = [a for a in targets if a.type_id.startswith("controller.ai.walker")]
    sensors = [a for a in targets if a.type_id.startswith("sensor.")]
    walkers = [a for a in targets if a.type_id.startswith("walker.")]
    vehicles = [a for a in targets if a.type_id.startswith("vehicle.")]

    for group in [controllers, sensors, walkers, vehicles]:
        for a in group:
            safe_destroy(a)

    time.sleep(1.0)


def set_role(bp, name):
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", name)


def choose_bp(bp_lib, filters, fallback):
    for f in filters:
        bps = list(bp_lib.filter(f))
        if bps:
            return random.choice(bps)

    bps = list(bp_lib.filter(fallback))
    if not bps:
        raise RuntimeError(f"Blueprint yok: {fallback}")

    return random.choice(bps)


def forward_right_location(base_tf, forward_m, right_m, z_add):
    yaw = math.radians(base_tf.rotation.yaw)

    fx = math.cos(yaw)
    fy = math.sin(yaw)

    rx = math.cos(yaw + math.pi / 2.0)
    ry = math.sin(yaw + math.pi / 2.0)

    return carla.Location(
        x=base_tf.location.x + fx * forward_m + rx * right_m,
        y=base_tf.location.y + fy * forward_m + ry * right_m,
        z=base_tf.location.z + z_add,
    )


def spawn_ego(world, bp_lib, spawn_index):
    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        raise RuntimeError("Spawn point yok")

    ego_bp = choose_bp(
        bp_lib,
        [
            "vehicle.tesla.model3",
            "vehicle.lincoln.mkz_2020",
            "vehicle.audi.tt",
            "vehicle.dodge.charger_2020",
        ],
        "vehicle.*",
    )

    set_role(ego_bp, f"{TAG}_ego")

    indices = [spawn_index] + list(range(len(spawn_points)))

    for idx in indices:
        idx = idx % len(spawn_points)

        tf = spawn_points[idx]
        tf.location.z += 0.35

        ego = world.try_spawn_actor(ego_bp, tf)

        if ego is None:
            continue

        try:
            ego.set_autopilot(False)
            ego.set_simulate_physics(False)
        except Exception:
            pass

        print(f"[SPAWN] Ego OK id={ego.id} spawn_index={idx} type={ego.type_id}")
        return ego, tf

    raise RuntimeError("Ego spawn edilemedi")


def spawn_camera(world, bp_lib, ego):
    cam_bp = bp_lib.find("sensor.camera.rgb")

    cam_bp.set_attribute("image_size_x", "960")
    cam_bp.set_attribute("image_size_y", "540")
    cam_bp.set_attribute("fov", "100")
    cam_bp.set_attribute("sensor_tick", "0.05")

    set_role(cam_bp, "rgb_front")

    cam_tf = carla.Transform(
        carla.Location(x=1.70, y=0.0, z=1.55),
        carla.Rotation(pitch=-2.0, yaw=0.0, roll=0.0),
    )

    cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    print(f"[SPAWN] Camera OK id={cam.id} role=rgb_front attach_to={ego.id}")
    return cam


def spawn_walker(world, bp_lib, ego_tf, forward_m, right_m, yaw_delta, name):
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))

    if not walker_bps:
        print("[SPAWN] Walker blueprint yok")
        return None

    random.shuffle(walker_bps)

    for bp in walker_bps[:8]:
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        set_role(bp, f"{TAG}_{name}")

        loc = forward_right_location(ego_tf, forward_m, right_m, 0.55)

        tf = carla.Transform(
            loc,
            carla.Rotation(
                pitch=0.0,
                yaw=ego_tf.rotation.yaw + yaw_delta,
                roll=0.0,
            ),
        )

        actor = world.try_spawn_actor(bp, tf)

        if actor is None:
            continue

        try:
            actor.set_simulate_physics(False)
        except Exception:
            pass

        print(
            f"[SPAWN] Walker OK id={actor.id} "
            f"name={name} fwd={forward_m} right={right_m} type={actor.type_id}"
        )
        return actor

    print(f"[SPAWN] Walker FAIL name={name} fwd={forward_m} right={right_m}")
    return None


def spawn_vehicle(world, bp_lib, ego_tf, forward_m, right_m, yaw_delta, name):
    vehicle_bp = choose_bp(
        bp_lib,
        [
            "vehicle.tesla.model3",
            "vehicle.lincoln.mkz_2020",
            "vehicle.audi.tt",
            "vehicle.dodge.charger_2020",
            "vehicle.bmw.grandtourer",
        ],
        "vehicle.*",
    )

    set_role(vehicle_bp, f"{TAG}_{name}")

    loc = forward_right_location(ego_tf, forward_m, right_m, 0.35)

    tf = carla.Transform(
        loc,
        carla.Rotation(
            pitch=0.0,
            yaw=ego_tf.rotation.yaw + yaw_delta,
            roll=0.0,
        ),
    )

    actor = world.try_spawn_actor(vehicle_bp, tf)

    if actor is None:
        print(f"[SPAWN] Vehicle FAIL name={name} fwd={forward_m} right={right_m}")
        return None

    try:
        actor.set_autopilot(False)
        actor.set_simulate_physics(False)
    except Exception:
        pass

    print(
        f"[SPAWN] Vehicle OK id={actor.id} "
        f"name={name} fwd={forward_m} right={right_m} type={actor.type_id}"
    )
    return actor


def set_spectator(world, ego_tf):
    spectator = world.get_spectator()

    yaw = math.radians(ego_tf.rotation.yaw)

    loc = carla.Location(
        x=ego_tf.location.x - math.cos(yaw) * 9.0,
        y=ego_tf.location.y - math.sin(yaw) * 9.0,
        z=ego_tf.location.z + 6.5,
    )

    rot = carla.Rotation(
        pitch=-32.0,
        yaw=ego_tf.rotation.yaw,
        roll=0.0,
    )

    spectator.set_transform(carla.Transform(loc, rot))


def set_world(world):
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    world.set_weather(
        carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            sun_altitude_angle=65.0,
            sun_azimuth_angle=20.0,
            fog_density=0.0,
            wetness=0.0,
        )
    )


def load_map(client, world, map_name):
    current = world.get_map().name.split("/")[-1]

    if current != map_name:
        print(f"[MAP] {current} -> {map_name}")
        world = client.load_world(map_name)
        time.sleep(3.0)
    else:
        print(f"[MAP] Aktif: {current}")

    return world


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Town04")
    parser.add_argument("--spawn-index", type=int, default=35)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--clear-only", action="store_true")
    parser.add_argument("--vehicles", action="store_true")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    world = client.get_world()

    if args.clear:
        clear_scene(world)

    if args.clear_only:
        return

    world = load_map(client, world, args.map)
    set_world(world)

    bp_lib = world.get_blueprint_library()

    ego, ego_tf = spawn_ego(world, bp_lib, args.spawn_index)
    cam = spawn_camera(world, bp_lib, ego)

    actors = [ego, cam]

    walker_specs = [
        (6.5, -2.4, 180.0, "person_1_left_close"),
        (7.5, 2.4, 180.0, "person_2_right_close"),
        (9.5, 0.0, 180.0, "person_3_center_close"),
        (11.5, -1.2, 180.0, "person_4_left_mid"),
        (13.0, 1.2, 180.0, "person_5_right_mid"),
    ]

    for fwd, right, yaw_delta, name in walker_specs:
        a = spawn_walker(world, bp_lib, ego_tf, fwd, right, yaw_delta, name)
        if a:
            actors.append(a)

    if args.vehicles:
        vehicle_specs = [
            (18.0, -4.2, 0.0, "side_vehicle_left"),
            (22.0, 4.2, 0.0, "side_vehicle_right"),
            (28.0, 0.0, 0.0, "far_vehicle_center"),
        ]

        for fwd, right, yaw_delta, name in vehicle_specs:
            a = spawn_vehicle(world, bp_lib, ego_tf, fwd, right, yaw_delta, name)
            if a:
                actors.append(a)

    set_spectator(world, ego_tf)

    print("")
    print("======================================================")
    print("ADAS CAMERA-FOV HUMAN TEST READY")
    print("======================================================")
    print(f"Map          : {world.get_map().name}")
    print(f"Ego id       : {ego.id}")
    print(f"Camera id    : {cam.id}")
    print(f"Camera role  : rgb_front")
    print(f"Actor count  : {len(actors)}")
    print("")
    print("Bu sahnede insanlar kameranın tam önüne spawn edilir.")
    print("Perception ekranında Persons > 0 görmek zorundasın.")
    print("JSON içinde label=person original=pedestrian gelmeli.")
    print("======================================================")
    print("")


if __name__ == "__main__":
    main()

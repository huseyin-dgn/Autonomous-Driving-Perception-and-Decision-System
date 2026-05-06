#!/usr/bin/env python3
import argparse
import random
import time
import math
import carla


TAG = "adas_ped_test"


def actor_role(actor):
    try:
        return actor.attributes.get("role_name", "")
    except Exception:
        return ""


def clear_scene(world):
    actors = world.get_actors()

    targets = []

    for actor in actors:
        tid = actor.type_id
        role = actor_role(actor)

        if role.startswith(TAG):
            targets.append(actor)
            continue

        if role in ["rgb_front", "ego_vehicle", "hero", "ego"]:
            targets.append(actor)
            continue

        if tid.startswith("sensor.camera"):
            targets.append(actor)
            continue

        if tid.startswith("vehicle."):
            targets.append(actor)
            continue

        if tid.startswith("walker.pedestrian"):
            targets.append(actor)
            continue

        if tid.startswith("controller.ai.walker"):
            targets.append(actor)
            continue

    print(f"[CLEAR] Silinecek actor sayısı: {len(targets)}")

    for actor in targets:
        try:
            print(f"[CLEAR] destroy id={actor.id} type={actor.type_id} role={actor_role(actor)}")
            actor.destroy()
        except Exception as e:
            print(f"[CLEAR] destroy hata id={actor.id}: {e}")

    time.sleep(1.0)


def get_blueprint(bp_lib, preferred_filters, fallback_filter):
    for f in preferred_filters:
        bps = list(bp_lib.filter(f))
        if bps:
            return random.choice(bps)

    bps = list(bp_lib.filter(fallback_filter))
    if not bps:
        raise RuntimeError(f"Blueprint bulunamadı: {fallback_filter}")

    return random.choice(bps)


def set_role(bp, role_name):
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role_name)


def rel_location(base_tf, forward_m=0.0, right_m=0.0, up_m=0.0):
    f = base_tf.get_forward_vector()
    r = base_tf.get_right_vector()

    return carla.Location(
        x=base_tf.location.x + f.x * forward_m + r.x * right_m,
        y=base_tf.location.y + f.y * forward_m + r.y * right_m,
        z=base_tf.location.z + up_m,
    )


def spawn_static_vehicle(world, bp_lib, ego_tf, forward_m, right_m, yaw_delta=0.0, role_suffix="vehicle"):
    vehicle_bp = get_blueprint(
        bp_lib,
        [
            "vehicle.tesla.model3",
            "vehicle.audi.tt",
            "vehicle.lincoln.mkz_2020",
            "vehicle.dodge.charger_2020",
        ],
        "vehicle.*",
    )

    set_role(vehicle_bp, f"{TAG}_{role_suffix}")

    loc = rel_location(ego_tf, forward_m, right_m, 0.35)

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
        print(f"[SPAWN] Araç spawn olmadı forward={forward_m} right={right_m}")
        return None

    try:
        actor.set_autopilot(False)
        actor.set_simulate_physics(False)
    except Exception:
        pass

    print(f"[SPAWN] Araç id={actor.id} type={actor.type_id} loc={loc}")
    return actor


def spawn_static_walker(world, bp_lib, ego_tf, forward_m, right_m, yaw_delta=180.0, role_suffix="ped"):
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))

    if not walker_bps:
        print("[SPAWN] Walker blueprint bulunamadı")
        return None

    walker_bp = random.choice(walker_bps)

    if walker_bp.has_attribute("is_invincible"):
        walker_bp.set_attribute("is_invincible", "false")

    set_role(walker_bp, f"{TAG}_{role_suffix}")

    loc = rel_location(ego_tf, forward_m, right_m, 0.55)

    tf = carla.Transform(
        loc,
        carla.Rotation(
            pitch=0.0,
            yaw=ego_tf.rotation.yaw + yaw_delta,
            roll=0.0,
        ),
    )

    actor = world.try_spawn_actor(walker_bp, tf)

    if actor is None:
        print(f"[SPAWN] Yaya spawn olmadı forward={forward_m} right={right_m}")
        return None

    try:
        actor.set_simulate_physics(False)
    except Exception:
        pass

    print(f"[SPAWN] Yaya id={actor.id} type={actor.type_id} loc={loc}")
    return actor


def spawn_rgb_front_camera(world, bp_lib, ego):
    cam_bp = bp_lib.find("sensor.camera.rgb")

    cam_bp.set_attribute("image_size_x", "1240")
    cam_bp.set_attribute("image_size_y", "720")
    cam_bp.set_attribute("fov", "90")
    cam_bp.set_attribute("sensor_tick", "0.05")

    set_role(cam_bp, "rgb_front")

    cam_tf = carla.Transform(
        carla.Location(x=1.65, y=0.0, z=1.55),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    )

    cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    print(f"[SPAWN] RGB front camera id={cam.id} role=rgb_front attach_to={ego.id}")
    return cam


def spawn_ego(world, bp_lib, spawn_index):
    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        raise RuntimeError("Map spawn point vermedi")

    indices = [spawn_index] + list(range(len(spawn_points)))

    ego_bp = get_blueprint(
        bp_lib,
        [
            "vehicle.tesla.model3",
            "vehicle.lincoln.mkz_2020",
            "vehicle.audi.tt",
        ],
        "vehicle.*",
    )

    set_role(ego_bp, f"{TAG}_ego")

    for idx in indices:
        idx = idx % len(spawn_points)
        tf = spawn_points[idx]
        tf.location.z += 0.35

        ego = world.try_spawn_actor(ego_bp, tf)

        if ego is not None:
            try:
                ego.set_autopilot(False)
                ego.set_simulate_physics(False)
            except Exception:
                pass

            print(f"[SPAWN] Ego id={ego.id} spawn_index={idx} type={ego.type_id}")
            return ego, tf

    raise RuntimeError("Ego araç spawn edilemedi")


def set_spectator(world, ego_tf):
    spectator = world.get_spectator()

    yaw_rad = math.radians(ego_tf.rotation.yaw)
    back_x = -math.cos(yaw_rad) * 9.0
    back_y = -math.sin(yaw_rad) * 9.0

    loc = carla.Location(
        x=ego_tf.location.x + back_x,
        y=ego_tf.location.y + back_y,
        z=ego_tf.location.z + 7.0,
    )

    rot = carla.Rotation(
        pitch=-35.0,
        yaw=ego_tf.rotation.yaw,
        roll=0.0,
    )

    spectator.set_transform(carla.Transform(loc, rot))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Town04")
    parser.add_argument("--spawn-index", type=int, default=35)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--clear-only", action="store_true")
    parser.add_argument("--keep-alive", action="store_true")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    world = client.get_world()

    current_map = world.get_map().name.split("/")[-1]

    if args.clear:
        clear_scene(world)

    if args.clear_only:
        return

    if args.map and current_map != args.map:
        print(f"[MAP] {current_map} -> {args.map} yükleniyor")
        world = client.load_world(args.map)
        time.sleep(3.0)
    else:
        print(f"[MAP] Aktif map: {current_map}")

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    world.set_weather(
        carla.WeatherParameters(
            cloudiness=10.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=0.0,
            sun_altitude_angle=55.0,
            sun_azimuth_angle=25.0,
            fog_density=0.0,
            wetness=0.0,
        )
    )

    bp_lib = world.get_blueprint_library()

    ego, ego_tf = spawn_ego(world, bp_lib, args.spawn_index)

    camera = spawn_rgb_front_camera(world, bp_lib, ego)

    actors = [ego, camera]

    pedestrian_specs = [
        (8.0, -1.8, 180.0, "ped_close_left"),
        (10.5, 1.8, 180.0, "ped_close_right"),
        (14.0, 0.0, 180.0, "ped_center"),
        (18.0, -3.2, 160.0, "ped_mid_left"),
        (22.0, 3.0, 200.0, "ped_mid_right"),
        (28.0, 0.8, 180.0, "ped_far_center"),
    ]

    for fwd, right, yaw_delta, name in pedestrian_specs:
        actor = spawn_static_walker(world, bp_lib, ego_tf, fwd, right, yaw_delta, name)
        if actor is not None:
            actors.append(actor)

    vehicle_specs = [
        (24.0, -4.2, 0.0, "veh_left_front"),
        (30.0, 4.2, 0.0, "veh_right_front"),
        (38.0, 0.0, 0.0, "veh_far_center"),
        (16.0, 6.0, -12.0, "veh_side_right"),
    ]

    for fwd, right, yaw_delta, name in vehicle_specs:
        actor = spawn_static_vehicle(world, bp_lib, ego_tf, fwd, right, yaw_delta, name)
        if actor is not None:
            actors.append(actor)

    set_spectator(world, ego_tf)

    print("")
    print("======================================================")
    print("ADAS PEDESTRIAN DETECTION TEST SCENE READY")
    print("======================================================")
    print(f"Map          : {world.get_map().name}")
    print(f"Ego id       : {ego.id}")
    print(f"Camera id    : {camera.id}")
    print(f"Camera role  : rgb_front")
    print(f"Actor count  : {len(actors)}")
    print("")
    print("Beklenen algılar:")
    print("- pedestrian -> perception içinde person olarak görünmeli")
    print("- vehicle")
    print("")
    print("Perception terminalinde şunu arayacağız:")
    print("MODEL_DET class_id=1 ... original=pedestrian mapped_label=person")
    print("======================================================")
    print("")

    if args.keep_alive:
        print("[KEEP-ALIVE] Script açık kalıyor. Çıkmak için CTRL+C.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("[KEEP-ALIVE] Çıkılıyor.")


if __name__ == "__main__":
    main()

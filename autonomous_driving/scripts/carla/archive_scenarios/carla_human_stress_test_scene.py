#!/usr/bin/env python3
import argparse
import random
import time
import math
import carla


TAG = "adas_human_stress"


def get_role(actor):
    try:
        return actor.attributes.get("role_name", "")
    except Exception:
        return ""


def safe_destroy(actor):
    try:
        if actor is not None and actor.is_alive:
            print(f"[CLEAR] destroy id={actor.id} type={actor.type_id} role={get_role(actor)}")
            actor.destroy()
    except Exception as e:
        print(f"[CLEAR] destroy hata id={getattr(actor, 'id', '?')}: {e}")


def clear_scene(world):
    actors = world.get_actors()

    sensors = []
    controllers = []
    walkers = []
    vehicles = []
    tagged = []

    for actor in actors:
        tid = actor.type_id
        role = get_role(actor)

        if role.startswith(TAG) or role in ["rgb_front", "ego_vehicle", "hero", "ego"]:
            tagged.append(actor)

        if tid.startswith("sensor.camera"):
            sensors.append(actor)
        elif tid.startswith("controller.ai.walker"):
            controllers.append(actor)
        elif tid.startswith("walker.pedestrian"):
            walkers.append(actor)
        elif tid.startswith("vehicle."):
            vehicles.append(actor)

    ordered = []
    for group in [controllers, sensors, walkers, vehicles, tagged]:
        for actor in group:
            if actor not in ordered:
                ordered.append(actor)

    print(f"[CLEAR] Toplam silinecek actor: {len(ordered)}")

    for actor in ordered:
        safe_destroy(actor)

    time.sleep(1.0)


def set_role(bp, role_name):
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role_name)


def choose_bp(bp_lib, filters, fallback):
    for f in filters:
        bps = list(bp_lib.filter(f))
        if bps:
            return random.choice(bps)

    bps = list(bp_lib.filter(fallback))
    if not bps:
        raise RuntimeError(f"Blueprint bulunamadı: {fallback}")

    return random.choice(bps)


def rel_location(base_tf, forward_m=0.0, right_m=0.0, up_m=0.0):
    f = base_tf.get_forward_vector()
    r = base_tf.get_right_vector()

    return carla.Location(
        x=base_tf.location.x + f.x * forward_m + r.x * right_m,
        y=base_tf.location.y + f.y * forward_m + r.y * right_m,
        z=base_tf.location.z + up_m,
    )


def make_transform_from_ego(ego_tf, forward_m, right_m, up_m, yaw_delta):
    loc = rel_location(ego_tf, forward_m, right_m, up_m)

    return carla.Transform(
        loc,
        carla.Rotation(
            pitch=0.0,
            yaw=ego_tf.rotation.yaw + yaw_delta,
            roll=0.0,
        ),
    )


def spawn_ego(world, bp_lib, spawn_index):
    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        raise RuntimeError("Spawn point bulunamadı")

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

    candidate_indices = [spawn_index] + list(range(len(spawn_points)))

    for idx in candidate_indices:
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

        print(f"[SPAWN] Ego id={ego.id} spawn_index={idx} type={ego.type_id}")
        return ego, tf

    raise RuntimeError("Ego spawn edilemedi")


def spawn_rgb_front_camera(world, bp_lib, ego):
    cam_bp = bp_lib.find("sensor.camera.rgb")

    cam_bp.set_attribute("image_size_x", "1240")
    cam_bp.set_attribute("image_size_y", "720")
    cam_bp.set_attribute("fov", "95")
    cam_bp.set_attribute("sensor_tick", "0.05")

    set_role(cam_bp, "rgb_front")

    cam_tf = carla.Transform(
        carla.Location(x=1.70, y=0.0, z=1.55),
        carla.Rotation(pitch=-2.0, yaw=0.0, roll=0.0),
    )

    cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    print(f"[SPAWN] RGB front camera id={cam.id} role=rgb_front attach_to={ego.id}")
    return cam


def spawn_static_vehicle(world, bp_lib, ego_tf, forward_m, right_m, yaw_delta=0.0, role_suffix="vehicle"):
    vehicle_bp = choose_bp(
        bp_lib,
        [
            "vehicle.tesla.model3",
            "vehicle.lincoln.mkz_2020",
            "vehicle.audi.tt",
            "vehicle.dodge.charger_2020",
            "vehicle.bmw.grandtourer",
            "vehicle.mercedes.coupe",
        ],
        "vehicle.*",
    )

    set_role(vehicle_bp, f"{TAG}_{role_suffix}")

    tf = make_transform_from_ego(
        ego_tf,
        forward_m=forward_m,
        right_m=right_m,
        up_m=0.35,
        yaw_delta=yaw_delta,
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

    print(f"[SPAWN] Araç id={actor.id} type={actor.type_id} fwd={forward_m} right={right_m}")
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

    tf = make_transform_from_ego(
        ego_tf,
        forward_m=forward_m,
        right_m=right_m,
        up_m=0.55,
        yaw_delta=yaw_delta,
    )

    actor = world.try_spawn_actor(walker_bp, tf)

    if actor is None:
        print(f"[SPAWN] Yaya spawn olmadı forward={forward_m} right={right_m}")
        return None

    try:
        actor.set_simulate_physics(False)
    except Exception:
        pass

    print(f"[SPAWN] Yaya id={actor.id} type={actor.type_id} fwd={forward_m} right={right_m}")
    return actor


def spawn_close_person_test(world, bp_lib, ego_tf):
    actors = []

    specs = [
        (7.0, -2.8, 180.0, "close_left_big"),
        (8.0, 2.4, 180.0, "close_right_big"),
        (10.0, 0.0, 180.0, "close_center"),
    ]

    for fwd, right, yaw, name in specs:
        a = spawn_static_walker(world, bp_lib, ego_tf, fwd, right, yaw, name)
        if a:
            actors.append(a)

    return actors


def spawn_medium_far_person_test(world, bp_lib, ego_tf):
    actors = []

    specs = [
        (14.0, -1.5, 180.0, "mid_left"),
        (16.0, 1.5, 180.0, "mid_right"),
        (21.0, 0.0, 180.0, "far_center"),
        (27.0, -2.0, 180.0, "far_left"),
        (32.0, 2.0, 180.0, "very_far_right"),
    ]

    for fwd, right, yaw, name in specs:
        a = spawn_static_walker(world, bp_lib, ego_tf, fwd, right, yaw, name)
        if a:
            actors.append(a)

    return actors


def spawn_occlusion_test(world, bp_lib, ego_tf):
    actors = []

    vehicle_specs = [
        (12.0, -0.4, 0.0, "occluder_center"),
        (18.0, 3.2, 0.0, "occluder_right"),
        (24.0, -3.4, 0.0, "occluder_left"),
    ]

    for fwd, right, yaw, name in vehicle_specs:
        a = spawn_static_vehicle(world, bp_lib, ego_tf, fwd, right, yaw, name)
        if a:
            actors.append(a)

    walker_specs = [
        (12.8, -2.0, 180.0, "peek_left_of_car"),
        (13.5, 1.5, 180.0, "peek_right_of_car"),
        (18.8, 4.7, 180.0, "behind_right_car"),
        (24.5, -5.0, 180.0, "behind_left_car"),
    ]

    for fwd, right, yaw, name in walker_specs:
        a = spawn_static_walker(world, bp_lib, ego_tf, fwd, right, yaw, name)
        if a:
            actors.append(a)

    return actors


def spawn_group_test(world, bp_lib, ego_tf):
    actors = []

    specs = [
        (17.0, -0.9, 180.0, "group_1"),
        (17.5, -0.1, 180.0, "group_2"),
        (18.0, 0.8, 180.0, "group_3"),
        (19.0, 1.5, 180.0, "group_4"),
    ]

    for fwd, right, yaw, name in specs:
        a = spawn_static_walker(world, bp_lib, ego_tf, fwd, right, yaw, name)
        if a:
            actors.append(a)

    return actors


def spawn_crossing_pose_test(world, bp_lib, ego_tf):
    actors = []

    specs = [
        (11.0, -3.8, 90.0, "crossing_left_to_right_1"),
        (13.0, -2.4, 90.0, "crossing_left_to_right_2"),
        (15.0, 3.8, -90.0, "crossing_right_to_left_1"),
        (19.0, 2.8, -90.0, "crossing_right_to_left_2"),
    ]

    for fwd, right, yaw, name in specs:
        a = spawn_static_walker(world, bp_lib, ego_tf, fwd, right, yaw, name)
        if a:
            actors.append(a)

    return actors


def spawn_vehicle_context(world, bp_lib, ego_tf):
    actors = []

    specs = [
        (20.0, -4.8, 0.0, "context_left_vehicle"),
        (28.0, 4.6, 0.0, "context_right_vehicle"),
        (36.0, 0.0, 0.0, "context_far_vehicle"),
    ]

    for fwd, right, yaw, name in specs:
        a = spawn_static_vehicle(world, bp_lib, ego_tf, fwd, right, yaw, name)
        if a:
            actors.append(a)

    return actors


def set_spectator(world, ego_tf):
    spectator = world.get_spectator()

    yaw_rad = math.radians(ego_tf.rotation.yaw)
    back_x = -math.cos(yaw_rad) * 10.0
    back_y = -math.sin(yaw_rad) * 10.0

    loc = carla.Location(
        x=ego_tf.location.x + back_x,
        y=ego_tf.location.y + back_y,
        z=ego_tf.location.z + 7.5,
    )

    rot = carla.Rotation(
        pitch=-35.0,
        yaw=ego_tf.rotation.yaw,
        roll=0.0,
    )

    spectator.set_transform(carla.Transform(loc, rot))


def set_world_runtime(world):
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    world.set_weather(
        carla.WeatherParameters(
            cloudiness=5.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=0.0,
            sun_altitude_angle=60.0,
            sun_azimuth_angle=40.0,
            fog_density=0.0,
            wetness=0.0,
        )
    )


def load_map_if_needed(client, world, map_name):
    current_map = world.get_map().name.split("/")[-1]

    if map_name and current_map != map_name:
        print(f"[MAP] {current_map} -> {map_name} yükleniyor")
        world = client.load_world(map_name)
        time.sleep(3.0)
    else:
        print(f"[MAP] Aktif map: {current_map}")

    return world


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Town03")
    parser.add_argument("--spawn-index", type=int, default=15)
    parser.add_argument(
        "--mode",
        default="all",
        choices=[
            "all",
            "close",
            "medium_far",
            "occlusion",
            "group",
            "crossing",
        ],
    )
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--clear-only", action="store_true")
    parser.add_argument("--keep-alive", action="store_true")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    world = client.get_world()

    if args.clear:
        clear_scene(world)

    if args.clear_only:
        return

    world = load_map_if_needed(client, world, args.map)
    set_world_runtime(world)

    bp_lib = world.get_blueprint_library()

    ego, ego_tf = spawn_ego(world, bp_lib, args.spawn_index)
    camera = spawn_rgb_front_camera(world, bp_lib, ego)

    actors = [ego, camera]

    if args.mode == "close":
        actors += spawn_close_person_test(world, bp_lib, ego_tf)

    elif args.mode == "medium_far":
        actors += spawn_medium_far_person_test(world, bp_lib, ego_tf)

    elif args.mode == "occlusion":
        actors += spawn_occlusion_test(world, bp_lib, ego_tf)

    elif args.mode == "group":
        actors += spawn_group_test(world, bp_lib, ego_tf)

    elif args.mode == "crossing":
        actors += spawn_crossing_pose_test(world, bp_lib, ego_tf)

    elif args.mode == "all":
        actors += spawn_close_person_test(world, bp_lib, ego_tf)
        actors += spawn_occlusion_test(world, bp_lib, ego_tf)
        actors += spawn_group_test(world, bp_lib, ego_tf)
        actors += spawn_crossing_pose_test(world, bp_lib, ego_tf)
        actors += spawn_vehicle_context(world, bp_lib, ego_tf)

    set_spectator(world, ego_tf)

    time.sleep(1.0)

    print("")
    print("======================================================")
    print("ADAS HUMAN STRESS TEST SCENE READY")
    print("======================================================")
    print(f"Map          : {world.get_map().name}")
    print(f"Mode         : {args.mode}")
    print(f"Ego id       : {ego.id}")
    print(f"Camera id    : {camera.id}")
    print(f"Camera role  : rgb_front")
    print(f"Actor count  : {len(actors)}")
    print("")
    print("Bu sahnede beklenen:")
    print("- label=person original=pedestrian çıktısı")
    print("- yakın yayalar yüksek conf")
    print("- orta/uzak yayalar daha düşük conf")
    print("- kısmen kapanan yayalar bazen düşük conf verebilir")
    print("")
    print("Perception ekranında Persons > 0 olmalı.")
    print("Detection JSON içinde label=person original=pedestrian görmelisin.")
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

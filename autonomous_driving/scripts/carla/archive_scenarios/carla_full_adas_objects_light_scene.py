#!/usr/bin/env python3
import argparse
import math
import random
import time
import carla


TAG = "adas_full_light_test"


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

    controllers = [a for a in targets if a.type_id.startswith("controller.ai.walker")]
    sensors = [a for a in targets if a.type_id.startswith("sensor.")]
    walkers = [a for a in targets if a.type_id.startswith("walker.")]
    vehicles = [a for a in targets if a.type_id.startswith("vehicle.")]

    ordered = controllers + sensors + walkers + vehicles

    print(f"[CLEAR] Silinecek actor sayısı: {len(ordered)}")

    for a in ordered:
        safe_destroy(a)

    time.sleep(1.0)


def set_role(bp, name):
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", name)


def choose_bp(bp_lib, filters, fallback=None):
    for f in filters:
        bps = list(bp_lib.filter(f))
        if bps:
            return random.choice(bps)

    if fallback:
        bps = list(bp_lib.filter(fallback))
        if bps:
            return random.choice(bps)

    return None


def fr_location(base_tf, forward_m, right_m, z_add):
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


def make_tf(base_tf, forward_m, right_m, z_add, yaw_delta):
    loc = fr_location(base_tf, forward_m, right_m, z_add)

    return carla.Transform(
        loc,
        carla.Rotation(
            pitch=0.0,
            yaw=base_tf.rotation.yaw + yaw_delta,
            roll=0.0,
        ),
    )


def set_world(world):
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    world.set_weather(
        carla.WeatherParameters(
            cloudiness=5.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            sun_altitude_angle=60.0,
            sun_azimuth_angle=35.0,
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


def state_from_text(light_state):
    s = light_state.lower().strip()

    if s == "red":
        return carla.TrafficLightState.Red

    if s == "yellow":
        return carla.TrafficLightState.Yellow

    if s == "green":
        return carla.TrafficLightState.Green

    raise ValueError(f"Geçersiz light_state: {light_state}")


def set_all_traffic_lights(world, light_state):
    target_state = state_from_text(light_state)

    lights = list(world.get_actors().filter("traffic.traffic_light*"))

    print(f"[LIGHT] Bulunan trafik ışığı sayısı: {len(lights)}")

    for tl in lights:
        try:
            tl.set_state(target_state)
            tl.set_red_time(9999.0)
            tl.set_yellow_time(9999.0)
            tl.set_green_time(9999.0)

            try:
                tl.freeze(True)
            except Exception:
                pass

        except Exception as e:
            print(f"[LIGHT] state set hata id={tl.id}: {e}")

    print(f"[LIGHT] Tüm ışıklar {light_state.upper()} yapıldı.")
    return lights


def get_candidate_ego_transforms_from_lights(world):
    candidates = []
    lights = list(world.get_actors().filter("traffic.traffic_light*"))

    for tl in lights:
        try:
            stop_wps = tl.get_stop_waypoints()
        except Exception:
            stop_wps = []

        for wp in stop_wps:
            base_tf = wp.transform
            f = base_tf.get_forward_vector()

            for back_dist in [14.0, 18.0, 22.0, 26.0]:
                loc = carla.Location(
                    x=base_tf.location.x - f.x * back_dist,
                    y=base_tf.location.y - f.y * back_dist,
                    z=base_tf.location.z + 0.40,
                )

                tf = carla.Transform(
                    loc,
                    carla.Rotation(
                        pitch=0.0,
                        yaw=base_tf.rotation.yaw,
                        roll=0.0,
                    ),
                )

                candidates.append((tl, tf, back_dist))

    return candidates


def spawn_ego_near_traffic_light(world, bp_lib):
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

    if ego_bp is None:
        raise RuntimeError("Ego vehicle blueprint bulunamadı")

    set_role(ego_bp, f"{TAG}_ego")

    candidates = get_candidate_ego_transforms_from_lights(world)
    random.shuffle(candidates)

    for tl, tf, back_dist in candidates:
        ego = world.try_spawn_actor(ego_bp, tf)

        if ego is None:
            continue

        try:
            ego.set_autopilot(False)
            ego.set_simulate_physics(False)
        except Exception:
            pass

        print(f"[SPAWN] Ego OK id={ego.id} near_light_id={tl.id} back_dist={back_dist}")
        return ego, tf, tl

    raise RuntimeError("Trafik ışığına göre ego spawn edilemedi")


def spawn_camera(world, bp_lib, ego):
    cam_bp = bp_lib.find("sensor.camera.rgb")

    cam_bp.set_attribute("image_size_x", "960")
    cam_bp.set_attribute("image_size_y", "540")
    cam_bp.set_attribute("fov", "105")
    cam_bp.set_attribute("sensor_tick", "0.05")

    set_role(cam_bp, "rgb_front")

    cam_tf = carla.Transform(
        carla.Location(x=1.70, y=0.0, z=1.55),
        carla.Rotation(pitch=-3.0, yaw=0.0, roll=0.0),
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

    for bp in walker_bps[:10]:
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        set_role(bp, f"{TAG}_{name}")

        tf = make_tf(ego_tf, forward_m, right_m, 0.55, yaw_delta)
        actor = world.try_spawn_actor(bp, tf)

        if actor is None:
            continue

        try:
            actor.set_simulate_physics(False)
        except Exception:
            pass

        print(f"[SPAWN] Walker OK id={actor.id} name={name} fwd={forward_m} right={right_m}")
        return actor

    print(f"[SPAWN] Walker FAIL name={name}")
    return None


def spawn_vehicle(world, bp_lib, ego_tf, forward_m, right_m, yaw_delta, name):
    bp = choose_bp(
        bp_lib,
        [
            "vehicle.lincoln.mkz_2020",
            "vehicle.tesla.model3",
            "vehicle.audi.tt",
            "vehicle.dodge.charger_2020",
            "vehicle.bmw.grandtourer",
        ],
        "vehicle.*",
    )

    if bp is None:
        print("[SPAWN] Vehicle blueprint yok")
        return None

    set_role(bp, f"{TAG}_{name}")

    tf = make_tf(ego_tf, forward_m, right_m, 0.35, yaw_delta)
    actor = world.try_spawn_actor(bp, tf)

    if actor is None:
        print(f"[SPAWN] Vehicle FAIL name={name}")
        return None

    try:
        actor.set_autopilot(False)
        actor.set_simulate_physics(False)
    except Exception:
        pass

    print(f"[SPAWN] Vehicle OK id={actor.id} name={name} fwd={forward_m} right={right_m}")
    return actor


def spawn_motorcycle(world, bp_lib, ego_tf, forward_m, right_m, yaw_delta, name):
    bp = choose_bp(
        bp_lib,
        [
            "vehicle.kawasaki.ninja",
            "vehicle.yamaha.yzf",
            "vehicle.harley-davidson.low_rider",
            "vehicle.vespa.zx125",
        ],
        None,
    )

    if bp is None:
        print("[SPAWN] Motorcycle blueprint yok")
        return None

    set_role(bp, f"{TAG}_{name}")

    tf = make_tf(ego_tf, forward_m, right_m, 0.35, yaw_delta)
    actor = world.try_spawn_actor(bp, tf)

    if actor is None:
        print(f"[SPAWN] Motorcycle FAIL name={name}")
        return None

    try:
        actor.set_autopilot(False)
        actor.set_simulate_physics(False)
    except Exception:
        pass

    print(f"[SPAWN] Motorcycle OK id={actor.id} type={actor.type_id} name={name} fwd={forward_m} right={right_m}")
    return actor


def set_spectator(world, ego_tf):
    spectator = world.get_spectator()
    yaw = math.radians(ego_tf.rotation.yaw)

    loc = carla.Location(
        x=ego_tf.location.x - math.cos(yaw) * 10.0,
        y=ego_tf.location.y - math.sin(yaw) * 10.0,
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
    parser.add_argument("--map", default="Town03")
    parser.add_argument("--light-state", default="red", choices=["red", "yellow", "green"])
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--clear-only", action="store_true")
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
    set_all_traffic_lights(world, args.light_state)

    bp_lib = world.get_blueprint_library()

    ego, ego_tf, light = spawn_ego_near_traffic_light(world, bp_lib)
    camera = spawn_camera(world, bp_lib, ego)

    actors = [ego, camera]

    walker_specs = [
        (7.0, -2.2, 180.0, "person_left_close"),
        (8.5, 2.2, 180.0, "person_right_close"),
        (11.0, -0.8, 180.0, "person_center_mid"),
    ]

    for fwd, right, yaw_delta, name in walker_specs:
        a = spawn_walker(world, bp_lib, ego_tf, fwd, right, yaw_delta, name)
        if a:
            actors.append(a)

    vehicle_specs = [
        (15.0, 3.8, 0.0, "vehicle_right"),
        (23.0, -4.2, 0.0, "vehicle_left_far"),
    ]

    for fwd, right, yaw_delta, name in vehicle_specs:
        a = spawn_vehicle(world, bp_lib, ego_tf, fwd, right, yaw_delta, name)
        if a:
            actors.append(a)

    moto = spawn_motorcycle(
        world,
        bp_lib,
        ego_tf,
        forward_m=10.5,
        right_m=0.9,
        yaw_delta=0.0,
        name="motorcycle_front",
    )

    if moto:
        actors.append(moto)

    set_spectator(world, ego_tf)

    time.sleep(1.0)

    print("")
    print("======================================================")
    print("ADAS FULL OBJECT + TRAFFIC LIGHT SCENE READY")
    print("======================================================")
    print(f"Map              : {world.get_map().name}")
    print(f"Light state      : {args.light_state.upper()}")
    print(f"Traffic light id : {light.id}")
    print(f"Ego id           : {ego.id}")
    print(f"Camera id        : {camera.id}")
    print(f"Camera role      : rgb_front")
    print(f"Actor count      : {len(actors)}")
    print("")
    print("Beklenen algılar:")
    print("- person / pedestrian")
    print("- vehicle")
    print("- motorcycle")
    print("- traffic_light + HSV state red/yellow/green")
    print("")
    print("Perception ekranında:")
    print("- Persons > 0")
    print("- Vehicles > 0")
    print("- Motorcycles > 0")
    print("- TrafficLight > 0")
    print("- Light State red/yellow/green")
    print("======================================================")
    print("")


if __name__ == "__main__":
    main()

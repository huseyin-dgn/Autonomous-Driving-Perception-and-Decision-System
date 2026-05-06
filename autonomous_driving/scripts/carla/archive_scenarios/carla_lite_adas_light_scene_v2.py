#!/usr/bin/env python3
import argparse
import math
import random
import time
import carla


TAG = "adas_lite_v2"


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
    targets = []

    for a in world.get_actors():
        tid = a.type_id
        r = role(a)

        if r.startswith(TAG) or r == "rgb_front":
            targets.append(a)
        elif tid.startswith("sensor.camera"):
            targets.append(a)
        elif tid.startswith("walker.pedestrian"):
            targets.append(a)
        elif tid.startswith("controller.ai.walker"):
            targets.append(a)
        elif tid.startswith("vehicle.") and r.startswith(TAG):
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


def light_state_from_text(s):
    s = s.lower().strip()

    if s == "red":
        return carla.TrafficLightState.Red

    if s == "yellow":
        return carla.TrafficLightState.Yellow

    if s == "green":
        return carla.TrafficLightState.Green

    raise ValueError(f"Geçersiz light_state: {s}")


def set_all_traffic_lights(world, light_state):
    target = light_state_from_text(light_state)
    lights = list(world.get_actors().filter("traffic.traffic_light*"))

    print(f"[LIGHT] Toplam trafik ışığı: {len(lights)}")
    print(f"[LIGHT] Tüm trafik ışıkları {light_state.upper()} yapılacak")

    for tl in lights:
        try:
            tl.set_state(target)
            tl.set_red_time(9999.0)
            tl.set_yellow_time(9999.0)
            tl.set_green_time(9999.0)

            try:
                tl.freeze(True)
            except Exception:
                pass

        except Exception as e:
            print(f"[LIGHT] set hata id={tl.id}: {e}")

    return lights


def find_light_spawn_candidate(world):
    traffic_lights = list(world.get_actors().filter("traffic.traffic_light*"))
    print(f"[LIGHT] Trafik ışığı sayısı: {len(traffic_lights)}")

    best_candidates = []

    for tl in traffic_lights:
        try:
            stop_wps = tl.get_stop_waypoints()
        except Exception:
            stop_wps = []

        for wp in stop_wps:
            base_tf = wp.transform
            f = base_tf.get_forward_vector()

            for back_dist in [10.0, 12.0, 14.0, 16.0]:
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

                best_candidates.append((tl, tf, back_dist))

    if not best_candidates:
        return None, None

    random.shuffle(best_candidates)
    tl, tf, dist = best_candidates[0]
    print(f"[LIGHT] Seçilen traffic_light id={tl.id} back_dist={dist}")
    return tl, tf


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


def set_world_light(world):
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    world.set_weather(
        carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            sun_altitude_angle=60.0,
            sun_azimuth_angle=35.0,
            fog_density=0.0,
            wetness=0.0,
        )
    )


def spawn_ego(world, bp_lib, ego_tf):
    bp = choose_bp(
        bp_lib,
        [
            "vehicle.tesla.model3",
            "vehicle.lincoln.mkz_2020",
            "vehicle.audi.tt",
        ],
        "vehicle.*",
    )

    if bp is None:
        raise RuntimeError("Ego blueprint bulunamadı")

    set_role(bp, f"{TAG}_ego")

    ego = world.try_spawn_actor(bp, ego_tf)

    if ego is None:
        raise RuntimeError("Ego spawn edilemedi")

    try:
        ego.set_autopilot(False)
        ego.set_simulate_physics(False)
    except Exception:
        pass

    print(f"[SPAWN] Ego OK id={ego.id} type={ego.type_id}")
    return ego


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

    print(f"[SPAWN] Camera OK id={cam.id} role=rgb_front")
    return cam


def try_spawn_at_positions(world, bp, ego_tf, specs, actor_name):
    for forward_m, right_m, z_add, yaw_delta in specs:
        tf = make_tf(ego_tf, forward_m, right_m, z_add, yaw_delta)
        actor = world.try_spawn_actor(bp, tf)

        if actor is not None:
            try:
                if actor.type_id.startswith("vehicle."):
                    actor.set_autopilot(False)
                    actor.set_simulate_physics(False)
                elif actor.type_id.startswith("walker."):
                    actor.set_simulate_physics(False)
            except Exception:
                pass

            print(
                f"[SPAWN] {actor_name} OK id={actor.id} type={actor.type_id} "
                f"fwd={forward_m} right={right_m}"
            )
            return actor

        print(f"[SPAWN] {actor_name} deneme başarısız fwd={forward_m} right={right_m}")

    print(f"[SPAWN] {actor_name} FAIL")
    return None



def spawn_walkers(world, bp_lib, ego_tf):
    bps = list(bp_lib.filter("walker.pedestrian.*"))

    if not bps:
        print("[SPAWN] Walker blueprint yok")
        return []

    random.shuffle(bps)

    person_specs = [
        (7.5, -2.0, 0.55, 180.0, "person_left"),
        (8.5, 0.0, 0.55, 180.0, "person_center"),
        (9.5, 2.0, 0.55, 180.0, "person_right"),
    ]

    spawned = []

    for idx, (forward_m, right_m, z_add, yaw_delta, name) in enumerate(person_specs, 1):
        ok = False

        random.shuffle(bps)

        for bp in bps[:12]:
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")

            set_role(bp, f"{TAG}_{name}")

            tf = make_tf(ego_tf, forward_m, right_m, z_add, yaw_delta)
            actor = world.try_spawn_actor(bp, tf)

            if actor is None:
                continue

            try:
                actor.set_simulate_physics(False)
            except Exception:
                pass

            print(
                f"[SPAWN] Person OK id={actor.id} type={actor.type_id} "
                f"name={name} fwd={forward_m} right={right_m}"
            )

            spawned.append(actor)
            ok = True
            break

        if not ok:
            print(f"[SPAWN] Person FAIL name={name}")

    return spawned



def spawn_vehicles(world, bp_lib, ego_tf):
    vehicle_bps = []

    for f in [
        "vehicle.lincoln.mkz_2020",
        "vehicle.tesla.model3",
        "vehicle.audi.tt",
        "vehicle.dodge.charger_2020",
        "vehicle.bmw.grandtourer",
    ]:
        vehicle_bps.extend(list(bp_lib.filter(f)))

    if not vehicle_bps:
        vehicle_bps = list(bp_lib.filter("vehicle.*"))

    if not vehicle_bps:
        print("[SPAWN] Vehicle blueprint yok")
        return []

    vehicle_specs = [
        (11.5, 3.4, 0.35, 0.0, "vehicle_right_close"),
        (16.0, -3.8, 0.35, 0.0, "vehicle_left_mid"),
    ]

    spawned = []

    for forward_m, right_m, z_add, yaw_delta, name in vehicle_specs:
        ok = False
        random.shuffle(vehicle_bps)

        for bp in vehicle_bps[:12]:
            set_role(bp, f"{TAG}_{name}")

            tf = make_tf(ego_tf, forward_m, right_m, z_add, yaw_delta)
            actor = world.try_spawn_actor(bp, tf)

            if actor is None:
                continue

            try:
                actor.set_autopilot(False)
                actor.set_simulate_physics(False)
            except Exception:
                pass

            print(
                f"[SPAWN] Vehicle OK id={actor.id} type={actor.type_id} "
                f"name={name} fwd={forward_m} right={right_m}"
            )

            spawned.append(actor)
            ok = True
            break

        if not ok:
            print(f"[SPAWN] Vehicle FAIL name={name}")

    return spawned


def spawn_motorcycle(world, bp_lib, ego_tf):
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

    set_role(bp, f"{TAG}_motorcycle")

    specs = [
        (8.5, -0.9, 0.35, 0.0),
        (8.5, 0.9, 0.35, 0.0),
        (9.5, -1.4, 0.35, 0.0),
        (9.5, 1.4, 0.35, 0.0),
    ]

    return try_spawn_at_positions(world, bp, ego_tf, specs, "Motorcycle")


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

    set_world_light(world)

    bp_lib = world.get_blueprint_library()

    tl, ego_tf = find_light_spawn_candidate(world)

    if tl is None or ego_tf is None:
        raise RuntimeError("Uygun trafik ışığı bulunamadı. Town03/Town04 açık olmalı.")

    set_all_traffic_lights(world, args.light_state)

    ego = spawn_ego(world, bp_lib, ego_tf)
    camera = spawn_camera(world, bp_lib, ego)

    actors = [ego, camera]

    persons = spawn_walkers(world, bp_lib, ego_tf)
    actors.extend(persons)

    vehicles = spawn_vehicles(world, bp_lib, ego_tf)
    actors.extend(vehicles)

    motorcycle = spawn_motorcycle(world, bp_lib, ego_tf)
    if motorcycle:
        actors.append(motorcycle)

    set_spectator(world, ego_tf)

    time.sleep(1.0)

    print("")
    print("======================================================")
    print("ADAS LITE V2 OBJECT + TRAFFIC LIGHT SCENE READY")
    print("======================================================")
    print(f"Map              : {world.get_map().name}")
    print(f"Light state      : {args.light_state.upper()}")
    print(f"Traffic light id : {tl.id}")
    print(f"Ego id           : {ego.id}")
    print(f"Camera id        : {camera.id}")
    print(f"Camera role      : rgb_front")
    print(f"Actor count      : {len(actors)}")
    print("")
    print("Beklenen algılar:")
    print("- person > 0")
    print("- vehicle > 0")
    print("- motorcycle > 0")
    print("- traffic_light > 0")
    print("- ACTIVE LIGHT seçilen renk")
    print("======================================================")
    print("")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import math
import random
import time
import carla


TAG = "adas_showcase"


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


def find_camera_base_from_traffic_light(world):
    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    print(f"[LIGHT] Trafik ışığı sayısı: {len(lights)}")

    candidates = []

    for tl in lights:
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
                    z=base_tf.location.z + 1.65,
                )

                tf = carla.Transform(
                    loc,
                    carla.Rotation(
                        pitch=-4.0,
                        yaw=base_tf.rotation.yaw,
                        roll=0.0,
                    ),
                )

                candidates.append((tl, tf, back_dist))

    if not candidates:
        raise RuntimeError("Uygun trafik ışığı/stop waypoint bulunamadı. Town03/Town04 açık olmalı.")

    random.shuffle(candidates)

    tl, cam_tf, dist = candidates[0]
    print(f"[CAMERA] Seçilen traffic_light id={tl.id}, back_dist={dist}")

    return tl, cam_tf


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


def make_actor_tf(camera_tf, forward_m, right_m, z_add, yaw_delta):
    loc = fr_location(camera_tf, forward_m, right_m, z_add)

    return carla.Transform(
        loc,
        carla.Rotation(
            pitch=0.0,
            yaw=camera_tf.rotation.yaw + yaw_delta,
            roll=0.0,
        ),
    )


def choose_bp(bp_lib, patterns):
    for pattern in patterns:
        bps = list(bp_lib.filter(pattern))
        if bps:
            return random.choice(bps)

    return None


def spawn_camera(world, bp_lib, camera_tf):
    cam_bp = bp_lib.find("sensor.camera.rgb")

    cam_bp.set_attribute("image_size_x", "960")
    cam_bp.set_attribute("image_size_y", "540")
    cam_bp.set_attribute("fov", "120")
    cam_bp.set_attribute("sensor_tick", "0.05")

    set_role(cam_bp, "rgb_front")

    cam = world.spawn_actor(cam_bp, camera_tf)

    print(f"[SPAWN] Camera OK id={cam.id} role=rgb_front")
    return cam


def spawn_persons(world, bp_lib, camera_tf):
    bps = list(bp_lib.filter("walker.pedestrian.*"))

    if not bps:
        print("[SPAWN] Person blueprint yok")
        return []

    specs = [
        (6.5, -1.8, -1.05, 180.0, "person_left"),
        (7.5, 0.0, -1.05, 180.0, "person_center"),
        (8.5, 1.8, -1.05, 180.0, "person_right"),
    ]

    actors = []

    for forward_m, right_m, z_add, yaw_delta, name in specs:
        ok = False
        random.shuffle(bps)

        for bp in bps[:15]:
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")

            set_role(bp, f"{TAG}_{name}")
            tf = make_actor_tf(camera_tf, forward_m, right_m, z_add, yaw_delta)

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

            actors.append(actor)
            ok = True
            break

        if not ok:
            print(f"[SPAWN] Person FAIL name={name}")

    return actors


def spawn_vehicles(world, bp_lib, camera_tf):
    patterns = [
        "vehicle.lincoln.mkz_2020",
        "vehicle.tesla.model3",
        "vehicle.dodge.charger_2020",
        "vehicle.audi.tt",
        "vehicle.bmw.grandtourer",
    ]

    bps = []
    for p in patterns:
        bps.extend(list(bp_lib.filter(p)))

    if not bps:
        bps = list(bp_lib.filter("vehicle.*"))

    if not bps:
        print("[SPAWN] Vehicle blueprint yok")
        return []

    specs = [
        (18.0, -4.5, -1.25, 0.0, "vehicle_far_left"),
        (20.0, 4.5, -1.25, 0.0, "vehicle_far_right"),
    ]

    actors = []

    for forward_m, right_m, z_add, yaw_delta, name in specs:
        ok = False
        random.shuffle(bps)

        for bp in bps[:15]:
            set_role(bp, f"{TAG}_{name}")
            tf = make_actor_tf(camera_tf, forward_m, right_m, z_add, yaw_delta)

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

            actors.append(actor)
            ok = True
            break

        if not ok:
            print(f"[SPAWN] Vehicle FAIL name={name}")

    return actors


def spawn_motorcycle(world, bp_lib, camera_tf):
    bp = choose_bp(
        bp_lib,
        [
            "vehicle.kawasaki.ninja",
            "vehicle.yamaha.yzf",
            "vehicle.harley-davidson.low_rider",
            "vehicle.vespa.zx125",
        ],
    )

    if bp is None:
        print("[SPAWN] Motorcycle blueprint yok")
        return None

    set_role(bp, f"{TAG}_motorcycle")

    specs = [
        (9.5, -0.6, -1.25, 0.0),
        (9.5, 0.6, -1.25, 0.0),
        (10.5, -1.0, -1.25, 0.0),
        (10.5, 1.0, -1.25, 0.0),
    ]

    for forward_m, right_m, z_add, yaw_delta in specs:
        tf = make_actor_tf(camera_tf, forward_m, right_m, z_add, yaw_delta)

        actor = world.try_spawn_actor(bp, tf)

        if actor is None:
            print(f"[SPAWN] Motorcycle deneme başarısız fwd={forward_m} right={right_m}")
            continue

        try:
            actor.set_autopilot(False)
            actor.set_simulate_physics(False)
        except Exception:
            pass

        print(
            f"[SPAWN] Motorcycle OK id={actor.id} type={actor.type_id} "
            f"fwd={forward_m} right={right_m}"
        )

        return actor

    print("[SPAWN] Motorcycle FAIL")
    return None


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
            sun_altitude_angle=60.0,
            sun_azimuth_angle=35.0,
            fog_density=0.0,
            wetness=0.0,
        )
    )


def set_spectator(world, camera_tf):
    spectator = world.get_spectator()

    yaw = math.radians(camera_tf.rotation.yaw)

    loc = carla.Location(
        x=camera_tf.location.x - math.cos(yaw) * 3.0,
        y=camera_tf.location.y - math.sin(yaw) * 3.0,
        z=camera_tf.location.z + 5.0,
    )

    rot = carla.Rotation(
        pitch=-35.0,
        yaw=camera_tf.rotation.yaw,
        roll=0.0,
    )

    spectator.set_transform(carla.Transform(loc, rot))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--light-state", default="yellow", choices=["red", "yellow", "green"])
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

    set_world(world)
    set_all_traffic_lights(world, args.light_state)

    bp_lib = world.get_blueprint_library()

    selected_light, camera_tf = find_camera_base_from_traffic_light(world)

    camera = spawn_camera(world, bp_lib, camera_tf)

    actors = [camera]

    persons = spawn_persons(world, bp_lib, camera_tf)
    actors.extend(persons)

    vehicles = spawn_vehicles(world, bp_lib, camera_tf)
    actors.extend(vehicles)

    motorcycle = spawn_motorcycle(world, bp_lib, camera_tf)
    if motorcycle:
        actors.append(motorcycle)

    set_spectator(world, camera_tf)

    time.sleep(1.0)

    print("")
    print("======================================================")
    print("ADAS SHOWCASE ALL OBJECTS SCENE READY")
    print("======================================================")
    print(f"Map              : {world.get_map().name}")
    print(f"Light state      : {args.light_state.upper()}")
    print(f"Selected light   : {selected_light.id}")
    print(f"Camera id        : {camera.id}")
    print(f"Camera role      : rgb_front")
    print(f"Actor count      : {len(actors)}")
    print("")
    print("Beklenen algılar:")
    print("- person >= 1")
    print("- vehicle >= 1")
    print("- motorcycle >= 1")
    print("- traffic_light >= 1")
    print("- ACTIVE LIGHT seçilen renk")
    print("======================================================")
    print("")


if __name__ == "__main__":
    main()

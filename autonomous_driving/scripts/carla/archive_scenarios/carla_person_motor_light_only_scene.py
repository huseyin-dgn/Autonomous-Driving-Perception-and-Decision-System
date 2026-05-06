#!/usr/bin/env python3
import argparse
import math
import random
import time
import carla


TAG = "adas_person_motor_light_only"

OLD_TAG_PREFIXES = [
    "adas_showcase",
    "adas_lite_v2",
    "adas_lite_scene",
    "adas_full_light_test",
    "adas_person_light_only",
    "adas_person_motor_light_only",
]


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

        if r == "rgb_front":
            targets.append(a)
        elif any(r.startswith(prefix) for prefix in OLD_TAG_PREFIXES):
            targets.append(a)
        elif tid.startswith("sensor.camera"):
            targets.append(a)
        elif tid.startswith("walker.pedestrian"):
            targets.append(a)
        elif tid.startswith("controller.ai.walker"):
            targets.append(a)
        elif tid.startswith("vehicle.") and any(r.startswith(prefix) for prefix in OLD_TAG_PREFIXES):
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
            sun_azimuth_angle=35.0,
            fog_density=0.0,
            wetness=0.0,
        )
    )


def find_camera_transform_near_light(world):
    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    candidates = []

    for tl in lights:
        try:
            stop_wps = tl.get_stop_waypoints()
        except Exception:
            stop_wps = []

        for wp in stop_wps:
            base_tf = wp.transform
            f = base_tf.get_forward_vector()

            for back_dist in [7.0, 8.5, 10.0, 12.0]:
                loc = carla.Location(
                    x=base_tf.location.x - f.x * back_dist,
                    y=base_tf.location.y - f.y * back_dist,
                    z=base_tf.location.z + 1.65,
                )

                cam_tf = carla.Transform(
                    loc,
                    carla.Rotation(
                        pitch=-4.0,
                        yaw=base_tf.rotation.yaw,
                        roll=0.0,
                    ),
                )

                candidates.append((tl, cam_tf, back_dist))

    if not candidates:
        raise RuntimeError("Uygun trafik ışığı/stop waypoint bulunamadı. Town03 veya Town04 açık olmalı.")

    random.shuffle(candidates)

    tl, cam_tf, dist = candidates[0]
    print(f"[CAMERA] Seçilen traffic_light id={tl.id}, back_dist={dist}")

    return tl, cam_tf


def fr_location(base_tf, forward_m, right_m):
    yaw = math.radians(base_tf.rotation.yaw)

    fx = math.cos(yaw)
    fy = math.sin(yaw)

    rx = math.cos(yaw + math.pi / 2.0)
    ry = math.sin(yaw + math.pi / 2.0)

    return carla.Location(
        x=base_tf.location.x + fx * forward_m + rx * right_m,
        y=base_tf.location.y + fy * forward_m + ry * right_m,
        z=base_tf.location.z,
    )


def project_to_road_transform(world, camera_tf, forward_m, right_m, z_add, yaw_delta):
    raw_loc = fr_location(camera_tf, forward_m, right_m)

    wp = world.get_map().get_waypoint(
        raw_loc,
        project_to_road=True,
        lane_type=carla.LaneType.Any,
    )

    if wp is not None:
        loc = carla.Location(
            x=wp.transform.location.x,
            y=wp.transform.location.y,
            z=wp.transform.location.z + z_add,
        )

        yaw = wp.transform.rotation.yaw + yaw_delta
    else:
        loc = carla.Location(
            x=raw_loc.x,
            y=raw_loc.y,
            z=camera_tf.location.z - 1.25 + z_add,
        )

        yaw = camera_tf.rotation.yaw + yaw_delta

    return carla.Transform(
        loc,
        carla.Rotation(
            pitch=0.0,
            yaw=yaw,
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
    cam_bp.set_attribute("image_size_y", "720")
    cam_bp.set_attribute("fov", "75")
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
        (5.8, -1.5, 0.15, 180.0, "person_left"),
        (6.8, 1.5, 0.15, 180.0, "person_right"),
    ]

    actors = []

    for forward_m, right_m, z_add, yaw_delta, name in specs:
        ok = False
        random.shuffle(bps)

        for bp in bps[:20]:
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")

            set_role(bp, f"{TAG}_{name}")

            tf = project_to_road_transform(world, camera_tf, forward_m, right_m, z_add, yaw_delta)
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



def spawn_motorcycle(world, bp_lib, camera_tf):
    motorcycle_bps = []

    for pattern in [
        "vehicle.kawasaki.ninja",
        "vehicle.yamaha.yzf",
        "vehicle.harley-davidson.low_rider",
        "vehicle.vespa.zx125",
    ]:
        motorcycle_bps.extend(list(bp_lib.filter(pattern)))

    if not motorcycle_bps:
        print("[SPAWN] Motorcycle blueprint yok")
        return None

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    target_specs = [
        (7.0, 0.0, 0.65, 0.0),
        (7.5, -0.7, 0.65, 0.0),
        (7.5, 0.7, 0.65, 0.0),
        (8.0, -1.0, 0.65, 0.0),
        (8.0, 1.0, 0.65, 0.0),
    ]

    random.shuffle(motorcycle_bps)

    for bp in motorcycle_bps:
        set_role(bp, f"{TAG}_motorcycle")

        actor = None

        for sp in spawn_points[:30]:
            safe_tf = carla.Transform(
                carla.Location(
                    x=sp.location.x,
                    y=sp.location.y,
                    z=sp.location.z + 0.50,
                ),
                sp.rotation,
            )

            actor = world.try_spawn_actor(bp, safe_tf)

            if actor is not None:
                break

        if actor is None:
            print(f"[SPAWN] Motorcycle safe spawn başarısız bp={bp.id}")
            continue

        try:
            actor.set_autopilot(False)
            actor.set_simulate_physics(False)
        except Exception:
            pass

        for forward_m, right_m, z_add, yaw_delta in target_specs:
            target_tf = project_to_road_transform(
                world,
                camera_tf,
                forward_m,
                right_m,
                z_add,
                yaw_delta,
            )

            try:
                actor.set_transform(target_tf)
                time.sleep(0.2)

                print(
                    f"[SPAWN] Motorcycle OK id={actor.id} type={actor.type_id} "
                    f"fwd={forward_m} right={right_m}"
                )

                return actor

            except Exception as e:
                print(f"[SPAWN] Motorcycle teleport hata: {e}")

        safe_destroy(actor)

    print("[SPAWN] Motorcycle FAIL")
    return None


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

    set_world(world)
    set_all_traffic_lights(world, args.light_state)

    bp_lib = world.get_blueprint_library()

    selected_light, camera_tf = find_camera_transform_near_light(world)

    camera = spawn_camera(world, bp_lib, camera_tf)

    actors = [camera]

    persons = spawn_persons(world, bp_lib, camera_tf)
    actors.extend(persons)

    motorcycle = spawn_motorcycle(world, bp_lib, camera_tf)
    if motorcycle:
        actors.append(motorcycle)

    set_spectator(world, camera_tf)

    time.sleep(1.0)

    print("")
    print("======================================================")
    print("ADAS PERSON + MOTORCYCLE + TRAFFIC LIGHT SCENE READY")
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
    print("- motorcycle >= 1")
    print("- traffic_light >= 1")
    print("- ACTIVE LIGHT seçilen renk")
    print("======================================================")
    print("")


if __name__ == "__main__":
    main()

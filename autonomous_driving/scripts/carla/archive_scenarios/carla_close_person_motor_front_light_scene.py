#!/usr/bin/env python3
import argparse
import math
import random
import time
import queue

import cv2
import numpy as np
import carla


TAG = "adas_close_person_motor_front_light"

OLD_PREFIXES = [
    "adas_showcase",
    "adas_lite",
    "adas_full_light_test",
    "adas_person_light_only",
    "adas_person_motor_light_only",
    "adas_front_light_person_motor",
    "adas_close_person_motor_front_light",
]


def role(actor):
    try:
        return actor.attributes.get("role_name", "")
    except Exception:
        return ""


def destroy(actor):
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

        if tid.startswith("sensor.camera"):
            targets.append(a)
        elif any(r.startswith(p) for p in OLD_PREFIXES):
            targets.append(a)
        elif tid.startswith("walker.pedestrian") and any(r.startswith(p) for p in OLD_PREFIXES):
            targets.append(a)
        elif tid.startswith("vehicle.") and any(r.startswith(p) for p in OLD_PREFIXES):
            targets.append(a)

    ordered = (
        [a for a in targets if a.type_id.startswith("sensor.")] +
        [a for a in targets if a.type_id.startswith("walker.")] +
        [a for a in targets if a.type_id.startswith("vehicle.")]
    )

    print(f"[CLEAR] Silinecek actor sayısı: {len(ordered)}")

    for a in ordered:
        destroy(a)

    time.sleep(1.0)


def set_role(bp, name):
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", name)


def set_weather(world):
    world.set_weather(
        carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            sun_altitude_angle=70.0,
            sun_azimuth_angle=35.0,
            fog_density=0.0,
            wetness=0.0,
        )
    )


def set_traffic_lights(world, state_text):
    state_text = state_text.lower().strip()

    if state_text == "red":
        state = carla.TrafficLightState.Red
    elif state_text == "yellow":
        state = carla.TrafficLightState.Yellow
    elif state_text == "green":
        state = carla.TrafficLightState.Green
    else:
        raise ValueError("light-state red/yellow/green olmalı")

    lights = list(world.get_actors().filter("traffic.traffic_light*"))

    print(f"[LIGHT] Toplam trafik ışığı: {len(lights)}")
    print(f"[LIGHT] Tüm trafik ışıkları {state_text.upper()} yapılacak")

    for tl in lights:
        try:
            tl.set_state(state)
            tl.set_red_time(9999.0)
            tl.set_yellow_time(9999.0)
            tl.set_green_time(9999.0)
            try:
                tl.freeze(True)
            except Exception:
                pass
        except Exception as e:
            print(f"[LIGHT] hata id={tl.id}: {e}")

    return lights


def look_at(src, dst):
    dx = dst.x - src.x
    dy = dst.y - src.y
    dz = dst.z - src.z

    yaw = math.degrees(math.atan2(dy, dx))
    dist = math.sqrt(dx * dx + dy * dy)
    pitch = -math.degrees(math.atan2(dz, dist))

    return carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)


def image_to_bgr(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr[:, :, :3].copy()


def color_score(frame, target):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    if target == "red":
        mask = (
            (((h >= 0) & (h <= 12)) | ((h >= 168) & (h <= 179))) &
            (s >= 80) &
            (v >= 100)
        )
    elif target == "yellow":
        mask = (
            (h >= 18) &
            (h <= 38) &
            (s >= 90) &
            (v >= 130)
        )
    elif target == "green":
        mask = (
            (h >= 42) &
            (h <= 95) &
            (s >= 60) &
            (v >= 80)
        )
    else:
        return 0

    hh, ww = mask.shape[:2]

    roi = mask[
        int(hh * 0.05):int(hh * 0.70),
        int(ww * 0.05):int(ww * 0.95),
    ]

    return int(roi.sum())


def capture_probe(world, bp_lib, tf, target):
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "640")
    cam_bp.set_attribute("image_size_y", "480")
    cam_bp.set_attribute("fov", "65")
    cam_bp.set_attribute("sensor_tick", "0.05")
    set_role(cam_bp, f"{TAG}_probe")

    q = queue.Queue()
    cam = world.try_spawn_actor(cam_bp, tf)

    if cam is None:
        return 0

    cam.listen(q.put)

    frame = None

    try:
        for _ in range(20):
            try:
                image = q.get(timeout=0.5)
                frame = image_to_bgr(image)
                break
            except queue.Empty:
                pass
    finally:
        try:
            cam.stop()
        except Exception:
            pass
        destroy(cam)

    if frame is None:
        return 0

    return color_score(frame, target)


def select_front_light_camera(world, bp_lib, target_color):
    lights = list(world.get_actors().filter("traffic.traffic_light*"))

    if not lights:
        raise RuntimeError("Trafik ışığı bulunamadı.")

    candidates = []

    for tl in lights:
        loc = tl.get_transform().location
        target = carla.Location(loc.x, loc.y, loc.z + 2.4)

        for radius in [7.0, 10.0, 13.0]:
            for angle_deg in range(0, 360, 45):
                a = math.radians(angle_deg)

                cam_loc = carla.Location(
                    x=loc.x + math.cos(a) * radius,
                    y=loc.y + math.sin(a) * radius,
                    z=loc.z + 1.7,
                )

                rot = look_at(cam_loc, target)
                cam_tf = carla.Transform(cam_loc, rot)

                candidates.append((tl, cam_tf, radius, angle_deg))

    random.shuffle(candidates)

    best = None
    best_score = -1

    for i, (tl, cam_tf, radius, angle_deg) in enumerate(candidates[:70], 1):
        score = capture_probe(world, bp_lib, cam_tf, target_color)

        print(
            f"[PROBE] {i:02d} tl={tl.id} radius={radius} angle={angle_deg} "
            f"target={target_color} score={score}"
        )

        if score > best_score:
            best_score = score
            best = (tl, cam_tf, radius, angle_deg, score)

    if best is None:
        raise RuntimeError("Uygun kamera açısı bulunamadı.")

    tl, cam_tf, radius, angle_deg, score = best

    print(
        f"[CAMERA] BEST tl={tl.id} radius={radius} angle={angle_deg} "
        f"target={target_color} score={score}"
    )

    return tl, cam_tf


def spawn_camera(world, bp_lib, tf):
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "960")
    cam_bp.set_attribute("image_size_y", "720")
    cam_bp.set_attribute("fov", "65")
    cam_bp.set_attribute("sensor_tick", "0.05")
    set_role(cam_bp, "rgb_front")

    cam = world.spawn_actor(cam_bp, tf)
    print(f"[SPAWN] Camera OK id={cam.id} role=rgb_front size=960x720")
    return cam


def front_location(cam_tf, forward_m, right_m, z):
    yaw = math.radians(cam_tf.rotation.yaw)

    fx = math.cos(yaw)
    fy = math.sin(yaw)

    rx = math.cos(yaw + math.pi / 2.0)
    ry = math.sin(yaw + math.pi / 2.0)

    return carla.Location(
        x=cam_tf.location.x + fx * forward_m + rx * right_m,
        y=cam_tf.location.y + fy * forward_m + ry * right_m,
        z=z,
    )


def ground_transform(world, cam_tf, forward_m, right_m, yaw_delta, z_add):
    rough = front_location(cam_tf, forward_m, right_m, cam_tf.location.z - 1.5)

    wp = world.get_map().get_waypoint(
        rough,
        project_to_road=True,
        lane_type=carla.LaneType.Any,
    )

    if wp is not None:
        loc = carla.Location(
            wp.transform.location.x,
            wp.transform.location.y,
            wp.transform.location.z + z_add,
        )
    else:
        loc = carla.Location(rough.x, rough.y, rough.z + z_add)

    yaw = cam_tf.rotation.yaw + yaw_delta

    return carla.Transform(
        loc,
        carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0),
    )


def spawn_walker(world, bp_lib, cam_tf, forward_m, right_m, name):
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
    random.shuffle(walker_bps)

    tf = ground_transform(world, cam_tf, forward_m, right_m, 180.0, 0.15)

    for bp in walker_bps[:30]:
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        set_role(bp, f"{TAG}_{name}")

        nav_loc = world.get_random_location_from_navigation()

        if nav_loc is not None:
            spawn_tf = carla.Transform(nav_loc, carla.Rotation())
            actor = world.try_spawn_actor(bp, spawn_tf)
        else:
            actor = world.try_spawn_actor(bp, tf)

        if actor is None:
            continue

        try:
            actor.set_simulate_physics(False)
        except Exception:
            pass

        actor.set_transform(tf)

        print(f"[SPAWN] Person OK id={actor.id} type={actor.type_id} name={name}")
        return actor

    print(f"[SPAWN] Person FAIL name={name}")
    return None


def spawn_motorcycle(world, bp_lib, cam_tf):
    patterns = [
        "vehicle.kawasaki.ninja",
        "vehicle.yamaha.yzf",
        "vehicle.harley-davidson.low_rider",
        "vehicle.vespa.zx125",
    ]

    bps = []

    for p in patterns:
        bps.extend(list(bp_lib.filter(p)))

    if not bps:
        print("[SPAWN] Motorcycle blueprint yok.")
        return None

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    random.shuffle(bps)

    target_tf = ground_transform(world, cam_tf, 6.5, 0.0, 90.0, 0.55)

    for bp in bps:
        set_role(bp, f"{TAG}_motorcycle")

        for sp in spawn_points[:60]:
            spawn_tf = carla.Transform(
                carla.Location(sp.location.x, sp.location.y, sp.location.z + 0.5),
                sp.rotation,
            )

            actor = world.try_spawn_actor(bp, spawn_tf)

            if actor is None:
                continue

            try:
                actor.set_autopilot(False)
                actor.set_simulate_physics(False)
            except Exception:
                pass

            actor.set_transform(target_tf)

            print(f"[SPAWN] Motorcycle OK id={actor.id} type={actor.type_id}")
            return actor

    print("[SPAWN] Motorcycle FAIL")
    return None


def set_spectator(world, cam_tf):
    spectator = world.get_spectator()

    yaw = math.radians(cam_tf.rotation.yaw)

    loc = carla.Location(
        x=cam_tf.location.x - math.cos(yaw) * 3.0,
        y=cam_tf.location.y - math.sin(yaw) * 3.0,
        z=cam_tf.location.z + 5.0,
    )

    rot = carla.Rotation(pitch=-35.0, yaw=cam_tf.rotation.yaw, roll=0.0)
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

    set_weather(world)

    bp_lib = world.get_blueprint_library()

    set_traffic_lights(world, args.light_state)

    selected_light, cam_tf = select_front_light_camera(world, bp_lib, args.light_state)

    camera = spawn_camera(world, bp_lib, cam_tf)

    actors = [camera]

    p1 = spawn_walker(world, bp_lib, cam_tf, 4.5, -1.2, "person_left")
    p2 = spawn_walker(world, bp_lib, cam_tf, 5.5, 1.4, "person_right")
    moto = spawn_motorcycle(world, bp_lib, cam_tf)

    for a in [p1, p2, moto]:
        if a is not None:
            actors.append(a)

    set_spectator(world, cam_tf)

    time.sleep(1.0)

    print("")
    print("====================================================")
    print("ADAS CLOSE PERSON + MOTOR + FRONT LIGHT READY")
    print("====================================================")
    print(f"Map            : {world.get_map().name}")
    print(f"Light state    : {args.light_state.upper()}")
    print(f"Selected light : {selected_light.id}")
    print(f"Camera id      : {camera.id}")
    print(f"Camera role    : rgb_front")
    print(f"Image size     : 960x720")
    print(f"Actor count    : {len(actors)}")
    print("")
    print("Beklenen:")
    print("- traffic_light red/yellow/green veya net değilse unknown")
    print("- person >= 1")
    print("- motorcycle >= 1")
    print("====================================================")
    print("")


if __name__ == "__main__":
    main()

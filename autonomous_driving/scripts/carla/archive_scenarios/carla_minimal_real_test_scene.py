#!/usr/bin/env python3
import argparse
import math
import random
import time
import carla

TAG = "adas_minimal_real_test"


def role(a):
    try:
        return a.attributes.get("role_name", "")
    except Exception:
        return ""


def clear_scene(world):
    targets = []
    for a in world.get_actors():
        r = role(a)
        if a.type_id.startswith("sensor.camera"):
            targets.append(a)
        elif r.startswith(TAG):
            targets.append(a)
        elif a.type_id.startswith("vehicle.") and r.startswith(TAG):
            targets.append(a)
        elif a.type_id.startswith("walker.pedestrian") and r.startswith(TAG):
            targets.append(a)

    print(f"[CLEAR] actor={len(targets)}")
    for a in targets:
        print(f"[CLEAR] destroy id={a.id} type={a.type_id} role={role(a)}")
        a.destroy()
    time.sleep(1.0)


def set_role(bp, name):
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", name)


def set_weather(world):
    world.set_weather(
        carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            sun_altitude_angle=70.0,
            sun_azimuth_angle=30.0,
            fog_density=0.0,
            wetness=0.0,
        )
    )


def set_lights(world, state_text):
    if state_text == "red":
        state = carla.TrafficLightState.Red
    elif state_text == "yellow":
        state = carla.TrafficLightState.Yellow
    elif state_text == "green":
        state = carla.TrafficLightState.Green
    else:
        raise RuntimeError("red/yellow/green olmalı")

    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    print(f"[LIGHT] total={len(lights)} state={state_text}")

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
            print("[LIGHT] hata:", e)

    return lights


def look_at(src, dst):
    dx = dst.x - src.x
    dy = dst.y - src.y
    dz = dst.z - src.z
    yaw = math.degrees(math.atan2(dy, dx))
    dist = math.sqrt(dx * dx + dy * dy)
    pitch = -math.degrees(math.atan2(dz, dist))
    return carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)


def norm_vec(dx, dy):
    d = math.sqrt(dx * dx + dy * dy)
    if d < 0.001:
        return 1.0, 0.0
    return dx / d, dy / d


def find_best_light_and_camera(world):
    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    candidates = []

    for tl in lights:
        tl_loc = tl.get_transform().location

        try:
            stop_wps = tl.get_stop_waypoints()
        except Exception:
            stop_wps = []

        for wp in stop_wps:
            stop_loc = wp.transform.location
            dx = tl_loc.x - stop_loc.x
            dy = tl_loc.y - stop_loc.y
            dist = math.sqrt(dx * dx + dy * dy)

            if 4.0 <= dist <= 30.0:
                candidates.append((dist, tl, wp))

    if not candidates:
        raise RuntimeError("Uygun trafik ışığı + stop waypoint bulunamadı.")

    candidates = sorted(candidates, key=lambda x: x[0])
    dist, light, stop_wp = candidates[len(candidates) // 2]

    light_loc = light.get_transform().location
    stop_loc = stop_wp.transform.location

    fx, fy = norm_vec(light_loc.x - stop_loc.x, light_loc.y - stop_loc.y)

    cam_loc = carla.Location(
        x=stop_loc.x - fx * 7.0,
        y=stop_loc.y - fy * 7.0,
        z=stop_loc.z + 1.7,
    )

    target = carla.Location(
        x=light_loc.x,
        y=light_loc.y,
        z=light_loc.z + 2.3,
    )

    cam_rot = look_at(cam_loc, target)
    cam_tf = carla.Transform(cam_loc, cam_rot)

    print(f"[SELECT] light_id={light.id} dist={dist:.2f}")
    print(f"[SELECT] stop=({stop_loc.x:.1f},{stop_loc.y:.1f},{stop_loc.z:.1f})")
    print(f"[SELECT] light=({light_loc.x:.1f},{light_loc.y:.1f},{light_loc.z:.1f})")
    print(f"[SELECT] cam=({cam_loc.x:.1f},{cam_loc.y:.1f},{cam_loc.z:.1f}) yaw={cam_rot.yaw:.1f}")

    return light, stop_wp, cam_tf, fx, fy


def spawn_camera(world, bp_lib, cam_tf):
    bp = bp_lib.find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", "960")
    bp.set_attribute("image_size_y", "720")
    bp.set_attribute("fov", "70")
    bp.set_attribute("sensor_tick", "0.05")
    set_role(bp, "rgb_front")

    cam = world.spawn_actor(bp, cam_tf)
    print(f"[SPAWN] camera id={cam.id} role=rgb_front 960x720")
    return cam


def make_tf_ahead(cam_tf, fx, fy, forward, side, z_base, yaw_offset):
    rx = -fy
    ry = fx

    loc = carla.Location(
        x=cam_tf.location.x + fx * forward + rx * side,
        y=cam_tf.location.y + fy * forward + ry * side,
        z=z_base,
    )

    yaw = math.degrees(math.atan2(fy, fx)) + yaw_offset

    return carla.Transform(
        loc,
        carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0),
    )


def spawn_person(world, bp_lib, cam_tf, fx, fy, z_base):
    bps = list(bp_lib.filter("walker.pedestrian.*"))
    random.shuffle(bps)

    tf = make_tf_ahead(cam_tf, fx, fy, forward=4.5, side=-1.0, z_base=z_base + 0.15, yaw_offset=180.0)

    for bp in bps:
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        set_role(bp, f"{TAG}_person")

        actor = world.try_spawn_actor(bp, tf)
        if actor is None:
            continue

        try:
            actor.set_simulate_physics(False)
        except Exception:
            pass

        actor.set_transform(tf)
        print(f"[SPAWN] person id={actor.id} type={actor.type_id}")
        return actor

    print("[SPAWN] person FAIL")
    return None


def spawn_motorcycle(world, bp_lib, cam_tf, fx, fy, z_base):
    names = [
        "vehicle.kawasaki.ninja",
        "vehicle.yamaha.yzf",
        "vehicle.harley-davidson.low_rider",
        "vehicle.vespa.zx125",
    ]

    bps = []
    for n in names:
        bps += list(bp_lib.filter(n))

    if not bps:
        print("[SPAWN] motorcycle blueprint yok")
        return None

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    target_tf = make_tf_ahead(cam_tf, fx, fy, forward=5.2, side=1.0, z_base=z_base + 0.55, yaw_offset=90.0)

    for bp in bps:
        set_role(bp, f"{TAG}_motorcycle")

        for sp in spawn_points[:80]:
            tmp = carla.Transform(
                carla.Location(sp.location.x, sp.location.y, sp.location.z + 0.5),
                sp.rotation,
            )

            actor = world.try_spawn_actor(bp, tmp)
            if actor is None:
                continue

            try:
                actor.set_autopilot(False)
                actor.set_simulate_physics(False)
            except Exception:
                pass

            actor.set_transform(target_tf)
            print(f"[SPAWN] motorcycle id={actor.id} type={actor.type_id}")
            return actor

    print("[SPAWN] motorcycle FAIL")
    return None


def set_spectator(world, cam_tf):
    spectator = world.get_spectator()
    spectator.set_transform(
        carla.Transform(
            carla.Location(
                cam_tf.location.x,
                cam_tf.location.y,
                cam_tf.location.z + 8.0,
            ),
            carla.Rotation(pitch=-60.0, yaw=cam_tf.rotation.yaw, roll=0.0),
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--light-state", choices=["red", "yellow", "green"], default="red")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--clear-only", action="store_true")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    if args.clear:
        clear_scene(world)

    if args.clear_only:
        return

    set_weather(world)
    set_lights(world, args.light_state)

    light, stop_wp, cam_tf, fx, fy = find_best_light_and_camera(world)

    camera = spawn_camera(world, bp_lib, cam_tf)

    z_base = stop_wp.transform.location.z

    person = spawn_person(world, bp_lib, cam_tf, fx, fy, z_base)
    motorcycle = spawn_motorcycle(world, bp_lib, cam_tf, fx, fy, z_base)

    set_spectator(world, cam_tf)

    time.sleep(1.0)

    print("")
    print("====================================================")
    print("MINIMAL REAL TEST READY")
    print("====================================================")
    print(f"Light state : {args.light_state.upper()}")
    print(f"Camera      : rgb_front 960x720")
    print(f"Person      : {'OK' if person else 'FAIL'}")
    print(f"Motorcycle  : {'OK' if motorcycle else 'FAIL'}")
    print("====================================================")


if __name__ == "__main__":
    main()

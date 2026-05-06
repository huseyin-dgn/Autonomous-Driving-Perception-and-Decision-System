#!/usr/bin/env python3
import argparse
import math
import random
import time
import carla

TAG = "adas_clean_visible_test"


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

    print(f"[CLEAR] actor={len(targets)}")
    for a in targets:
        print(f"[CLEAR] destroy id={a.id} type={a.type_id} role={role(a)}")
        try:
            a.destroy()
        except Exception as e:
            print("[CLEAR] hata:", e)
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
            sun_azimuth_angle=20.0,
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
    print(f"[LIGHT] total={len(lights)} state={state_text.upper()}")

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
        except Exception:
            pass

    return lights


def look_at(src, dst):
    dx = dst.x - src.x
    dy = dst.y - src.y
    dz = dst.z - src.z
    yaw = math.degrees(math.atan2(dy, dx))
    dist = math.sqrt(dx * dx + dy * dy)
    pitch = -math.degrees(math.atan2(dz, dist))
    return carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)


def pick_good_light(world):
    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    if not lights:
        raise RuntimeError("Trafik ışığı yok.")

    candidates = []

    for tl in lights:
        loc = tl.get_transform().location
        if loc.z < -1:
            continue

        score = 0

        try:
            stop_wps = tl.get_stop_waypoints()
            score += len(stop_wps) * 100
        except Exception:
            stop_wps = []

        score -= abs(loc.z - 5.0) * 3
        score -= abs(loc.x) * 0.01
        score -= abs(loc.y) * 0.01

        candidates.append((score, tl, stop_wps))

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    tl = candidates[0][1]
    stop_wps = candidates[0][2]

    loc = tl.get_transform().location
    print(f"[LIGHT] selected id={tl.id} loc=({loc.x:.1f},{loc.y:.1f},{loc.z:.1f}) stop_wps={len(stop_wps)}")

    return tl, stop_wps


def camera_from_light(light, stop_wps, side):
    light_tf = light.get_transform()
    light_loc = light_tf.location

    target = carla.Location(
        x=light_loc.x,
        y=light_loc.y,
        z=light_loc.z + 2.1,
    )

    if stop_wps:
        stop_loc = stop_wps[0].transform.location
        dx = stop_loc.x - light_loc.x
        dy = stop_loc.y - light_loc.y
        d = math.sqrt(dx * dx + dy * dy)

        if d > 0.1:
            ux = dx / d
            uy = dy / d
        else:
            fv = light_tf.get_forward_vector()
            ux = -fv.x
            uy = -fv.y
    else:
        fv = light_tf.get_forward_vector()
        ux = -fv.x
        uy = -fv.y

    cam_loc = carla.Location(
        x=light_loc.x + ux * 18.0 * side,
        y=light_loc.y + uy * 18.0 * side,
        z=max(1.7, light_loc.z + 1.2),
    )

    cam_rot = look_at(cam_loc, target)

    return carla.Transform(cam_loc, cam_rot)


def basis(cam_tf):
    yaw = math.radians(cam_tf.rotation.yaw)
    fx = math.cos(yaw)
    fy = math.sin(yaw)
    rx = math.cos(yaw + math.pi / 2.0)
    ry = math.sin(yaw + math.pi / 2.0)
    return fx, fy, rx, ry


def visible_tf(cam_tf, forward, right, z, yaw_offset):
    fx, fy, rx, ry = basis(cam_tf)

    loc = carla.Location(
        x=cam_tf.location.x + fx * forward + rx * right,
        y=cam_tf.location.y + fy * forward + ry * right,
        z=z,
    )

    yaw = cam_tf.rotation.yaw + yaw_offset

    return carla.Transform(
        loc,
        carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0),
    )


def spawn_camera(world, bp_lib, cam_tf):
    bp = bp_lib.find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", "960")
    bp.set_attribute("image_size_y", "720")
    bp.set_attribute("fov", "65")
    bp.set_attribute("sensor_tick", "0.05")
    set_role(bp, "rgb_front")

    cam = world.spawn_actor(bp, cam_tf)
    print(f"[SPAWN] camera id={cam.id} role=rgb_front 960x720")
    print(f"[CAMERA] loc=({cam_tf.location.x:.1f},{cam_tf.location.y:.1f},{cam_tf.location.z:.1f}) yaw={cam_tf.rotation.yaw:.1f} pitch={cam_tf.rotation.pitch:.1f}")
    return cam


def spawn_person(world, bp_lib, cam_tf, ground_z):
    bps = list(bp_lib.filter("walker.pedestrian.*"))
    random.shuffle(bps)

    tf = visible_tf(
        cam_tf,
        forward=8.0,
        right=-2.2,
        z=ground_z + 0.15,
        yaw_offset=180.0,
    )

    for bp in bps:
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        set_role(bp, f"{TAG}_person")

        a = world.try_spawn_actor(bp, tf)
        if a:
            try:
                a.set_simulate_physics(False)
            except Exception:
                pass
            a.set_transform(tf)
            print(f"[SPAWN] person id={a.id} type={a.type_id}")
            return a

    print("[SPAWN] person FAIL")
    return None


def spawn_motorcycle(world, bp_lib, cam_tf, ground_z):
    names = [
        "vehicle.kawasaki.ninja",
        "vehicle.yamaha.yzf",
        "vehicle.harley-davidson.low_rider",
        "vehicle.vespa.zx125",
    ]

    bps = []
    for n in names:
        bps.extend(list(bp_lib.filter(n)))

    if not bps:
        print("[SPAWN] motorcycle bp yok")
        return None

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    tf = visible_tf(
        cam_tf,
        forward=9.5,
        right=2.2,
        z=ground_z + 0.55,
        yaw_offset=90.0,
    )

    for bp in bps:
        set_role(bp, f"{TAG}_motorcycle")

        for sp in spawn_points[:100]:
            tmp = carla.Transform(
                carla.Location(sp.location.x, sp.location.y, sp.location.z + 0.5),
                sp.rotation,
            )
            a = world.try_spawn_actor(bp, tmp)
            if not a:
                continue

            try:
                a.set_autopilot(False)
                a.set_simulate_physics(False)
            except Exception:
                pass

            a.set_transform(tf)
            print(f"[SPAWN] motorcycle id={a.id} type={a.type_id}")
            return a

    print("[SPAWN] motorcycle FAIL")
    return None


def spawn_vehicle(world, bp_lib, cam_tf, ground_z):
    names = [
        "vehicle.tesla.model3",
        "vehicle.lincoln.mkz_2020",
        "vehicle.audi.tt",
        "vehicle.dodge.charger_2020",
    ]

    bps = []
    for n in names:
        bps.extend(list(bp_lib.filter(n)))

    if not bps:
        bps = list(bp_lib.filter("vehicle.*"))

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    tf = visible_tf(
        cam_tf,
        forward=13.5,
        right=0.0,
        z=ground_z + 0.45,
        yaw_offset=180.0,
    )

    for bp in bps:
        set_role(bp, f"{TAG}_vehicle")

        for sp in spawn_points[:100]:
            tmp = carla.Transform(
                carla.Location(sp.location.x, sp.location.y, sp.location.z + 0.5),
                sp.rotation,
            )
            a = world.try_spawn_actor(bp, tmp)
            if not a:
                continue

            try:
                a.set_autopilot(False)
                a.set_simulate_physics(False)
            except Exception:
                pass

            a.set_transform(tf)
            print(f"[SPAWN] vehicle id={a.id} type={a.type_id}")
            return a

    print("[SPAWN] vehicle FAIL")
    return None


def set_spectator(world, cam_tf):
    world.get_spectator().set_transform(
        carla.Transform(
            carla.Location(
                cam_tf.location.x,
                cam_tf.location.y,
                cam_tf.location.z + 7.0,
            ),
            carla.Rotation(
                pitch=-55.0,
                yaw=cam_tf.rotation.yaw,
                roll=0.0,
            ),
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--light-state", choices=["red", "yellow", "green"], default="red")
    parser.add_argument("--side", type=int, choices=[1, -1], default=1)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    if args.clear:
        clear_scene(world)

    set_weather(world)
    set_lights(world, args.light_state)

    light, stop_wps = pick_good_light(world)
    cam_tf = camera_from_light(light, stop_wps, args.side)

    spawn_camera(world, bp_lib, cam_tf)

    ground_z = 0.0
    if stop_wps:
        ground_z = stop_wps[0].transform.location.z

    person = spawn_person(world, bp_lib, cam_tf, ground_z)
    motorcycle = spawn_motorcycle(world, bp_lib, cam_tf, ground_z)
    vehicle = spawn_vehicle(world, bp_lib, cam_tf, ground_z)

    set_spectator(world, cam_tf)

    time.sleep(1.0)

    print("")
    print("====================================================")
    print("CLEAN VISIBLE TEST READY")
    print("====================================================")
    print(f"Light state : {args.light_state.upper()}")
    print(f"Side        : {args.side}")
    print(f"Person      : {'OK' if person else 'FAIL'}")
    print(f"Motorcycle  : {'OK' if motorcycle else 'FAIL'}")
    print(f"Vehicle     : {'OK' if vehicle else 'FAIL'}")
    print("====================================================")


if __name__ == "__main__":
    main()

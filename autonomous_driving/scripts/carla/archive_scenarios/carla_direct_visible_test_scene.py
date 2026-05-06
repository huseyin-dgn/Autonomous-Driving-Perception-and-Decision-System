#!/usr/bin/env python3
import argparse
import math
import random
import time
import carla

TAG = "adas_direct_visible_test"


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
            sun_azimuth_angle=40.0,
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
        raise RuntimeError("light-state red/yellow/green olmalı")

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


def pick_light(world):
    lights = list(world.get_actors().filter("traffic.traffic_light*"))

    if not lights:
        raise RuntimeError("Trafik ışığı bulunamadı.")

    usable = []

    for tl in lights:
        loc = tl.get_transform().location

        if loc.z < -1.0:
            continue

        usable.append(tl)

    if not usable:
        usable = lights

    usable = sorted(
        usable,
        key=lambda t: abs(t.get_transform().location.x) + abs(t.get_transform().location.y)
    )

    tl = usable[len(usable) // 2]
    loc = tl.get_transform().location

    print(f"[LIGHT] selected id={tl.id} loc=({loc.x:.1f},{loc.y:.1f},{loc.z:.1f})")
    return tl


def spawn_camera(world, bp_lib, light):
    light_loc = light.get_transform().location

    target = carla.Location(
        x=light_loc.x,
        y=light_loc.y,
        z=light_loc.z + 2.2,
    )

    # Işığın tam karşısında, biraz geride kamera.
    # Önceki gibi yola/waypoint'e güvenmiyoruz.
    cam_loc = carla.Location(
        x=light_loc.x + 18.0,
        y=light_loc.y - 10.0,
        z=light_loc.z + 1.8,
    )

    cam_rot = look_at(cam_loc, target)
    cam_tf = carla.Transform(cam_loc, cam_rot)

    bp = bp_lib.find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", "960")
    bp.set_attribute("image_size_y", "720")
    bp.set_attribute("fov", "75")
    bp.set_attribute("sensor_tick", "0.05")
    set_role(bp, "rgb_front")

    cam = world.spawn_actor(bp, cam_tf)

    print(f"[SPAWN] camera id={cam.id} role=rgb_front 960x720")
    print(f"[CAMERA] loc=({cam_loc.x:.1f},{cam_loc.y:.1f},{cam_loc.z:.1f}) yaw={cam_rot.yaw:.1f} pitch={cam_rot.pitch:.1f}")

    return cam, cam_tf


def basis(cam_tf):
    yaw = math.radians(cam_tf.rotation.yaw)

    fx = math.cos(yaw)
    fy = math.sin(yaw)

    rx = math.cos(yaw + math.pi / 2.0)
    ry = math.sin(yaw + math.pi / 2.0)

    return fx, fy, rx, ry


def make_visible_tf(cam_tf, forward, right, z_offset, yaw_offset):
    fx, fy, rx, ry = basis(cam_tf)

    loc = carla.Location(
        x=cam_tf.location.x + fx * forward + rx * right,
        y=cam_tf.location.y + fy * forward + ry * right,
        z=cam_tf.location.z - 1.55 + z_offset,
    )

    yaw = cam_tf.rotation.yaw + yaw_offset

    return carla.Transform(
        loc,
        carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0)
    )


def spawn_person(world, bp_lib, cam_tf):
    bps = list(bp_lib.filter("walker.pedestrian.*"))
    random.shuffle(bps)

    # Kameranın sol-önünde, net yakın.
    tf = make_visible_tf(cam_tf, forward=5.0, right=-2.0, z_offset=0.15, yaw_offset=180.0)

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


def spawn_motorcycle(world, bp_lib, cam_tf):
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

    # Kameranın sağ-önünde, kişiden ayrı.
    target_tf = make_visible_tf(cam_tf, forward=7.0, right=1.8, z_offset=0.55, yaw_offset=90.0)

    for bp in bps:
        set_role(bp, f"{TAG}_motorcycle")

        for sp in spawn_points[:80]:
            tmp = carla.Transform(
                carla.Location(sp.location.x, sp.location.y, sp.location.z + 0.5),
                sp.rotation
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


def spawn_vehicle(world, bp_lib, cam_tf):
    names = [
        "vehicle.tesla.model3",
        "vehicle.audi.tt",
        "vehicle.lincoln.mkz_2020",
        "vehicle.dodge.charger_2020",
    ]

    bps = []
    for n in names:
        bps += list(bp_lib.filter(n))

    if not bps:
        bps = list(bp_lib.filter("vehicle.*"))

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    # Ortada ama daha uzakta. İnsan/motorla çakışmasın.
    target_tf = make_visible_tf(cam_tf, forward=10.5, right=0.0, z_offset=0.45, yaw_offset=180.0)

    for bp in bps:
        set_role(bp, f"{TAG}_vehicle")

        for sp in spawn_points[:100]:
            tmp = carla.Transform(
                carla.Location(sp.location.x, sp.location.y, sp.location.z + 0.5),
                sp.rotation
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

            print(f"[SPAWN] vehicle id={actor.id} type={actor.type_id}")
            return actor

    print("[SPAWN] vehicle FAIL")
    return None


def set_spectator(world, cam_tf):
    spectator = world.get_spectator()

    spectator.set_transform(
        carla.Transform(
            carla.Location(
                x=cam_tf.location.x,
                y=cam_tf.location.y,
                z=cam_tf.location.z + 7.0,
            ),
            carla.Rotation(
                pitch=-55.0,
                yaw=cam_tf.rotation.yaw,
                roll=0.0,
            )
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

    light = pick_light(world)
    camera, cam_tf = spawn_camera(world, bp_lib, light)

    person = spawn_person(world, bp_lib, cam_tf)
    motorcycle = spawn_motorcycle(world, bp_lib, cam_tf)
    vehicle = spawn_vehicle(world, bp_lib, cam_tf)

    set_spectator(world, cam_tf)

    time.sleep(1.0)

    print("")
    print("====================================================")
    print("DIRECT VISIBLE TEST READY")
    print("====================================================")
    print(f"Light state : {args.light_state.upper()}")
    print("Camera      : rgb_front 960x720")
    print(f"Person      : {'OK' if person else 'FAIL'}")
    print(f"Motorcycle  : {'OK' if motorcycle else 'FAIL'}")
    print(f"Vehicle     : {'OK' if vehicle else 'FAIL'}")
    print("")
    print("Beklenen görüntü:")
    print("- Solda/önde 1 insan")
    print("- Sağda/önde 1 motorsiklet")
    print("- Ortada/ileride 1 araba")
    print("- Üst/ileride trafik ışığı")
    print("====================================================")
    print("")


if __name__ == "__main__":
    main()

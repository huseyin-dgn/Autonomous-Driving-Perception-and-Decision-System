#!/usr/bin/env python3
import argparse
import math
import time
import random
import carla

TAG = "adas_minimal_light_person_motor_test"

OLD_TAGS = [
    "adas_minimal_light_person_motor_test",
    "adas_close_person_motor_front_light",
    "adas_person_motor_light_only",
    "adas_person_light_only",
    "adas_showcase",
    "adas_lite",
]


def role(actor):
    try:
        return actor.attributes.get("role_name", "")
    except Exception:
        return ""


def clear_scene(world):
    targets = []
    for a in world.get_actors():
        r = role(a)
        if a.type_id.startswith("sensor.camera"):
            targets.append(a)
        elif any(r.startswith(t) for t in OLD_TAGS):
            targets.append(a)

    print(f"[CLEAR] Silinecek actor sayısı: {len(targets)}")
    for a in targets:
        try:
            print(f"[CLEAR] destroy id={a.id} type={a.type_id} role={role(a)}")
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
            sun_altitude_angle=65.0,
            sun_azimuth_angle=20.0,
            fog_density=0.0,
            wetness=0.0,
        )
    )


def set_all_lights(world, state_text):
    if state_text == "red":
        state = carla.TrafficLightState.Red
    elif state_text == "yellow":
        state = carla.TrafficLightState.Yellow
    elif state_text == "green":
        state = carla.TrafficLightState.Green
    else:
        raise RuntimeError("light-state red/yellow/green olmalı")

    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    print(f"[LIGHT] Toplam trafik ışığı: {len(lights)}")
    print(f"[LIGHT] Hepsi {state_text.upper()} yapılacak")

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


def find_good_light(world):
    lights = list(world.get_actors().filter("traffic.traffic_light*"))

    if not lights:
        raise RuntimeError("Trafik ışığı bulunamadı.")

    # Çok kenarda olmayan, şehir içinde kalan bir ışık seç.
    lights = sorted(
        lights,
        key=lambda tl: abs(tl.get_transform().location.x) + abs(tl.get_transform().location.y)
    )

    selected = lights[len(lights) // 2]
    loc = selected.get_transform().location

    print(f"[LIGHT] Seçilen ışık id={selected.id} loc=({loc.x:.1f},{loc.y:.1f},{loc.z:.1f})")
    return selected


def spawn_camera(world, bp_lib, light):
    light_loc = light.get_transform().location

    target = carla.Location(
        x=light_loc.x,
        y=light_loc.y,
        z=light_loc.z + 2.5,
    )

    # Işığa yakın ve net bakacak kamera.
    cam_loc = carla.Location(
        x=light_loc.x - 8.0,
        y=light_loc.y - 3.0,
        z=light_loc.z + 1.7,
    )

    rot = look_at(cam_loc, target)
    tf = carla.Transform(cam_loc, rot)

    bp = bp_lib.find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", "960")
    bp.set_attribute("image_size_y", "720")
    bp.set_attribute("fov", "70")
    bp.set_attribute("sensor_tick", "0.05")
    set_role(bp, "rgb_front")

    cam = world.spawn_actor(bp, tf)
    print(f"[SPAWN] Camera OK id={cam.id} role=rgb_front 960x720")
    return cam, tf


def forward_right_location(cam_tf, forward, right, z):
    yaw = math.radians(cam_tf.rotation.yaw)

    fx = math.cos(yaw)
    fy = math.sin(yaw)

    rx = math.cos(yaw + math.pi / 2.0)
    ry = math.sin(yaw + math.pi / 2.0)

    return carla.Location(
        x=cam_tf.location.x + fx * forward + rx * right,
        y=cam_tf.location.y + fy * forward + ry * right,
        z=z,
    )


def ground_tf(world, cam_tf, forward, right, yaw_delta, z_add):
    rough = forward_right_location(cam_tf, forward, right, cam_tf.location.z)

    wp = world.get_map().get_waypoint(
        rough,
        project_to_road=True,
        lane_type=carla.LaneType.Any,
    )

    if wp:
        loc = carla.Location(
            x=wp.transform.location.x,
            y=wp.transform.location.y,
            z=wp.transform.location.z + z_add,
        )
    else:
        loc = carla.Location(rough.x, rough.y, rough.z + z_add)

    return carla.Transform(
        loc,
        carla.Rotation(pitch=0.0, yaw=cam_tf.rotation.yaw + yaw_delta, roll=0.0),
    )


def spawn_person(world, bp_lib, cam_tf):
    bps = list(bp_lib.filter("walker.pedestrian.*"))
    random.shuffle(bps)

    tf = ground_tf(world, cam_tf, forward=5.0, right=-1.0, yaw_delta=180.0, z_add=0.15)

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

        print(f"[SPAWN] Person OK id={actor.id} type={actor.type_id}")
        return actor

    print("[SPAWN] Person FAIL")
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
        bps += list(bp_lib.filter(p))

    if not bps:
        print("[SPAWN] Motorcycle blueprint yok.")
        return None

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    target_tf = ground_tf(world, cam_tf, forward=5.7, right=1.1, yaw_delta=90.0, z_add=0.45)

    for bp in bps:
        set_role(bp, f"{TAG}_motorcycle")

        for sp in spawn_points[:80]:
            tmp_tf = carla.Transform(
                carla.Location(sp.location.x, sp.location.y, sp.location.z + 0.5),
                sp.rotation,
            )

            actor = world.try_spawn_actor(bp, tmp_tf)
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
        x=cam_tf.location.x - math.cos(yaw) * 4.0,
        y=cam_tf.location.y - math.sin(yaw) * 4.0,
        z=cam_tf.location.z + 4.0,
    )

    rot = carla.Rotation(pitch=-25.0, yaw=cam_tf.rotation.yaw, roll=0.0)
    spectator.set_transform(carla.Transform(loc, rot))


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
    set_all_lights(world, args.light_state)

    light = find_good_light(world)
    cam, cam_tf = spawn_camera(world, bp_lib, light)

    person = spawn_person(world, bp_lib, cam_tf)
    moto = spawn_motorcycle(world, bp_lib, cam_tf)

    set_spectator(world, cam_tf)

    time.sleep(1.0)

    print("")
    print("====================================================")
    print("MINIMAL TEST SCENE READY")
    print("====================================================")
    print(f"Light state : {args.light_state.upper()}")
    print(f"Camera      : rgb_front 960x720")
    print(f"Person      : {'OK' if person else 'FAIL'}")
    print(f"Motorcycle  : {'OK' if moto else 'FAIL'}")
    print("")
    print("Bu sahnede beklenen:")
    print("- 1 traffic_light")
    print("- 1 person")
    print("- 1 motorcycle")
    print("====================================================")
    print("")


if __name__ == "__main__":
    main()

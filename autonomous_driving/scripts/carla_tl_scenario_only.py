#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path

import carla


STATE_MAP = {
    "red": carla.TrafficLightState.Red,
    "yellow": carla.TrafficLightState.Yellow,
    "green": carla.TrafficLightState.Green,
}


def loc_dict(l):
    return {"x": float(l.x), "y": float(l.y), "z": float(l.z)}


def rot_dict(r):
    return {
        "pitch": float(r.pitch),
        "yaw": float(r.yaw),
        "roll": float(r.roll),
    }


def look_at(src, dst):
    dx = dst.x - src.x
    dy = dst.y - src.y
    dz = dst.z - src.z

    yaw = math.degrees(math.atan2(dy, dx))
    dist_xy = math.sqrt(dx * dx + dy * dy)
    pitch = math.degrees(math.atan2(dz, max(0.001, dist_xy)))

    return carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)


def destroy_dynamic(world):
    count = 0

    for a in world.get_actors():
        if (
            a.type_id.startswith("vehicle.")
            or a.type_id.startswith("walker.")
            or a.type_id.startswith("sensor.")
            or a.type_id.startswith("controller.ai.walker")
        ):
            try:
                a.destroy()
                count += 1
            except Exception:
                pass

    print(f"[SCENE] destroyed dynamic actors: {count}")


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


def set_all_lights(lights, state_name):
    state = STATE_MAP[state_name]

    for l in lights:
        try:
            l.freeze(False)
            l.set_state(state)
            l.set_red_time(9999.0)
            l.set_yellow_time(9999.0)
            l.set_green_time(9999.0)
            l.freeze(True)
        except Exception:
            pass

    print(f"[SCENE] ALL_LIGHTS_STATE={state_name.upper()}")


def get_light_target(light, target_z_add):
    tf = light.get_transform()

    try:
        boxes = light.get_light_boxes()
        points = []

        for box in boxes:
            for v in box.get_world_vertices(tf):
                points.append(v)

        if points:
            xs = [p.x for p in points]
            ys = [p.y for p in points]
            zs = [p.z for p in points]

            return carla.Location(
                x=sum(xs) / len(xs),
                y=sum(ys) / len(ys),
                z=sum(zs) / len(zs),
            )
    except Exception:
        pass

    loc = tf.location
    return carla.Location(
        x=loc.x,
        y=loc.y,
        z=loc.z + target_z_add,
    )


def make_camera_transform(target, angle_deg, distance, z_offset):
    a = math.radians(angle_deg)

    cam_loc = carla.Location(
        x=target.x + math.cos(a) * distance,
        y=target.y + math.sin(a) * distance,
        z=target.z + z_offset,
    )

    cam_rot = look_at(cam_loc, target)

    return carla.Transform(cam_loc, cam_rot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Town10HD")
    parser.add_argument("--reload-map", action="store_true")
    parser.add_argument("--destroy-dynamic", action="store_true")

    parser.add_argument("--light-index", type=int, default=0)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--distance", type=float, default=6.0)
    parser.add_argument("--z-offset", type=float, default=0.0)
    parser.add_argument("--target-z-add", type=float, default=5.0)

    parser.add_argument("--state", choices=["red", "yellow", "green"], default="red")
    parser.add_argument("--cycle", action="store_true")
    parser.add_argument("--seconds", type=float, default=6.0)

    parser.add_argument("--out", default="/tmp/adas_tl_only_scene.json")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    if args.reload_map:
        print(f"[SCENE] loading map: {args.map}")
        world = client.load_world(args.map)
        time.sleep(3.0)
    else:
        world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    set_weather(world)

    if args.destroy_dynamic:
        destroy_dynamic(world)

    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    lights.sort(key=lambda a: (a.get_transform().location.x, a.get_transform().location.y))

    if not lights:
        raise RuntimeError("Map içinde trafik ışığı bulunamadı. Town10HD veya Town03 dene.")

    if args.light_index < 0 or args.light_index >= len(lights):
        raise RuntimeError(f"Geçersiz light-index. Bulunan ışık sayısı: {len(lights)}")

    selected = lights[args.light_index]
    target = get_light_target(selected, args.target_z_add)

    cam_tf = make_camera_transform(
        target=target,
        angle_deg=args.angle,
        distance=args.distance,
        z_offset=args.z_offset,
    )

    world.get_spectator().set_transform(cam_tf)

    payload = {
        "host": args.host,
        "port": args.port,
        "map": world.get_map().name,
        "light_count": len(lights),
        "selected_light_index": args.light_index,
        "selected_light_id": selected.id,
        "angle": args.angle,
        "distance": args.distance,
        "z_offset": args.z_offset,
        "target": loc_dict(target),
        "camera_transform": {
            "location": loc_dict(cam_tf.location),
            "rotation": rot_dict(cam_tf.rotation),
        },
    }

    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("===================================================")
    print("[SCENE] TRAFFIC LIGHT SCENE READY")
    print(f"[SCENE] map         : {world.get_map().name}")
    print(f"[SCENE] light count : {len(lights)}")
    print(f"[SCENE] selected    : index={args.light_index}, id={selected.id}")
    print(f"[SCENE] angle       : {args.angle}")
    print(f"[SCENE] distance    : {args.distance}")
    print(f"[SCENE] z_offset    : {args.z_offset}")
    print(f"[SCENE] json        : {args.out}")
    print("===================================================")

    if args.cycle:
        states = ["red", "yellow", "green"]
        i = 0

        while True:
            s = states[i % len(states)]
            set_all_lights(lights, s)
            i += 1
            time.sleep(args.seconds)
    else:
        set_all_lights(lights, args.state)

        while True:
            time.sleep(1.0)


if __name__ == "__main__":
    main()

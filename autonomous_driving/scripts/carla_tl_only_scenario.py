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


def loc_to_dict(l):
    return {"x": float(l.x), "y": float(l.y), "z": float(l.z)}


def rot_to_dict(r):
    return {
        "pitch": float(r.pitch),
        "yaw": float(r.yaw),
        "roll": float(r.roll),
    }


def transform_location(tf, local_loc):
    p = carla.Location(
        x=float(local_loc.x),
        y=float(local_loc.y),
        z=float(local_loc.z),
    )
    tf.transform(p)
    return p


def look_at_rotation(src, dst):
    dx = dst.x - src.x
    dy = dst.y - src.y
    dz = dst.z - src.z

    yaw = math.degrees(math.atan2(dy, dx))
    dist_xy = math.sqrt(dx * dx + dy * dy)
    pitch = math.degrees(math.atan2(dz, max(0.001, dist_xy)))

    return carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)


def get_target_location(light, fallback_z_add):
    tf = light.get_transform()

    try:
        boxes = light.get_light_boxes()
        if boxes:
            world_points = []
            for box in boxes:
                world_points.append(transform_location(tf, box.location))

            world_points.sort(key=lambda p: p.z, reverse=True)
            return world_points[0]
    except Exception:
        pass

    loc = tf.location
    return carla.Location(
        x=loc.x,
        y=loc.y,
        z=loc.z + fallback_z_add,
    )


def get_camera_transform(light, mode, distance, camera_height, target_z_add):
    tf = light.get_transform()
    target = get_target_location(light, target_z_add)

    if mode == "trigger":
        try:
            trig = light.trigger_volume
            cam_loc = transform_location(tf, trig.location)
            cam_loc.z += camera_height
        except Exception:
            mode = "minus"

    if mode != "trigger":
        yaw_rad = math.radians(tf.rotation.yaw)

        forward = carla.Vector3D(
            x=math.cos(yaw_rad),
            y=math.sin(yaw_rad),
            z=0.0,
        )

        right = carla.Vector3D(
            x=math.cos(yaw_rad + math.pi / 2.0),
            y=math.sin(yaw_rad + math.pi / 2.0),
            z=0.0,
        )

        if mode == "plus":
            direction = forward
        elif mode == "minus":
            direction = carla.Vector3D(-forward.x, -forward.y, 0.0)
        elif mode == "right":
            direction = right
        elif mode == "left":
            direction = carla.Vector3D(-right.x, -right.y, 0.0)
        else:
            direction = carla.Vector3D(-forward.x, -forward.y, 0.0)

        cam_loc = carla.Location(
            x=target.x + direction.x * distance,
            y=target.y + direction.y * distance,
            z=target.z + camera_height,
        )

    cam_rot = look_at_rotation(cam_loc, target)
    return carla.Transform(cam_loc, cam_rot), target


def destroy_dynamic_actors(world):
    actors = world.get_actors()
    to_destroy = []

    for a in actors:
        if (
            a.type_id.startswith("vehicle.")
            or a.type_id.startswith("walker.")
            or a.type_id.startswith("sensor.")
            or a.type_id.startswith("controller.ai.walker")
        ):
            to_destroy.append(a)

    for a in to_destroy:
        try:
            a.destroy()
        except Exception:
            pass

    print(f"[SCENE] Destroyed dynamic actors: {len(to_destroy)}")


def set_weather(world):
    world.set_weather(
        carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            sun_altitude_angle=55.0,
            sun_azimuth_angle=30.0,
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
            l.freeze(True)
            l.set_red_time(9999.0)
            l.set_yellow_time(9999.0)
            l.set_green_time(9999.0)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Town10HD")
    parser.add_argument("--reload-map", action="store_true")
    parser.add_argument("--destroy-dynamic", action="store_true")

    parser.add_argument("--light-index", type=int, default=0)
    parser.add_argument("--state", choices=["red", "yellow", "green"], default="red")
    parser.add_argument("--cycle", action="store_true")
    parser.add_argument("--seconds", type=float, default=6.0)

    parser.add_argument(
        "--camera-mode",
        choices=["trigger", "minus", "plus", "left", "right"],
        default="trigger",
    )
    parser.add_argument("--distance", type=float, default=8.0)
    parser.add_argument("--camera-height", type=float, default=1.7)
    parser.add_argument("--target-z-add", type=float, default=3.0)

    parser.add_argument("--out", default="/tmp/adas_tl_only_scene.json")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    if args.reload_map:
        print(f"[SCENE] Loading map: {args.map}")
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
        destroy_dynamic_actors(world)

    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    lights.sort(key=lambda a: (a.get_transform().location.x, a.get_transform().location.y))

    if not lights:
        raise RuntimeError("CARLA map içinde traffic light bulunamadı. Town10HD/Town03 dene.")

    if args.light_index < 0 or args.light_index >= len(lights):
        raise RuntimeError(f"--light-index geçersiz. Bulunan ışık sayısı: {len(lights)}")

    selected = lights[args.light_index]
    cam_tf, target = get_camera_transform(
        selected,
        args.camera_mode,
        args.distance,
        args.camera_height,
        args.target_z_add,
    )

    world.get_spectator().set_transform(cam_tf)

    payload = {
        "host": args.host,
        "port": args.port,
        "map": world.get_map().name,
        "light_count": len(lights),
        "selected_light_id": selected.id,
        "selected_light_index": args.light_index,
        "camera_mode": args.camera_mode,
        "camera_transform": {
            "location": loc_to_dict(cam_tf.location),
            "rotation": rot_to_dict(cam_tf.rotation),
        },
        "target_location": loc_to_dict(target),
    }

    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("===================================================")
    print("[SCENE] TRAFFIC LIGHT ONLY SCENE READY")
    print(f"[SCENE] Map              : {world.get_map().name}")
    print(f"[SCENE] Traffic lights   : {len(lights)}")
    print(f"[SCENE] Selected light   : index={args.light_index}, id={selected.id}")
    print(f"[SCENE] Camera mode      : {args.camera_mode}")
    print(f"[SCENE] Scene JSON       : {args.out}")
    print("===================================================")

    if args.cycle:
        cycle = ["red", "yellow", "green"]
        i = 0
        while True:
            state_name = cycle[i % len(cycle)]
            set_all_lights(lights, state_name)
            print(f"[SCENE] ALL_LIGHTS_STATE={state_name.upper()}")
            i += 1
            time.sleep(args.seconds)
    else:
        set_all_lights(lights, args.state)
        print(f"[SCENE] ALL_LIGHTS_STATE={args.state.upper()}")
        print("[SCENE] Holding scene. Ctrl+C ile kapat.")
        while True:
            time.sleep(1.0)


if __name__ == "__main__":
    main()

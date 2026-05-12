#!/usr/bin/env python3
import argparse
import itertools
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


def dist(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


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


def get_light_target(light, fallback_z_add=5.0):
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
        z=loc.z + fallback_z_add,
    )


def find_best_triplet(lights, targets):
    best = None

    for combo in itertools.combinations(range(len(lights)), 3):
        a, b, c = combo

        d01 = dist(targets[a], targets[b])
        d02 = dist(targets[a], targets[c])
        d12 = dist(targets[b], targets[c])

        max_d = max(d01, d02, d12)
        sum_d = d01 + d02 + d12

        # Birbirine yakın 3 ışığı seç.
        score = max_d * 10.0 + sum_d

        if best is None or score < best[0]:
            best = (score, combo, (d01, d02, d12))

    return best[1], best[2]


def set_light_state(light, state_name):
    state = STATE_MAP[state_name]

    try:
        light.freeze(False)
        light.set_state(state)
        light.set_red_time(9999.0)
        light.set_yellow_time(9999.0)
        light.set_green_time(9999.0)
        light.freeze(True)
    except Exception as exc:
        print(f"[WARN] light state set failed id={light.id}: {exc}")


def make_camera_transform(center, angle_deg, distance, z_offset):
    a = math.radians(angle_deg)

    cam_loc = carla.Location(
        x=center.x + math.cos(a) * distance,
        y=center.y + math.sin(a) * distance,
        z=center.z + z_offset,
    )

    cam_rot = look_at(cam_loc, center)

    return carla.Transform(cam_loc, cam_rot)


def parse_indices(text):
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 3:
        raise RuntimeError("--indices tam 3 değer içermeli. Örnek: --indices 0,1,2")
    return [int(p) for p in parts]


def parse_states(text):
    parts = [p.strip().lower() for p in str(text).split(",") if p.strip()]
    if len(parts) != 3:
        raise RuntimeError("--states tam 3 değer içermeli. Örnek: --states red,yellow,green")

    for p in parts:
        if p not in STATE_MAP:
            raise RuntimeError(f"Geçersiz state: {p}")

    return parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Town10HD")
    parser.add_argument("--reload-map", action="store_true")
    parser.add_argument("--destroy-dynamic", action="store_true")

    parser.add_argument("--indices", default="")
    parser.add_argument("--states", default="red,yellow,green")

    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--distance", type=float, default=12.0)
    parser.add_argument("--z-offset", type=float, default=0.0)
    parser.add_argument("--target-z-add", type=float, default=5.0)

    parser.add_argument("--out", default="/tmp/adas_tl_only_scene.json")
    args = parser.parse_args()

    selected_states = parse_states(args.states)

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

    if len(lights) < 3:
        raise RuntimeError(f"Map içinde en az 3 trafik ışığı yok. Bulunan: {len(lights)}")

    targets = [get_light_target(l, args.target_z_add) for l in lights]

    if args.indices.strip():
        selected_indices = parse_indices(args.indices)
    else:
        selected_indices, distances = find_best_triplet(lights, targets)
        selected_indices = list(selected_indices)
        print(f"[SCENE] auto selected triplet distances={distances}")

    for idx in selected_indices:
        if idx < 0 or idx >= len(lights):
            raise RuntimeError(f"Geçersiz light index: {idx}. Bulunan ışık sayısı: {len(lights)}")

    selected_targets = [targets[i] for i in selected_indices]

    center = carla.Location(
        x=sum(t.x for t in selected_targets) / 3.0,
        y=sum(t.y for t in selected_targets) / 3.0,
        z=sum(t.z for t in selected_targets) / 3.0,
    )

    # Önce tüm ışıkları red yap, seçili üç ışığı istenen renklere ayarla.
    for l in lights:
        set_light_state(l, "red")

    selected_payload = []

    for i, state_name in zip(selected_indices, selected_states):
        light = lights[i]
        set_light_state(light, state_name)

        selected_payload.append({
            "index": int(i),
            "id": int(light.id),
            "state": state_name,
            "target": loc_dict(targets[i]),
        })

        print(f"[SCENE] selected index={i}, id={light.id}, state={state_name}")

    cam_tf = make_camera_transform(
        center=center,
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
        "selected_lights": selected_payload,
        "angle": args.angle,
        "distance": args.distance,
        "z_offset": args.z_offset,
        "center": loc_dict(center),
        "camera_transform": {
            "location": loc_dict(cam_tf.location),
            "rotation": rot_dict(cam_tf.rotation),
        },
    }

    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("===================================================")
    print("[SCENE] THREE TRAFFIC LIGHTS READY")
    print(f"[SCENE] map         : {world.get_map().name}")
    print(f"[SCENE] light count : {len(lights)}")
    print(f"[SCENE] indices     : {selected_indices}")
    print(f"[SCENE] states      : {selected_states}")
    print(f"[SCENE] angle       : {args.angle}")
    print(f"[SCENE] distance    : {args.distance}")
    print(f"[SCENE] json        : {args.out}")
    print("===================================================")

    while True:
        # State bozulmasın diye arada tekrar uygula.
        for i, state_name in zip(selected_indices, selected_states):
            set_light_state(lights[i], state_name)

        time.sleep(2.0)


if __name__ == "__main__":
    main()

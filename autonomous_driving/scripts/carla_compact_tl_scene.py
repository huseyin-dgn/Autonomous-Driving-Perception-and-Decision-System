#!/usr/bin/env python3
import argparse
import math
import random
import time

import carla


ADAS_TAG = "adas_compact_tl_test"


def get_bp(world, preferred, contains=None):
    bps = world.get_blueprint_library()

    for name in preferred:
        found = bps.find(name) if name in [bp.id for bp in bps] else None
        if found is not None:
            return found

    if contains:
        matches = [bp for bp in bps if contains.lower() in bp.id.lower()]
        if matches:
            return random.choice(matches)

    return None


def set_role(bp, role):
    if bp is not None and bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role)


def vec_len(v):
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def yaw_to_forward(yaw_deg):
    yaw = math.radians(yaw_deg)
    return carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)


def yaw_to_right(yaw_deg):
    yaw = math.radians(yaw_deg + 90.0)
    return carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)


def add_vec(loc, f, amount):
    return carla.Location(
        x=loc.x + f.x * amount,
        y=loc.y + f.y * amount,
        z=loc.z + f.z * amount,
    )


def local_to_world(base_loc, yaw_deg, forward_m, right_m, up_m=0.0):
    f = yaw_to_forward(yaw_deg)
    r = yaw_to_right(yaw_deg)
    return carla.Location(
        x=base_loc.x + f.x * forward_m + r.x * right_m,
        y=base_loc.y + f.y * forward_m + r.y * right_m,
        z=base_loc.z + up_m,
    )


def look_at_yaw(src, dst):
    dx = dst.x - src.x
    dy = dst.y - src.y
    return math.degrees(math.atan2(dy, dx))


def cleanup(world):
    victims = []

    for actor in world.get_actors():
        role = actor.attributes.get("role_name", "")
        if role.startswith(ADAS_TAG):
            victims.append(actor)

    if victims:
        print(f"[CLEANUP] destroying {len(victims)} old ADAS compact actors")
        for actor in victims:
            try:
                actor.destroy()
            except Exception:
                pass
        time.sleep(0.5)


def choose_traffic_light(world, index=None):
    lights = list(world.get_actors().filter("traffic.traffic_light*"))

    if not lights:
        raise RuntimeError("Mapte traffic light actor bulunamadı. Town içeren bir CARLA map açman gerekiyor.")

    lights = sorted(lights, key=lambda a: a.id)

    if index is not None:
        index = max(0, min(index, len(lights) - 1))
        return lights[index], len(lights)

    candidates = []
    for tl in lights:
        loc = tl.get_transform().location
        if loc.z < 2.0:
            continue
        candidates.append(tl)

    if not candidates:
        candidates = lights

    return candidates[len(candidates) // 2], len(lights)


def freeze_light(tl, state):
    state = state.lower().strip()

    if state == "red":
        tl.set_state(carla.TrafficLightState.Red)
    elif state == "green":
        tl.set_state(carla.TrafficLightState.Green)
    elif state == "yellow":
        tl.set_state(carla.TrafficLightState.Yellow)
    else:
        tl.set_state(carla.TrafficLightState.Red)

    tl.freeze(True)
    print(f"[TL] id={tl.id} state={state} frozen=True loc={tl.get_transform().location}")


def spawn_actor(world, bp, transform, role, physics=False):
    if bp is None:
        print(f"[WARN] blueprint yok: {role}")
        return None

    set_role(bp, role)

    actor = world.try_spawn_actor(bp, transform)

    if actor is None:
        print(f"[WARN] spawn başarısız: {role} tf={transform}")
        return None

    if hasattr(actor, "set_simulate_physics"):
        actor.set_simulate_physics(bool(physics))

    print(f"[SPAWN] {role:18s} id={actor.id} type={actor.type_id}")
    return actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--tl-state", default="yellow", choices=["red", "yellow", "green"])
    parser.add_argument("--tl-index", type=int, default=None)
    parser.add_argument("--ego-distance", type=float, default=18.0)
    parser.add_argument("--camera-yaw-offset", type=float, default=0.0)
    parser.add_argument("--sign-yaw-offset", type=float, default=180.0)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    world.set_weather(
        carla.WeatherParameters(
            cloudiness=5.0,
            precipitation=0.0,
            sun_altitude_angle=55.0,
            sun_azimuth_angle=20.0,
            fog_density=0.0,
        )
    )

    cleanup(world)

    tl, total_lights = choose_traffic_light(world, args.tl_index)
    freeze_light(tl, args.tl_state)

    tl_tf = tl.get_transform()
    tl_loc = tl_tf.location
    tl_yaw = tl_tf.rotation.yaw

    f = yaw_to_forward(tl_yaw)

    ego_loc = carla.Location(
        x=tl_loc.x - f.x * args.ego_distance,
        y=tl_loc.y - f.y * args.ego_distance,
        z=tl_loc.z - 2.2,
    )

    ego_yaw = look_at_yaw(ego_loc, tl_loc) + args.camera_yaw_offset

    ego_tf = carla.Transform(
        ego_loc,
        carla.Rotation(pitch=0.0, yaw=ego_yaw, roll=0.0),
    )

    print(f"[INFO] total traffic lights={total_lights}")
    print(f"[INFO] selected tl id={tl.id}")
    print(f"[INFO] ego loc={ego_loc} yaw={ego_yaw:.2f}")

    vehicle_bp = get_bp(
        world,
        [
            "vehicle.audi.tt",
            "vehicle.tesla.model3",
            "vehicle.lincoln.mkz_2020",
            "vehicle.mercedes.coupe",
        ],
        contains="vehicle",
    )

    ego_bp = get_bp(
        world,
        [
            "vehicle.tesla.model3",
            "vehicle.audi.tt",
            "vehicle.lincoln.mkz_2020",
        ],
        contains="vehicle",
    )

    motorcycle_bps = []
    for name in [
        "vehicle.yamaha.yzf",
        "vehicle.kawasaki.ninja",
        "vehicle.harley-davidson.low_rider",
        "vehicle.vespa.zx125",
    ]:
        bp = get_bp(world, [name])
        if bp is not None:
            motorcycle_bps.append(bp)

    if not motorcycle_bps:
        motorcycle_bps = [bp for bp in world.get_blueprint_library() if "vehicle" in bp.id and ("yamaha" in bp.id or "ninja" in bp.id or "harley" in bp.id or "vespa" in bp.id)]

    walker_bp = get_bp(world, ["walker.pedestrian.0001"], contains="walker.pedestrian")

    sign_bp = None
    for sign_name in [
        "static.prop.trafficwarning",
        "static.prop.streetsign01",
        "static.prop.streetsign04",
        "static.prop.streetsign",
        "static.prop.busstop",
    ]:
        try:
            sign_bp = world.get_blueprint_library().find(sign_name)
            if sign_bp is not None:
                break
        except Exception:
            pass

    spawn_actor(world, ego_bp, ego_tf, f"{ADAS_TAG}_ego_vehicle", physics=False)

    car_loc = local_to_world(ego_loc, ego_yaw, 8.0, -1.2, -0.1)
    car_tf = carla.Transform(car_loc, carla.Rotation(yaw=ego_yaw))
    spawn_actor(world, vehicle_bp, car_tf, f"{ADAS_TAG}_front_vehicle", physics=False)

    motor_positions = [
        (5.5, 3.0, 0.0),
        (8.0, 3.9, 0.0),
        (10.5, 4.8, 0.0),
    ]

    for i, (forward_m, right_m, yaw_add) in enumerate(motor_positions):
        if motorcycle_bps:
            bp = motorcycle_bps[i % len(motorcycle_bps)]
            loc = local_to_world(ego_loc, ego_yaw, forward_m, right_m, -0.1)
            tf = carla.Transform(loc, carla.Rotation(yaw=ego_yaw + yaw_add))
            spawn_actor(world, bp, tf, f"{ADAS_TAG}_motorcycle_{i+1}", physics=False)

    person_loc = local_to_world(ego_loc, ego_yaw, 10.0, -3.2, -0.1)
    person_tf = carla.Transform(person_loc, carla.Rotation(yaw=ego_yaw + 90.0))
    spawn_actor(world, walker_bp, person_tf, f"{ADAS_TAG}_person_1", physics=False)

    if sign_bp is not None:
        sign_loc = local_to_world(ego_loc, ego_yaw, 12.0, 3.0, 1.0)
        sign_tf = carla.Transform(
            sign_loc,
            carla.Rotation(pitch=0.0, yaw=ego_yaw + args.sign_yaw_offset, roll=0.0),
        )
        spawn_actor(world, sign_bp, sign_tf, f"{ADAS_TAG}_traffic_sign_1", physics=False)
    else:
        print("[WARN] Levha blueprint bulunamadı.")

    spectator = world.get_spectator()
    spectator_loc = local_to_world(ego_loc, ego_yaw, -6.0, 0.0, 5.0)
    spectator_tf = carla.Transform(
        spectator_loc,
        carla.Rotation(pitch=-20.0, yaw=ego_yaw, roll=0.0),
    )
    spectator.set_transform(spectator_tf)

    print("")
    print("========== COMPACT TL SCENE READY ==========")
    print(f"traffic_light_id={tl.id}")
    print(f"traffic_light_state={args.tl_state}")
    print(f"ego_distance={args.ego_distance}")
    print("actors: 1 ego, 1 vehicle, 1 person, 3 motorcycles, 1 sign")
    print("===========================================")


if __name__ == "__main__":
    main()

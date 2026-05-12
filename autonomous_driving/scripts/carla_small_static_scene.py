#!/usr/bin/env python3
import argparse
import math
import os
import sys
import time
import traceback

try:
    import carla
except Exception:
    for p in [
        os.path.expanduser("~/CARLA_DISK/PythonAPI/carla"),
        os.path.expanduser("~/CARLA_DISK/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg"),
        os.path.expanduser("~/CARLA_DISK/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg"),
    ]:
        if os.path.exists(p):
            sys.path.append(p)
    import carla


ROLE_PREFIX = "adas_small_test"


def get_bp(bp_lib, patterns):
    for p in patterns:
        bps = bp_lib.filter(p)
        if bps:
            return bps[0]
    return None


def set_role(bp, role):
    if bp is not None and bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role)


def fvec(yaw):
    r = math.radians(yaw)
    return carla.Vector3D(math.cos(r), math.sin(r), 0.0)


def rvec(yaw):
    r = math.radians(yaw + 90.0)
    return carla.Vector3D(math.cos(r), math.sin(r), 0.0)


def move_tf(base, forward=0.0, right=0.0, z=0.3, yaw_offset=0.0):
    f = fvec(base.rotation.yaw)
    r = rvec(base.rotation.yaw)

    loc = carla.Location(
        x=base.location.x + f.x * forward + r.x * right,
        y=base.location.y + f.y * forward + r.y * right,
        z=base.location.z + z,
    )

    rot = carla.Rotation(
        pitch=0.0,
        yaw=base.rotation.yaw + yaw_offset,
        roll=0.0,
    )

    return carla.Transform(loc, rot)


def destroy_old_scene(world, clear_all_dynamic):
    actors = world.get_actors()
    to_destroy = []

    for a in actors:
        tid = a.type_id
        role = a.attributes.get("role_name", "")

        if role.startswith(ROLE_PREFIX):
            to_destroy.append(a)
        elif clear_all_dynamic and (
            tid.startswith("vehicle.") or
            tid.startswith("walker.") or
            tid.startswith("sensor.")
        ):
            to_destroy.append(a)

    for a in to_destroy:
        try:
            print(f"[DESTROY] id={a.id} type={a.type_id}")
            a.destroy()
        except Exception:
            pass


def try_spawn(world, bp, tf, name, physics=False):
    if bp is None:
        print(f"[WARN] Blueprint yok: {name}")
        return None

    actor = world.try_spawn_actor(bp, tf)
    if actor is None:
        print(f"[WARN] Spawn olmadı: {name}")
        return None

    try:
        actor.set_simulate_physics(bool(physics))
    except Exception:
        pass

    print(f"[SPAWN] {name:<18} id={actor.id:<5} type={actor.type_id}")
    return actor


def set_light(tl, state):
    if tl is None:
        return

    state = state.lower().strip()

    if state == "green":
        s = carla.TrafficLightState.Green
    elif state == "yellow":
        s = carla.TrafficLightState.Yellow
    else:
        s = carla.TrafficLightState.Red

    try:
        tl.set_state(s)
        tl.set_red_time(999.0)
        tl.set_yellow_time(999.0)
        tl.set_green_time(999.0)
        tl.freeze(True)
    except Exception as e:
        print("[WARN] Traffic light state ayarlanamadı:", e)


def get_stop_wp(tl, carla_map):
    try:
        wps = tl.get_stop_waypoints()
        if wps:
            return wps[0]
    except Exception:
        pass

    try:
        return carla_map.get_waypoint(
            tl.get_transform().location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
    except Exception:
        return None


def choose_base_near_traffic_light(world, carla_map):
    traffic_lights = list(world.get_actors().filter("traffic.traffic_light*"))
    print(f"[INFO] Map traffic light count: {len(traffic_lights)}")

    best = None

    for tl in traffic_lights:
        stop_wp = get_stop_wp(tl, carla_map)
        if stop_wp is None:
            continue

        prev = stop_wp.previous(18.0)
        if not prev:
            continue

        ego_wp = prev[0]
        ego_tf = ego_wp.transform
        ego_tf.location.z += 0.35

        best = (tl, ego_tf)
        break

    if best is not None:
        return best

    spawn_points = carla_map.get_spawn_points()
    if not spawn_points:
        raise RuntimeError("Spawn point bulunamadı.")

    print("[WARN] Traffic light bulunamadı. İlk spawn point kullanılacak.")
    return None, spawn_points[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--light", default="red", choices=["red", "yellow", "green"])
    parser.add_argument("--no-clear", action="store_true")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    carla_map = world.get_map()

    print("[INFO] Connected.")
    print("[INFO] Map:", carla_map.name)

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    try:
        world.set_weather(carla.WeatherParameters.ClearNoon)
    except Exception:
        pass

    destroy_old_scene(world, clear_all_dynamic=not args.no_clear)

    bp_lib = world.get_blueprint_library()

    ego_bp = get_bp(bp_lib, ["vehicle.tesla.model3", "vehicle.lincoln.mkz_2017", "vehicle.audi.tt", "vehicle.*"])
    car_bp = get_bp(bp_lib, ["vehicle.audi.tt", "vehicle.lincoln.mkz_2017", "vehicle.tesla.model3", "vehicle.*"])

    motor1_bp = get_bp(bp_lib, ["vehicle.yamaha.yzf", "vehicle.kawasaki.ninja", "vehicle.harley-davidson.low_rider"])
    motor2_bp = get_bp(bp_lib, ["vehicle.kawasaki.ninja", "vehicle.yamaha.yzf", "vehicle.harley-davidson.low_rider"])
    motor3_bp = get_bp(bp_lib, ["vehicle.harley-davidson.low_rider", "vehicle.yamaha.yzf", "vehicle.kawasaki.ninja"])

    person_bp = get_bp(bp_lib, ["walker.pedestrian.0001", "walker.pedestrian.0002", "walker.pedestrian.*"])
    sign_bp = get_bp(bp_lib, ["static.prop.trafficwarning", "static.prop.streetsign04", "static.prop.streetsign01", "static.prop.streetsign"])

    set_role(ego_bp, f"{ROLE_PREFIX}_ego")
    set_role(car_bp, f"{ROLE_PREFIX}_front_car")
    set_role(motor1_bp, f"{ROLE_PREFIX}_motor_left")
    set_role(motor2_bp, f"{ROLE_PREFIX}_motor_right")
    set_role(motor3_bp, f"{ROLE_PREFIX}_motor_far")
    set_role(person_bp, f"{ROLE_PREFIX}_person_right")
    set_role(sign_bp, f"{ROLE_PREFIX}_sign_right")

    tl, ego_tf = choose_base_near_traffic_light(world, carla_map)
    set_light(tl, args.light)

    ego = try_spawn(world, ego_bp, ego_tf, "ego_vehicle", physics=False)
    if ego is None:
        raise RuntimeError("Ego araç spawn edilemedi.")

    actors = [ego]

    placements = [
        ("front_car", car_bp, move_tf(ego_tf, forward=13.0, right=-1.2, z=0.35, yaw_offset=0.0)),
        ("motor_left", motor1_bp, move_tf(ego_tf, forward=9.0, right=-3.7, z=0.35, yaw_offset=0.0)),
        ("motor_right", motor2_bp, move_tf(ego_tf, forward=10.5, right=3.6, z=0.35, yaw_offset=0.0)),
        ("motor_far", motor3_bp, move_tf(ego_tf, forward=16.0, right=-3.9, z=0.35, yaw_offset=0.0)),
        ("person_right", person_bp, move_tf(ego_tf, forward=12.0, right=5.0, z=0.55, yaw_offset=180.0)),
        ("traffic_sign_right", sign_bp, move_tf(ego_tf, forward=11.0, right=6.0, z=0.20, yaw_offset=180.0)),
    ]

    for name, bp, tf in placements:
        actor = try_spawn(world, bp, tf, name, physics=False)
        if actor is not None:
            actors.append(actor)

    spectator = world.get_spectator()
    spec_tf = move_tf(ego_tf, forward=-7.0, right=0.0, z=5.0, yaw_offset=0.0)
    spec_tf.rotation.pitch = -18.0
    spectator.set_transform(spec_tf)

    if tl is not None:
        print(f"[INFO] Traffic light selected id={tl.id}, forced_state={args.light}")
    else:
        print("[WARN] Map traffic light yok. Sahne kuruldu ama gerçek trafik ışığı yok.")

    print("")
    print("SAHNE KURULDU.")
    print("Publisher ayrı çalışacak.")
    print("Ego role_name:", f"{ROLE_PREFIX}_ego")
    print("Actor count:", len(actors))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()

#!/usr/bin/env python3
import argparse
import random
import sys
import time
import math
import carla


def log(msg):
    print(msg, flush=True)


def tick(world, n=1):
    for _ in range(n):
        try:
            world.tick()
        except RuntimeError:
            world.wait_for_tick()


def set_attr(bp, key, value):
    if bp.has_attribute(key):
        bp.set_attribute(key, str(value))


def attr_int(bp, key, default=None):
    try:
        if not bp.has_attribute(key):
            return default
        a = bp.get_attribute(key)
        if hasattr(a, "as_int"):
            return a.as_int()
        return int(str(a))
    except Exception:
        return default


def destroy_dynamic_actors(client, world):
    ids = []

    for a in world.get_actors():
        tid = a.type_id

        if (
            tid.startswith("vehicle.")
            or tid.startswith("walker.pedestrian")
            or tid.startswith("controller.ai.walker")
            or tid.startswith("sensor.")
        ):
            ids.append(a.id)

    if ids:
        log(f"[CLEAN] Dynamic actor temizleniyor: {len(ids)}")
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in ids], True)
        time.sleep(0.8)


def is_car_bp(bp):
    tid = bp.id.lower()
    wheels = attr_int(bp, "number_of_wheels", None)

    if wheels != 4:
        return False

    banned = ["kawasaki", "yamaha", "vespa", "harley", "bike", "motorcycle"]
    return not any(x in tid for x in banned)


def choose_car_bps(bp_lib):
    preferred = [
        "vehicle.lincoln.mkz_2020",
        "vehicle.tesla.model3",
        "vehicle.audi.tt",
        "vehicle.dodge.charger_2020",
        "vehicle.mercedes.coupe",
        "vehicle.bmw.grandtourer",
    ]

    out = []

    for bp_id in preferred:
        found = bp_lib.filter(bp_id)
        if found and is_car_bp(found[0]):
            out.append(found[0])

    for bp in bp_lib.filter("vehicle.*"):
        if is_car_bp(bp) and bp not in out:
            out.append(bp)

    if not out:
        raise RuntimeError("Araba blueprint bulunamadı.")

    while len(out) < 3:
        out.append(out[0])

    return out


def choose_motor_bps(bp_lib):
    preferred = [
        "vehicle.kawasaki.ninja",
        "vehicle.yamaha.yzf",
        "vehicle.harley-davidson.low_rider",
        "vehicle.vespa.zx125",
    ]

    out = []

    for bp_id in preferred:
        found = bp_lib.filter(bp_id)
        if found:
            out.append(found[0])

    for bp in bp_lib.filter("vehicle.*"):
        tid = bp.id.lower()
        wheels = attr_int(bp, "number_of_wheels", None)

        if (
            wheels == 2
            or "kawasaki" in tid
            or "yamaha" in tid
            or "vespa" in tid
            or "harley" in tid
        ):
            if bp not in out:
                out.append(bp)

    if not out:
        raise RuntimeError("Motosiklet blueprint bulunamadı.")

    while len(out) < 2:
        out.append(out[0])

    return out


def choose_ped_bps(bp_lib):
    preferred = [
        "walker.pedestrian.0001",
        "walker.pedestrian.0002",
        "walker.pedestrian.0003",
        "walker.pedestrian.0004",
        "walker.pedestrian.0010",
        "walker.pedestrian.0011",
    ]

    out = []

    for bp_id in preferred:
        found = bp_lib.filter(bp_id)
        if found:
            out.append(found[0])

    for bp in bp_lib.filter("walker.pedestrian.*"):
        if bp not in out:
            out.append(bp)

    if not out:
        raise RuntimeError("Pedestrian blueprint bulunamadı.")

    while len(out) < 2:
        out.append(out[0])

    return out


def yaw_diff(a, b):
    d = abs(a - b) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return d


def build_straight_chain(start_wp, step_m=2.0, count=55):
    chain = [start_wp]
    cur = start_wp
    start_yaw = start_wp.transform.rotation.yaw

    for _ in range(count - 1):
        nxts = cur.next(step_m)

        if not nxts:
            break

        nxts = sorted(nxts, key=lambda w: yaw_diff(start_yaw, w.transform.rotation.yaw))
        nxt = nxts[0]

        if nxt.lane_type != carla.LaneType.Driving:
            break

        if yaw_diff(start_yaw, nxt.transform.rotation.yaw) > 8.0:
            break

        chain.append(nxt)
        cur = nxt

    return chain


def score_waypoint(wp):
    chain = build_straight_chain(wp, step_m=2.0, count=55)

    if len(chain) < 35:
        return -999999.0, chain

    start_yaw = chain[0].transform.rotation.yaw
    max_yaw = max(yaw_diff(start_yaw, x.transform.rotation.yaw) for x in chain)
    lane_width = getattr(wp, "lane_width", 3.0)

    score = len(chain) * 20.0 + lane_width * 20.0 - max_yaw * 50.0

    return score, chain


def find_best_straight_base(world):
    spawn_points = world.get_map().get_spawn_points()
    candidates = []

    for sp in spawn_points:
        wp = world.get_map().get_waypoint(
            sp.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if wp is None:
            continue

        score, chain = score_waypoint(wp)

        if score > -999999.0:
            candidates.append((score, wp, chain))

    if not candidates:
        raise RuntimeError("Düz yol için uygun waypoint bulunamadı.")

    candidates.sort(key=lambda x: x[0], reverse=True)

    score, wp, chain = candidates[0]
    log(f"[BASE] Düz yol seçildi. chain_len={len(chain)}, score={score:.1f}")

    tf = wp.transform
    tf.location.z += 0.45
    tf.rotation.pitch = 0.0
    tf.rotation.roll = 0.0

    return tf


def get_traffic_lights(world):
    lights = list(world.get_actors().filter("traffic.traffic_light*"))

    if not lights:
        lights = [a for a in world.get_actors() if "traffic_light" in a.type_id.lower()]

    return lights


def parse_light_state(name):
    name = name.lower().strip()

    if name == "red":
        return carla.TrafficLightState.Red
    if name == "yellow":
        return carla.TrafficLightState.Yellow
    if name == "green":
        return carla.TrafficLightState.Green

    raise RuntimeError("--light red | yellow | green olmalı")


def set_all_lights(world, state):
    lights = get_traffic_lights(world)

    for light in lights:
        try:
            light.set_state(state)
        except Exception:
            pass

        try:
            light.freeze(True)
        except Exception:
            pass

        try:
            light.set_red_time(9999.0)
            light.set_yellow_time(9999.0)
            light.set_green_time(9999.0)
        except Exception:
            pass

    log(f"[LIGHT] Trafik ışıkları sabitlendi: {len(lights)}")


def rel_tf(base_tf, forward, right, yaw_add=0.0, z_add=0.0):
    fwd = base_tf.get_forward_vector()
    rgt = base_tf.get_right_vector()

    loc = carla.Location(
        x=base_tf.location.x + fwd.x * forward + rgt.x * right,
        y=base_tf.location.y + fwd.y * forward + rgt.y * right,
        z=base_tf.location.z + z_add,
    )

    rot = carla.Rotation(
        pitch=0.0,
        yaw=base_tf.rotation.yaw + yaw_add,
        roll=0.0,
    )

    return carla.Transform(loc, rot)


def spawn_or_move(world, bp, tf, name):
    actor = world.try_spawn_actor(bp, tf)

    if actor is not None:
        log(f"[SPAWN] {name}: id={actor.id}, type={actor.type_id}")
        return actor

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    for sp in spawn_points[:80]:
        actor = world.try_spawn_actor(bp, sp)

        if actor is not None:
            try:
                actor.set_simulate_physics(False)
            except Exception:
                pass

            time.sleep(0.1)
            actor.set_transform(tf)
            log(f"[SPAWN+MOVE] {name}: id={actor.id}, type={actor.type_id}")
            return actor

    raise RuntimeError(f"{name} spawn edilemedi.")


def freeze_actor(actor):
    try:
        actor.set_autopilot(False)
    except Exception:
        pass

    try:
        actor.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0,
                hand_brake=True,
            )
        )
    except Exception:
        pass

    try:
        actor.set_simulate_physics(False)
    except Exception:
        pass


def set_spectator(world, ego_tf):
    spectator = world.get_spectator()
    fwd = ego_tf.get_forward_vector()

    loc = carla.Location(
        x=ego_tf.location.x - fwd.x * 10.0,
        y=ego_tf.location.y - fwd.y * 10.0,
        z=ego_tf.location.z + 5.0,
    )

    rot = carla.Rotation(
        pitch=-17.0,
        yaw=ego_tf.rotation.yaw,
        roll=0.0,
    )

    spectator.set_transform(carla.Transform(loc, rot))


def print_summary(world):
    log("")
    log("========== ACTOR CHECK ==========")

    for a in world.get_actors().filter("vehicle.*"):
        role = a.attributes.get("role_name", "")
        log(f"VEHICLE id={a.id} type={a.type_id} role_name={role}")

    for a in world.get_actors().filter("walker.pedestrian.*"):
        role = a.attributes.get("role_name", "")
        log(f"PEDESTRIAN id={a.id} type={a.type_id} role_name={role}")

    log("=================================")
    log("")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town01")
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--light", default="red", choices=["red", "yellow", "green"])

    parser.add_argument("--ego-speed", type=float, default=1.2)
    parser.add_argument("--move-distance", type=float, default=18.0)
    parser.add_argument("--hz", type=float, default=10.0)

    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(40.0)

    if args.reload:
        log(f"[WORLD] Harita yükleniyor: {args.town}")
        world = client.load_world(args.town)
        time.sleep(2.5)
    else:
        world = client.get_world()
        log(f"[WORLD] Mevcut harita kullanılıyor: {world.get_map().name}")

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    world.set_weather(
        carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=0.0,
            sun_altitude_angle=65.0,
            sun_azimuth_angle=35.0,
            fog_density=0.0,
            wetness=0.0,
        )
    )

    destroy_dynamic_actors(client, world)
    tick(world, 8)

    state = parse_light_state(args.light)
    set_all_lights(world, state)

    bp_lib = world.get_blueprint_library()

    car_bps = choose_car_bps(bp_lib)
    motor_bps = choose_motor_bps(bp_lib)
    ped_bps = choose_ped_bps(bp_lib)

    base_tf = find_best_straight_base(world)

    ego_bp = car_bps[0]
    car1_bp = car_bps[1]
    car2_bp = car_bps[2]
    motor1_bp = motor_bps[0]
    motor2_bp = motor_bps[1]
    ped1_bp = ped_bps[0]
    ped2_bp = ped_bps[1]

    set_attr(ego_bp, "role_name", "ego_vehicle")
    set_attr(car1_bp, "role_name", "adas_target_car_1")
    set_attr(car2_bp, "role_name", "adas_target_car_2")
    set_attr(motor1_bp, "role_name", "adas_target_motorcycle_1")
    set_attr(motor2_bp, "role_name", "adas_target_motorcycle_2")
    set_attr(ped1_bp, "role_name", "adas_target_pedestrian_1")
    set_attr(ped2_bp, "role_name", "adas_target_pedestrian_2")

    set_attr(ego_bp, "color", "0,0,255")
    set_attr(car1_bp, "color", "255,0,0")
    set_attr(car2_bp, "color", "255,255,255")
    set_attr(motor1_bp, "color", "255,255,0")
    set_attr(motor2_bp, "color", "0,255,255")
    set_attr(ped1_bp, "is_invincible", "false")
    set_attr(ped2_bp, "is_invincible", "false")

    ego = spawn_or_move(world, ego_bp, rel_tf(base_tf, 0.0, 0.0, 0.0, 0.0), "EGO_CAR")

    car1 = spawn_or_move(world, car1_bp, rel_tf(base_tf, 19.0, 0.0, 0.0, 0.05), "CAR_1_FRONT_CENTER")
    car2 = spawn_or_move(world, car2_bp, rel_tf(base_tf, 33.0, 2.5, 0.0, 0.05), "CAR_2_FRONT_RIGHT")

    motor1 = spawn_or_move(world, motor1_bp, rel_tf(base_tf, 13.0, -2.3, 0.0, 0.05), "MOTOR_1_LEFT")
    motor2 = spawn_or_move(world, motor2_bp, rel_tf(base_tf, 27.0, -3.2, 0.0, 0.05), "MOTOR_2_LEFT_FAR")

    ped1 = spawn_or_move(world, ped1_bp, rel_tf(base_tf, 11.0, -4.6, 165.0, 0.15), "PED_1_LEFT")
    ped2 = spawn_or_move(world, ped2_bp, rel_tf(base_tf, 24.0, -5.2, 165.0, 0.15), "PED_2_LEFT_FAR")

    actors = [ego, car1, car2, motor1, motor2, ped1, ped2]

    for actor in actors:
        freeze_actor(actor)

    tick(world, 10)
    print_summary(world)

    log("===================================================")
    log("CLEAN STRAIGHT ADAS SCENE READY")
    log(f"Map              : {world.get_map().name}")
    log(f"Ego speed        : {args.ego_speed:.2f} m/s")
    log(f"Move distance    : {args.move_distance:.2f} m")
    log("Objects          : 2 cars + 2 motorcycles + 2 pedestrians")
    log("Layout           : cars önde, motorlar solda, insanlar solda")
    log("Mode             : hedefler ego önünde sabit görüş alanında tutulur")
    log("Stop             : CTRL+C")
    log("===================================================")

    dt = 1.0 / max(args.hz, 1.0)
    start = time.monotonic()
    last_print = 0.0

    try:
        while True:
            elapsed = time.monotonic() - start

            forward = min(args.move_distance, elapsed * args.ego_speed)

            ego_tf = rel_tf(base_tf, forward, 0.0, 0.0, 0.0)

            ego.set_transform(ego_tf)

            car1.set_transform(rel_tf(ego_tf, 19.0, 0.0, 0.0, 0.05))
            car2.set_transform(rel_tf(ego_tf, 33.0, 2.5, 0.0, 0.05))

            motor1.set_transform(rel_tf(ego_tf, 13.0, -2.3, 0.0, 0.05))
            motor2.set_transform(rel_tf(ego_tf, 27.0, -3.2, 0.0, 0.05))

            ped1.set_transform(rel_tf(ego_tf, 11.0, -4.6, 165.0, 0.15))
            ped2.set_transform(rel_tf(ego_tf, 24.0, -5.2, 165.0, 0.15))

            set_all_lights(world, state)
            set_spectator(world, ego_tf)

            if elapsed - last_print >= 3.0:
                last_print = elapsed
                log(
                    f"[MOVE] t={elapsed:.1f}s ego_forward={forward:.1f}m "
                    f"visible=2car+2motor+2ped"
                )

            tick(world, 1)
            time.sleep(dt)

    except KeyboardInterrupt:
        log("")
        log("[STOP] Sahne durduruldu.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[ERROR] {e}")
        sys.exit(1)

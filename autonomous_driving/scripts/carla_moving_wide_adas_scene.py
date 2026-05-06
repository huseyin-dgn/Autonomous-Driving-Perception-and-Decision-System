#!/usr/bin/env python3
import argparse
import math
import random
import sys
import time
import carla


def log(msg):
    print(msg, flush=True)


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


def tick(world, n=1):
    for _ in range(n):
        try:
            world.tick()
        except RuntimeError:
            world.wait_for_tick()


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
        log(f"[CLEAN] Dinamik actor temizleniyor: {len(ids)}")
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in ids], True)
        time.sleep(0.8)


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


def set_light_state(light, state):
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


def is_car_bp(bp):
    tid = bp.id.lower()
    wheels = attr_int(bp, "number_of_wheels", None)

    if wheels != 4:
        return False

    banned = ["kawasaki", "yamaha", "vespa", "harley", "bike", "motorcycle"]
    return not any(x in tid for x in banned)


def choose_ego_car_bp(bp_lib):
    preferred = [
        "vehicle.lincoln.mkz_2020",
        "vehicle.tesla.model3",
        "vehicle.audi.tt",
        "vehicle.dodge.charger_2020",
        "vehicle.mercedes.coupe",
        "vehicle.bmw.grandtourer",
    ]

    for bp_id in preferred:
        found = bp_lib.filter(bp_id)
        if found and is_car_bp(found[0]):
            return found[0]

    for bp in bp_lib.filter("vehicle.*"):
        if is_car_bp(bp):
            return bp

    raise RuntimeError("4 teker ego araç blueprint bulunamadı.")


def choose_target_car_bp(bp_lib):
    preferred = [
        "vehicle.tesla.model3",
        "vehicle.audi.tt",
        "vehicle.dodge.charger_2020",
        "vehicle.mercedes.coupe",
        "vehicle.bmw.grandtourer",
        "vehicle.lincoln.mkz_2020",
    ]

    for bp_id in preferred:
        found = bp_lib.filter(bp_id)
        if found and is_car_bp(found[0]):
            return found[0]

    return choose_ego_car_bp(bp_lib)


def choose_motorcycle_bp(bp_lib):
    preferred = [
        "vehicle.kawasaki.ninja",
        "vehicle.yamaha.yzf",
        "vehicle.harley-davidson.low_rider",
        "vehicle.vespa.zx125",
    ]

    for bp_id in preferred:
        found = bp_lib.filter(bp_id)
        if found:
            return found[0]

    for bp in bp_lib.filter("vehicle.*"):
        tid = bp.id.lower()
        wheels = attr_int(bp, "number_of_wheels", None)

        if wheels == 2 or "kawasaki" in tid or "yamaha" in tid or "vespa" in tid or "harley" in tid:
            return bp

    raise RuntimeError("Motosiklet blueprint bulunamadı.")


def choose_pedestrian_bp(bp_lib):
    preferred = [
        "walker.pedestrian.0001",
        "walker.pedestrian.0002",
        "walker.pedestrian.0003",
        "walker.pedestrian.0004",
        "walker.pedestrian.0010",
    ]

    for bp_id in preferred:
        found = bp_lib.filter(bp_id)
        if found:
            return found[0]

    found = bp_lib.filter("walker.pedestrian.*")
    if not found:
        raise RuntimeError("Pedestrian blueprint bulunamadı.")

    return found[0]


def count_driving_lanes(wp):
    count = 1

    cur = wp
    for _ in range(4):
        try:
            cur = cur.get_left_lane()
        except Exception:
            cur = None

        if cur is None:
            break

        if cur.lane_type == carla.LaneType.Driving:
            count += 1
        else:
            break

    cur = wp
    for _ in range(4):
        try:
            cur = cur.get_right_lane()
        except Exception:
            cur = None

        if cur is None:
            break

        if cur.lane_type == carla.LaneType.Driving:
            count += 1
        else:
            break

    return count


def make_tf_before_light(light, back_distance):
    try:
        stop_wps = light.get_stop_waypoints()
    except Exception:
        stop_wps = []

    if not stop_wps:
        return []

    out = []

    for wp in stop_wps:
        for dist in [back_distance, back_distance + 5.0, back_distance + 10.0, 22.0, 28.0, 34.0]:
            prevs = wp.previous(dist)

            if prevs:
                tf = prevs[0].transform
                used_wp = prevs[0]
            else:
                tf = wp.transform
                used_wp = wp
                fwd = tf.get_forward_vector()
                tf.location.x -= fwd.x * dist
                tf.location.y -= fwd.y * dist
                tf.location.z -= fwd.z * dist

            tf.location.z += 0.45
            tf.rotation.pitch = 0.0
            tf.rotation.roll = 0.0

            lane_count = count_driving_lanes(used_wp)
            lane_width = getattr(used_wp, "lane_width", 3.0)

            score = lane_count * 100.0 + lane_width * 10.0 + dist
            out.append((score, light, tf, lane_count, lane_width))

    return out


def find_wide_base_transform(world, lights, back_distance):
    candidates = []

    for light in lights:
        candidates.extend(make_tf_before_light(light, back_distance))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        score, light, tf, lane_count, lane_width = candidates[0]
        log(f"[BASE] Geniş yol adayı seçildi: lane_count={lane_count}, lane_width={lane_width:.2f}, score={score:.1f}")
        return light, tf

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("Spawn point bulunamadı.")

    log("[BASE] Traffic light stop waypoint bulunamadı, spawn point fallback kullanılıyor.")
    return lights[0], spawn_points[0]


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
        x=ego_tf.location.x - fwd.x * 9.0,
        y=ego_tf.location.y - fwd.y * 9.0,
        z=ego_tf.location.z + 5.2,
    )

    rot = carla.Rotation(
        pitch=-18.0,
        yaw=ego_tf.rotation.yaw,
        roll=0.0,
    )

    spectator.set_transform(carla.Transform(loc, rot))


def loop_distance(elapsed, speed, max_distance):
    period = max(12.0, (max_distance / max(speed, 0.1)) * 2.0)
    phase = (elapsed % period) / period
    return 0.5 * max_distance * (1.0 - math.cos(2.0 * math.pi * phase))


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
    parser.add_argument("--town", default="Town05")
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--light", default="red", choices=["red", "yellow", "green"])

    parser.add_argument("--ego-back", type=float, default=34.0)
    parser.add_argument("--ego-speed", type=float, default=1.2)
    parser.add_argument("--loop-distance", type=float, default=16.0)
    parser.add_argument("--hz", type=float, default=8.0)

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

    bp_lib = world.get_blueprint_library()

    lights = get_traffic_lights(world)
    if not lights:
        raise RuntimeError("Traffic light bulunamadı. Town05 veya Town01 ile dene.")

    state = parse_light_state(args.light)

    for light in lights:
        set_light_state(light, state)

    chosen_light, base_tf = find_wide_base_transform(world, lights, args.ego_back)
    set_light_state(chosen_light, state)

    ego_bp = choose_ego_car_bp(bp_lib)
    set_attr(ego_bp, "role_name", "ego_vehicle")
    set_attr(ego_bp, "color", "0,0,255")

    car_bp = choose_target_car_bp(bp_lib)
    set_attr(car_bp, "role_name", "adas_target_car")
    set_attr(car_bp, "color", "255,0,0")

    motor_bp = choose_motorcycle_bp(bp_lib)
    set_attr(motor_bp, "role_name", "adas_target_motorcycle")
    set_attr(motor_bp, "color", "255,255,0")

    ped_bp = choose_pedestrian_bp(bp_lib)
    set_attr(ped_bp, "role_name", "adas_target_pedestrian")
    set_attr(ped_bp, "is_invincible", "false")

    ego = spawn_or_move(
        world,
        ego_bp,
        rel_tf(base_tf, 0.0, 0.0, 0.0, 0.0),
        "EGO_CAR",
    )

    target_car = spawn_or_move(
        world,
        car_bp,
        rel_tf(base_tf, 26.0, 0.0, 0.0, 0.05),
        "TARGET_CAR_CENTER",
    )

    motorcycle = spawn_or_move(
        world,
        motor_bp,
        rel_tf(base_tf, 18.0, -3.0, 0.0, 0.05),
        "TARGET_MOTORCYCLE_LEFT",
    )

    pedestrian = spawn_or_move(
        world,
        ped_bp,
        rel_tf(base_tf, 15.5, -6.2, 160.0, 0.15),
        "TARGET_PEDESTRIAN_LEFT",
    )

    freeze_actor(ego)
    freeze_actor(target_car)
    freeze_actor(motorcycle)

    try:
        pedestrian.set_simulate_physics(False)
    except Exception:
        pass

    tick(world, 10)
    print_summary(world)

    log("===================================================")
    log("CARLA MOVING WIDE ADAS SCENE READY")
    log(f"Map              : {world.get_map().name}")
    log(f"Traffic light    : {args.light.upper()}")
    log(f"Ego speed        : {args.ego_speed:.2f} m/s")
    log(f"Loop distance    : {args.loop_distance:.2f} m")
    log("Layout           : geniş yol seçilir; motor + insan sol tarafta")
    log("Objects          : ego + center car + left motorcycle + left pedestrian + traffic light")
    log("Stop             : CTRL+C")
    log("===================================================")

    dt = 1.0 / max(args.hz, 1.0)
    start = time.monotonic()
    last_print = 0.0

    try:
        while True:
            elapsed = time.monotonic() - start
            d = loop_distance(elapsed, args.ego_speed, args.loop_distance)

            ego_tf = rel_tf(base_tf, d, 0.0, 0.0, 0.0)

            car_tf = rel_tf(
                base_tf,
                d + 26.0 + 0.8 * math.sin(elapsed * 0.30),
                0.0,
                0.0,
                0.05,
            )

            motor_tf = rel_tf(
                base_tf,
                d + 18.0 + 0.8 * math.sin(elapsed * 0.45),
                -3.0,
                0.0,
                0.05,
            )

            ped_tf = rel_tf(
                base_tf,
                d + 15.5,
                -6.2 + 0.5 * math.sin(elapsed * 0.35),
                160.0,
                0.15,
            )

            ego.set_transform(ego_tf)
            target_car.set_transform(car_tf)
            motorcycle.set_transform(motor_tf)
            pedestrian.set_transform(ped_tf)

            set_light_state(chosen_light, state)
            set_spectator(world, ego_tf)

            if elapsed - last_print >= 3.0:
                last_print = elapsed
                log(
                    f"[MOVE] t={elapsed:.1f}s ego_d={d:.1f}m "
                    f"car=center+26m motor=left+18m pedestrian=left+15.5m"
                )

            tick(world, 1)
            time.sleep(dt)

    except KeyboardInterrupt:
        log("")
        log("[STOP] Hareketli geniş yol sahnesi durduruldu.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[ERROR] {e}")
        sys.exit(1)

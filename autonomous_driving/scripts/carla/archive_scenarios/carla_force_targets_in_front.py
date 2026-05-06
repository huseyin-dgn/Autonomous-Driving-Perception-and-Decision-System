#!/usr/bin/env python3
import argparse
import random
import sys
import time
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

        if wheels == 2 or "kawasaki" in tid or "yamaha" in tid or "vespa" in tid or "harley" in tid:
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


def find_ego(world, bp_lib):
    vehicles = list(world.get_actors().filter("vehicle.*"))

    for v in vehicles:
        if v.attributes.get("role_name", "") == "ego_vehicle":
            log(f"[EGO] Mevcut ego bulundu: id={v.id}, type={v.type_id}")
            return v

    for v in vehicles:
        tid = v.type_id.lower()
        if "kawasaki" not in tid and "yamaha" not in tid and "vespa" not in tid and "harley" not in tid:
            log(f"[EGO] role_name yok, ilk araba ego yapılıyor: id={v.id}, type={v.type_id}")
            return v

    car_bp = choose_car_bps(bp_lib)[0]
    set_attr(car_bp, "role_name", "ego_vehicle")
    set_attr(car_bp, "color", "0,0,255")

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("Ego spawn point bulunamadı.")

    ego = world.spawn_actor(car_bp, spawn_points[0])
    log(f"[EGO] Yeni ego spawn edildi: id={ego.id}, type={ego.type_id}")
    return ego


def clean_old_targets(client, world):
    ids = []

    for a in world.get_actors():
        role = a.attributes.get("role_name", "")
        if role.startswith("adas_target_"):
            ids.append(a.id)

    if ids:
        log(f"[CLEAN] Eski hedef actor temizleniyor: {len(ids)}")
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in ids], True)
        time.sleep(0.5)


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
        freeze_actor(actor)
        return actor

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    for sp in spawn_points[:80]:
        actor = world.try_spawn_actor(bp, sp)
        if actor is not None:
            freeze_actor(actor)
            time.sleep(0.1)
            actor.set_transform(tf)
            log(f"[SPAWN+MOVE] {name}: id={actor.id}, type={actor.type_id}")
            return actor

    raise RuntimeError(f"{name} spawn edilemedi.")


def set_all_lights_red(world):
    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    if not lights:
        lights = [a for a in world.get_actors() if "traffic_light" in a.type_id.lower()]

    for light in lights:
        try:
            light.set_state(carla.TrafficLightState.Red)
            light.freeze(True)
            light.set_red_time(9999.0)
            light.set_yellow_time(9999.0)
            light.set_green_time(9999.0)
        except Exception:
            pass

    log(f"[LIGHT] Kırmızıya sabitlenen ışık sayısı: {len(lights)}")


def set_spectator(world, ego_tf):
    spectator = world.get_spectator()
    fwd = ego_tf.get_forward_vector()

    loc = carla.Location(
        x=ego_tf.location.x - fwd.x * 8.0,
        y=ego_tf.location.y - fwd.y * 8.0,
        z=ego_tf.location.z + 4.5,
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
    parser.add_argument("--ego-speed", type=float, default=1.0)
    parser.add_argument("--move-distance", type=float, default=12.0)
    parser.add_argument("--hz", type=float, default=10.0)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()

    log(f"[WORLD] Mevcut harita: {world.get_map().name}")

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

    bp_lib = world.get_blueprint_library()

    clean_old_targets(client, world)
    tick(world, 3)

    ego = find_ego(world, bp_lib)
    freeze_actor(ego)

    car_bps = choose_car_bps(bp_lib)
    motor_bps = choose_motor_bps(bp_lib)
    ped_bps = choose_ped_bps(bp_lib)

    car1_bp = car_bps[1]
    car2_bp = car_bps[2]
    motor1_bp = motor_bps[0]
    motor2_bp = motor_bps[1]
    ped1_bp = ped_bps[0]
    ped2_bp = ped_bps[1]

    set_attr(car1_bp, "role_name", "adas_target_car_1")
    set_attr(car2_bp, "role_name", "adas_target_car_2")
    set_attr(motor1_bp, "role_name", "adas_target_motorcycle_1")
    set_attr(motor2_bp, "role_name", "adas_target_motorcycle_2")
    set_attr(ped1_bp, "role_name", "adas_target_pedestrian_1")
    set_attr(ped2_bp, "role_name", "adas_target_pedestrian_2")

    set_attr(car1_bp, "color", "255,0,0")
    set_attr(car2_bp, "color", "255,255,255")
    set_attr(motor1_bp, "color", "255,255,0")
    set_attr(motor2_bp, "color", "0,255,255")
    set_attr(ped1_bp, "is_invincible", "false")
    set_attr(ped2_bp, "is_invincible", "false")

    ego_tf = ego.get_transform()

    car1 = spawn_or_move(world, car1_bp, rel_tf(ego_tf, 18.0, 0.0, 0.0, 0.05), "CAR_1_FRONT")
    car2 = spawn_or_move(world, car2_bp, rel_tf(ego_tf, 28.0, 2.0, 0.0, 0.05), "CAR_2_FRONT_RIGHT")

    motor1 = spawn_or_move(world, motor1_bp, rel_tf(ego_tf, 12.0, -1.6, 0.0, 0.05), "MOTOR_1_LEFT")
    motor2 = spawn_or_move(world, motor2_bp, rel_tf(ego_tf, 22.0, -2.4, 0.0, 0.05), "MOTOR_2_LEFT")

    ped1 = spawn_or_move(world, ped1_bp, rel_tf(ego_tf, 9.5, -2.8, 180.0, 0.15), "PED_1_LEFT")
    ped2 = spawn_or_move(world, ped2_bp, rel_tf(ego_tf, 17.0, -3.2, 180.0, 0.15), "PED_2_LEFT")

    set_all_lights_red(world)
    tick(world, 5)
    print_summary(world)

    log("===================================================")
    log("FORCED FRONT ADAS SCENE READY")
    log("Objects: 2 cars + 2 motorcycles + 2 pedestrians")
    log("Layout : hedefler mevcut ego kamerasının önüne zorla konur")
    log("Move   : ego yavaş hareket eder, hedefler görüşte tutulur")
    log("Stop   : CTRL+C")
    log("===================================================")

    start_tf = ego.get_transform()
    start_time = time.monotonic()
    dt = 1.0 / max(args.hz, 1.0)
    last_print = 0.0

    try:
        while True:
            elapsed = time.monotonic() - start_time
            d = min(args.move_distance, elapsed * args.ego_speed)

            ego_tf = rel_tf(start_tf, d, 0.0, 0.0, 0.0)
            ego.set_transform(ego_tf)

            car1.set_transform(rel_tf(ego_tf, 18.0, 0.0, 0.0, 0.05))
            car2.set_transform(rel_tf(ego_tf, 28.0, 2.0, 0.0, 0.05))

            motor1.set_transform(rel_tf(ego_tf, 12.0, -1.6, 0.0, 0.05))
            motor2.set_transform(rel_tf(ego_tf, 22.0, -2.4, 0.0, 0.05))

            ped1.set_transform(rel_tf(ego_tf, 9.5, -2.8, 180.0, 0.15))
            ped2.set_transform(rel_tf(ego_tf, 17.0, -3.2, 180.0, 0.15))

            set_all_lights_red(world)
            set_spectator(world, ego_tf)

            if elapsed - last_print >= 3.0:
                last_print = elapsed
                log(f"[MOVE] t={elapsed:.1f}s ego_forward={d:.1f} visible=2car+2motor+2ped")

            tick(world, 1)
            time.sleep(dt)

    except KeyboardInterrupt:
        log("")
        log("[STOP] Forced front sahne durduruldu.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[ERROR] {e}")
        sys.exit(1)

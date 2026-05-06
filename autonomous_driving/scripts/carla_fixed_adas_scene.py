#!/usr/bin/env python3
import argparse
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
        attr = bp.get_attribute(key)
        if hasattr(attr, "as_int"):
            return attr.as_int()
        return int(str(attr))
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
        log(f"[CLEAN] Dinamik actor temizleniyor: {len(ids)}")
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in ids], True)
        time.sleep(0.8)


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


def parse_light_state(name):
    name = name.lower().strip()
    if name == "red":
        return carla.TrafficLightState.Red
    if name == "yellow":
        return carla.TrafficLightState.Yellow
    if name == "green":
        return carla.TrafficLightState.Green
    raise RuntimeError("--light red | yellow | green olmalı")


def tick(world, n=5):
    for _ in range(n):
        try:
            world.tick()
        except RuntimeError:
            world.wait_for_tick()


def get_traffic_lights(world):
    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    if not lights:
        lights = [a for a in world.get_actors() if "traffic_light" in a.type_id.lower()]
    return lights


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
        if found:
            bp = found[0]
            wheels = attr_int(bp, "number_of_wheels", 4)
            if wheels == 4:
                return bp

    cars = []
    for bp in bp_lib.filter("vehicle.*"):
        wheels = attr_int(bp, "number_of_wheels", None)
        tid = bp.id.lower()
        if wheels == 4 and "bike" not in tid and "motorcycle" not in tid and "vespa" not in tid and "kawasaki" not in tid and "yamaha" not in tid:
            cars.append(bp)

    if not cars:
        raise RuntimeError("4 teker ego araç blueprint bulunamadı.")

    return cars[0]


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
        if found:
            bp = found[0]
            wheels = attr_int(bp, "number_of_wheels", 4)
            if wheels == 4:
                return bp

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

    motors = []
    for bp in bp_lib.filter("vehicle.*"):
        wheels = attr_int(bp, "number_of_wheels", None)
        tid = bp.id.lower()
        if wheels == 2 or "kawasaki" in tid or "yamaha" in tid or "harley" in tid or "vespa" in tid:
            motors.append(bp)

    if not motors:
        raise RuntimeError("Motosiklet blueprint bulunamadı.")

    return motors[0]


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


def transform_back_from_light_stop(light, back_distance):
    stop_wps = []

    try:
        stop_wps = light.get_stop_waypoints()
    except Exception:
        stop_wps = []

    if not stop_wps:
        return None

    wp = stop_wps[0]
    prevs = wp.previous(back_distance)

    if prevs:
        tf = prevs[0].transform
    else:
        tf = wp.transform
        fwd = tf.get_forward_vector()
        tf.location.x -= fwd.x * back_distance
        tf.location.y -= fwd.y * back_distance
        tf.location.z -= fwd.z * back_distance

    tf.location.z += 0.45
    tf.rotation.pitch = 0.0
    tf.rotation.roll = 0.0
    return tf


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


def freeze_vehicle(actor):
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


def spawn_direct_or_teleport(world, bp, target_tf, name):
    actor = world.try_spawn_actor(bp, target_tf)

    if actor is not None:
        log(f"[SPAWN] {name}: id={actor.id}, type={actor.type_id}")
        return actor

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    for sp in spawn_points[:30]:
        actor = world.try_spawn_actor(bp, sp)
        if actor is not None:
            try:
                actor.set_simulate_physics(False)
            except Exception:
                pass

            time.sleep(0.1)
            actor.set_transform(target_tf)
            log(f"[SPAWN+MOVE] {name}: id={actor.id}, type={actor.type_id}")
            return actor

    raise RuntimeError(f"{name} spawn edilemedi.")


def find_ego_transform(world, lights, back_distance):
    random.shuffle(lights)

    for light in lights:
        tf = transform_back_from_light_stop(light, back_distance)
        if tf is not None:
            return light, tf

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("Spawn point bulunamadı.")

    return lights[0], spawn_points[0]


def set_spectator(world, ego_tf):
    spectator = world.get_spectator()
    fwd = ego_tf.get_forward_vector()

    loc = carla.Location(
        x=ego_tf.location.x - fwd.x * 8.0,
        y=ego_tf.location.y - fwd.y * 8.0,
        z=ego_tf.location.z + 4.5,
    )

    rot = carla.Rotation(
        pitch=-18.0,
        yaw=ego_tf.rotation.yaw,
        roll=0.0,
    )

    spectator.set_transform(carla.Transform(loc, rot))


def print_actor_summary(world):
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
    parser.add_argument("--ego-back", type=float, default=18.0)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    if args.reload:
        log(f"[WORLD] Hafif harita yükleniyor: {args.town}")
        world = client.load_world(args.town)
        time.sleep(2.0)
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
    tick(world, 5)

    bp_lib = world.get_blueprint_library()

    lights = get_traffic_lights(world)
    if not lights:
        raise RuntimeError(
            "Bu world içinde traffic light yok. CARLA'yı açtıktan sonra şu komutla dene: "
            "python3 scripts/carla_fixed_adas_scene.py --reload --town Town01 --light red"
        )

    state = parse_light_state(args.light)

    for light in lights:
        set_light_state(light, state)

    chosen_light, ego_tf = find_ego_transform(world, lights, args.ego_back)
    set_light_state(chosen_light, state)

    ego_bp = choose_ego_car_bp(bp_lib)
    set_attr(ego_bp, "role_name", "ego_vehicle")
    set_attr(ego_bp, "color", "0,0,255")

    ego = spawn_direct_or_teleport(world, ego_bp, ego_tf, "EGO_CAR")
    freeze_vehicle(ego)

    car_bp = choose_target_car_bp(bp_lib)
    set_attr(car_bp, "role_name", "adas_target_car")
    set_attr(car_bp, "color", "255,0,0")

    motor_bp = choose_motorcycle_bp(bp_lib)
    set_attr(motor_bp, "role_name", "adas_target_motorcycle")
    set_attr(motor_bp, "color", "255,255,0")

    ped_bp = choose_pedestrian_bp(bp_lib)
    set_attr(ped_bp, "role_name", "adas_target_pedestrian")
    set_attr(ped_bp, "is_invincible", "false")

    car_tf = rel_tf(
        ego_tf,
        forward=15.0,
        right=0.2,
        yaw_add=0.0,
        z_add=0.05,
    )

    motor_tf = rel_tf(
        ego_tf,
        forward=10.5,
        right=-2.0,
        yaw_add=0.0,
        z_add=0.05,
    )

    ped_tf = rel_tf(
        ego_tf,
        forward=8.0,
        right=2.4,
        yaw_add=180.0,
        z_add=0.15,
    )

    target_car = spawn_direct_or_teleport(world, car_bp, car_tf, "TARGET_CAR")
    motorcycle = spawn_direct_or_teleport(world, motor_bp, motor_tf, "TARGET_MOTORCYCLE")
    pedestrian = spawn_direct_or_teleport(world, ped_bp, ped_tf, "TARGET_PEDESTRIAN")

    freeze_vehicle(target_car)
    freeze_vehicle(motorcycle)

    try:
        pedestrian.set_simulate_physics(False)
    except Exception:
        pass

    tick(world, 10)
    set_spectator(world, ego_tf)
    tick(world, 5)

    print_actor_summary(world)

    log("===================================================")
    log("CARLA CLOSE ADAS SCENE READY")
    log(f"Map              : {world.get_map().name}")
    log(f"Traffic light    : {args.light.upper()}")
    log(f"Ego              : {ego.id} / {ego.type_id} / role_name=ego_vehicle")
    log(f"Target car       : {target_car.id} / {target_car.type_id}")
    log(f"Motorcycle       : {motorcycle.id} / {motorcycle.type_id}")
    log(f"Pedestrian       : {pedestrian.id} / {pedestrian.type_id}")
    log("Layout           : hepsi ego kameranın önüne yakın kondu")
    log("===================================================")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[ERROR] {e}")
        sys.exit(1)

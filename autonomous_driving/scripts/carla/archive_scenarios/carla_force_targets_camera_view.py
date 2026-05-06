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
        "vehicle.tesla.model3",
        "vehicle.audi.tt",
        "vehicle.dodge.charger_2020",
        "vehicle.lincoln.mkz_2020",
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

    while len(out) < 2:
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


def clean_old_targets(client, world):
    ids = []

    for a in world.get_actors():
        role = a.attributes.get("role_name", "")
        if role.startswith("adas_target_"):
            ids.append(a.id)

    if ids:
        log(f"[CLEAN] Eski hedefler siliniyor: {len(ids)}")
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in ids], True)
        time.sleep(0.5)


def find_rgb_camera(world):
    cams = list(world.get_actors().filter("sensor.camera.rgb"))

    if not cams:
        return None

    def cam_score(cam):
        role = cam.attributes.get("role_name", "").lower()
        score = 0

        if "front" in role:
            score += 100
        if "rgb" in role:
            score += 20
        if "adas" in role:
            score += 20

        return score

    cams.sort(key=cam_score, reverse=True)
    cam = cams[0]
    log(f"[CAMERA] Aktif RGB kamera: id={cam.id}, role={cam.attributes.get('role_name', '')}")
    return cam


def find_ground_z(world):
    vehicles = list(world.get_actors().filter("vehicle.*"))

    for v in vehicles:
        role = v.attributes.get("role_name", "")
        if role == "ego_vehicle":
            return v.get_transform().location.z

    if vehicles:
        return vehicles[0].get_transform().location.z

    return 0.5


def rel_camera_tf(cam_tf, ground_z, forward, right, yaw_add=0.0, z_add=0.0):
    yaw = cam_tf.rotation.yaw

    base_rot = carla.Rotation(
        pitch=0.0,
        yaw=yaw,
        roll=0.0,
    )

    base_tf = carla.Transform(cam_tf.location, base_rot)

    fwd = base_tf.get_forward_vector()
    rgt = base_tf.get_right_vector()

    loc = carla.Location(
        x=cam_tf.location.x + fwd.x * forward + rgt.x * right,
        y=cam_tf.location.y + fwd.y * forward + rgt.y * right,
        z=ground_z + z_add,
    )

    rot = carla.Rotation(
        pitch=0.0,
        yaw=yaw + yaw_add,
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


def set_lights_red(world):
    lights = list(world.get_actors().filter("traffic.traffic_light*"))

    for light in lights:
        try:
            light.set_state(carla.TrafficLightState.Red)
            light.freeze(True)
            light.set_red_time(9999.0)
            light.set_yellow_time(9999.0)
            light.set_green_time(9999.0)
        except Exception:
            pass

    log(f"[LIGHT] Kırmızıya sabitlenen ışık: {len(lights)}")


def print_summary(world):
    log("")
    log("========== TARGET CHECK ==========")

    for a in world.get_actors():
        role = a.attributes.get("role_name", "")
        if role.startswith("adas_target_"):
            log(f"{role}: id={a.id}, type={a.type_id}")

    log("==================================")
    log("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--hz", type=float, default=10.0)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    world = client.get_world()
    log(f"[WORLD] {world.get_map().name}")

    bp_lib = world.get_blueprint_library()

    cam = None
    for _ in range(50):
        cam = find_rgb_camera(world)
        if cam is not None:
            break
        log("[WAIT] RGB camera bekleniyor. Publisher açık mı?")
        time.sleep(0.2)

    if cam is None:
        raise RuntimeError("RGB camera bulunamadı. Önce publisher'ı çalıştır.")

    clean_old_targets(client, world)
    tick(world, 3)

    car_bps = choose_car_bps(bp_lib)
    motor_bps = choose_motor_bps(bp_lib)
    ped_bps = choose_ped_bps(bp_lib)

    car1_bp = car_bps[0]
    car2_bp = car_bps[1]
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

    ground_z = find_ground_z(world)
    cam_tf = cam.get_transform()

    car1 = spawn_or_move(world, car1_bp, rel_camera_tf(cam_tf, ground_z, 20.0, 2.2, 0.0, 0.05), "CAR_1_RIGHT")
    car2 = spawn_or_move(world, car2_bp, rel_camera_tf(cam_tf, ground_z, 24.0, -4.2, 0.0, 0.05), "CAR_2_LEFT_FAR")

    motor1 = spawn_or_move(world, motor1_bp, rel_camera_tf(cam_tf, ground_z, 10.0, -0.7, 28.0, 0.05), "MOTOR_1_CENTER_LEFT")
    motor2 = spawn_or_move(world, motor2_bp, rel_camera_tf(cam_tf, ground_z, 14.0, 0.9, -28.0, 0.05), "MOTOR_2_CENTER_RIGHT")

    ped1 = spawn_or_move(world, ped1_bp, rel_camera_tf(cam_tf, ground_z, 12.0, -3.4, 180.0, 0.15), "PED_1_LEFT")
    ped2 = spawn_or_move(world, ped2_bp, rel_camera_tf(cam_tf, ground_z, 18.0, -4.0, 180.0, 0.15), "PED_2_LEFT_FAR")

    set_lights_red(world)
    print_summary(world)

    log("===================================================")
    log("CAMERA VIEW TARGET SCENE READY")
    log("2 car + 2 motorcycle + 2 pedestrian")
    log("Hedefler aktif RGB kameranın görüş alanına göre konur.")
    log("Bu script açık kalmalı.")
    log("Stop: CTRL+C")
    log("===================================================")

    dt = 1.0 / max(args.hz, 1.0)

    try:
        while True:
            cam_tf = cam.get_transform()
            ground_z = find_ground_z(world)

            car1.set_transform(rel_camera_tf(cam_tf, ground_z, 20.0, 2.2, 0.0, 0.05))
            car2.set_transform(rel_camera_tf(cam_tf, ground_z, 24.0, -4.2, 0.0, 0.05))

            motor1.set_transform(rel_camera_tf(cam_tf, ground_z, 10.0, -0.7, 28.0, 0.05))
            motor2.set_transform(rel_camera_tf(cam_tf, ground_z, 14.0, 0.9, -28.0, 0.05))

            ped1.set_transform(rel_camera_tf(cam_tf, ground_z, 12.0, -3.4, 180.0, 0.15))
            ped2.set_transform(rel_camera_tf(cam_tf, ground_z, 18.0, -4.0, 180.0, 0.15))

            set_lights_red(world)
            tick(world, 1)
            time.sleep(dt)

    except KeyboardInterrupt:
        log("")
        log("[STOP] Camera-view target sahnesi durduruldu.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[ERROR] {e}")
        sys.exit(1)

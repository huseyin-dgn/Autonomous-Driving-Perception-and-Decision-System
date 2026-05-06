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


def choose_sign_bp(bp_lib):
    preferred = [
        "traffic.speed_limit.30",
        "traffic.speed_limit.40",
        "traffic.speed_limit.50",
        "traffic.stop",
        "traffic.yield",
    ]

    for bp_id in preferred:
        found = bp_lib.filter(bp_id)
        if found:
            return found[0]

    found = bp_lib.filter("traffic.*")
    for bp in found:
        tid = bp.id.lower()
        if "speed_limit" in tid or "stop" in tid or "yield" in tid:
            return bp

    return None


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
        if v.attributes.get("role_name", "") == "ego_vehicle":
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


def place_clean_traffic_light(world, cam_tf, ground_z, light_yaw_add):
    lights = list(world.get_actors().filter("traffic.traffic_light*"))

    if not lights:
        log("[LIGHT] Haritada traffic light actor bulunamadı.")
        return None

    target = lights[0]

    # Diğer ışıkları kadrajdan uzaklaştırmaya çalış.
    # CARLA bazı haritalarda trafik ışığını taşımaya izin verir, bazılarında izin vermez.
    for i, light in enumerate(lights):
        try:
            if light.id != target.id:
                far_tf = rel_camera_tf(cam_tf, ground_z, 120.0 + i * 3.0, 30.0, 0.0, 0.0)
                light.set_transform(far_tf)
        except Exception:
            pass

    try:
        tf = rel_camera_tf(cam_tf, ground_z, 12.5, 4.2, light_yaw_add, 0.0)
        target.set_transform(tf)
    except Exception:
        pass

    try:
        target.set_state(carla.TrafficLightState.Red)
        target.freeze(True)
        target.set_red_time(9999.0)
        target.set_yellow_time(9999.0)
        target.set_green_time(9999.0)
    except Exception:
        pass

    log(f"[LIGHT] Temiz trafik ışığı yerleştirildi: id={target.id}, yaw_add={light_yaw_add}")
    return target


def spawn_or_place_traffic_sign(world, bp_lib, cam_tf, ground_z):
    sign_bp = choose_sign_bp(bp_lib)

    if sign_bp is not None:
        set_attr(sign_bp, "role_name", "adas_target_traffic_sign")
        tf = rel_camera_tf(cam_tf, ground_z, 13.0, 4.8, 180.0, 2.0)
        try:
            sign = world.try_spawn_actor(sign_bp, tf)
            if sign is not None:
                log(f"[SIGN] Trafik işareti spawn edildi: id={sign.id}, type={sign.type_id}")
                return sign
        except Exception:
            pass

    # Spawn olmazsa var olan işareti taşımayı dene.
    candidates = []
    for a in world.get_actors():
        tid = a.type_id.lower()
        if "traffic.speed_limit" in tid or "traffic.stop" in tid or "traffic.yield" in tid:
            candidates.append(a)

    if candidates:
        sign = candidates[0]
        try:
            sign.set_transform(rel_camera_tf(cam_tf, ground_z, 13.0, 4.8, 180.0, 2.0))
            log(f"[SIGN] Mevcut trafik işareti taşındı: id={sign.id}, type={sign.type_id}")
            return sign
        except Exception:
            pass

    log("[SIGN] Trafik işareti oluşturulamadı/taşınamadı.")
    return None


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
    parser.add_argument("--light-yaw-add", type=float, default=0.0)
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
        log("[WAIT] RGB camera bekleniyor. Önce publisher açık olmalı.")
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

    cam_tf = cam.get_transform()
    ground_z = find_ground_z(world)

    # Arabalar daha net: merkeze ve sağ tarafa alınır.
    car1 = spawn_or_move(world, car1_bp, rel_camera_tf(cam_tf, ground_z, 19.5, 3.0, 0.0, 0.05), "CAR_1_RIGHT_CLEAR")
    car2 = spawn_or_move(world, car2_bp, rel_camera_tf(cam_tf, ground_z, 31.0, 0.0, 0.0, 0.05), "CAR_2_CENTER_FAR_CLEAR")

    # Motorlardan biri sol, biri sağ tarafta. Arabaların üstüne binmeyecek.
    motor1 = spawn_or_move(world, motor1_bp, rel_camera_tf(cam_tf, ground_z, 10.0, -3.0, 32.0, 0.05), "MOTOR_1_LEFT")
    motor2 = spawn_or_move(world, motor2_bp, rel_camera_tf(cam_tf, ground_z, 13.0, 2.0, -32.0, 0.05), "MOTOR_2_RIGHT")

    # İnsanlardan biri sol, biri sağ tarafta.
    ped1 = spawn_or_move(world, ped1_bp, rel_camera_tf(cam_tf, ground_z, 11.0, -5.4, 180.0, 0.15), "PED_1_LEFT")
    ped2 = spawn_or_move(world, ped2_bp, rel_camera_tf(cam_tf, ground_z, 16.0, 5.2, 180.0, 0.15), "PED_2_RIGHT")

    sign = spawn_or_place_traffic_sign(world, bp_lib, cam_tf, ground_z)
    light = place_clean_traffic_light(world, cam_tf, ground_z, args.light_yaw_add)

    print_summary(world)

    log("===================================================")
    log("CAMERA VIEW FULL TEST SCENE READY")
    log("2 car + 2 motorcycle + 2 pedestrian + traffic sign + traffic light")
    log("Motorlar sola çekildi, arabalar daha net bırakıldı.")
    log("Trafik işareti sağ tarafa, trafik ışığı sağ üst/ön tarafa konur.")
    log("Bu script açık kalmalı.")
    log("Stop: CTRL+C")
    log("===================================================")

    dt = 1.0 / max(args.hz, 1.0)
    last_light_fix = 0.0

    try:
        while True:
            cam_tf = cam.get_transform()
            ground_z = find_ground_z(world)

            car1.set_transform(rel_camera_tf(cam_tf, ground_z, 19.5, 3.0, 0.0, 0.05))
            car2.set_transform(rel_camera_tf(cam_tf, ground_z, 31.0, 0.0, 0.0, 0.05))

            motor1.set_transform(rel_camera_tf(cam_tf, ground_z, 10.0, -3.0, 32.0, 0.05))
            motor2.set_transform(rel_camera_tf(cam_tf, ground_z, 13.0, 2.0, -32.0, 0.05))

            ped1.set_transform(rel_camera_tf(cam_tf, ground_z, 11.0, -5.4, 180.0, 0.15))
            ped2.set_transform(rel_camera_tf(cam_tf, ground_z, 16.0, 5.2, 180.0, 0.15))

            if sign is not None:
                try:
                    sign.set_transform(rel_camera_tf(cam_tf, ground_z, 13.0, 4.8, 180.0, 2.0))
                except Exception:
                    pass

            now = time.monotonic()
            if light is not None and now - last_light_fix > 0.5:
                last_light_fix = now
                try:
                    light.set_transform(rel_camera_tf(cam_tf, ground_z, 12.5, 4.2, args.light_yaw_add, 0.0))
                    light.set_state(carla.TrafficLightState.Red)
                    light.freeze(True)
                except Exception:
                    pass

            tick(world, 1)
            time.sleep(dt)

    except KeyboardInterrupt:
        log("")
        log("[STOP] Full test sahnesi durduruldu.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[ERROR] {e}")
        sys.exit(1)

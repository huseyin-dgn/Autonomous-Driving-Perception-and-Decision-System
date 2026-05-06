#!/usr/bin/env python3
import argparse
import math
import random
import time
import carla


SPAWNED = []


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def yaw_diff_deg(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


def forward_vec(yaw_deg):
    r = math.radians(yaw_deg)
    return carla.Vector3D(math.cos(r), math.sin(r), 0.0)


def right_vec(yaw_deg):
    r = math.radians(yaw_deg + 90.0)
    return carla.Vector3D(math.cos(r), math.sin(r), 0.0)


def add_vec(loc, vec, scale):
    return carla.Location(
        x=loc.x + vec.x * scale,
        y=loc.y + vec.y * scale,
        z=loc.z + vec.z * scale,
    )


def set_sync_free(world):
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)


def clear_world(world):
    actors = world.get_actors()

    destroy_ids = []

    for pattern in [
        "vehicle.*",
        "walker.pedestrian.*",
        "controller.ai.walker",
        "sensor.camera.*",
        "static.prop.trafficwarning",
        "static.prop.streetsign",
        "static.prop.streetsign01",
        "static.prop.streetsign04",
        "static.prop.busstop",
        "static.prop.busstoplb",
    ]:
        for actor in actors.filter(pattern):
            destroy_ids.append(actor.id)

    if destroy_ids:
        client = world.get_client()
        client.apply_batch_sync(
            [carla.command.DestroyActor(x) for x in destroy_ids],
            True,
        )

    print(f"[CLEAR] Temizlenen actor: {len(destroy_ids)}")


def choose_straight_spawn(world):
    amap = world.get_map()
    spawn_points = amap.get_spawn_points()

    best = None
    best_score = -999999

    for sp in spawn_points:
        try:
            wp = amap.get_waypoint(
                sp.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )

            if wp is None:
                continue

            yaw0 = wp.transform.rotation.yaw
            cur = wp
            ok = True
            yaw_penalty = 0.0
            junction_penalty = 0.0

            for _ in range(10):
                nxt = cur.next(5.0)
                if not nxt:
                    ok = False
                    break
                cur = nxt[0]

                dy = abs(yaw_diff_deg(cur.transform.rotation.yaw, yaw0))
                yaw_penalty += dy

                if cur.is_junction:
                    junction_penalty += 15.0

            if not ok:
                continue

            score = 100.0 - yaw_penalty - junction_penalty

            if score > best_score:
                best_score = score
                best = sp

        except Exception:
            continue

    if best is None:
        best = spawn_points[0]

    best.location.z += 0.50
    return best


def get_bp(bp_lib, preferred, fallback_filter):
    for name in preferred:
        try:
            return bp_lib.find(name)
        except Exception:
            pass

    candidates = list(bp_lib.filter(fallback_filter))
    if not candidates:
        return None

    return random.choice(candidates)


def spawn_vehicle(world, bp, transform, role_name):
    if bp is None:
        return None

    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role_name)

    actor = world.try_spawn_actor(bp, transform)

    if actor is None:
        transform.location.z += 0.5
        actor = world.try_spawn_actor(bp, transform)

    if actor is not None:
        try:
            actor.set_autopilot(False)
            actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        except Exception:
            pass

        SPAWNED.append(actor)
        print(f"[SPAWN] vehicle {role_name}: id={actor.id} {actor.type_id}")

    return actor


def spawn_front_vehicle(world, bp, base_wp, distance, lateral_offset, role_name):
    nxt = base_wp.next(distance)
    if not nxt:
        return None

    wp = nxt[0]
    tf = wp.transform
    rv = right_vec(tf.rotation.yaw)
    tf.location = add_vec(tf.location, rv, lateral_offset)
    tf.location.z += 0.35
    tf.rotation.pitch = 0.0
    tf.rotation.roll = 0.0

    return spawn_vehicle(world, bp, tf, role_name)


def spawn_walker(world, bp, base_wp, distance, lateral_offset, name):
    if bp is None:
        return None

    nxt = base_wp.next(distance)
    if not nxt:
        return None

    wp = nxt[0]
    tf = wp.transform

    rv = right_vec(tf.rotation.yaw)
    tf.location = add_vec(tf.location, rv, lateral_offset)
    tf.location.z += 0.65

    tf.rotation.yaw = tf.rotation.yaw + 180.0
    tf.rotation.pitch = 0.0
    tf.rotation.roll = 0.0

    if bp.has_attribute("is_invincible"):
        bp.set_attribute("is_invincible", "false")

    actor = world.try_spawn_actor(bp, tf)

    if actor is not None:
        SPAWNED.append(actor)
        print(f"[SPAWN] pedestrian {name}: id={actor.id} {actor.type_id}")
    else:
        print(f"[WARN] pedestrian spawn olmadı: {name}")

    return actor


def spawn_sign(world, bp_lib, base_wp):
    sign_names = [
        "static.prop.trafficwarning",
        "static.prop.streetsign04",
        "static.prop.streetsign01",
        "static.prop.streetsign",
        "static.prop.busstop",
        "static.prop.busstoplb",
    ]

    candidates = []

    for name in sign_names:
        try:
            candidates.append(bp_lib.find(name))
        except Exception:
            pass

    if not candidates:
        print("[WARN] Uygun tabela blueprint bulunamadı")
        return None

    for dist in [18.0, 24.0, 30.0]:
        for offset in [4.2, -4.2, 5.5, -5.5]:
            nxt = base_wp.next(dist)
            if not nxt:
                continue

            wp = nxt[0]
            tf = wp.transform
            rv = right_vec(tf.rotation.yaw)
            tf.location = add_vec(tf.location, rv, offset)
            tf.location.z += 0.20
            tf.rotation.yaw = tf.rotation.yaw + (90.0 if offset > 0 else -90.0)
            tf.rotation.pitch = 0.0
            tf.rotation.roll = 0.0

            for bp in candidates:
                actor = world.try_spawn_actor(bp, tf)

                if actor is not None:
                    SPAWNED.append(actor)
                    print(f"[SPAWN] sign/prop: id={actor.id} {actor.type_id}")
                    return actor

    print("[WARN] tabela spawn edilemedi")
    return None


def force_traffic_lights_red(world):
    count = 0

    for tl in world.get_actors().filter("traffic.traffic_light*"):
        try:
            tl.set_state(carla.TrafficLightState.Red)
            tl.set_red_time(9999.0)
            tl.freeze(True)
            count += 1
        except Exception:
            pass

    print(f"[TRAFFIC_LIGHT] Red + freeze edilen ışık: {count}")


def drive_ego_slow(world, ego, target_kmh):
    target_ms = target_kmh / 3.6
    amap = world.get_map()

    while True:
        try:
            vel = ego.get_velocity()
            speed = math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)

            loc = ego.get_location()
            wp = amap.get_waypoint(
                loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )

            steer = 0.0

            if wp is not None:
                nxt = wp.next(8.0)

                if nxt:
                    target_wp = nxt[0]
                    car_yaw = ego.get_transform().rotation.yaw
                    target_yaw = target_wp.transform.rotation.yaw
                    yaw_err = yaw_diff_deg(target_yaw, car_yaw)
                    steer = clamp(yaw_err / 45.0, -0.25, 0.25)

            if speed < target_ms * 0.80:
                throttle = 0.32
                brake = 0.0
            elif speed < target_ms:
                throttle = 0.18
                brake = 0.0
            elif speed < target_ms * 1.20:
                throttle = 0.05
                brake = 0.0
            else:
                throttle = 0.0
                brake = 0.25

            ego.apply_control(
                carla.VehicleControl(
                    throttle=float(throttle),
                    steer=float(steer),
                    brake=float(brake),
                    hand_brake=False,
                    reverse=False,
                )
            )

            time.sleep(0.05)

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            print("[EGO_CONTROL] hata:", exc)
            time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Town03")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--ego-speed", type=float, default=4.0)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    if args.map:
        print(f"[WORLD] Map yükleniyor: {args.map}")
        world = client.load_world(args.map)
        time.sleep(2.0)
    else:
        world = client.get_world()

    set_sync_free(world)

    world.set_weather(
        carla.WeatherParameters(
            cloudiness=5.0,
            precipitation=0.0,
            sun_altitude_angle=65.0,
            sun_azimuth_angle=35.0,
            fog_density=0.0,
            wetness=0.0,
        )
    )

    if args.clear:
        clear_world(world)
        time.sleep(1.0)

    bp_lib = world.get_blueprint_library()

    ego_bp = get_bp(
        bp_lib,
        ["vehicle.toyota.prius", "vehicle.tesla.model3", "vehicle.audi.tt"],
        "vehicle.*",
    )

    front_bp = get_bp(
        bp_lib,
        ["vehicle.tesla.model3", "vehicle.audi.tt", "vehicle.lincoln.mkz_2020", "vehicle.toyota.prius"],
        "vehicle.*",
    )

    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
    walker_bp_1 = walker_bps[0] if len(walker_bps) > 0 else None
    walker_bp_2 = walker_bps[5] if len(walker_bps) > 5 else walker_bp_1
    walker_bp_3 = walker_bps[10] if len(walker_bps) > 10 else walker_bp_1

    ego_tf = choose_straight_spawn(world)

    if ego_bp.has_attribute("role_name"):
        ego_bp.set_attribute("role_name", "ego")

    ego = world.try_spawn_actor(ego_bp, ego_tf)

    if ego is None:
        raise RuntimeError("Ego araç spawn edilemedi")

    SPAWNED.append(ego)
    print(f"[SPAWN] EGO: id={ego.id} {ego.type_id}")
    print(f"[EGO] speed target: {args.ego_speed} km/h")

    amap = world.get_map()
    base_wp = amap.get_waypoint(
        ego.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    spawn_front_vehicle(world, front_bp, base_wp, 24.0, 0.0, "front_vehicle_1")
    spawn_front_vehicle(world, front_bp, base_wp, 42.0, -3.5, "front_vehicle_2")
    spawn_front_vehicle(world, front_bp, base_wp, 55.0, 3.5, "front_vehicle_3")

    spawn_walker(world, walker_bp_1, base_wp, 20.0, 2.4, "right_road_person")
    spawn_walker(world, walker_bp_2, base_wp, 31.0, -2.6, "left_road_person")
    spawn_walker(world, walker_bp_3, base_wp, 38.0, 4.6, "sidewalk_person")

    spawn_sign(world, bp_lib, base_wp)
    force_traffic_lights_red(world)

    print("")
    print("==========================================")
    print("STABLE ADAS CARLA SCENE HAZIR")
    print("==========================================")
    print("Ego yavaş gider.")
    print("Önde araçlar, yayalar, tabela prop ve kırmızı trafik ışıkları var.")
    print("Bu terminal açık kalmalı.")
    print("==========================================")
    print("")

    try:
        drive_ego_slow(world, ego, args.ego_speed)

    except KeyboardInterrupt:
        print("\n[EXIT] Sahne kapatılıyor...")

    finally:
        for actor in reversed(SPAWNED):
            try:
                actor.destroy()
            except Exception:
                pass

        print("[CLEANUP] Spawn edilen aktörler silindi.")


if __name__ == "__main__":
    main()

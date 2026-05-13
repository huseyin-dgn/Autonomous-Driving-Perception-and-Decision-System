#!/usr/bin/env python3
import argparse
import math
import random
import time

import carla


ROLE_PREFIX = "adas_rg_ped_motor"


def clamp_angle(angle):
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def angle_diff(a, b):
    return abs(clamp_angle(a - b))


def average_yaw(yaws):
    sx = 0.0
    sy = 0.0
    for yaw in yaws:
        r = math.radians(yaw)
        sx += math.cos(r)
        sy += math.sin(r)
    return math.degrees(math.atan2(sy, sx))


def forward_vec(yaw_deg):
    r = math.radians(yaw_deg)
    return carla.Vector3D(math.cos(r), math.sin(r), 0.0)


def right_vec(yaw_deg):
    return forward_vec(yaw_deg + 90.0)


def yaw_to(src, dst):
    return math.degrees(math.atan2(dst.y - src.y, dst.x - src.x))


def loc_add(loc, vec, scale):
    return carla.Location(
        x=loc.x + vec.x * scale,
        y=loc.y + vec.y * scale,
        z=loc.z + vec.z * scale,
    )


def set_role(bp, role_name):
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role_name)


def set_color_if_possible(bp, color):
    if bp.has_attribute("color"):
        colors = bp.get_attribute("color").recommended_values
        if color in colors:
            bp.set_attribute("color", color)
        elif colors:
            bp.set_attribute("color", random.choice(colors))


def destroy_actor(actor):
    try:
        if actor.type_id.startswith("sensor."):
            actor.stop()
    except Exception:
        pass

    try:
        actor.destroy()
        return True
    except Exception:
        return False


def clean_scene(world, clean_static_props=False):
    destroyed = 0

    actors = list(world.get_actors())

    for actor in actors:
        type_id = actor.type_id
        role = actor.attributes.get("role_name", "")

        should_destroy = False

        if type_id.startswith("sensor."):
            should_destroy = True

        elif type_id.startswith("vehicle."):
            should_destroy = True

        elif type_id.startswith("walker.pedestrian."):
            should_destroy = True

        elif role.startswith(ROLE_PREFIX):
            should_destroy = True

        elif clean_static_props and type_id.startswith("static.prop."):
            should_destroy = True

        if should_destroy:
            if destroy_actor(actor):
                destroyed += 1

    print(f"[CLEAN] Temizlenen actor sayısı: {destroyed}")


def get_blueprint(bp_lib, candidates, fallback_filter=None):
    for bp_id in candidates:
        found = bp_lib.filter(bp_id)
        if found:
            return random.choice(found)

    if fallback_filter:
        found = bp_lib.filter(fallback_filter)
        if found:
            return random.choice(found)

    return None


def get_motorcycle_blueprints(bp_lib):
    patterns = [
        "vehicle.kawasaki.ninja",
        "vehicle.yamaha.yzf",
        "vehicle.harley-davidson.low_rider",
        "vehicle.vespa.zx125",
        "vehicle.*ninja*",
        "vehicle.*yamaha*",
        "vehicle.*harley*",
        "vehicle.*vespa*",
        "vehicle.*bike*",
        "vehicle.*motor*",
    ]

    result = []
    seen = set()

    for pattern in patterns:
        for bp in bp_lib.filter(pattern):
            if bp.id not in seen:
                seen.add(bp.id)
                result.append(bp)

    return result


def ground_location(world, loc, extra_z=0.20):
    carla_map = world.get_map()

    try:
        wp = carla_map.get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Any,
        )
        if wp is not None:
            return carla.Location(
                x=loc.x,
                y=loc.y,
                z=wp.transform.location.z + extra_z,
            )
    except Exception:
        pass

    return carla.Location(x=loc.x, y=loc.y, z=loc.z + extra_z)


def list_traffic_lights(world):
    traffic_lights = list(world.get_actors().filter("traffic.traffic_light*"))

    print(f"[INFO] Haritadaki trafik ışığı sayısı: {len(traffic_lights)}")

    for idx, tl in enumerate(traffic_lights):
        tf = tl.get_transform()
        state = "unknown"

        try:
            state = str(tl.get_state())
        except Exception:
            pass

        print(
            f"[{idx:03d}] id={tl.id} "
            f"loc=({tf.location.x:.2f}, {tf.location.y:.2f}, {tf.location.z:.2f}) "
            f"yaw={tf.rotation.yaw:.1f} "
            f"state={state}"
        )


def select_traffic_lights(world, red_id=None, green_id=None):
    traffic_lights = list(world.get_actors().filter("traffic.traffic_light*"))

    if len(traffic_lights) < 2:
        raise RuntimeError(
            "Bu haritada en az 2 trafik ışığı bulunamadı. Town03 / Town05 gibi ışıklı bir map açman lazım."
        )

    by_id = {tl.id: tl for tl in traffic_lights}

    if red_id is not None and green_id is not None:
        if red_id not in by_id:
            raise RuntimeError(f"red-id bulunamadı: {red_id}")
        if green_id not in by_id:
            raise RuntimeError(f"green-id bulunamadı: {green_id}")
        return by_id[red_id], by_id[green_id]

    best_pair = None
    best_score = 1e18

    for i in range(len(traffic_lights)):
        for j in range(i + 1, len(traffic_lights)):
            a = traffic_lights[i]
            b = traffic_lights[j]

            atf = a.get_transform()
            btf = b.get_transform()

            dist = atf.location.distance(btf.location)
            if dist < 2.0 or dist > 35.0:
                continue

            yaw_d = angle_diff(atf.rotation.yaw, btf.rotation.yaw)

            # Aynı yöne bakan ve birbirine yakın iki ışık daha iyi.
            score = abs(dist - 8.0) * 4.0 + yaw_d * 0.8

            if yaw_d > 80.0:
                score += 80.0

            if score < best_score:
                best_score = score
                best_pair = (a, b)

    if best_pair is None:
        # Son çare: ilk iki ışık.
        return traffic_lights[0], traffic_lights[1]

    return best_pair


def freeze_and_set_lights(world, red_tl, green_tl):
    all_lights = list(world.get_actors().filter("traffic.traffic_light*"))

    for tl in all_lights:
        try:
            tl.freeze(True)
        except Exception:
            pass

    try:
        red_tl.set_state(carla.TrafficLightState.Red)
        red_tl.set_red_time(999.0)
        red_tl.set_yellow_time(1.0)
        red_tl.set_green_time(1.0)
        red_tl.freeze(True)
    except Exception as exc:
        print(f"[WARN] Kırmızı ışık ayarlanamadı: {exc}")

    try:
        green_tl.set_state(carla.TrafficLightState.Green)
        green_tl.set_green_time(999.0)
        green_tl.set_yellow_time(1.0)
        green_tl.set_red_time(1.0)
        green_tl.freeze(True)
    except Exception as exc:
        print(f"[WARN] Yeşil ışık ayarlanamadı: {exc}")

    print(
        f"[LIGHT] RED   id={red_tl.id}   loc={red_tl.get_transform().location}"
    )
    print(
        f"[LIGHT] GREEN id={green_tl.id} loc={green_tl.get_transform().location}"
    )


def spawn_ego_near_lights(world, bp_lib, red_tl, green_tl, ego_distance, view_yaw_offset):
    red_tf = red_tl.get_transform()
    green_tf = green_tl.get_transform()

    mid = carla.Location(
        x=(red_tf.location.x + green_tf.location.x) / 2.0,
        y=(red_tf.location.y + green_tf.location.y) / 2.0,
        z=(red_tf.location.z + green_tf.location.z) / 2.0,
    )

    light_face_yaw = average_yaw([red_tf.rotation.yaw, green_tf.rotation.yaw])

    # Varsayım: ışığın ön yüzüne bakmak için ışığın yaw yönünün tersinden yaklaşılır.
    base_view_yaw = light_face_yaw + 180.0 + view_yaw_offset

    ego_bp = get_blueprint(
        bp_lib,
        [
            "vehicle.tesla.model3",
            "vehicle.lincoln.mkz_2020",
            "vehicle.audi.tt",
            "vehicle.dodge.charger_2020",
        ],
        fallback_filter="vehicle.*",
    )

    if ego_bp is None:
        raise RuntimeError("Ego aracı için vehicle blueprint bulunamadı.")

    set_role(ego_bp, "ego")
    set_color_if_possible(ego_bp, "0,0,0")

    view_forward = forward_vec(base_view_yaw)
    view_right = right_vec(base_view_yaw)

    distance_trials = [
        ego_distance,
        ego_distance + 4.0,
        ego_distance - 4.0,
        ego_distance + 8.0,
        ego_distance - 7.0,
    ]

    lateral_trials = [0.0, -1.5, 1.5, -3.0, 3.0]

    for dist in distance_trials:
        if dist < 8.0:
            continue

        for lateral in lateral_trials:
            ego_loc = loc_add(mid, view_forward, -dist)
            ego_loc = loc_add(ego_loc, view_right, lateral)
            ego_loc = ground_location(world, ego_loc, extra_z=0.35)

            ego_yaw = yaw_to(ego_loc, mid)

            ego_tf = carla.Transform(
                ego_loc,
                carla.Rotation(pitch=0.0, yaw=ego_yaw, roll=0.0),
            )

            ego = world.try_spawn_actor(ego_bp, ego_tf)

            if ego is not None:
                try:
                    ego.set_autopilot(False)
                    ego.set_simulate_physics(False)
                except Exception:
                    pass

                print(
                    f"[EGO] Spawn edildi: id={ego.id}, loc={ego_tf.location}, yaw={ego_tf.rotation.yaw:.1f}"
                )
                return ego

    raise RuntimeError(
        "Ego spawn edilemedi. Trafik ışığı çevresinde çakışma var. "
        "Farklı --view-yaw-offset veya --ego-distance dene."
    )


def relative_location(world, ego_tf, forward_m, right_m, extra_z=0.25):
    fwd = forward_vec(ego_tf.rotation.yaw)
    right = right_vec(ego_tf.rotation.yaw)

    loc = carla.Location(
        x=ego_tf.location.x,
        y=ego_tf.location.y,
        z=ego_tf.location.z,
    )

    loc = loc_add(loc, fwd, forward_m)
    loc = loc_add(loc, right, right_m)

    return ground_location(world, loc, extra_z=extra_z)


def spawn_pedestrian(world, bp_lib, transform, index):
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
    if not walker_bps:
        raise RuntimeError("walker.pedestrian.* blueprint bulunamadı.")

    bp = random.choice(walker_bps)

    if bp.has_attribute("is_invincible"):
        bp.set_attribute("is_invincible", "false")

    set_role(bp, f"{ROLE_PREFIX}_pedestrian_{index}")

    actor = world.try_spawn_actor(bp, transform)

    if actor is None:
        raise RuntimeError(f"Pedestrian spawn edilemedi: index={index}")

    try:
        actor.set_simulate_physics(False)
    except Exception:
        pass

    print(f"[PED] Spawn edildi: id={actor.id}, loc={transform.location}")
    return actor



def project_to_driving_transform(world, loc, yaw, z_offset=0.65):
    """
    Motorsiklet vehicle olduğu için kaldırım/boş alan yerine yola oturtuyoruz.
    try_spawn_actor yol dışı veya çakışmalı noktada None döndürür.
    """
    try:
        wp = world.get_map().get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if wp is not None:
            base = wp.transform.location
            return carla.Transform(
                carla.Location(
                    x=base.x,
                    y=base.y,
                    z=base.z + z_offset,
                ),
                carla.Rotation(
                    pitch=0.0,
                    yaw=yaw,
                    roll=0.0,
                ),
            )
    except Exception:
        pass

    return carla.Transform(
        carla.Location(
            x=loc.x,
            y=loc.y,
            z=loc.z + z_offset,
        ),
        carla.Rotation(
            pitch=0.0,
            yaw=yaw,
            roll=0.0,
        ),
    )


def spawn_motorcycle(world, bp_lib, transform_candidates, index):
    motor_bps = get_motorcycle_blueprints(bp_lib)

    if not motor_bps:
        raise RuntimeError(
            "Motorsiklet blueprint bulunamadı. CARLA blueprintlerinde kawasaki/yamaha/harley/vespa yok."
        )

    # Her ihtimale karşı birkaç motor blueprint'i deniyoruz.
    random.shuffle(motor_bps)

    for raw_tf in transform_candidates:
        for z_offset in [0.55, 0.75, 1.00]:
            road_tf = project_to_driving_transform(
                world,
                raw_tf.location,
                raw_tf.rotation.yaw,
                z_offset=z_offset,
            )

            for bp in motor_bps:
                set_role(bp, f"{ROLE_PREFIX}_motorcycle_{index}")
                set_color_if_possible(bp, "255,0,0" if index == 1 else "0,0,255")

                actor = world.try_spawn_actor(bp, road_tf)

                if actor is not None:
                    try:
                        actor.set_autopilot(False)
                        actor.set_simulate_physics(False)
                    except Exception:
                        pass

                    print(
                        f"[MOTOR] Spawn edildi: id={actor.id}, type={actor.type_id}, "
                        f"loc={road_tf.location}, yaw={road_tf.rotation.yaw:.1f}"
                    )
                    return actor

    raise RuntimeError(
        f"Motorsiklet spawn edilemedi: index={index}. "
        "Tüm yol-projected alternatifler başarısız oldu."
    )


def spawn_targets(world, bp_lib, ego):
    ego_tf = ego.get_transform()
    ego_yaw = ego_tf.rotation.yaw

    actors = []

    # İnsanlar yakın ve ayrı. Bunlar zaten çalıştı.
    ped_layout = [
        (6.5, -2.8),
        (8.5, 2.8),
    ]

    for i, (fwd_m, right_m) in enumerate(ped_layout, start=1):
        loc = relative_location(world, ego_tf, fwd_m, right_m, extra_z=0.35)

        tf = carla.Transform(
            loc,
            carla.Rotation(
                pitch=0.0,
                yaw=ego_yaw + 180.0,
                roll=0.0,
            ),
        )

        actors.append(spawn_pedestrian(world, bp_lib, tf, i))

    # Motorsikletler için tek nokta değil, alternatif liste veriyoruz.
    # CARLA bazı noktalarda vehicle spawn etmiyor.
    motor_1_candidates = []
    motor_2_candidates = []

    # Motor 1: yakın-sol/orta bölgede
    for fwd_m, right_m in [
        (9.5, -0.3),
        (10.5, -0.8),
        (11.5, -1.2),
        (12.5, -0.2),
        (13.5, -1.5),
        (15.0, -0.5),
    ]:
        loc = relative_location(world, ego_tf, fwd_m, right_m, extra_z=0.50)
        motor_1_candidates.append(
            carla.Transform(
                loc,
                carla.Rotation(
                    pitch=0.0,
                    yaw=ego_yaw + 90.0,
                    roll=0.0,
                ),
            )
        )

    # Motor 2: yakın-sağ/orta bölgede
    for fwd_m, right_m in [
        (13.5, 0.6),
        (14.5, 1.0),
        (15.5, 1.4),
        (16.5, 0.3),
        (18.0, 1.6),
        (20.0, 0.8),
    ]:
        loc = relative_location(world, ego_tf, fwd_m, right_m, extra_z=0.50)
        motor_2_candidates.append(
            carla.Transform(
                loc,
                carla.Rotation(
                    pitch=0.0,
                    yaw=ego_yaw - 90.0,
                    roll=0.0,
                ),
            )
        )

    actors.append(spawn_motorcycle(world, bp_lib, motor_1_candidates, 1))
    actors.append(spawn_motorcycle(world, bp_lib, motor_2_candidates, 2))

    return actors


def set_weather(world):
    try:
        world.set_weather(
            carla.WeatherParameters(
                cloudiness=0.0,
                precipitation=0.0,
                sun_altitude_angle=45.0,
                sun_azimuth_angle=30.0,
                fog_density=0.0,
                wetness=0.0,
            )
        )
    except Exception:
        pass


def set_spectator(world, ego):
    try:
        ego_tf = ego.get_transform()
        fwd = forward_vec(ego_tf.rotation.yaw)

        loc = carla.Location(
            x=ego_tf.location.x - fwd.x * 7.0,
            y=ego_tf.location.y - fwd.y * 7.0,
            z=ego_tf.location.z + 4.0,
        )

        tf = carla.Transform(
            loc,
            carla.Rotation(
                pitch=-18.0,
                yaw=ego_tf.rotation.yaw,
                roll=0.0,
            ),
        )

        world.get_spectator().set_transform(tf)
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30.0)

    parser.add_argument("--list-lights", action="store_true")

    parser.add_argument("--red-id", type=int, default=None)
    parser.add_argument("--green-id", type=int, default=None)

    parser.add_argument("--ego-distance", type=float, default=20.0)

    # Eğer ışıkların arkasını görürsen:
    # 0, 90, -90, 180 değerlerini dene.
    parser.add_argument("--view-yaw-offset", type=float, default=0.0)

    # Eski senaryodan static prop kaldıysa bunu açabilirsin.
    parser.add_argument("--clean-static-props", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    print("[INFO] CARLA bağlantısı başarılı.")
    print(f"[INFO] Map: {world.get_map().name}")

    if args.list_lights:
        list_traffic_lights(world)
        return

    clean_scene(world, clean_static_props=args.clean_static_props)
    time.sleep(0.5)

    set_weather(world)

    red_tl, green_tl = select_traffic_lights(
        world,
        red_id=args.red_id,
        green_id=args.green_id,
    )

    freeze_and_set_lights(world, red_tl, green_tl)

    ego = spawn_ego_near_lights(
        world,
        bp_lib,
        red_tl,
        green_tl,
        ego_distance=args.ego_distance,
        view_yaw_offset=args.view_yaw_offset,
    )

    targets = spawn_targets(world, bp_lib, ego)

    set_spectator(world, ego)

    print("")
    print("==========================================")
    print("ADAS RED/GREEN + PEDESTRIAN + MOTORCYCLE SCENARIO READY")
    print("==========================================")
    print(f"Ego id: {ego.id}")
    print(f"Red traffic light id: {red_tl.id}")
    print(f"Green traffic light id: {green_tl.id}")
    print(f"Target actors: {[actor.id for actor in targets]}")
    print("")
    print("Publisher bunu bulacak:")
    print('  role_name="ego"')
    print("")
    print("ROS topic publisher tarafından basılacak:")
    print("  /adas/camera/front/image_raw")
    print("==========================================")


if __name__ == "__main__":
    main()

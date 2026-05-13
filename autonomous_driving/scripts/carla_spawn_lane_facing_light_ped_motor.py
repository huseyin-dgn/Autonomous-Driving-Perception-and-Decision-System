#!/usr/bin/env python3
import argparse
import math
import random
import time

import carla


ROLE_PREFIX = "adas_lane_light_test"


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


def clean_scene(world):
    destroyed = 0

    for actor in list(world.get_actors()):
        tid = actor.type_id
        role = actor.attributes.get("role_name", "")

        if (
            tid.startswith("sensor.")
            or tid.startswith("vehicle.")
            or tid.startswith("walker.pedestrian.")
            or role.startswith(ROLE_PREFIX)
        ):
            if destroy_actor(actor):
                destroyed += 1

    print(f"[CLEAN] Temizlenen actor sayısı: {destroyed}")


def forward_vec(yaw_deg):
    r = math.radians(yaw_deg)
    return carla.Vector3D(math.cos(r), math.sin(r), 0.0)


def right_vec(yaw_deg):
    return forward_vec(yaw_deg + 90.0)


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
        values = bp.get_attribute("color").recommended_values
        if color in values:
            bp.set_attribute("color", color)
        elif values:
            bp.set_attribute("color", random.choice(values))


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

    out = []
    seen = set()

    for p in patterns:
        for bp in bp_lib.filter(p):
            if bp.id not in seen:
                seen.add(bp.id)
                out.append(bp)

    return out


def print_lights(world):
    tls = list(world.get_actors().filter("traffic.traffic_light*"))
    print(f"[INFO] Trafik ışığı sayısı: {len(tls)}")

    for i, tl in enumerate(tls):
        tf = tl.get_transform()
        try:
            state = tl.get_state()
        except Exception:
            state = "unknown"

        stop_count = 0
        try:
            stop_count = len(tl.get_stop_waypoints())
        except Exception:
            stop_count = 0

        print(
            f"[{i:03d}] id={tl.id} "
            f"loc=({tf.location.x:.2f}, {tf.location.y:.2f}, {tf.location.z:.2f}) "
            f"yaw={tf.rotation.yaw:.1f} "
            f"stop_wp={stop_count} "
            f"state={state}"
        )


def get_stop_waypoint_for_light(tl):
    try:
        stop_wps = tl.get_stop_waypoints()
        if stop_wps:
            return stop_wps[0]
    except Exception:
        pass

    return None


def choose_good_traffic_light(world, preferred_id=None):
    tls = list(world.get_actors().filter("traffic.traffic_light*"))

    if not tls:
        raise RuntimeError("Bu map içinde traffic.traffic_light bulunamadı.")

    if preferred_id is not None:
        for tl in tls:
            if tl.id == preferred_id:
                wp = get_stop_waypoint_for_light(tl)
                if wp is None:
                    raise RuntimeError(f"Seçilen ışığın stop waypoint'i yok: id={preferred_id}")
                return tl, wp

        raise RuntimeError(f"Verilen light id bulunamadı: {preferred_id}")

    candidates = []

    for tl in tls:
        wp = get_stop_waypoint_for_light(tl)
        if wp is None:
            continue

        tf = tl.get_transform()
        wtf = wp.transform

        dist = tf.location.distance(wtf.location)

        # Çok uzak stop waypoint saçma olabilir.
        if dist > 40.0:
            continue

        # Işık ile stop çizgisi arası 5-25m civarı iyidir.
        score = abs(dist - 14.0)

        candidates.append((score, tl, wp))

    if not candidates:
        raise RuntimeError("Stop waypoint'i olan uygun trafik ışığı bulunamadı.")

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][2]


def get_green_light_from_group(red_tl):
    try:
        group = red_tl.get_group_traffic_lights()
        group = [x for x in group if x.id != red_tl.id]

        if group:
            # Red ışığa yakın olanı yeşil test ışığı yap.
            group.sort(
                key=lambda x: x.get_transform().location.distance(
                    red_tl.get_transform().location
                )
            )
            return group[0]
    except Exception:
        pass

    return None


def freeze_lights(world, red_tl, green_tl=None):
    for tl in world.get_actors().filter("traffic.traffic_light*"):
        try:
            tl.freeze(True)
        except Exception:
            pass

    try:
        red_tl.set_state(carla.TrafficLightState.Red)
        red_tl.set_red_time(999.0)
        red_tl.freeze(True)
    except Exception as exc:
        print(f"[WARN] Red light set edilemedi: {exc}")

    if green_tl is not None:
        try:
            green_tl.set_state(carla.TrafficLightState.Green)
            green_tl.set_green_time(999.0)
            green_tl.freeze(True)
        except Exception as exc:
            print(f"[WARN] Green light set edilemedi: {exc}")

    print(f"[LIGHT] RED   id={red_tl.id} loc={red_tl.get_transform().location}")

    if green_tl is not None:
        print(f"[LIGHT] GREEN id={green_tl.id} loc={green_tl.get_transform().location}")
    else:
        print("[LIGHT] GREEN bulunamadı, sadece RED kullanılacak.")


def spawn_ego_on_lane(world, bp_lib, stop_wp, back_distance):
    ego_bp = None

    for name in [
        "vehicle.tesla.model3",
        "vehicle.lincoln.mkz_2020",
        "vehicle.audi.tt",
        "vehicle.dodge.charger_2020",
    ]:
        bps = bp_lib.filter(name)
        if bps:
            ego_bp = bps[0]
            break

    if ego_bp is None:
        ego_bp = random.choice(bp_lib.filter("vehicle.*"))

    set_role(ego_bp, "ego")
    set_color_if_possible(ego_bp, "0,0,0")

    # Stop çizgisinden geriye doğru git.
    wp = stop_wp
    prevs = wp.previous(back_distance)

    if prevs:
        ego_wp = prevs[0]
    else:
        ego_wp = wp

    tf = ego_wp.transform

    spawn_tf = carla.Transform(
        carla.Location(
            x=tf.location.x,
            y=tf.location.y,
            z=tf.location.z + 0.40,
        ),
        carla.Rotation(
            pitch=0.0,
            yaw=tf.rotation.yaw,
            roll=0.0,
        ),
    )

    ego = world.try_spawn_actor(ego_bp, spawn_tf)

    if ego is None:
        # Biraz daha geriye dene.
        for d in [back_distance + 5.0, back_distance + 10.0, back_distance + 15.0]:
            prevs = stop_wp.previous(d)
            if not prevs:
                continue

            ego_wp = prevs[0]
            tf = ego_wp.transform

            spawn_tf = carla.Transform(
                carla.Location(
                    x=tf.location.x,
                    y=tf.location.y,
                    z=tf.location.z + 0.40,
                ),
                carla.Rotation(
                    pitch=0.0,
                    yaw=tf.rotation.yaw,
                    roll=0.0,
                ),
            )

            ego = world.try_spawn_actor(ego_bp, spawn_tf)
            if ego is not None:
                break

    if ego is None:
        raise RuntimeError("Ego araç şerit üstüne spawn edilemedi.")

    try:
        ego.set_autopilot(False)
        ego.set_simulate_physics(False)
    except Exception:
        pass

    print(
        f"[EGO] Şeritte düz spawn edildi: id={ego.id}, "
        f"loc={spawn_tf.location}, yaw={spawn_tf.rotation.yaw:.1f}"
    )

    return ego


def lane_relative_location(world, ego_tf, forward_m, right_m, z_offset=0.35):
    yaw = ego_tf.rotation.yaw
    fwd = forward_vec(yaw)
    right = right_vec(yaw)

    loc = carla.Location(
        x=ego_tf.location.x,
        y=ego_tf.location.y,
        z=ego_tf.location.z,
    )

    loc = loc_add(loc, fwd, forward_m)
    loc = loc_add(loc, right, right_m)

    wp = world.get_map().get_waypoint(
        loc,
        project_to_road=True,
        lane_type=carla.LaneType.Any,
    )

    if wp is not None:
        return carla.Location(
            x=loc.x,
            y=loc.y,
            z=wp.transform.location.z + z_offset,
        )

    return carla.Location(x=loc.x, y=loc.y, z=loc.z + z_offset)



def spawn_pedestrian(world, bp_lib, loc, yaw, index):
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))

    if not walker_bps:
        raise RuntimeError("walker.pedestrian.* blueprint yok.")

    random.shuffle(walker_bps)

    # CARLA walker spawn bazı noktalarda None döndürür.
    # Bu yüzden tek nokta değil, yakın çevrede alternatif noktalar deneniyor.
    candidate_locs = []

    for fwd_shift, right_shift, z_add in [
        (0.0, 0.0, 0.35),
        (0.5, 0.0, 0.35),
        (-0.5, 0.0, 0.35),
        (0.0, 0.5, 0.35),
        (0.0, -0.5, 0.35),
        (0.8, 0.4, 0.35),
        (0.8, -0.4, 0.35),
        (-0.8, 0.4, 0.35),
        (-0.8, -0.4, 0.35),
        (1.2, 0.8, 0.45),
        (1.2, -0.8, 0.45),
        (-1.2, 0.8, 0.45),
        (-1.2, -0.8, 0.45),
    ]:
        c = carla.Location(x=loc.x, y=loc.y, z=loc.z)
        c = loc_add(c, forward_vec(yaw), fwd_shift)
        c = loc_add(c, right_vec(yaw), right_shift)
        c.z = loc.z + z_add
        candidate_locs.append(c)

    for c_loc in candidate_locs:
        tf = carla.Transform(
            c_loc,
            carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0),
        )

        for bp in walker_bps:
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")

            set_role(bp, f"{ROLE_PREFIX}_pedestrian_{index}")

            actor = world.try_spawn_actor(bp, tf)

            if actor is not None:
                try:
                    actor.set_simulate_physics(False)
                except Exception:
                    pass

                print(f"[PED] id={actor.id} loc={tf.location}, yaw={tf.rotation.yaw:.1f}")
                return actor

    raise RuntimeError(
        f"Pedestrian spawn edilemedi: {index}. "
        "Yakın çevredeki tüm alternatif noktalar başarısız oldu."
    )


def spawn_motorcycle(world, bp_lib, loc, yaw, index):
    motor_bps = get_motorcycle_blueprints(bp_lib)

    if not motor_bps:
        raise RuntimeError("Motorsiklet blueprint bulunamadı.")

    random.shuffle(motor_bps)

    # ÖNEMLİ:
    # Önce raw konumu deniyoruz.
    # Çünkü CARLA get_waypoint/project_to_road bazen iki motoru aynı lane merkezine yaklaştırıyor.
    # Bu da ikinci motorun görünmesini/algılanmasını bozuyor.
    candidate_locs = []

    for fwd_shift, right_shift, z_add in [
        (0.0, 0.0, 0.35),
        (0.4, 0.0, 0.35),
        (-0.4, 0.0, 0.35),
        (0.0, 0.4, 0.35),
        (0.0, -0.4, 0.35),
        (0.6, 0.4, 0.45),
        (-0.6, -0.4, 0.45),
        (0.0, 0.0, 0.60),
    ]:
        base = carla.Location(x=loc.x, y=loc.y, z=loc.z)
        base = loc_add(base, forward_vec(yaw), fwd_shift)
        base = loc_add(base, right_vec(yaw), right_shift)
        base.z = loc.z + z_add
        candidate_locs.append(base)

    yaw_candidates = [
        yaw,
        yaw + 8.0,
        yaw - 8.0,
        yaw + 180.0,
    ]

    for c_loc in candidate_locs:
        for c_yaw in yaw_candidates:
            tf = carla.Transform(
                c_loc,
                carla.Rotation(pitch=0.0, yaw=c_yaw, roll=0.0),
            )

            for bp in motor_bps:
                set_role(bp, f"{ROLE_PREFIX}_motorcycle_{index}")
                set_color_if_possible(bp, "255,0,0" if index == 1 else "0,0,255")

                actor = world.try_spawn_actor(bp, tf)

                if actor is not None:
                    try:
                        actor.set_autopilot(False)
                        actor.set_simulate_physics(False)
                    except Exception:
                        pass

                    print(
                        f"[MOTOR] RAW id={actor.id} type={actor.type_id} "
                        f"loc={tf.location}, yaw={tf.rotation.yaw:.1f}"
                    )
                    return actor

    # Raw başarısız olursa son çare olarak yola project et.
    wp = world.get_map().get_waypoint(
        loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if wp is not None:
        base = wp.transform.location
        lane_yaw = wp.transform.rotation.yaw

        fallback_locs = [
            carla.Location(base.x, base.y, base.z + 0.60),
            carla.Location(base.x, base.y, base.z + 0.80),
            carla.Location(base.x, base.y, base.z + 1.00),
        ]

        fallback_yaws = [
            yaw,
            yaw + 90.0,
            yaw - 90.0,
            lane_yaw,
            lane_yaw + 180.0,
        ]

        for c_loc in fallback_locs:
            for c_yaw in fallback_yaws:
                tf = carla.Transform(
                    c_loc,
                    carla.Rotation(pitch=0.0, yaw=c_yaw, roll=0.0),
                )

                for bp in motor_bps:
                    set_role(bp, f"{ROLE_PREFIX}_motorcycle_{index}")
                    set_color_if_possible(bp, "255,0,0" if index == 1 else "0,0,255")

                    actor = world.try_spawn_actor(bp, tf)

                    if actor is not None:
                        try:
                            actor.set_autopilot(False)
                            actor.set_simulate_physics(False)
                        except Exception:
                            pass

                        print(
                            f"[MOTOR] ROAD_FALLBACK id={actor.id} type={actor.type_id} "
                            f"loc={tf.location}, yaw={tf.rotation.yaw:.1f}"
                        )
                        return actor

    raise RuntimeError(f"Motorsiklet spawn edilemedi: {index}")





def spawn_targets(world, bp_lib, ego):
    """
    LIGHT-ONLY TEST.

    Bu testte insan, motorsiklet, araç, tabela spawn edilmiyor.
    Sadece CARLA map üzerindeki trafik ışıkları ve ego araç kalıyor.

    Amaç:
    - traffic_light detection
    - red / green / yellow state classification
    - active traffic light seçimi
    """

    print("[TARGETS] Light-only test: pedestrian/motorcycle/vehicle/sign spawn edilmeyecek.")
    return []


def set_weather(world):
    try:
        world.set_weather(
            carla.WeatherParameters(
                cloudiness=0.0,
                precipitation=0.0,
                sun_altitude_angle=50.0,
                sun_azimuth_angle=25.0,
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
            x=ego_tf.location.x - fwd.x * 8.0,
            y=ego_tf.location.y - fwd.y * 8.0,
            z=ego_tf.location.z + 5.0,
        )

        world.get_spectator().set_transform(
            carla.Transform(
                loc,
                carla.Rotation(
                    pitch=-20.0,
                    yaw=ego_tf.rotation.yaw,
                    roll=0.0,
                ),
            )
        )
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30.0)

    parser.add_argument("--list-lights", action="store_true")
    parser.add_argument("--light-id", type=int, default=None)

    parser.add_argument("--back-distance", type=float, default=24.0)

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
        print_lights(world)
        return

    clean_scene(world)
    time.sleep(0.5)

    set_weather(world)

    red_tl, stop_wp = choose_good_traffic_light(world, preferred_id=args.light_id)
    green_tl = get_green_light_from_group(red_tl)

    freeze_lights(world, red_tl, green_tl)

    ego = spawn_ego_on_lane(
        world,
        bp_lib,
        stop_wp,
        back_distance=args.back_distance,
    )

    targets = spawn_targets(world, bp_lib, ego)

    set_spectator(world, ego)

    print("")
    print("==========================================")
    print("LANE-FACING LIGHT TEST READY")
    print("==========================================")
    print("Ego araç artık şeritte dümdüz duruyor.")
    print("Kamera publisher bu ego araca bağlanacak.")
    print(f"Ego id: {ego.id}")
    print(f"Red light id: {red_tl.id}")
    if green_tl is not None:
        print(f"Green light id: {green_tl.id}")
    print(f"Target actors: {[a.id for a in targets]}")
    print("==========================================")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import math
import random
import carla

ROLE_PREFIX = "adas_simple_layout"


def local_to_world(base_tf, x, y, z=0.0):
    yaw = math.radians(base_tf.rotation.yaw)

    bx = base_tf.location.x
    by = base_tf.location.y
    bz = base_tf.location.z

    wx = bx + x * math.cos(yaw) - y * math.sin(yaw)
    wy = by + x * math.sin(yaw) + y * math.cos(yaw)
    wz = bz + z

    return carla.Location(wx, wy, wz)


def find_ego(world):
    for v in world.get_actors().filter("vehicle.*"):
        role = v.attributes.get("role_name", "").lower()
        if "ego" in role:
            return v

    vehicles = list(world.get_actors().filter("vehicle.*"))
    return vehicles[0] if vehicles else None


def clear_old(world):
    killed = 0

    for a in list(world.get_actors()):
        role = a.attributes.get("role_name", "")

        if role.startswith(ROLE_PREFIX):
            try:
                a.destroy()
                killed += 1
            except Exception:
                pass

    print(f"[CLEAR] old simple layout actors removed: {killed}")


def remove_front_non_ego_vehicles(world, ego):
    ego_tf = ego.get_transform()
    killed = 0

    for v in list(world.get_actors().filter("vehicle.*")):
        if v.id == ego.id:
            continue

        role = v.attributes.get("role_name", "")

        # Sadece bizim önceki test araçlarını ve ego önündeki yakın araçları kaldır
        loc = v.get_location()
        yaw = math.radians(ego_tf.rotation.yaw)

        dx = loc.x - ego_tf.location.x
        dy = loc.y - ego_tf.location.y

        lx = dx * math.cos(yaw) + dy * math.sin(yaw)
        ly = -dx * math.sin(yaw) + dy * math.cos(yaw)

        if role.startswith("adas_") or (0.0 < lx < 45.0 and abs(ly) < 10.0):
            try:
                print(f"[REMOVE VEHICLE] id={v.id} role={role} lx={lx:.1f} ly={ly:.1f}")
                v.destroy()
                killed += 1
            except Exception:
                pass

    print(f"[REMOVE VEHICLE] removed count={killed}")


def get_vehicle_bp(bp_lib):
    names = [
        "vehicle.tesla.model3",
        "vehicle.lincoln.mkz_2020",
        "vehicle.audi.tt",
        "vehicle.bmw.grandtourer",
        "vehicle.nissan.patrol",
    ]

    for name in names:
        try:
            return bp_lib.find(name)
        except Exception:
            pass

    vehicles = list(bp_lib.filter("vehicle.*"))
    return random.choice(vehicles) if vehicles else None


def spawn_straight_car(world, bp_lib, ego_tf):
    bp = get_vehicle_bp(bp_lib)

    if bp is None:
        print("[WARN] vehicle blueprint bulunamadı")
        return None

    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", f"{ROLE_PREFIX}_front_car")

    if bp.has_attribute("color"):
        colors = bp.get_attribute("color").recommended_values
        if colors:
            bp.set_attribute("color", random.choice(colors))

    # Önde, şerit içinde, düz duran araba
    x = 16.0
    y = 0.0

    loc = local_to_world(ego_tf, x, y, 0.35)

    tf = carla.Transform(
        loc,
        carla.Rotation(
            pitch=0.0,
            yaw=ego_tf.rotation.yaw,
            roll=0.0
        )
    )

    actor = world.try_spawn_actor(bp, tf)

    if actor is None:
        tf.location.z += 0.5
        actor = world.try_spawn_actor(bp, tf)

    if actor:
        try:
            actor.set_autopilot(False)
            actor.set_simulate_physics(False)
        except Exception:
            pass

        print(f"[CAR] id={actor.id} x={x} y={y} düz şekilde spawn edildi")

    return actor


def get_first_bp(bp_lib, candidates):
    for name in candidates:
        try:
            return bp_lib.find(name)
        except Exception:
            pass

    return None


def spawn_stop_sign_left(world, bp_lib, ego_tf):
    candidates = [
        "static.prop.trafficwarning",
        "static.prop.streetsign04",
        "static.prop.streetsign01",
        "static.prop.streetsign",
    ]

    bp = get_first_bp(bp_lib, candidates)

    if bp is None:
        print("[WARN] DUR tabelası yok. CARLA içindeki generic traffic/street sign deneniyor.")
        print("[INFO] Blueprint listesini görmek için:")
        print("python3 scripts/carla_simple_car_stop_peds.py --list-bps")
        return None

    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", f"{ROLE_PREFIX}_stop_sign_left")

    # Sol taraf: y negatif
    x = 12.0
    y = -5.0

    loc = local_to_world(ego_tf, x, y, 0.35)

    tf = carla.Transform(
        loc,
        carla.Rotation(
            pitch=0.0,
            yaw=ego_tf.rotation.yaw + 180.0,
            roll=0.0
        )
    )

    actor = world.try_spawn_actor(bp, tf)

    if actor is None:
        tf.location.z += 0.5
        actor = world.try_spawn_actor(bp, tf)

    if actor:
        print(f"[STOP SIGN] id={actor.id} bp={bp.id} x={x} y={y}")

    return actor


def spawn_pedestrian(world, bp_lib, ego_tf, idx, x, y):
    walkers = list(bp_lib.filter("walker.pedestrian.*"))

    if not walkers:
        print("[WARN] walker blueprint yok")
        return None

    bp = random.choice(walkers)

    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", f"{ROLE_PREFIX}_right_ped_{idx}")

    loc = local_to_world(ego_tf, x, y, 0.85)

    tf = carla.Transform(
        loc,
        carla.Rotation(
            pitch=0.0,
            yaw=ego_tf.rotation.yaw + 180.0,
            roll=0.0
        )
    )

    actor = world.try_spawn_actor(bp, tf)

    if actor is None:
        tf.location.z += 0.5
        actor = world.try_spawn_actor(bp, tf)

    if actor:
        try:
            actor.set_simulate_physics(False)
        except Exception:
            pass

        print(f"[PED] id={actor.id} idx={idx} x={x} y={y}")

    return actor


def spawn_two_pedestrians_right(world, bp_lib, ego_tf):
    # Sağ taraf: y pozitif
    layout = [
        (10.0, 5.8),
        (15.0, 6.2),
    ]

    count = 0

    for idx, (x, y) in enumerate(layout):
        if spawn_pedestrian(world, bp_lib, ego_tf, idx, x, y):
            count += 1

    print(f"[PED] right pedestrians spawned: {count}")


def list_bps(bp_lib):
    for bp in bp_lib:
        bid = bp.id.lower()
        if "traffic" in bid or "sign" in bid or "stop" in bid or "yield" in bid or "speed" in bid:
            print(bp.id)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--list-bps", action="store_true")
    parser.add_argument("--clear-only", action="store_true")
    args = parser.parse_args()

    random.seed(303)

    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)

    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    if args.list_bps:
        list_bps(bp_lib)
        return

    ego = find_ego(world)

    if ego is None:
        raise RuntimeError("Ego vehicle bulunamadı. Önce ana sahneyi kur.")

    ego_tf = ego.get_transform()

    print(f"[EGO] id={ego.id} yaw={ego_tf.rotation.yaw}")

    clear_old(world)

    if args.clear_only:
        return

    remove_front_non_ego_vehicles(world, ego)

    ego_tf = ego.get_transform()

    spawn_straight_car(world, bp_lib, ego_tf)
    spawn_stop_sign_left(world, bp_lib, ego_tf)
    spawn_two_pedestrians_right(world, bp_lib, ego_tf)

    print("")
    print("========== SIMPLE LAYOUT READY ==========")
    print("1 front straight car")
    print("1 stop sign on left")
    print("2 pedestrians on right")
    print("=========================================")
    print("")


if __name__ == "__main__":
    main()

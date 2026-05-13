#!/usr/bin/env python3
import carla
import time
import math
import numpy as np
import cv2
import os

OUT_DIR = "/tmp/carla_real_tl_test"
os.makedirs(OUT_DIR, exist_ok=True)

def image_to_bgr(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr[:, :, :3][:, :, ::-1].copy()

def color_score(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    red = cv2.countNonZero(red1 | red2)

    yellow = cv2.countNonZero(cv2.inRange(hsv, (18, 80, 80), (38, 255, 255)))
    green = cv2.countNonZero(cv2.inRange(hsv, (40, 60, 60), (90, 255, 255)))

    return red, yellow, green

def yaw_to_target(src, dst):
    dx = dst.x - src.x
    dy = dst.y - src.y
    return math.degrees(math.atan2(dy, dx))

def get_forward_vector(yaw_deg):
    yaw = math.radians(yaw_deg)
    return carla.Vector3D(math.cos(yaw), math.sin(yaw), 0)

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()

    traffic_lights = list(world.get_actors().filter("traffic.traffic_light*"))
    print(f"[INFO] Traffic light count: {len(traffic_lights)}")

    if not traffic_lights:
        print("[ERROR] Haritada traffic light yok.")
        return

    bp_lib = world.get_blueprint_library()

    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "1280")
    camera_bp.set_attribute("image_size_y", "720")
    camera_bp.set_attribute("fov", "70")

    spawn_points = world.get_map().get_spawn_points()
    ego = world.try_spawn_actor(vehicle_bp, spawn_points[0])
    if ego is None:
        ego = world.spawn_actor(vehicle_bp, spawn_points[1])

    camera_tf = carla.Transform(
        carla.Location(x=1.5, z=1.7),
        carla.Rotation(pitch=0)
    )
    camera = world.spawn_actor(camera_bp, camera_tf, attach_to=ego)

    latest = {"img": None}

    def cb(image):
        latest["img"] = image_to_bgr(image)

    camera.listen(cb)

    states = [
        ("red", carla.TrafficLightState.Red),
        ("yellow", carla.TrafficLightState.Yellow),
        ("green", carla.TrafficLightState.Green),
    ]

    best = None

    for tl in traffic_lights:
        tl_loc = tl.get_transform().location

        # Trafik ışığının karşısına ego aracı koyuyoruz.
        # Farklı uzaklıkları dener.
        for dist in [10, 14, 18, 22]:
            # Işığın kendi yönünün tersinden bakmayı dener
            tl_yaw = tl.get_transform().rotation.yaw
            back_yaw = tl_yaw + 180.0
            fwd = get_forward_vector(back_yaw)

            ego_loc = carla.Location(
                x=tl_loc.x + fwd.x * dist,
                y=tl_loc.y + fwd.y * dist,
                z=tl_loc.z
            )

            yaw = yaw_to_target(ego_loc, tl_loc)

            ego.set_transform(
                carla.Transform(
                    ego_loc,
                    carla.Rotation(yaw=yaw)
                )
            )

            time.sleep(0.5)

            scores = {}

            for name, state in states:
                tl.set_state(state)
                tl.freeze(True)

                time.sleep(0.7)

                img = latest["img"]
                if img is None:
                    continue

                # Merkez bölgeden renk analizi
                h, w = img.shape[:2]
                crop = img[int(h*0.15):int(h*0.85), int(w*0.25):int(w*0.75)]

                r, y, g = color_score(crop)
                scores[name] = (r, y, g)

                save_path = os.path.join(
                    OUT_DIR,
                    f"tl_{tl.id}_dist_{dist}_{name}.png"
                )
                cv2.imwrite(save_path, img)

            if len(scores) != 3:
                continue

            red_ok = scores["red"][0] > scores["red"][1] and scores["red"][0] > scores["red"][2]
            yellow_ok = scores["yellow"][1] > scores["yellow"][0] and scores["yellow"][1] > scores["yellow"][2]
            green_ok = scores["green"][2] > scores["green"][0] and scores["green"][2] > scores["green"][1]

            print(f"\n[TL {tl.id}] dist={dist}")
            print("RED score    :", scores["red"])
            print("YELLOW score :", scores["yellow"])
            print("GREEN score  :", scores["green"])
            print("OK:", red_ok, yellow_ok, green_ok)

            if red_ok and yellow_ok and green_ok:
                best = (tl, dist, ego.get_transform(), scores)
                break

        if best:
            break

    if best is None:
        print("\n[FAIL] Gerçek kırmızı/sarı/yeşil görünen uygun ışık bulunamadı.")
        print(f"[INFO] Görseller: {OUT_DIR}")
    else:
        tl, dist, ego_tf, scores = best
        print("\n[SUCCESS] UYGUN TRAFİK IŞIĞI BULUNDU")
        print("Traffic light id:", tl.id)
        print("Distance:", dist)
        print("Ego transform:")
        print(ego_tf)
        print("Scores:", scores)
        print(f"Görseller: {OUT_DIR}")

        tl.set_state(carla.TrafficLightState.Green)
        tl.freeze(True)

    print("\nKapatmak için CTRL+C")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    camera.stop()
    camera.destroy()
    ego.destroy()

if __name__ == "__main__":
    main()

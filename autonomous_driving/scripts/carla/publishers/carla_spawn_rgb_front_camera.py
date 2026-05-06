#!/usr/bin/env python3
import time
import carla


def find_ego(world):
    vehicles = world.get_actors().filter("vehicle.*")

    for actor in vehicles:
        if actor.attributes.get("role_name", "") == "ego":
            return actor

    if len(vehicles) > 0:
        return vehicles[0]

    return None


def destroy_old_rgb_front(world):
    count = 0

    for actor in world.get_actors().filter("sensor.camera.rgb"):
        role = actor.attributes.get("role_name", "")
        if role == "rgb_front":
            actor.destroy()
            count += 1

    print(f"[CAMERA] Eski rgb_front sensor temizlendi: {count}")


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)

    world = client.get_world()

    ego = find_ego(world)

    if ego is None:
        raise RuntimeError("Ego araç bulunamadı. Önce carla_straight_front_scene.py çalışmalı.")

    destroy_old_rgb_front(world)
    time.sleep(0.5)

    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("role_name", "rgb_front")
    bp.set_attribute("image_size_x", "960")
    bp.set_attribute("image_size_y", "540")
    bp.set_attribute("fov", "70")
    bp.set_attribute("sensor_tick", "0.05")

    camera_tf = carla.Transform(
        carla.Location(x=1.8, y=0.0, z=1.45),
        carla.Rotation(pitch=-4.0, yaw=0.0, roll=0.0),
    )

    cam = world.spawn_actor(bp, camera_tf, attach_to=ego)

    print("[CAMERA] rgb_front kamera spawn edildi.")
    print("[CAMERA] Ego:", ego.id, ego.type_id)
    print("[CAMERA] Camera:", cam.id, cam.type_id)
    print("[CAMERA] Bu terminal açık kalabilir ama şart değil.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[CAMERA] Çıkılıyor. Kamera silinmedi.")


if __name__ == "__main__":
    main()

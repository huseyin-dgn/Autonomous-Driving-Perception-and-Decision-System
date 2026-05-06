#!/usr/bin/env python3

import time
import argparse
import carla


def find_ego(world):
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.attributes.get("role_name") == "ego_vehicle":
            return actor
    return None


def draw_light_column(world, base_loc, state, label):
    lifetime = 0.20

    # Direk
    pole_bottom = carla.Location(base_loc.x, base_loc.y, base_loc.z)
    pole_top = carla.Location(base_loc.x, base_loc.y, base_loc.z + 2.2)
    world.debug.draw_line(
        pole_bottom,
        pole_top,
        thickness=0.06,
        color=carla.Color(30, 30, 30),
        life_time=lifetime,
        persistent_lines=False,
    )

    # Gövde
    body_center = carla.Location(base_loc.x, base_loc.y, base_loc.z + 2.55)
    extent = carla.Vector3D(0.25, 0.08, 0.65)
    rot = carla.Rotation()

    world.debug.draw_box(
        carla.BoundingBox(body_center, extent),
        rot,
        thickness=0.04,
        color=carla.Color(20, 20, 20),
        life_time=lifetime,
        persistent_lines=False,
    )

    red_loc = carla.Location(base_loc.x, base_loc.y, base_loc.z + 3.00)
    yellow_loc = carla.Location(base_loc.x, base_loc.y, base_loc.z + 2.55)
    green_loc = carla.Location(base_loc.x, base_loc.y, base_loc.z + 2.10)

    dim_red = carla.Color(80, 0, 0)
    dim_yellow = carla.Color(80, 80, 0)
    dim_green = carla.Color(0, 80, 0)

    on_red = carla.Color(255, 0, 0)
    on_yellow = carla.Color(255, 220, 0)
    on_green = carla.Color(0, 255, 0)

    red_color = on_red if state == "red" else dim_red
    yellow_color = on_yellow if state == "yellow" else dim_yellow
    green_color = on_green if state == "green" else dim_green

    for loc, color in [
        (red_loc, red_color),
        (yellow_loc, yellow_color),
        (green_loc, green_color),
    ]:
        world.debug.draw_point(
            loc,
            size=0.22,
            color=color,
            life_time=lifetime,
            persistent_lines=False,
        )

    world.debug.draw_string(
        carla.Location(base_loc.x, base_loc.y, base_loc.z + 3.45),
        label.upper(),
        draw_shadow=True,
        color=carla.Color(255, 255, 255),
        life_time=lifetime,
        persistent_lines=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=float, default=12.0)
    parser.add_argument("--side", type=float, default=3.0)
    parser.add_argument("--spacing", type=float, default=1.4)
    args = parser.parse_args()

    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()

    ego = find_ego(world)

    if ego is None:
        raise RuntimeError("ego_vehicle bulunamadı. Önce ego/kamera senaryosunu kur.")

    print("3 gerçek sahne ışığı çiziliyor: RED / YELLOW / GREEN")
    print("Bu terminal açık kalacak. Kapatırsan ışıklar gider.")

    while True:
        ego_tf = ego.get_transform()
        fwd = ego_tf.get_forward_vector()
        right = ego_tf.get_right_vector()

        center = ego_tf.location + fwd * args.distance + right * args.side

        red_base = center - right * args.spacing
        yellow_base = center
        green_base = center + right * args.spacing

        red_base.z = ego_tf.location.z
        yellow_base.z = ego_tf.location.z
        green_base.z = ego_tf.location.z

        draw_light_column(world, red_base, "red", "red")
        draw_light_column(world, yellow_base, "yellow", "yellow")
        draw_light_column(world, green_base, "green", "green")

        time.sleep(0.10)


if __name__ == "__main__":
    main()

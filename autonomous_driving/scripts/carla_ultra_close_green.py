#!/usr/bin/env python3
import carla
import math
import time

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()
bp = world.get_blueprint_library()

# temizle
for a in world.get_actors():
    if a.type_id.startswith("vehicle.") or a.type_id.startswith("sensor.camera"):
        try:
            a.destroy()
        except:
            pass

# ışıkları GREEN yap
for tl in world.get_actors().filter("traffic.traffic_light*"):
    tl.freeze(False)
    tl.set_state(carla.TrafficLightState.Green)
    tl.set_green_time(99999.0)
    tl.freeze(True)

TARGET_ID = 16
tl = world.get_actor(TARGET_ID)

loc = tl.get_location()

# direkt ışığın dibine araç
ego_loc = carla.Location(
    x=loc.x + 6.0,
    y=loc.y,
    z=0.4
)

yaw = 180.0

veh_bp = bp.find("vehicle.tesla.model3")
veh_bp.set_attribute("role_name", "ego_vehicle")

ego = world.spawn_actor(
    veh_bp,
    carla.Transform(
        ego_loc,
        carla.Rotation(yaw=yaw)
    )
)

cam_bp = bp.find("sensor.camera.rgb")
cam_bp.set_attribute("role_name", "rgb_front")
cam_bp.set_attribute("image_size_x", "1920")
cam_bp.set_attribute("image_size_y", "1080")

# ZOOM
cam_bp.set_attribute("fov", "18")

cam = world.spawn_actor(
    cam_bp,
    carla.Transform(
        carla.Location(x=2.2, z=2.4),
        carla.Rotation(pitch=-1.5)
    ),
    attach_to=ego
)

print("ULTRA CLOSE GREEN READY")
print("ego:", ego.id)
print("camera:", cam.id)
print("light:", tl.id)

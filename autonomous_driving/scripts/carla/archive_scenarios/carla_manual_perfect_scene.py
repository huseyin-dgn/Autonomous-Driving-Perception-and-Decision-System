#!/usr/bin/env python3
import carla
import time
import random

client = carla.Client("localhost", 2000)
client.set_timeout(20.0)

world = client.get_world()
bp = world.get_blueprint_library()

# ---------------------------------------------------
# CLEAR
# ---------------------------------------------------

for a in world.get_actors():
    role = a.attributes.get("role_name", "")

    if (
        role.startswith("adas_manual")
        or a.type_id.startswith("sensor.camera")
    ):
        try:
            a.destroy()
        except:
            pass

time.sleep(1)

# ---------------------------------------------------
# WEATHER
# ---------------------------------------------------

world.set_weather(
    carla.WeatherParameters(
        cloudiness=0.0,
        precipitation=0.0,
        fog_density=0.0,
        wetness=0.0,
        sun_altitude_angle=70.0,
    )
)

# ---------------------------------------------------
# TRAFFIC LIGHTS -> RED
# ---------------------------------------------------

for tl in world.get_actors().filter("traffic.traffic_light*"):
    try:
        tl.set_state(carla.TrafficLightState.Red)
        tl.freeze(True)
    except:
        pass

# ---------------------------------------------------
# CAMERA
# ---------------------------------------------------

cam_bp = bp.find("sensor.camera.rgb")

cam_bp.set_attribute("image_size_x", "960")
cam_bp.set_attribute("image_size_y", "720")
cam_bp.set_attribute("fov", "72")

cam_tf = carla.Transform(
    carla.Location(x=-95.0, y=25.5, z=2.2),
    carla.Rotation(pitch=-6.0, yaw=0.0, roll=0.0)
)

camera = world.spawn_actor(cam_bp, cam_tf)
print("[OK] CAMERA")

# spectator
world.get_spectator().set_transform(
    carla.Transform(
        carla.Location(x=-100.0, y=25.5, z=10.0),
        carla.Rotation(pitch=-30.0, yaw=0.0)
    )
)

# ---------------------------------------------------
# PERSON
# ---------------------------------------------------

walker_bp = random.choice(
    bp.filter("walker.pedestrian.*")
)

if walker_bp.has_attribute("is_invincible"):
    walker_bp.set_attribute("is_invincible", "false")

person_tf = carla.Transform(
    carla.Location(x=-76.0, y=23.0, z=1.0),
    carla.Rotation(yaw=180.0)
)

person = world.try_spawn_actor(walker_bp, person_tf)

if person:
    print("[OK] PERSON")

# ---------------------------------------------------
# MOTORCYCLE
# ---------------------------------------------------

motor_bp = bp.find("vehicle.kawasaki.ninja")

motor_tf = carla.Transform(
    carla.Location(x=-72.0, y=29.0, z=0.6),
    carla.Rotation(yaw=180.0)
)

motor = world.try_spawn_actor(motor_bp, motor_tf)

if motor:
    motor.set_autopilot(False)
    print("[OK] MOTOR")

# ---------------------------------------------------
# VEHICLE
# ---------------------------------------------------

veh_bp = bp.find("vehicle.tesla.model3")

veh_tf = carla.Transform(
    carla.Location(x=-55.0, y=25.5, z=0.6),
    carla.Rotation(yaw=180.0)
)

veh = world.try_spawn_actor(veh_bp, veh_tf)

if veh:
    veh.set_autopilot(False)
    print("[OK] VEHICLE")

print("")
print("===================================")
print("MANUAL PERFECT TEST READY")
print("===================================")
print("LEFT  : person")
print("CENTER: vehicle")
print("RIGHT : motorcycle")
print("RED LIGHT visible ahead")
print("===================================")

while True:
    time.sleep(1)

import carla
import math
import time

HOST = "localhost"
PORT = 2000
TIMEOUT = 20.0

LIGHT_INDEX = 0
DISTANCE = 6.0
CAMERA_Z = 5.2

client = carla.Client(HOST, PORT)
client.set_timeout(TIMEOUT)
world = client.get_world()
bp = world.get_blueprint_library()

print("[INFO] CARLA bağlantısı başarılı.")
print("[INFO] Map:", world.get_map().name)

# Eski test actorlerini temizle
removed = 0
for a in world.get_actors():
    try:
        rn = a.attributes.get("role_name", "")
        if rn in ["rgb_front", "adas_green_test"]:
            a.destroy()
            removed += 1
    except Exception:
        pass

print("[CLEAN] Temizlenen actor:", removed)

# Tüm trafik ışıklarını GREEN yap
tls = list(world.get_actors().filter("traffic.traffic_light*"))
if not tls:
    raise RuntimeError("Trafik ışığı bulunamadı.")

tls = sorted(tls, key=lambda x: x.id)

for tl in tls:
    tl.freeze(False)
    tl.set_state(carla.TrafficLightState.Green)
    tl.set_green_time(99999.0)
    tl.set_yellow_time(0.1)
    tl.set_red_time(0.1)
    tl.freeze(True)

time.sleep(0.5)

target = tls[LIGHT_INDEX]
tf = target.get_transform()
loc = tf.location
yaw = tf.rotation.yaw

# Işığın baktığı yönün karşısına kamera koy
rad = math.radians(yaw)
forward = carla.Vector3D(math.cos(rad), math.sin(rad), 0.0)

cam_loc = carla.Location(
    x=loc.x + forward.x * DISTANCE,
    y=loc.y + forward.y * DISTANCE,
    z=CAMERA_Z
)

cam_rot = carla.Rotation(
    pitch=0.0,
    yaw=yaw + 180.0,
    roll=0.0
)

cam_bp = bp.find("sensor.camera.rgb")
cam_bp.set_attribute("role_name", "rgb_front")
cam_bp.set_attribute("image_size_x", "1280")
cam_bp.set_attribute("image_size_y", "720")
cam_bp.set_attribute("fov", "70")
cam_bp.set_attribute("sensor_tick", "0.05")

camera = world.spawn_actor(
    cam_bp,
    carla.Transform(cam_loc, cam_rot)
)

print("===================================================")
print("[GREEN TEST SCENE READY]")
print("[SCENE] selected light id:", target.id)
print("[SCENE] light state:", target.get_state())
print("[SCENE] light loc:", loc)
print("[SCENE] light yaw:", yaw)
print("[SCENE] camera id:", camera.id)
print("[SCENE] camera loc:", cam_loc)
print("[SCENE] camera yaw:", cam_rot.yaw)
print("[SCENE] role_name:", camera.attributes.get("role_name"))
print("===================================================")
print("Şimdi publisher aç.")

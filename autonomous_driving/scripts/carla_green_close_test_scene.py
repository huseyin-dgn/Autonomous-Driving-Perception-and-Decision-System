#!/usr/bin/env python3
import carla
import time
import math
import rclpy

HOST = "localhost"
PORT = 2000
TIMEOUT = 10.0

TARGET_LIGHT_ID = 16
DISTANCE = 18.0
SIDE_OFFSET = 0.0

def yaw_to_target(src, dst):
    dx = dst.x - src.x
    dy = dst.y - src.y
    return math.degrees(math.atan2(dy, dx))

def destroy_old(world):
    count = 0
    for a in world.get_actors():
        if a.type_id.startswith("vehicle.") or a.type_id.startswith("sensor.camera"):
            try:
                a.destroy()
                count += 1
            except Exception:
                pass
    print(f"[CLEAN] destroyed={count}")

def force_all_green(world):
    for tl in world.get_actors().filter("traffic.traffic_light*"):
        tl.freeze(False)
        tl.set_state(carla.TrafficLightState.Green)
        tl.set_green_time(99999.0)
        tl.set_yellow_time(0.1)
        tl.set_red_time(0.1)
        tl.freeze(True)
    print("[LIGHT] all lights forced GREEN")


def main():
    rclpy.init()
    node = None

    try:
        node = CarlaCameraScreenAndROS()
        node.show_loop()

    except KeyboardInterrupt:
        print("[INFO] Publisher kullanıcı tarafından durduruldu.")

    except rclpy.executors.ExternalShutdownException:
        print("[INFO] ROS shutdown yakalandı.")

    except Exception as exc:
        print(f"[ERROR] Publisher hata: {exc}")
        raise

    finally:
        try:
            if node is not None:
                node.destroy_node()
        except Exception:
            pass

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
if __name__ == "__main__":
    main()
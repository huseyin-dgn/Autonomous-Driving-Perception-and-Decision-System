import json
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_sim.carla_loader import load_carla


class PID:
    def __init__(self, kp, ki, kd, mn, mx):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.mn = float(mn)
        self.mx = float(mx)
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, error, dt):
        dt = max(dt, 1e-3)
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        value = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(self.mn, min(self.mx, value))


class TeknofestRouteAgentNode(Node):
    def __init__(self):
        super().__init__("teknofest_route_agent_node")

        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 120.0)
        self.declare_parameter("ego_role_name", "ego_vehicle")

        self.declare_parameter("decision_topic", "/adas/decision")
        self.declare_parameter("mission_topic", "/adas/teknofest/mission")
        self.declare_parameter("debug_topic", "/adas/teknofest/route_agent_debug")
        self.declare_parameter("collision_topic", "/adas/events/collision")
        self.declare_parameter("collision_halt_s", 5.0)

        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("max_speed_mps", 3.0)
        self.declare_parameter("go_speed_mps", 2.0)
        self.declare_parameter("slow_speed_mps", 0.8)
        self.declare_parameter("parking_speed_mps", 0.45)
        self.declare_parameter("lookahead_m", 6.0)
        self.declare_parameter("max_steer", 0.55)
        self.declare_parameter("steer_kp", 0.025)
        self.declare_parameter("mission_stop_override", True)
        self.declare_parameter("ignore_decision_for_mission_test", False)

        self.carla_root = self.get_parameter("carla_root").value
        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.ego_role_name = self.get_parameter("ego_role_name").value

        self.decision_topic = self.get_parameter("decision_topic").value
        self.mission_topic = self.get_parameter("mission_topic").value
        self.debug_topic = self.get_parameter("debug_topic").value
        self.collision_topic = self.get_parameter("collision_topic").value
        self.collision_halt_s = float(self.get_parameter("collision_halt_s").value)
        self.collision_until = 0.0
        self.last_collision = None

        self.max_speed_mps = float(self.get_parameter("max_speed_mps").value)
        self.go_speed_mps = float(self.get_parameter("go_speed_mps").value)
        self.slow_speed_mps = float(self.get_parameter("slow_speed_mps").value)
        self.parking_speed_mps = float(self.get_parameter("parking_speed_mps").value)
        self.lookahead_m = float(self.get_parameter("lookahead_m").value)
        self.max_steer = float(self.get_parameter("max_steer").value)
        self.steer_kp = float(self.get_parameter("steer_kp").value)
        self.mission_stop_override = bool(self.get_parameter("mission_stop_override").value)
        self.ignore_decision_for_mission_test = bool(self.get_parameter("ignore_decision_for_mission_test").value)

        self.carla = load_carla(self.carla_root)
        self.client = self.carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.ego = self.wait_for_ego()

        self.latest_decision = {
            "decision": "STOP",
            "risk": "UNKNOWN",
            "target_speed": 0.0,
            "reason": "initial",
        }
        self.latest_mission = None
        self.last_decision_time = 0.0
        self.last_mission_time = 0.0

        self.speed_pid = PID(kp=0.55, ki=0.02, kd=0.04, mn=-1.0, mx=0.65)

        self.debug_pub = self.create_publisher(String, self.debug_topic, 10)

        self.create_subscription(String, self.decision_topic, self.decision_cb, 10)
        self.create_subscription(String, self.mission_topic, self.mission_cb, 10)
        self.create_subscription(String, self.collision_topic, self.collision_cb, 10)

        self.prev_time = time.time()
        rate = float(self.get_parameter("control_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self.tick)

        self.get_logger().info("TEKNOFEST route agent hazır.")

    def wait_for_ego(self):
        for _ in range(300):
            vehicles = self.world.get_actors().filter("vehicle.*")
            for vehicle in vehicles:
                if vehicle.attributes.get("role_name", "") == self.ego_role_name:
                    return vehicle
            time.sleep(0.2)

        raise RuntimeError("Ego vehicle bulunamadı.")

    def decision_cb(self, msg):
        try:
            self.latest_decision = json.loads(msg.data)
            self.last_decision_time = time.time()
        except Exception as exc:
            self.get_logger().warning(f"decision parse hatası: {exc}")

    def mission_cb(self, msg):
        try:
            self.latest_mission = json.loads(msg.data)
            self.last_mission_time = time.time()
        except Exception as exc:
            self.get_logger().warning(f"mission parse hatası: {exc}")

    def collision_cb(self, msg):
        self.last_collision = msg.data
        self.collision_until = time.time() + self.collision_halt_s
        self.speed_pid.reset()
        self.get_logger().warning(f"COLLISION HALT: {msg.data}")

    def get_speed(self):
        v = self.ego.get_velocity()
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    def clamp(self, value, mn, mx):
        return max(mn, min(mx, float(value)))

    def normalize_angle(self, angle):
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle

    def get_target_steer(self):
        """
        Basit heading-only yerine waypoint lokasyonuna pure-pursuit benzeri yönelme.
        Bu, aracın yol merkezinden kopup kaldırım/direk tarafına sürüklenmesini azaltır.
        """
        transform = self.ego.get_transform()
        waypoint = self.map.get_waypoint(
            transform.location,
            project_to_road=True,
            lane_type=self.carla.LaneType.Driving,
        )

        if waypoint is None:
            return 0.0

        next_waypoints = waypoint.next(self.lookahead_m)
        if not next_waypoints:
            return 0.0

        target_wp = next_waypoints[0]
        vehicle_loc = transform.location
        target_loc = target_wp.transform.location

        dx = target_loc.x - vehicle_loc.x
        dy = target_loc.y - vehicle_loc.y

        yaw = math.radians(transform.rotation.yaw)
        local_x = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        local_y = math.sin(-yaw) * dx + math.cos(-yaw) * dy

        # Hedef arkadaysa agresif U dönüşü yapmasın.
        if local_x < 0.5:
            local_x = 0.5

        angle = math.atan2(local_y, local_x)
        steer = self.clamp(1.25 * angle, -self.max_steer, self.max_steer)
        return steer

    def resolve_target_speed(self):

        now = time.time()

        if self.latest_mission is None or now - self.last_mission_time > 3.0:
            return 0.0, "mission_missing_or_timeout"

        stage = str(self.latest_mission.get("stage", "UNKNOWN"))
        must_stop = bool(self.latest_mission.get("must_stop", False))
        distance_to_target = self.latest_mission.get("distance_to_target_m", None)

        if stage in {"COMPLETED", "FAILED"}:
            return 0.0, f"mission_{stage.lower()}"

        if self.mission_stop_override and must_stop:
            return 0.0, f"mission_stop_stage:{stage}"

        if stage == "PARKING":
            return self.parking_speed_mps, "parking_slow"

        if self.ignore_decision_for_mission_test:
            target_speed = self.go_speed_mps
            return self.clamp(target_speed, 0.0, self.max_speed_mps), "mission_test_ignore_decision"

        if now - self.last_decision_time > 2.0:
            return 0.0, "decision_timeout"

        decision = str(self.latest_decision.get("decision", "STOP")).upper()
        reason = str(self.latest_decision.get("reason", "unknown"))

        if decision == "STOP":
            return 0.0, f"decision_stop:{reason}"

        if decision == "SLOW":
            return self.slow_speed_mps, f"decision_slow:{reason}"

        target_speed = self.latest_decision.get("target_speed", self.go_speed_mps)

        try:
            target_speed = float(target_speed)
        except Exception:
            target_speed = self.go_speed_mps

        if distance_to_target is not None and distance_to_target < 6.0:
            target_speed = min(target_speed, 0.7)

        return self.clamp(target_speed, 0.0, self.max_speed_mps), f"decision_go:{reason}"

    def tick(self):
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now

        target_speed, speed_reason = self.resolve_target_speed()
        current_speed = self.get_speed()

        if time.time() < self.collision_until:
            target_speed = 0.0
            speed_reason = "collision_halt"

        control = self.carla.VehicleControl()
        control.manual_gear_shift = False
        control.hand_brake = False

        if target_speed <= 0.01:
            control.throttle = 0.0
            control.brake = 1.0
            control.steer = 0.0 if current_speed < 0.2 else self.get_target_steer()
            self.speed_pid.reset()
        else:
            speed_error = target_speed - current_speed
            pid_out = self.speed_pid.step(speed_error, dt)

            if pid_out >= 0.0:
                control.throttle = self.clamp(pid_out, 0.0, 0.65)
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = self.clamp(abs(pid_out), 0.0, 1.0)

            control.steer = self.get_target_steer()

        self.ego.apply_control(control)

        payload = {
            "stamp": now,
            "mission_stage": self.latest_mission.get("stage") if self.latest_mission else None,
            "distance_to_target_m": self.latest_mission.get("distance_to_target_m") if self.latest_mission else None,
            "decision": self.latest_decision.get("decision"),
            "decision_reason": self.latest_decision.get("reason"),
            "target_speed_mps": round(target_speed, 3),
            "current_speed_mps": round(current_speed, 3),
            "speed_reason": speed_reason,
            "throttle": round(control.throttle, 3),
            "brake": round(control.brake, 3),
            "steer": round(control.steer, 3),
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.debug_pub.publish(msg)

        self.get_logger().info(
            f"[TEKNOFEST ROUTE] stage={payload['mission_stage']} "
            f"target={target_speed:.2f} speed={current_speed:.2f} "
            f"throttle={control.throttle:.2f} brake={control.brake:.2f} "
            f"steer={control.steer:.2f} reason={speed_reason}",
            throttle_duration_sec=0.5,
        )


def main(args=None):
    rclpy.init(args=args)
    node = TeknofestRouteAgentNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

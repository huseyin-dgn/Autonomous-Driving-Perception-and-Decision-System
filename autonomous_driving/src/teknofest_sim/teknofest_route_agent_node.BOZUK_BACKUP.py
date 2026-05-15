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

        self.declare_parameter("carla_root", "/mnt/carla/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 20.0)
        self.declare_parameter("ego_role_name", "ego_vehicle")

        self.declare_parameter("decision_topic", "/adas/decision")
        self.declare_parameter("mission_topic", "/adas/teknofest/mission")
        self.declare_parameter("debug_topic", "/adas/teknofest/route_agent_debug")

        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("max_speed_mps", 3.0)
        self.declare_parameter("go_speed_mps", 2.0)
        self.declare_parameter("slow_speed_mps", 0.8)
        self.declare_parameter("parking_speed_mps", 0.45)
        self.declare_parameter("lookahead_m", 6.0)
        self.declare_parameter("max_steer", 0.55)
        self.declare_parameter("steer_kp", 0.025)
        self.declare_parameter("mission_stop_override", True)

        self.carla_root = self.get_parameter("carla_root").value
        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.ego_role_name = self.get_parameter("ego_role_name").value

        self.decision_topic = self.get_parameter("decision_topic").value
        self.mission_topic = self.get_parameter("mission_topic").value
        self.debug_topic = self.get_parameter("debug_topic").value

        self.max_speed_mps = float(self.get_parameter("max_speed_mps").value)
        self.go_speed_mps = float(self.get_parameter("go_speed_mps").value)
        self.slow_speed_mps = float(self.get_parameter("slow_speed_mps").value)
        self.parking_speed_mps = float(self.get_parameter("parking_speed_mps").value)
        self.lookahead_m = float(self.get_parameter("lookahead_m").value)
        self.max_steer = float(self.get_parameter("max_steer").value)
        self.steer_kp = float(self.get_parameter("steer_kp").value)
        self.mission_stop_override = bool(self.get_parameter("mission_stop_override").value)

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

        rate = float(self.get_parameter("control_rate_hz").value)
        self.prev_time = time.time()
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self.tick)

        self.get_logger().info("TEKNOFEST route agent hazır.")

    def wait_for_ego(self):
        for _ in range(100):
            for vehicle in self.world.get_actors().filter("vehicle.*"):
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
        transform = self.ego.get_transform()
        waypoint = self.map.get_waypoint(
            transform.location,
            project_to_road=True,
            lane_type=self.carla.LaneType.Driving,
        )

        if waypoint is None:
            return 0.0, "no_waypoint"

        nxt = waypoint.next(self.lookahead_m)
        if not nxt:
            return 0.0, "no_next_waypoint"

        target_wp = nxt[0]
        target_yaw = target_wp.transform.rotation.yaw
        vehicle_yaw = transform.rotation.yaw

        heading_error = self.normalize_angle(target_yaw - vehicle_yaw)
        steer = self.clamp(self.steer_kp * heading_error, -self.max_steer, self.max_steer)

        return steer, f"lane_keep_heading_error:{heading_error:.2f}"

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

    def hard_stop_control(self):
        control = self.carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 1.0
        control.steer = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False
        return control

    def tick(self):
        # RUNTIME_DEFAULTS_FIX:
        # Farklı commit/patch karışınca bazı runtime alanları __init__ içinde oluşmayabiliyor.
        # Node crash olmasın diye tick başında güvenli varsayılanları garanti ediyoruz.
        if not hasattr(self, "collision_until"):
            self.collision_until = 0.0
        if not hasattr(self, "last_collision"):
            self.last_collision = None
        if not hasattr(self, "last_throttle_cmd"):
            self.last_throttle_cmd = 0.0
        if not hasattr(self, "last_brake_cmd"):
            self.last_brake_cmd = 0.0
        if not hasattr(self, "current_lane_debug"):
            self.current_lane_debug = {
                "enabled": False,
                "used": False,
                "reason": "runtime_default",
            }

        target_speed, reason = self.resolve_target_speed()
        current_speed = self.get_speed()

        if time.time() < self.collision_until:
            control = self.hard_stop_control()
            reason = "collision_halt"
            target_speed = 0.0
            self.last_throttle_cmd = 0.0
            self.last_brake_cmd = 0.0

        elif target_speed <= 0.01:
            control = self.hard_stop_control()
            self.last_throttle_cmd = 0.0
            self.last_brake_cmd = 0.0

        else:
            ok = self.set_agent_destination_if_needed()

            if not ok:
                control = self.hard_stop_control()
                reason = "route_missing_stop"
                target_speed = 0.0
                self.last_throttle_cmd = 0.0
                self.last_brake_cmd = 0.0

            else:
                try:
                    self.agent.set_target_speed(target_speed * 3.6)
                except Exception:
                    pass

                try:
                    control = self.agent.run_step(debug=False)
                except TypeError:
                    control = self.agent.run_step()

                basic_steer = self.clamp(control.steer, -self.max_steer, self.max_steer)
                control.steer = self.apply_lane_assist_to_steer(basic_steer, target_speed)

                # SMOOTH_LONGITUDINAL_FIX:
                # BasicAgent steer iyi; throttle/brake düşük hızda gaz-fren-gaz-fren yapıyor.
                # Bu yüzden gaz/freni kendimiz yumuşatıyoruz.
                if not hasattr(self, "last_throttle_cmd"):
                    self.last_throttle_cmd = 0.0

                if not hasattr(self, "last_brake_cmd"):
                    self.last_brake_cmd = 0.0

                speed_error = float(target_speed) - float(current_speed)
                overspeed = float(current_speed) - float(target_speed)

                desired_throttle = 0.0
                desired_brake = 0.0

                if speed_error > 0.20:
                    desired_throttle = 0.055 + 0.14 * speed_error
                    desired_throttle = self.clamp(desired_throttle, 0.055, 0.23)
                    desired_brake = 0.0

                elif overspeed <= 0.55:
                    desired_throttle = 0.018 if current_speed < target_speed else 0.0
                    desired_brake = 0.0

                else:
                    desired_throttle = 0.0
                    desired_brake = self.clamp(0.10 * (overspeed - 0.55), 0.0, 0.08)

                def _slew(cur, dst, step):
                    cur = float(cur)
                    dst = float(dst)
                    step = abs(float(step))

                    if dst > cur:
                        return min(dst, cur + step)

                    if dst < cur:
                        return max(dst, cur - step)

                    return cur

                throttle_cmd = _slew(self.last_throttle_cmd, desired_throttle, 0.030)
                brake_cmd = _slew(self.last_brake_cmd, desired_brake, 0.035)

                if brake_cmd > 0.001:
                    throttle_cmd = 0.0

                self.last_throttle_cmd = throttle_cmd
                self.last_brake_cmd = brake_cmd

                control.throttle = self.clamp(throttle_cmd, 0.0, 0.23)
                control.brake = self.clamp(brake_cmd, 0.0, 0.08)
                control.hand_brake = False
                control.manual_gear_shift = False

        try:
            control.throttle = self.clamp(control.throttle, 0.0, 0.23)
            control.brake = self.clamp(control.brake, 0.0, 1.0)
            control.steer = self.clamp(control.steer, -0.26, 0.26)
        except Exception:
            pass

        self.ego.apply_control(control)

        target = self.get_target() or {}
        mission_dist = self.latest_mission.get("distance_to_target_m") if self.latest_mission else None

        payload = {
            "stamp": time.time(),
            "mission_stage": self.get_stage(),
            "task_index": self.latest_mission.get("task_index") if self.latest_mission else None,
            "target_name": target.get("name"),
            "distance_to_target_m": mission_dist,
            "target_speed_mps": round(target_speed, 3),
            "current_speed_mps": round(current_speed, 3),
            "throttle": round(control.throttle, 3),
            "brake": round(control.brake, 3),
            "steer": round(control.steer, 3),
            "route_status": self.route_status,
            "lane": self.current_lane_debug,
            "reason": reason,
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.debug_pub.publish(msg)

        self.get_logger().info(
            f"[TEKNOFEST ROUTE] stage={payload['mission_stage']} "
            f"target_name={payload['target_name']} "
            f"dist={payload['distance_to_target_m']} "
            f"target={payload['target_speed_mps']:.2f} "
            f"speed={payload['current_speed_mps']:.2f} "
            f"throttle={payload['throttle']:.2f} "
            f"brake={payload['brake']:.2f} "
            f"steer={payload['steer']:.2f} "
            f"lane={payload['lane'].get('reason') if isinstance(payload['lane'], dict) else payload['lane']} "
            f"route={payload['route_status']} "
            f"reason={payload['reason']}",
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
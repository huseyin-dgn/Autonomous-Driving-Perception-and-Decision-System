import json
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_sim.carla_loader import load_carla


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
        self.declare_parameter("collision_halt_s", 4.0)

        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("max_speed_mps", 1.3)
        self.declare_parameter("go_speed_mps", 0.9)
        self.declare_parameter("slow_speed_mps", 0.5)
        self.declare_parameter("parking_speed_mps", 0.35)

        # BasicAgent direksiyonu kendi üretir. Bunu çok düşük tutarsan araç dönemiyor.
        self.declare_parameter("max_steer", 0.70)
        self.declare_parameter("lane_assist_enabled", True)
        self.declare_parameter("lane_topic", "/adas/lane/assist")
        self.declare_parameter("lane_min_confidence", 0.60)
        self.declare_parameter("lane_fresh_timeout_s", 0.50)
        self.declare_parameter("lane_blend_straight", 0.35)
        self.declare_parameter("lane_blend_turn", 0.12)
        self.declare_parameter("lane_turn_steer_threshold", 0.28)
        self.declare_parameter("lane_allowed_stages", "GO_TO_TASK,GO_TO_PARK")

        self.declare_parameter("mission_stop_override", True)
        self.declare_parameter("ignore_decision_for_mission_test", True)

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
        self.max_steer = float(self.get_parameter("max_steer").value)

        self.lane_assist_enabled = bool(self.get_parameter("lane_assist_enabled").value)
        self.lane_topic = self.get_parameter("lane_topic").value
        self.lane_min_confidence = float(self.get_parameter("lane_min_confidence").value)
        self.lane_fresh_timeout_s = float(self.get_parameter("lane_fresh_timeout_s").value)
        self.lane_blend_straight = float(self.get_parameter("lane_blend_straight").value)
        self.lane_blend_turn = float(self.get_parameter("lane_blend_turn").value)
        self.lane_turn_steer_threshold = float(self.get_parameter("lane_turn_steer_threshold").value)
        self.lane_allowed_stages = [
            x.strip() for x in str(self.get_parameter("lane_allowed_stages").value).split(",") if x.strip()
        ]

        self.mission_stop_override = bool(self.get_parameter("mission_stop_override").value)
        self.ignore_decision_for_mission_test = False

        self.carla = load_carla(self.carla_root)
        self.client = self.carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.ego = self.wait_for_ego()

        self.BasicAgent = self.load_basic_agent()
        self.agent = self.BasicAgent(self.ego, target_speed=self.go_speed_mps * 3.6)

        self.configure_agent_ignore_rules()

        self.latest_decision = {
            "decision": "STOP",
            "risk": "UNKNOWN",
            "target_speed": 0.0,
            "reason": "initial",
        }
        self.latest_mission = None
        self.last_decision_time = 0.0
        self.last_mission_time = 0.0

        self.active_target_key = None
        self.active_destination = None
        self.route_status = "not_planned"

        self.latest_lane = None
        self.last_lane_time = 0.0
        self.current_lane_debug = {
            "enabled": self.lane_assist_enabled,
            "used": False,
            "reason": "initial",
        }

        self.debug_pub = self.create_publisher(String, self.debug_topic, 10)

        self.create_subscription(String, self.decision_topic, self.decision_cb, 10)
        self.create_subscription(String, self.mission_topic, self.mission_cb, 10)
        self.create_subscription(String, self.collision_topic, self.collision_cb, 10)
        self.create_subscription(String, self.lane_topic, self.lane_cb, 10)

        rate = float(self.get_parameter("control_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self.tick)

        self.get_logger().info("TEKNOFEST route agent hazır: CARLA BasicAgent lane follower aktif.")

    def load_basic_agent(self):
        possible_paths = [
            os.path.join(self.carla_root, "PythonAPI", "carla"),
            os.path.join(self.carla_root, "PythonAPI"),
            os.path.expanduser("~/CARLA_DISK/PythonAPI/carla"),
            os.path.expanduser("~/İndirilenler/PythonAPI/carla"),
        ]

        for p in possible_paths:
            if os.path.isdir(p) and p not in sys.path:
                sys.path.append(p)

        try:
            from agents.navigation.basic_agent import BasicAgent
            self.get_logger().info("BasicAgent import OK.")
            return BasicAgent
        except Exception as e:
            raise RuntimeError(f"BasicAgent import edilemedi. PythonAPI/carla/agents yolu yok veya hatalı: {e}")

    def configure_agent_ignore_rules(self):
        for method_name in ["ignore_traffic_lights", "ignore_stop_signs", "ignore_vehicles"]:
            try:
                if hasattr(self.agent, method_name):
                    getattr(self.agent, method_name)(True)
                    self.get_logger().info(f"BasicAgent {method_name}(True)")
            except Exception as e:
                self.get_logger().warning(f"{method_name} ayarlanamadı: {e}")

    def wait_for_ego(self):
        for _ in range(300):
            vehicles = self.world.get_actors().filter("vehicle.*")
            for vehicle in vehicles:
                if vehicle.attributes.get("role_name", "") == self.ego_role_name:
                    self.get_logger().info(f"Ego bulundu: id={vehicle.id}")
                    return vehicle
            time.sleep(0.2)

        raise RuntimeError("Ego vehicle bulunamadı.")

    def decision_cb(self, msg):
        try:
            data = json.loads(msg.data)

            # Off-lane person filter:
            # Kamera genişliği 960. center_x oranı 0.30-0.70 dışındaysa kişi şerit dışında kabul edilir.
            try:
                if str(data.get("decision", "")).upper() == "STOP" and "person" in str(data.get("reason", "")):
                    person = data.get("active_person")
                    if isinstance(person, dict):
                        cx = person.get("center_x")
                        bottom = person.get("bottom_ratio")

                        if cx is None:
                            bbox = person.get("bbox")
                            if isinstance(bbox, list) and len(bbox) >= 4:
                                cx = (float(bbox[0]) + float(bbox[2])) / 2.0

                        if cx is not None:
                            cx_ratio = float(cx) / 960.0
                            bottom_ok = True if bottom is None else float(bottom) >= 0.45
                            in_lane = 0.30 <= cx_ratio <= 0.70 and bottom_ok

                            if not in_lane:
                                old_reason = data.get("reason")
                                data["decision"] = "GO"
                                data["risk"] = "LOW"
                                data["target_speed"] = 0.8
                                data["reason"] = (
                                    f"route_offlane_person_filtered:"
                                    f"old={old_reason},cx={cx_ratio:.3f},bottom={bottom}"
                                )
                                data["person_risk"] = "offlane_filtered"
            except Exception as exc:
                self.get_logger().warning(f"offlane person filter hata: {exc}")

            self.latest_decision = data
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
        self.get_logger().warning(f"COLLISION HALT: {msg.data}")

    def lane_cb(self, msg):
        try:
            self.latest_lane = json.loads(msg.data)
            self.last_lane_time = time.time()
        except Exception as exc:
            self.get_logger().warning(f"lane assist parse hatası: {exc}")

    def apply_lane_assist_to_steer(self, basic_steer, target_speed):
        now = time.time()
        stage = self.get_stage()

        debug = {
            "enabled": bool(self.lane_assist_enabled),
            "used": False,
            "reason": "not_used",
            "basic_steer": round(float(basic_steer), 4),
            "final_steer": round(float(basic_steer), 4),
            "lane_confidence": None,
            "lane_steer": None,
            "lane_offset_norm": None,
            "blend": 0.0,
        }

        if not self.lane_assist_enabled:
            debug["reason"] = "disabled"
            self.current_lane_debug = debug
            return basic_steer

        if stage not in self.lane_allowed_stages:
            debug["reason"] = f"stage_not_allowed:{stage}"
            self.current_lane_debug = debug
            return basic_steer

        if target_speed <= 0.05:
            debug["reason"] = "target_speed_zero"
            self.current_lane_debug = debug
            return basic_steer

        if self.latest_lane is None or now - self.last_lane_time > self.lane_fresh_timeout_s:
            debug["reason"] = "lane_timeout"
            self.current_lane_debug = debug
            return basic_steer

        lane_detected = bool(self.latest_lane.get("lane_detected", False))
        conf = float(self.latest_lane.get("confidence", 0.0))
        lane_steer = float(self.latest_lane.get("lane_steer", 0.0))
        offset_norm = float(self.latest_lane.get("offset_norm", 0.0))

        debug["lane_confidence"] = round(conf, 3)
        debug["lane_steer"] = round(lane_steer, 4)
        debug["lane_offset_norm"] = round(offset_norm, 4)

        if not lane_detected:
            debug["reason"] = "lane_not_detected"
            self.current_lane_debug = debug
            return basic_steer

        if conf < self.lane_min_confidence:
            debug["reason"] = f"low_conf:{conf:.3f}"
            self.current_lane_debug = debug
            return basic_steer

        # Keskin dönüşte lane etkisini azalt. Düz yolda daha fazla hizalasın.
        if abs(basic_steer) >= self.lane_turn_steer_threshold:
            blend = self.lane_blend_turn
            debug["reason"] = "used_turn_low_blend"
        else:
            blend = self.lane_blend_straight
            debug["reason"] = "used_straight_blend"

        blend = self.clamp(blend, 0.0, 0.75)
        final_steer = (1.0 - blend) * basic_steer + blend * lane_steer
        final_steer = self.clamp(final_steer, -self.max_steer, self.max_steer)

        debug["used"] = True
        debug["blend"] = round(float(blend), 3)
        debug["final_steer"] = round(float(final_steer), 4)

        self.current_lane_debug = debug
        return final_steer

    def get_speed(self):
        v = self.ego.get_velocity()
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    def clamp(self, value, mn, mx):
        return max(mn, min(mx, float(value)))

    def get_stage(self):
        if not self.latest_mission:
            return None
        return self.latest_mission.get("stage")

    def get_target(self):
        if not self.latest_mission:
            return None
        return self.latest_mission.get("target")

    def get_target_key(self):
        target = self.get_target()
        if not target or not self.latest_mission:
            return None

        return (
            str(self.latest_mission.get("stage")) + "|" +
            str(self.latest_mission.get("task_index")) + "|" +
            str(target.get("name")) + "|" +
            str(round(float(target.get("lat", 0.0)), 10)) + "|" +
            str(round(float(target.get("lon", 0.0)), 10))
        )

    def mission_geo_to_carla_location_near_ego(self, target):
        ego_loc = self.ego.get_location()

        base_geo = self.map.transform_to_geolocation(ego_loc)

        geo_x = self.map.transform_to_geolocation(
            self.carla.Location(x=ego_loc.x + 1.0, y=ego_loc.y, z=ego_loc.z)
        )
        geo_y = self.map.transform_to_geolocation(
            self.carla.Location(x=ego_loc.x, y=ego_loc.y + 1.0, z=ego_loc.z)
        )

        lat0 = float(base_geo.latitude)
        lon0 = float(base_geo.longitude)

        lat_dx = float(geo_x.latitude) - lat0
        lon_dx = float(geo_x.longitude) - lon0
        lat_dy = float(geo_y.latitude) - lat0
        lon_dy = float(geo_y.longitude) - lon0

        target_lat = float(target["lat"])
        target_lon = float(target["lon"])

        dlat = target_lat - lat0
        dlon = target_lon - lon0

        det = lat_dx * lon_dy - lat_dy * lon_dx

        if abs(det) < 1e-16:
            self.get_logger().warning("Geo inverse det çok küçük.")
            return ego_loc

        dx = (dlat * lon_dy - lat_dy * dlon) / det
        dy = (lat_dx * dlon - dlat * lon_dx) / det

        dx = self.clamp(dx, -500.0, 500.0)
        dy = self.clamp(dy, -500.0, 500.0)

        return self.carla.Location(x=ego_loc.x + dx, y=ego_loc.y + dy, z=ego_loc.z)

    def destination_from_target(self, target):
        raw_loc = self.mission_geo_to_carla_location_near_ego(target)

        wp = self.map.get_waypoint(
            raw_loc,
            project_to_road=True,
            lane_type=self.carla.LaneType.Driving,
        )

        if wp is None:
            self.get_logger().warning("Target waypoint bulunamadı, raw location kullanılacak.")
            return raw_loc

        loc = wp.transform.location

        # Sağ şerit / sağ tarafa yanaşma offset'i.
        # Pozitif değer waypoint'in sağ vektörüne doğru kaydırır.
        try:
            right_vec = wp.transform.get_right_vector()
            lane_shift_m = 2.2
            shifted_x = loc.x + right_vec.x * lane_shift_m
            shifted_y = loc.y + right_vec.y * lane_shift_m
            return self.carla.Location(x=shifted_x, y=shifted_y, z=loc.z + 0.2)
        except Exception:
            return self.carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.2)

    def set_agent_destination_if_needed(self):
        target = self.get_target()
        key = self.get_target_key()

        if target is None or key is None:
            self.route_status = "mission_target_missing"
            return False

        if key == self.active_target_key:
            return True

        dest = self.destination_from_target(target)
        start = self.ego.get_location()

        try:
            self.agent.set_destination(dest)
        except TypeError:
            self.agent.set_destination(dest, start_location=start)
        except Exception:
            try:
                self.agent.set_destination(start, dest)
            except Exception as e:
                self.route_status = f"set_destination_failed:{e}"
                self.get_logger().error(self.route_status)
                return False

        self.active_target_key = key
        self.active_destination = dest
        self.route_status = f"basic_agent_route_to:{target.get('name')}"

        self.get_logger().info(
            f"BasicAgent destination set: stage={self.get_stage()} "
            f"target={target.get('name')} dest=({dest.x:.2f},{dest.y:.2f},{dest.z:.2f})"
        )

        return True

    def hard_stop_control(self):
        control = self.carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 1.0
        control.steer = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False
        return control

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

        if False and self.ignore_decision_for_mission_test:
            target_speed = self.go_speed_mps
        else:
            if now - self.last_decision_time > 2.0:
                return 0.0, "decision_timeout"

            decision = str(self.latest_decision.get("decision", "STOP")).upper()
            reason = str(self.latest_decision.get("reason", "unknown"))

            if decision == "STOP":
                return 0.0, f"decision_stop:{reason}"

            if decision == "SLOW":
                return self.slow_speed_mps, f"decision_slow:{reason}"

            try:
                target_speed = float(self.latest_decision.get("target_speed", self.go_speed_mps))
            except Exception:
                target_speed = self.go_speed_mps

        if distance_to_target is not None and distance_to_target < 8.0:
            target_speed = min(target_speed, 0.55)

        return self.clamp(target_speed, 0.0, self.max_speed_mps), "mission_test_ignore_decision"

    def tick(self):
        target_speed, reason = self.resolve_target_speed()
        current_speed = self.get_speed()

        if time.time() < self.collision_until:
            control = self.hard_stop_control()
            reason = "collision_halt"
            target_speed = 0.0
        elif target_speed <= 0.01:
            control = self.hard_stop_control()
        else:
            ok = self.set_agent_destination_if_needed()
            if not ok:
                control = self.hard_stop_control()
                reason = "route_missing_stop"
                target_speed = 0.0
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
                control.throttle = self.clamp(control.throttle, 0.0, 0.75)
                control.brake = self.clamp(control.brake, 0.0, 1.0)
                control.hand_brake = False
                control.manual_gear_shift = False

        # HARD SAFETY CLAMP: araç kontrolden çıkmasın diye hız/gaz sınırı
        try:
            control.throttle = self.clamp(control.throttle, 0.0, 0.40)
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
            f"target={target_speed:.2f} speed={current_speed:.2f} "
            f"throttle={control.throttle:.2f} brake={control.brake:.2f} "
            f"steer={control.steer:.2f} lane={self.current_lane_debug.get('reason')} route={self.route_status} "
            f"reason={reason}",
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

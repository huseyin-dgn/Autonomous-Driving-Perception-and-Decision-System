import json
import time
from dataclasses import asdict

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String

from teknofest_sim.geojson_mission import (
    haversine_meters,
    load_mission_geojson,
    mission_to_dict,
)


class TeknofestMissionNode(Node):
    def __init__(self):
        super().__init__("teknofest_mission_node")

        self.declare_parameter("mission_geojson", "missions/teknofest_round3.geojson")
        self.declare_parameter("round_name", "round_3")
        self.declare_parameter("gnss_topic", "/adas/localization/gnss")
        self.declare_parameter("mission_topic", "/adas/teknofest/mission")
        self.declare_parameter("event_topic", "/adas/teknofest/events")

        self.declare_parameter("point_pass_tolerance_m", 1.0)
        self.declare_parameter("passenger_stop_min_s", 15.0)
        self.declare_parameter("passenger_stop_max_s", 20.0)
        self.declare_parameter("park_time_limit_s", 180.0)

        self.mission_geojson = self.get_parameter("mission_geojson").value
        self.round_name = self.get_parameter("round_name").value

        self.gnss_topic = self.get_parameter("gnss_topic").value
        self.mission_topic = self.get_parameter("mission_topic").value
        self.event_topic = self.get_parameter("event_topic").value

        self.point_pass_tolerance_m = float(self.get_parameter("point_pass_tolerance_m").value)
        self.passenger_stop_min_s = float(self.get_parameter("passenger_stop_min_s").value)
        self.passenger_stop_max_s = float(self.get_parameter("passenger_stop_max_s").value)
        self.park_time_limit_s = float(self.get_parameter("park_time_limit_s").value)

        self.mission = load_mission_geojson(self.mission_geojson, self.round_name)

        self.current_lat = None
        self.current_lon = None

        self.stage = "GO_TO_TASK"
        self.route_index = 0
        self.completed_task_count = 0
        self.stop_started_at = None
        self.park_entry_reached_at = None
        self.completed = False

        self.route_points = self.build_route_points()

        self.pub = self.create_publisher(String, self.mission_topic, 10)
        self.event_pub = self.create_publisher(String, self.event_topic, 10)

        self.create_subscription(NavSatFix, self.gnss_topic, self.gnss_cb, 10)
        self.timer = self.create_timer(0.2, self.tick)

        self.get_logger().info(
            f"TEKNOFEST mission loaded: {json.dumps(mission_to_dict(self.mission), ensure_ascii=False)}"
        )

        self.get_logger().info(
            "ROUTE_SEQUENCE_FIX active route="
            + json.dumps(
                [
                    {
                        "index": i,
                        "name": p.name,
                        "kind": self.point_kind(p),
                        "nokta_id": p.nokta_id,
                    }
                    for i, p in enumerate(self.route_points)
                ],
                ensure_ascii=False,
            )
        )

    def publish_event(self, event_type: str, payload: dict):
        msg = String()
        data = {
            "stamp": time.time(),
            "event_type": event_type,
            **payload,
        }
        msg.data = json.dumps(data, ensure_ascii=False)
        self.event_pub.publish(msg)
        self.get_logger().info(f"[MISSION EVENT] {msg.data}")

    def gnss_cb(self, msg: NavSatFix):
        self.current_lat = float(msg.latitude)
        self.current_lon = float(msg.longitude)

    def point_kind(self, point):
        name = str(point.name or "").lower()

        if name == "park_giris" or name.startswith("park"):
            return "park"

        if name.startswith("gorev_") or name.startswith("passenger_"):
            return "task"

        return "via"

    def build_route_points(self):
        points = [
            p for name, p in self.mission.raw_points.items()
            if str(name).lower() != "start"
        ]

        points = sorted(points, key=lambda p: int(p.nokta_id))

        # Park her zaman en sonda olsun.
        non_park = [p for p in points if self.point_kind(p) != "park"]
        park = [p for p in points if self.point_kind(p) == "park"]

        route = non_park + park

        if not route:
            route = list(self.mission.task_points) + [self.mission.park_entry]

        return route

    def current_route_point(self):
        if self.route_index < len(self.route_points):
            return self.route_points[self.route_index]

        return self.mission.park_entry

    def current_route_kind(self):
        return self.point_kind(self.current_route_point())

    def advance_route_index(self):
        if self.route_index < len(self.route_points) - 1:
            self.route_index += 1
            return True

        return False

    def target_point(self):
        return self.current_route_point()

    def distance_to_target(self):
        if self.current_lat is None or self.current_lon is None:
            return None

        target = self.target_point()
        return haversine_meters(
            self.current_lat,
            self.current_lon,
            target.lat,
            target.lon,
        )

    def tick(self):
        dist = self.distance_to_target()
        target = self.target_point()
        kind = self.current_route_kind()
        now = time.time()

        if dist is not None and not self.completed:
            if self.stage == "GO_TO_TASK":
                if kind == "via" and dist <= self.point_pass_tolerance_m:
                    self.publish_event(
                        "route_via_reached",
                        {
                            "target": asdict(target),
                            "route_index": self.route_index,
                            "distance_m": round(dist, 3),
                        },
                    )

                    self.advance_route_index()
                    target = self.target_point()
                    kind = self.current_route_kind()

                    if kind == "park":
                        self.stage = "GO_TO_PARK"
                    else:
                        self.stage = "GO_TO_TASK"

                elif kind == "task" and dist <= self.point_pass_tolerance_m:
                    self.stage = "PASSENGER_STOP"
                    self.stop_started_at = now
                    self.publish_event(
                        "passenger_stop_started",
                        {
                            "target": asdict(target),
                            "task_index": self.completed_task_count,
                            "route_index": self.route_index,
                            "distance_m": round(dist, 3),
                        },
                    )

                elif kind == "park" and dist <= self.point_pass_tolerance_m:
                    self.stage = "PARKING"
                    self.park_entry_reached_at = now
                    self.publish_event(
                        "park_entry_reached",
                        {
                            "target": asdict(target),
                            "route_index": self.route_index,
                            "distance_m": round(dist, 3),
                        },
                    )

            elif self.stage == "PASSENGER_STOP":
                elapsed = now - self.stop_started_at

                if elapsed >= self.passenger_stop_min_s:
                    self.publish_event(
                        "passenger_stop_completed",
                        {
                            "target": asdict(target),
                            "task_index": self.completed_task_count,
                            "route_index": self.route_index,
                            "stop_elapsed_s": round(elapsed, 3),
                            "valid_stop_window": elapsed <= self.passenger_stop_max_s,
                        },
                    )

                    self.completed_task_count += 1
                    self.stop_started_at = None

                    self.advance_route_index()
                    target = self.target_point()
                    kind = self.current_route_kind()

                    if kind == "park":
                        self.stage = "GO_TO_PARK"
                    else:
                        self.stage = "GO_TO_TASK"

            elif self.stage == "GO_TO_PARK":
                if kind == "via" and dist <= self.point_pass_tolerance_m:
                    self.publish_event(
                        "route_via_reached",
                        {
                            "target": asdict(target),
                            "route_index": self.route_index,
                            "distance_m": round(dist, 3),
                        },
                    )

                    self.advance_route_index()

                elif kind == "park" and dist <= self.point_pass_tolerance_m:
                    self.stage = "PARKING"
                    self.park_entry_reached_at = now
                    self.publish_event(
                        "park_entry_reached",
                        {
                            "target": asdict(target),
                            "route_index": self.route_index,
                            "distance_m": round(dist, 3),
                        },
                    )

            elif self.stage == "PARKING":
                elapsed = now - self.park_entry_reached_at

                if elapsed >= 8.0:
                    self.completed = True
                    self.stage = "COMPLETED"
                    self.publish_event(
                        "mission_completed",
                        {
                            "park_elapsed_s": round(elapsed, 3),
                            "within_park_time_limit": elapsed <= self.park_time_limit_s,
                        },
                    )

                elif elapsed > self.park_time_limit_s:
                    self.completed = True
                    self.stage = "FAILED"
                    self.publish_event(
                        "park_timeout",
                        {
                            "park_elapsed_s": round(elapsed, 3),
                        },
                    )

        target = self.target_point()
        kind = self.current_route_kind()

        out = {
            "stamp": now,
            "mission": mission_to_dict(self.mission),
            "stage": self.stage,
            "task_index": self.completed_task_count,
            "route_index": self.route_index,
            "route_kind": kind,
            "target": asdict(target),
            "distance_to_target_m": round(dist, 3) if dist is not None else None,
            "must_stop": self.stage in {"PASSENGER_STOP", "PARKING"},
            "completed": self.completed,
            "passenger_stop_elapsed_s": round(now - self.stop_started_at, 3)
            if self.stop_started_at is not None else None,
            "park_elapsed_s": round(now - self.park_entry_reached_at, 3)
            if self.park_entry_reached_at is not None else None,
        }

        msg = String()
        msg.data = json.dumps(out, ensure_ascii=False)
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TeknofestMissionNode()

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

import json
import math
import random
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_sim.carla_loader import load_carla


class TeknofestScenarioNode(Node):
    def __init__(self):
        super().__init__("teknofest_scenario_node")

        self.declare_parameter("carla_root", "/mnt/carla/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 20.0)
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter("scenario_round", "round_3")
        self.declare_parameter("traffic_manager_port", 8000)
        self.declare_parameter("destroy_on_shutdown", True)

        self.declare_parameter("npc_vehicle_count", 6)
        self.declare_parameter("walker_count", 4)
        self.declare_parameter("static_obstacle_count", 4)
        self.declare_parameter("dynamic_crossing_enabled", True)

        self.carla_root = self.get_parameter("carla_root").value
        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.ego_role_name = self.get_parameter("ego_role_name").value
        self.scenario_round = self.get_parameter("scenario_round").value
        self.traffic_manager_port = int(self.get_parameter("traffic_manager_port").value)
        self.destroy_on_shutdown = bool(self.get_parameter("destroy_on_shutdown").value)

        self.npc_vehicle_count = int(self.get_parameter("npc_vehicle_count").value)
        self.walker_count = int(self.get_parameter("walker_count").value)
        self.static_obstacle_count = int(self.get_parameter("static_obstacle_count").value)
        self.dynamic_crossing_enabled = bool(self.get_parameter("dynamic_crossing_enabled").value)

        self.carla = load_carla(self.carla_root)
        self.client = self.carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.bp_lib = self.world.get_blueprint_library()
        self.created_actors = []

        self.ego = self.wait_for_ego()
        self.tm = self.client.get_trafficmanager(self.traffic_manager_port)
        self.tm.set_global_distance_to_leading_vehicle(3.0)

        self.status_pub = self.create_publisher(String, "/adas/teknofest/scenario_status", 10)

        self.spawn_all()
        self.timer = self.create_timer(1.0, self.publish_status)

        self.get_logger().info("TEKNOFEST scenario node hazır.")

    def wait_for_ego(self):
        for _ in range(100):
            for vehicle in self.world.get_actors().filter("vehicle.*"):
                if vehicle.attributes.get("role_name", "") == self.ego_role_name:
                    return vehicle
            time.sleep(0.2)
        raise RuntimeError("Ego vehicle bulunamadı. Önce carla_world_manager_node çalışmalı.")

    def add_actor(self, actor):
        if actor is not None:
            self.created_actors.append(actor)
        return actor

    def get_route_waypoints_ahead(self, count=120, step_m=4.0):
        ego_wp = self.map.get_waypoint(
            self.ego.get_location(),
            project_to_road=True,
            lane_type=self.carla.LaneType.Driving,
        )

        if ego_wp is None:
            return []

        waypoints = [ego_wp]
        current = ego_wp

        for _ in range(count - 1):
            nxt = current.next(step_m)
            if not nxt:
                break
            current = random.choice(nxt)
            waypoints.append(current)

        return waypoints

    def spawn_static_obstacles(self):
        obstacle_blueprints = []

        for pattern in [
            "static.prop.trafficcone*",
            "static.prop.streetbarrier*",
            "static.prop.warningconstruction*",
            "static.prop.constructioncone*",
        ]:
            obstacle_blueprints.extend(list(self.bp_lib.filter(pattern)))

        if not obstacle_blueprints:
            self.get_logger().warning("Statik engel blueprint bulunamadı.")
            return

        waypoints = self.get_route_waypoints_ahead(count=80, step_m=5.0)
        usable = waypoints[8:60:10]

        spawned = 0
        for wp in usable:
            if spawned >= self.static_obstacle_count:
                break

            bp = random.choice(obstacle_blueprints)

            # Şeridin kenarına yakın ama yol içinde engel oluştur.
            right_vec = wp.transform.get_right_vector()
            loc = wp.transform.location + self.carla.Location(
                x=right_vec.x * random.choice([-0.7, 0.7]),
                y=right_vec.y * random.choice([-0.7, 0.7]),
                z=0.15,
            )

            transform = self.carla.Transform(
                loc,
                self.carla.Rotation(
                    pitch=0.0,
                    yaw=wp.transform.rotation.yaw + random.uniform(-15.0, 15.0),
                    roll=0.0,
                ),
            )

            actor = self.world.try_spawn_actor(bp, transform)
            if actor is not None:
                self.add_actor(actor)
                spawned += 1

        self.get_logger().info(f"Statik engel spawn edildi: {spawned}")

    def spawn_npc_vehicles(self):
        vehicle_bps = [
            bp for bp in self.bp_lib.filter("vehicle.*")
            if bp.has_attribute("number_of_wheels")
            and int(bp.get_attribute("number_of_wheels")) == 4
        ]

        spawn_points = list(self.map.get_spawn_points())
        random.shuffle(spawn_points)

        spawned = 0
        ego_loc = self.ego.get_location()

        for sp in spawn_points:
            if spawned >= self.npc_vehicle_count:
                break

            if sp.location.distance(ego_loc) < 25.0:
                continue

            bp = random.choice(vehicle_bps)
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "teknofest_npc_vehicle")

            actor = self.world.try_spawn_actor(bp, sp)
            if actor is None:
                continue

            actor.set_autopilot(True, self.traffic_manager_port)
            self.add_actor(actor)
            spawned += 1

        self.get_logger().info(f"NPC araç spawn edildi: {spawned}")

    def spawn_walkers(self):
        walker_bps = list(self.bp_lib.filter("walker.pedestrian.*"))
        controller_bp = self.bp_lib.find("controller.ai.walker")

        spawned = 0
        ego_loc = self.ego.get_location()

        for _ in range(self.walker_count * 5):
            if spawned >= self.walker_count:
                break

            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                continue

            if loc.distance(ego_loc) < 15.0:
                continue

            walker_bp = random.choice(walker_bps)
            walker = self.world.try_spawn_actor(walker_bp, self.carla.Transform(loc))

            if walker is None:
                continue

            controller = self.world.try_spawn_actor(
                controller_bp,
                self.carla.Transform(),
                walker,
            )

            if controller is not None:
                controller.start()
                target = self.world.get_random_location_from_navigation()
                if target is not None:
                    controller.go_to_location(target)
                controller.set_max_speed(random.uniform(0.7, 1.4))
                self.add_actor(controller)

            self.add_actor(walker)
            spawned += 1

        self.get_logger().info(f"Yaya spawn edildi: {spawned}")

    def spawn_dynamic_crossing_obstacle(self):
        if not self.dynamic_crossing_enabled:
            return

        walker_bps = list(self.bp_lib.filter("walker.pedestrian.*"))
        controller_bp = self.bp_lib.find("controller.ai.walker")
        waypoints = self.get_route_waypoints_ahead(count=50, step_m=4.0)

        if len(waypoints) < 15:
            return

        target_wp = waypoints[14]
        right_vec = target_wp.transform.get_right_vector()

        start_loc = target_wp.transform.location + self.carla.Location(
            x=right_vec.x * 5.0,
            y=right_vec.y * 5.0,
            z=0.2,
        )

        end_loc = target_wp.transform.location + self.carla.Location(
            x=-right_vec.x * 5.0,
            y=-right_vec.y * 5.0,
            z=0.2,
        )

        walker = self.world.try_spawn_actor(
            random.choice(walker_bps),
            self.carla.Transform(start_loc),
        )

        if walker is None:
            return

        controller = self.world.try_spawn_actor(
            controller_bp,
            self.carla.Transform(),
            walker,
        )

        if controller is not None:
            controller.start()
            controller.go_to_location(end_loc)
            controller.set_max_speed(1.1)
            self.add_actor(controller)

        self.add_actor(walker)
        self.get_logger().info("Dinamik geçiş engeli/yaya spawn edildi.")

    def spawn_sign_markers(self):
        """
        CARLA'da özel Türk trafik tabelası assetleri yoksa bile video ve logda görünür olacak
        temsilî tabela/işaret noktaları oluşturur. Asıl karar mekanizması perception_node +
        decision_node üzerinden çalışmaya devam eder.
        """
        prop_patterns = [
            "static.prop.trafficwarning",
            "static.prop.trafficcone*",
            "static.prop.warningconstruction*",
        ]

        props = []
        for p in prop_patterns:
            props.extend(list(self.bp_lib.filter(p)))

        if not props:
            return

        waypoints = self.get_route_waypoints_ahead(count=70, step_m=5.0)
        selected = waypoints[10:50:15]

        for i, wp in enumerate(selected):
            right_vec = wp.transform.get_right_vector()
            loc = wp.transform.location + self.carla.Location(
                x=right_vec.x * 3.0,
                y=right_vec.y * 3.0,
                z=0.4,
            )
            transform = self.carla.Transform(
                loc,
                self.carla.Rotation(yaw=wp.transform.rotation.yaw - 90.0),
            )

            actor = self.world.try_spawn_actor(random.choice(props), transform)
            if actor is not None:
                self.add_actor(actor)

        self.get_logger().info(f"Tabela/işaret marker spawn edildi: {len(selected)}")

    def spawn_all(self):
        self.spawn_static_obstacles()
        self.spawn_npc_vehicles()
        self.spawn_walkers()
        self.spawn_dynamic_crossing_obstacle()
        self.spawn_sign_markers()

    def publish_status(self):
        payload = {
            "stamp": time.time(),
            "scenario_round": self.scenario_round,
            "created_actor_count": len(self.created_actors),
            "static_or_prop_count": len([
                a for a in self.created_actors
                if a.type_id.startswith("static.") or a.type_id.startswith("traffic.")
            ]),
            "vehicle_count": len([
                a for a in self.created_actors
                if a.type_id.startswith("vehicle.")
            ]),
            "walker_count": len([
                a for a in self.created_actors
                if a.type_id.startswith("walker.")
            ]),
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)

    def destroy_node(self):
        if self.destroy_on_shutdown:
            for actor in reversed(getattr(self, "created_actors", [])):
                try:
                    if actor.is_alive:
                        actor.destroy()
                except Exception:
                    pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TeknofestScenarioNode()

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
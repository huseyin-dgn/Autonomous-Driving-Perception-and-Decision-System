import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class ControlNode(Node):
    def __init__(self):
        super().__init__("control_node")

        self.declare_parameter("decision_topic", "/adas/decision")
        self.declare_parameter("cmd_topic", "/cmd_vel")

        self.decision_topic = self.get_parameter("decision_topic").value
        self.cmd_topic = self.get_parameter("cmd_topic").value

        self.publisher = self.create_publisher(Twist, self.cmd_topic, 10)

        self.subscription = self.create_subscription(
            String,
            self.decision_topic,
            self.decision_callback,
            10,
        )

        self.get_logger().info(f"control_node başladı: {self.decision_topic} -> {self.cmd_topic}")

    def decision_callback(self, msg: String) -> None:
        cmd = Twist()

        try:
            data = json.loads(msg.data)
            decision = str(data.get("decision", "STOP")).upper()
            risk = data.get("risk", "UNKNOWN")
            distance_est = data.get("distance_est", None)
        except Exception:
            decision = msg.data.strip().upper()
            risk = "UNKNOWN"
            distance_est = None

        if decision == "STOP":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        elif decision == "SLOW":
            cmd.linear.x = 0.8
            cmd.angular.z = 0.0
        elif decision == "GO":
            cmd.linear.x = 3.0
            cmd.angular.z = 0.0
        elif decision == "LEFT":
            cmd.linear.x = 0.8
            cmd.angular.z = 0.5
        elif decision == "RIGHT":
            cmd.linear.x = 0.8
            cmd.angular.z = -0.5
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        self.publisher.publish(cmd)

        self.get_logger().info(
            f"[CONTROL] decision={decision} risk={risk} distance_est={distance_est} linear={cmd.linear.x}",
            throttle_duration_sec=0.5,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControlNode()
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
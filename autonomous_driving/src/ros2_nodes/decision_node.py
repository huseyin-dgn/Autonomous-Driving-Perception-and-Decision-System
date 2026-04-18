import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class DecisionNode(Node):
    def __init__(self):
        super().__init__("decision_node")

        self.subscription = self.create_subscription(
            String,
            "/driving_decision",
            self.decision_callback,
            10,
        )
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        self.get_logger().info("decision_node started. Listening /driving_decision")

    def decision_callback(self, msg: String) -> None:
        cmd = Twist()

        decision = msg.data.strip().upper()

        if decision == "STOP":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        elif decision == "SLOW":
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0
        elif decision == "GO":
            cmd.linear.x = 1.0
            cmd.angular.z = 0.0
        elif decision == "LEFT":
            cmd.linear.x = 0.4
            cmd.angular.z = 0.5
        elif decision == "RIGHT":
            cmd.linear.x = 0.4
            cmd.angular.z = -0.5
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        self.publisher.publish(cmd)
        self.get_logger().info(f"Applied decision: {decision}", throttle_duration_sec=0.5)


def main(args=None):
    rclpy.init(args=args)
    node = DecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class ControlNode(Node):
    def __init__(self):
        super().__init__("control_node")
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        self.timer = self.create_timer(0.2, self.publish_stop)
        self.get_logger().info("control_node started. Publishing STOP to /cmd_vel")

    def publish_stop(self) -> None:
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.publisher.publish(msg)


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
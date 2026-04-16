#!/usr/bin/env python3
import math
import time
import argparse

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class CircleTraj(Node):
    def __init__(self):
        super().__init__("circle_traj")
        self.pub = self.create_publisher(Float64MultiArray, "/target_ee_pose", 10)

    def publish_pose(self, x, z, theta):
        msg = Float64MultiArray()
        msg.data = [float(x), float(z), float(theta)]
        self.pub.publish(msg)


def main():
    ap = argparse.ArgumentParser(description="Publish a circular (x,z) trajectory to /target_ee_pose.")
    ap.add_argument("--cx", type=float, default=0.70, help="circle center x (m)")
    ap.add_argument("--cz", type=float, default=0.00, help="circle center z (m)")
    ap.add_argument("--r", type=float, default=0.10, help="radius (m)")
    ap.add_argument("--theta", type=float, default=0.0, help="constant end-effector theta (rad)")
    ap.add_argument("--hz", type=float, default=20.0, help="publish rate (Hz)")
    ap.add_argument("--laps", type=float, default=1.0, help="number of circles to do")
    ap.add_argument("--seconds_per_lap", type=float, default=10.0, help="duration of one circle (s)")
    ap.add_argument("--ccw", action="store_true", help="counter-clockwise (default clockwise)")
    args = ap.parse_args()

    rclpy.init()
    node = CircleTraj()

    # Small warmup so publishers connect
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.05)

    dt = 1.0 / args.hz
    direction = 1.0 if args.ccw else -1.0
    omega = direction * (2.0 * math.pi / args.seconds_per_lap)

    t0 = time.time()
    t_end = t0 + args.laps * args.seconds_per_lap

    node.get_logger().info(
        f"Publishing circle: center=({args.cx},{args.cz}) r={args.r} theta={args.theta} "
        f"rate={args.hz}Hz laps={args.laps} seconds_per_lap={args.seconds_per_lap} "
        f"{'CCW' if args.ccw else 'CW'}"
    )

    try:
        while rclpy.ok() and time.time() < t_end:
            t = time.time() - t0
            a = omega * t
            x = args.cx + args.r * math.cos(a)
            z = args.cz + args.r * math.sin(a)
            node.publish_pose(x, z, args.theta)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_pose(args.cx + args.r, args.cz, args.theta)  # end nicely
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
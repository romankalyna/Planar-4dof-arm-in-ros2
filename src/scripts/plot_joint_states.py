#!/usr/bin/env python3
import csv
import argparse

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStateLogger(Node):
    def __init__(self, outfile: str, hz: float):
        super().__init__("joint_state_logger")
        self.outfile = outfile
        self.period = 1.0 / float(hz)

        self.t0 = self.get_clock().now()
        self.last_log_t = None
        self.rows_since_flush = 0
        self.flush_every = 50  # flush every N rows, not every row

        self.f = open(self.outfile, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(
            ["t"]
            + [f"q{i+1}_rad" for i in range(4)]
            + [f"dq{i+1}_rad_s" for i in range(4)]
        )

        self.sub = self.create_subscription(JointState, "/joint_states", self.on_js, 10)
        self.get_logger().info(f"Logging /joint_states to {self.outfile} at {hz} Hz")

    def on_js(self, msg: JointState):
        if len(msg.position) < 4 or len(msg.velocity) < 4:
            return

        now = self.get_clock().now()
        t = (now - self.t0).nanoseconds * 1e-9

        if self.last_log_t is not None and (t - self.last_log_t) < self.period:
            return  # throttle

        self.last_log_t = t
        self.w.writerow([t] + list(msg.position[:4]) + list(msg.velocity[:4]))

        self.rows_since_flush += 1
        if self.rows_since_flush >= self.flush_every:
            self.f.flush()
            self.rows_since_flush = 0

    def close(self):
        try:
            self.f.flush()
            self.f.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="joint_states_log.csv", help="output CSV filename")
    ap.add_argument("--hz", type=float, default=50.0, help="logging rate (Hz). 20-100 recommended")
    args = ap.parse_args()

    rclpy.init()
    node = JointStateLogger(args.out, args.hz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
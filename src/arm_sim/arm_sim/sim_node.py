import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

from arm_dynamics.integrate import rk4_step
from arm_dynamics.dynamics import ddq_rigid


class ArmSimNode(Node):
    def __init__(self):
        super().__init__("arm_sim_node")

        self.get_logger().info("ArmSim using ddq_rigid (M + (optional C) + g + D)")

        # Simulation timestep (seconds). 0.002 -> 500 updates per second.
        self.dt = 0.002

        self.joint_names = ["joint1", "joint2", "joint3", "joint4"]

        # Initial joint angles (degrees -> radians).
        q0_deg = np.array([45.0, -45.0, 45.0, -45.0])
        q0 = np.deg2rad(q0_deg)
        self.get_logger().warn(f"DEBUG q0 = {q0.tolist()}")

        # Initial joint velocities (rad/s).
        dq0 = np.zeros(4)

        # State vector: x = [q(4), dq(4)]
        self.x = np.hstack([q0, dq0])

        # Latest torque command (Nm).
        self.tau = np.zeros(4)

        # Torque timeout (deadman switch)
        self.tau_timeout_s = 0.2  # seconds
        self.last_tau_time = self.get_clock().now()

        # -----------------------------
        # NEW: startup freeze
        # -----------------------------
        # Hold the exact initial condition for a short time so that:
        # - the first /joint_states messages reflect the intended q0
        # - loggers and RViz see the true initial condition
        self.start_time = self.get_clock().now()
        self.startup_freeze_s = 0.5  # seconds

        self.sub = self.create_subscription(
            Float64MultiArray, "/joint_torque_cmd", self.on_tau, 10
        )
        self.pub = self.create_publisher(JointState, "/joint_states", 10)

        self.timer = self.create_timer(self.dt, self.step)

    def on_tau(self, msg: Float64MultiArray):
        arr = np.array(msg.data, dtype=float)
        if arr.shape[0] == 4:
            self.tau = arr
            self.last_tau_time = self.get_clock().now()

    def _publish_joint_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.x[:4].tolist()
        msg.velocity = self.x[4:].tolist()
        self.pub.publish(msg)

    def step(self):
        # NEW: freeze at initial condition for startup_freeze_s seconds
        age0 = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        if age0 < self.startup_freeze_s:
            self.tau[:] = 0.0
            self._publish_joint_state()
            return

        # 1) If we haven't received torque commands recently, set torque to zero.
        age_ns = (self.get_clock().now() - self.last_tau_time).nanoseconds
        if age_ns > int(self.tau_timeout_s * 1e9):
            self.tau[:] = 0.0

        # 2) Integrate forward by dt using RK4
        self.x = rk4_step(self.x, self.tau, self.dt, ddq_rigid)

        # 3) Wrap angles to [-pi, pi]
        self.x[:4] = (self.x[:4] + np.pi) % (2 * np.pi) - np.pi

        # 4) Publish JointState
        self._publish_joint_state()


def main(args=None):
    rclpy.init(args=args)
    node = ArmSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
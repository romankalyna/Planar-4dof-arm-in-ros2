import csv
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# ----------------------------
# NEW: gravity compensation imports
# ----------------------------
# Note to myself:
# Use the SAME dynamics model/params as the simulator, so gravity feedforward matches the plant.
from arm_dynamics.dynamics import gravity_vector
from arm_dynamics.params import default_params


def wrap_to_pi(a):
    """
    Note to myself:
    Angles in the sim are wrapped to [-pi, pi].
    If I do (qd - q) directly, I can get a huge error near the boundary:
      example: qd=+3.13, q=-3.13  -> naive error = 6.26 rad (WRONG)
    This function converts any angle to the equivalent angle in [-pi, pi],
    so the error becomes the "shortest way around".
    """
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class ArmControlNode(Node):
    def __init__(self):
        super().__init__("arm_control_node")

        # How often I run the controller (500 Hz)
        self.dt = 0.002

        # Torque safety limits (Nm). I clamp output torque to these ranges.
        self.tau_max = np.array([60.0, 40.0, 20.0, 10.0], dtype=float)

        # ------------------------------------------------------------
        # IMPORTANT STARTUP BEHAVIOR CHANGE:
        # ------------------------------------------------------------
        # Note to myself:
        # Start holding the current pose; don't jump on startup.
        self.qd = None
        self.qd_cmd = None

        # Gains
        self.Kp = np.diag([30.0, 25.0, 18.0, 10.0])
        self.Kd = np.diag([18.0, 15.0, 12.0, 9.0])
        self.Ki = np.diag([0.0, 0.0, 0.0, 0.1])
        #self.Kp = np.diag([0, 0, 0, 0])
        #self.Kd = np.diag([0, 0, 0, 0])
        #self.Ki = np.diag([0, 0, 0, 0])


        # Integral state for the I term
        self.ei = np.zeros(4, dtype=float)
        self.ei_limit = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

        # These help stop wobble from the integral near the target
        self.i_zone = np.deg2rad(3.0)   # only integrate when I'm close to the target
        self.int_leak = 0.2             # slowly decay the integral over time

        # Measured state from /joint_states
        self.q = None
        self.dq = None

        # Velocity filter to reduce noise in the D term
        self.dq_f = np.zeros(4, dtype=float)
        self.dq_filter_alpha = 0.2

        # Target smoothing (rate limiting)
        self.qd_rate_limit = np.deg2rad([60.0, 60.0, 60.0, 60.0])  # rad/s

        # ----------------------------
        # NEW: store dynamics parameters once
        # ----------------------------
        # Note to myself:
        # gravity_vector(q, p) needs the same params as the sim (g0=-9.81, masses, lengths, etc).
        self.p = default_params()

        # ROS pub/sub
        self.pub = self.create_publisher(Float64MultiArray, "/joint_torque_cmd", 10)
        self.sub = self.create_subscription(JointState, "/joint_states", self.on_js, 10)
        self.target_sub = self.create_subscription(Float64MultiArray, "/target_position", self.on_target, 10)

        # Controller timer
        self.timer = self.create_timer(self.dt, self.loop)

        # CSV log (time, q, dq, tau)
        self.t0 = self.get_clock().now()
        self.f = open("arm_log.csv", "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(
            ["t"]
            + [f"q{i+1}_rad" for i in range(4)]
            + [f"dq{i+1}_rad_s" for i in range(4)]
            + [f"tau{i+1}_Nm" for i in range(4)]
        )

    def on_js(self, msg: JointState):
        """
        Note to myself:
        Update measured state (q, dq).
        On the first message, latch target to current pose to avoid startup jump.
        """
        if len(msg.position) >= 4 and len(msg.velocity) >= 4:
            self.q = np.array(msg.position[:4], dtype=float)
            self.dq = np.array(msg.velocity[:4], dtype=float)

            if self.qd is None:
                self.qd = self.q.copy()
                self.qd_cmd = self.q.copy()
                self.ei[:] = 0.0
                self.get_logger().info("Controller startup: holding current joint position (no motion).")

    def on_target(self, msg: Float64MultiArray):
        """
        Note to myself:
        IK gives a new joint target in radians.
        I store it in qd_cmd; the loop rate-limits qd toward qd_cmd.
        """
        if len(msg.data) == 4:
            # ----------------------------
            # NEW: wrap target angles
            # ----------------------------
            # Note to myself:
            # Keep qd_cmd in [-pi, pi] so it's consistent with wrapped sim joints.
            new_cmd = wrap_to_pi(np.array(msg.data, dtype=float))

            if self.qd is None:
                self.qd = new_cmd.copy()

            self.qd_cmd = new_cmd
            self.get_logger().info(f"New IK target received (rad): {self.qd_cmd}")

    def loop(self):
        if self.q is None or self.dq is None or self.qd is None or self.qd_cmd is None:
            return

        # 1) Smooth the target (rate limit)
        dq_allowed = self.qd_rate_limit * self.dt
        step = np.clip(self.qd_cmd - self.qd, -dq_allowed, dq_allowed)
        self.qd = self.qd + step

        # 2) Position error (shortest-angle)
        e = wrap_to_pi(self.qd - self.q)

        # Filter velocity
        self.dq_f = (1.0 - self.dq_filter_alpha) * self.dq_f + self.dq_filter_alpha * self.dq

        # Desired velocity is zero
        de = -self.dq_f

        # Integral leak
        self.ei *= (1.0 - self.int_leak * self.dt)

        # I-zone integration
        in_i_zone = np.abs(e) < self.i_zone
        self.ei[in_i_zone] += e[in_i_zone] * self.dt
        self.ei = np.clip(self.ei, -self.ei_limit, self.ei_limit)

        # ----------------------------
        # NEW: gravity compensation feedforward
        # ----------------------------
        # Note to myself:
        # Sim dynamics: M ddq + c + g + D dq = tau
        # So commanding tau = tau_pid + g(q) cancels gravity.
        tau_pid = (self.Kp @ e) + (self.Kd @ de) + (self.Ki @ self.ei)
        tau_g = gravity_vector(self.q, self.p)

        tau_unsat = tau_pid + tau_g

        # Clamp torque
        tau = np.clip(tau_unsat, -self.tau_max, self.tau_max)

        # Anti-windup
        saturated = np.abs(tau - tau_unsat) > 1e-9
        self.ei[saturated] *= 0.9

        # Publish torque
        out = Float64MultiArray()
        out.data = tau.tolist()
        self.pub.publish(out)

        # Log
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        self.w.writerow([t] + self.q.tolist() + self.dq.tolist() + tau.tolist())
        self.f.flush()


def main(args=None):
    rclpy.init(args=args)
    node = ArmControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.f.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
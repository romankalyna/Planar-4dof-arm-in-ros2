import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from rclpy.duration import Duration

from tf2_ros import Buffer, TransformListener


def wrap_to_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def pitch_from_quaternion(qx, qy, qz, qw):
    """
    Extract pitch (rotation about Y) from quaternion (x,y,z,w).
    ROS RPY convention: roll=X, pitch=Y, yaw=Z.
    pitch = asin(2*(w*y - z*x))
    """
    t2 = 2.0 * (qw * qy - qz * qx)
    t2 = np.clip(t2, -1.0, 1.0)
    return float(np.arcsin(t2))


class IKNode(Node):
    def __init__(self):
        super().__init__("ik_node")

        # Publishers
        self.actual_ee_pub = self.create_publisher(Float64MultiArray, "/actual_ee_pose", 10)
        self.target_q_pub = self.create_publisher(Float64MultiArray, "/target_position", 10)

        # Link lengths (must match sim/URDF)
        self.l = np.array([0.35, 0.30, 0.25, 0.20], dtype=float)

        # Current joints
        self.current_q = np.zeros(4, dtype=float)

        # Subscribers
        self.js_sub = self.create_subscription(JointState, "/joint_states", self.on_js, 10)
        self.target_ee_sub = self.create_subscription(
            Float64MultiArray, "/target_ee_pose", self.on_target_ee, 10
        )

        # IK params
        self.max_iters = 200
        self.tolerance = 1e-4
        self.damping = 0.05
        self.max_step = 0.2
        self.alpha = 0.5

        # Publish-for-2-seconds behavior
        self.last_target_q = None
        self.publish_target_until = None
        self.target_publish_period = 0.05  # 20 Hz
        self.target_timer = self.create_timer(self.target_publish_period, self.publish_target_loop)

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.base_frame = "base_link"
        self.tip_frame = "ee_link"  # IMPORTANT: use tip frame, not link4

        # Make TF lookup robust
        self.tf_timeout = Duration(seconds=1.0)

        # Reduce log spam: only warn once per second if TF missing
        self._last_tf_warn_time = None

        self.get_logger().info(f"IK Node started! TF actual pose: {self.base_frame} -> {self.tip_frame}")
        self.get_logger().info("Send [x, z, theta] to /target_ee_pose")

    def publish_target_loop(self):
        if self.last_target_q is None or self.publish_target_until is None:
            return
        if self.get_clock().now() > self.publish_target_until:
            return

        out_msg = Float64MultiArray()
        out_msg.data = self.last_target_q.tolist()
        self.target_q_pub.publish(out_msg)

    def on_js(self, msg: JointState):
        if len(msg.position) < 4:
            return

        self.current_q = np.array(msg.position[:4], dtype=float)

        # Publish actual pose from TF so it matches RViz/TF
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tip_frame,
                rclpy.time.Time(),  # latest
                timeout=self.tf_timeout,
            )

            x = tf.transform.translation.x
            z = tf.transform.translation.z

            q = tf.transform.rotation
            pitch = pitch_from_quaternion(q.x, q.y, q.z, q.w)
            theta = wrap_to_pi(pitch)

            out_msg = Float64MultiArray()
            out_msg.data = [float(x), float(z), float(theta)]
            self.actual_ee_pub.publish(out_msg)

        except Exception as ex:
            # If TF is not available, do FK fallback, but ALSO WARN so you know it's not TF.
            now = self.get_clock().now()
            if self._last_tf_warn_time is None or (now - self._last_tf_warn_time) > Duration(seconds=1.0):
                self.get_logger().warn(f"TF lookup failed ({self.base_frame}->{self.tip_frame}); using FK fallback. Error: {ex}")
                self._last_tf_warn_time = now

            current_ee = self.forward_kinematics(self.current_q)
            out_msg = Float64MultiArray()
            out_msg.data = current_ee.tolist()
            self.actual_ee_pub.publish(out_msg)

    def forward_kinematics(self, q):
        q = np.asarray(q, dtype=float)
        th = np.cumsum(q)
        x = np.sum(self.l * np.cos(th))
        z = np.sum(self.l * np.sin(th))
        theta = th[-1]
        return np.array([x, z, theta], dtype=float)

    def jacobian(self, q):
        q = np.asarray(q, dtype=float)
        th = np.cumsum(q)

        J = np.zeros((3, 4), dtype=float)
        for k in range(4):
            J[0, k] = -np.sum(self.l[k:] * np.sin(th[k:]))
            J[1, k] = np.sum(self.l[k:] * np.cos(th[k:]))
            J[2, k] = 1.0
        return J

    def damped_ls_delta_q(self, J, error):
        lam = self.damping
        JJt = J @ J.T
        A = JJt + (lam * lam) * np.eye(JJt.shape[0])
        return J.T @ np.linalg.solve(A, error)

    def on_target_ee(self, msg: Float64MultiArray):
        if len(msg.data) != 3:
            self.get_logger().error("Please send exactly 3 values: [x, z, theta]")
            return

        # User command convention: TF/base_link (+Z is up in RViz)
        target_pose = np.array(msg.data, dtype=float)

        # Map TF z to IK-model z (your model uses opposite sign)
        target_pose[1] *= -1.0

        self.get_logger().info(
            f"Target user (TF): {np.array(msg.data, dtype=float).tolist()} -> Target used by IK model: {target_pose.tolist()}"
        )

        q = self.current_q.copy()

        for i in range(self.max_iters):
            current_pose = self.forward_kinematics(q)
            error = target_pose - current_pose
            error[2] = wrap_to_pi(error[2])

            if np.linalg.norm(error) < self.tolerance:
                q_out = wrap_to_pi(q)
                fk_check = self.forward_kinematics(q_out)
                self.get_logger().info(f"NOTE: FK(q_out) = {fk_check}, target = {target_pose}")

                self.last_target_q = q_out
                self.publish_target_until = self.get_clock().now() + Duration(seconds=2.0)
                self.get_logger().info(f"IK solved in {i} steps! Publishing /target_position for 2 seconds.")
                return

            J = self.jacobian(q)
            delta_q = self.damped_ls_delta_q(J, error)
            delta_q = np.clip(delta_q, -self.max_step, self.max_step)
            q = wrap_to_pi(q + self.alpha * delta_q)

        self.get_logger().warning("IK Solver failed to converge! Target might be out of reach.")


def main(args=None):
    rclpy.init(args=args)
    node = IKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
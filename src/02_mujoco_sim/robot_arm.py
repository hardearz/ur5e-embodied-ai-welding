"""
Industrial UR5e Manipulator Workcell Simulation in MuJoCo.

Features:
  1. DeepMind Official UR5e Robotic Arm model with mesh assets and dynamic parameters.
  2. Industrial workcell with workstation table and seam-welding workpiece.
  3. Real-time Closed-Loop Differential Inverse Kinematics (DLS-IK) with joint limit safety.
  4. Multi-Camera Sensor Rig:
     - Third-person Workspace Overview Camera (workspace_cam)
     - Eye-in-Hand Wrist Camera (wrist_cam) mounted directly above the tool tip.
  5. Continuous trajectory execution and video dashboard recorder.

Key Robotics Concepts Demonstrated:
  - Coordinate transformations & forward kinematics.
  - Jacobian Pseudo-Inverse with Damped Least Squares (Levenberg-Marquardt).
  - Closed-loop Cartesian velocity control to eliminate steady-state dynamic drift.
  - Multi-camera perception pipelines essential for Visuomotor Policies (Diffusion Policy / ACT).
"""

import os
import mujoco
import numpy as np
import cv2
from typing import Tuple, List, Optional


class UR5eIndustrialWorkcell:
    def __init__(self, model_path: str = "models/ur5e/workcell.xml"):
        if not os.path.isabs(model_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, model_path)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        # Pre-allocated scratch MjData for fast, allocation-free IK & FK queries
        self._ik_data = mujoco.MjData(self.model)

        # Cache IDs
        self.tcp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name.replace("_joint", ""))
                             for name in self.joint_names]

        # Camera renderers (360p each -> combined 960x360 dashboard)
        self.renderers = {
            "workspace_cam": mujoco.Renderer(self.model, height=360, width=480),
            "wrist_cam": mujoco.Renderer(self.model, height=360, width=480),
        }

        # Nominal forward-facing home posture (over workstation table)
        # shoulder_pan ~ 180 deg (facing forward), shoulder_lift ~ -69 deg, elbow ~ 92 deg, wrist_1 ~ -113 deg, wrist_2 ~ -90 deg
        self.home_qpos = np.array([np.pi, -1.2, 1.6, -1.97, -1.57, 0.0])
        self.reset()

    def reset(self):
        """Resets robot to nominal forward-facing home configuration."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = self.home_qpos.copy()
        self.data.ctrl[:6] = self.home_qpos.copy()
        mujoco.mj_forward(self.model, self.data)

    def get_tcp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the Tool-Center-Point position (3,) and 3x3 rotation matrix."""
        pos = self.data.site_xpos[self.tcp_id].copy()
        R = self.data.site_xmat[self.tcp_id].reshape(3, 3).copy()
        return pos, R

    def compute_jacobian(self, data_obj: Optional[mujoco.MjData] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the 3xnv linear (translational) and 3xnv angular (rotational) Jacobians
        for the Tool Center Point (TCP).
        """
        d = self.data if data_obj is None else data_obj
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, d, jacp, jacr, self.tcp_id)
        return jacp, jacr

    def solve_ik(
        self,
        target_pos: np.ndarray,
        q_seed: Optional[np.ndarray] = None,
        max_iters: int = 25,
        tol: float = 0.0005,
        damping: float = 0.015,
    ) -> np.ndarray:
        """
        Solves Inverse Kinematics using Damped Least Squares (DLS) with joint limit clamping.
        q_seed: starting configuration for the local numerical search.
        Uses pre-allocated self._ik_data to avoid repeated C heap allocations in real-time loops.
        """
        temp_data = self._ik_data
        temp_data.qpos[:6] = self.data.qpos[:6] if q_seed is None else q_seed

        for _ in range(max_iters):
            mujoco.mj_forward(self.model, temp_data)
            curr = temp_data.site_xpos[self.tcp_id]
            err = target_pos - curr
            if np.linalg.norm(err) < tol:
                break

            jacp, _ = self.compute_jacobian(temp_data)
            J = jacp[:, :6]
            lambda_sq = damping ** 2
            JJT = J @ J.T + lambda_sq * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, err)

            temp_data.qpos[:6] += np.clip(dq, -0.15, 0.15)
            # Clip within physical joint limits
            for j in range(6):
                temp_data.qpos[j] = np.clip(
                    temp_data.qpos[j],
                    self.model.jnt_range[j, 0],
                    self.model.jnt_range[j, 1]
                )

        return temp_data.qpos[:6].copy()

    def step_control(self, target_pos: np.ndarray, feedback_gain: float = 0.8):
        """
        Executes one closed-loop control step:
          1. Solves kinematically accurate joint target via DLS-IK.
          2. Adds Cartesian feedback compensation for dynamic compliance / gravity droop.
          3. Applies position command and advances physics.
        """
        # Desired joint angles from kinematics
        q_des = self.solve_ik(target_pos, q_seed=self.data.qpos[:6])

        # Closed-loop Cartesian feedback term: J_pinv @ (target - actual)
        curr_pos, _ = self.get_tcp_pose()
        cart_err = target_pos - curr_pos
        if np.linalg.norm(cart_err) > 0.0005:
            jacp, _ = self.compute_jacobian()
            J = jacp[:, :6]
            JJT = J @ J.T + 0.02**2 * np.eye(3)
            q_corr = J.T @ np.linalg.solve(JJT, cart_err)
            q_cmd = q_des + feedback_gain * q_corr
        else:
            q_cmd = q_des

        self.data.ctrl[:6] = q_cmd
        mujoco.mj_step(self.model, self.data)

    def render_frame(self, camera_name: str) -> np.ndarray:
        """Renders an RGB frame from the requested camera."""
        renderer = self.renderers[camera_name]
        renderer.update_scene(self.data, camera=camera_name)
        return renderer.render()

    def render_dashboard(self) -> np.ndarray:
        """
        Renders a dual-camera dashboard frame (Workspace Camera + Eye-in-Hand Wrist Camera).
        Annotates with current TCP position and telemetry.
        """
        img_workspace = self.render_frame("workspace_cam")
        img_wrist = self.render_frame("wrist_cam")

        # Convert to BGR for OpenCV annotations
        bgr_workspace = cv2.cvtColor(img_workspace, cv2.COLOR_RGB2BGR)
        bgr_wrist = cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)

        # Add camera labels
        cv2.putText(bgr_workspace, "Workspace Overview (Camera 0)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bgr_wrist, "Eye-in-Hand Wrist Cam (Camera 1)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 220, 255), 2, cv2.LINE_AA)

        # Crosshair in center of wrist camera to show tool alignment
        h, w = bgr_wrist.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.drawMarker(bgr_wrist, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

        # Telemetry overlay on workspace view
        tcp_pos, _ = self.get_tcp_pose()
        telemetry = f"TCP: X={tcp_pos[0]:.3f} Y={tcp_pos[1]:.3f} Z={tcp_pos[2]:.3f}"
        cv2.putText(bgr_workspace, telemetry, (15, 335),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2, cv2.LINE_AA)

        # Combine side-by-side: [Workspace | Wrist]
        composite = np.hstack([bgr_workspace, bgr_wrist])
        return composite


def run_industrial_welding_demo(output_video: str = "outputs/ur5e_welding_demo.mp4"):
    """
    Demonstrates the UR5e executing an industrial welding seam task:
      1. Approach Seam: Safe transit from home down to weld start.
      2. Welding Seam: Continuous linear path along the weld seam line.
      3. Retract Tool: Vertical lift back to inspection height.
    Records the dual-camera dashboard to MP4 video.
    """
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    env = UR5eIndustrialWorkcell()

    print("=" * 70)
    print("UR5e INDUSTRIAL WORKCELL: SEAM WELDING TRAJECTORY")
    print("=" * 70)
    initial_tcp, _ = env.get_tcp_pose()
    print(f"Nominal Home TCP Position: {np.round(initial_tcp, 4)}")

    # Non-intersecting industrial trajectory waypoints (in meters) across full 320 mm seam
    above_start = np.array([0.52, -0.16, 0.530])
    weld_start  = np.array([0.52, -0.16, 0.435])
    weld_end    = np.array([0.52,  0.16, 0.435])
    retract     = np.array([0.50,  0.16, 0.600])

    waypoints = [
        ("Transit to Seam", above_start, 140),
        ("Touchdown",       weld_start,   80),
        ("Welding Seam",    weld_end,    320),
        ("Crater Dwell",    weld_end,     32),
        ("Retract Tool",    retract,      90),
    ]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video, fourcc, 30.0, (960, 360))

    frame_count = 0
    total_steps = sum(steps for _, _, steps in waypoints)
    print(f"Executing {len(waypoints)} trajectory phases over {total_steps} control cycles...")

    curr_target = initial_tcp.copy()

    for phase_name, target_wp, num_steps in waypoints:
        start_wp = curr_target.copy()
        print(f"\n>> Phase: {phase_name} -> Target: {np.round(target_wp, 3)}")

        for step in range(num_steps):
            # Smooth interpolation (Cosine S-curve profile for smooth acceleration)
            t = (step + 1) / num_steps
            alpha = 0.5 * (1 - np.cos(np.pi * t))
            curr_target = (1 - alpha) * start_wp + alpha * target_wp

            # Inner physics steps per control step (2 sub-steps at dt=0.002s -> 4ms per control cycle)
            for _ in range(2):
                env.step_control(curr_target)

            # Record dashboard frame at ~30 Hz (every 4 control cycles)
            if step % 4 == 0:
                frame = env.render_dashboard()
                cv2.putText(frame, f"Phase: {phase_name}", (390, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
                video_writer.write(frame)
                frame_count += 1

        final_tcp, _ = env.get_tcp_pose()
        err_mm = np.linalg.norm(final_tcp - target_wp) * 1000
        print(f"   Phase Complete! Final TCP: {np.round(final_tcp, 3)}, Error: {err_mm:.1f} mm | Contacts: {env.data.ncon}")

    video_writer.release()
    print("\n" + "=" * 70)
    print(f"Video saved successfully to: {output_video}")
    print(f"Total Frames Rendered: {frame_count}")
    print(f"Final Collisions: {env.data.ncon}")
    print("=" * 70)


if __name__ == "__main__":
    run_industrial_welding_demo()

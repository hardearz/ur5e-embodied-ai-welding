"""
EXERCISE 07: Closed-Loop Visuomotor Policy Evaluation (Autonomous Rollout)

In this exercise, you will close the loop!
The robot receives NO waypoints, NO pre-computed trajectory, and NO Inverse Kinematics.
It must autonomously guide itself down to the workpiece, track the seam, and retract,
driven entirely by its Eye-in-Hand Wrist Camera and Proprioception via the Visuomotor Policy.

Run this script with:
    robots/bin/python exercises/07_closed_loop_rollout.py
"""

import os
import sys
import shutil
import subprocess
import importlib.util
import numpy as np
import cv2
import torch
import mujoco

# Dynamic import helper for codebase modules
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _import_file(mod_name, rel_path):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(base_dir, rel_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

robot_arm_mod = _import_file("robot_arm", "src/02_mujoco_sim/robot_arm.py")
UR5eIndustrialWorkcell = robot_arm_mod.UR5eIndustrialWorkcell

ex6_mod = _import_file("ex6", "exercises/06_visuomotor_policy_training.py")
VisuomotorPolicy = ex6_mod.VisuomotorPolicy

IMG_HEIGHT, IMG_WIDTH = 120, 160


# ==============================================================================
# TODO 1: Observation Preprocessing & Policy Inference
# ==============================================================================
def predict_action(
    policy: VisuomotorPolicy,
    wrist_frame: np.ndarray,
    qpos: np.ndarray,
    device: str
) -> np.ndarray:
    """
    Transforms raw sensory observations into the neural policy's input tensors,
    executes an inference forward pass with no gradients, and returns the commanded joint angles.
    
    Args:
        policy: Trained PyTorch VisuomotorPolicy model.
        wrist_frame: Raw RGB image array from wrist camera (H, W, 3) with values in [0, 255].
        qpos: Current joint angles array (6,) from robot encoders.
        device: 'cuda' or 'cpu'.
        
    Steps:
      1. Resize wrist_frame to (IMG_WIDTH, IMG_HEIGHT) = (160, 120).
      
      2. Convert from HWC to CHW format: np.transpose(..., (2, 0, 1)).
      3. Cast to float32 and normalize RGB values to [0.0, 1.0].
      4. Convert image to PyTorch Tensor, add batch dimension -> shape (1, 3, 120, 160), move to device.
      5. Convert qpos to PyTorch float32 Tensor, add batch dimension -> shape (1, 6), move to device.
      6. With torch.no_grad():
           pred_action = policy(image_tensor, qpos_tensor)
      7. Return pred_action as a 1D numpy array of shape (6,).
    """
    # TODO 1: Implement observation preprocessing and policy inference!
    image = cv2.resize(wrist_frame, (IMG_WIDTH, IMG_HEIGHT))
    image = np.transpose(image,(2, 0, 1))
    image = image.astype(np.float32) / 255
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(device)
    qpos_tensor = torch.from_numpy(qpos).float().unsqueeze(0).to(device)
    with torch.no_grad():
        pred_action = policy(image_tensor, qpos_tensor).squeeze(0).cpu().numpy()

    return pred_action


# ==============================================================================
# TODO 2: Autonomous Closed-Loop Rollout Execution Loop
# ==============================================================================
def run_closed_loop_rollout(
    env: UR5eIndustrialWorkcell,
    policy: VisuomotorPolicy,
    max_steps: int = 140,
    device: str = "cpu"
) -> dict:
    """
    Executes an autonomous closed-loop rollout:
      - Starts at env.reset()
      - For each step in max_steps:
          1. Render wrist_cam: env.render_frame("wrist_cam")
          2. Read joint positions: env.data.qpos[:6]
          3. Predict action: action = predict_action(policy, wrist_raw, qpos, device)
          4. Command robot: env.data.ctrl[:6] = action
          5. Advance physics: call mujoco.mj_step(env.model, env.data) twice (matching 2 physics steps per demo step)
          6. Record telemetry: TCP position, dashboard video frame, joint angles
          
    Returns:
        dict with keys:
          - "tcp_positions": list of (3,) arrays of TCP coordinates
          - "dashboard_frames": list of rendered dashboard frames for video/GIF
          - "qpos_history": list of (6,) joint angle arrays
    """
    env.reset()
    policy.eval()

    tcp_positions = []
    dashboard_frames = []
    qpos_history = []

    print(f"Executing autonomous rollout for {max_steps} control cycles...")

    for step in range(max_steps):
        # 1. Capture current observation
        wrist_raw = env.render_frame("wrist_cam")
        qpos = env.data.qpos[:6].copy()

        # 2. Predict next joint action via visuomotor policy
        # TODO 2: Call predict_action(...) to get commanded joint targets
        action = predict_action(policy, wrist_raw, qpos, device)

        # 3. Apply action to robot actuators
        # TODO 2: Set env.data.ctrl[:6] = action
        env.data.ctrl[:6] = action

        # 4. Advance physics simulation by 20 physics substeps (25 Hz control loop, dt = 0.04s)
        for _ in range(20):
            mujoco.mj_step(env.model, env.data)

        # Record telemetry
        tcp_pos, _ = env.get_tcp_pose()
        tcp_positions.append(tcp_pos.copy())
        qpos_history.append(env.data.qpos[:6].copy())

        # Render dashboard frame with live overlay
        dash_frame = env.render_dashboard()
        # Add autonomous mode badge & step counter
        cv2.putText(dash_frame, f"AUTONOMOUS ROLLOUT | STEP {step:03d}/{max_steps}", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        dashboard_frames.append(dash_frame)

    return {
        "tcp_positions": np.array(tcp_positions),
        "dashboard_frames": dashboard_frames,
        "qpos_history": np.array(qpos_history),
    }


# ==============================================================================
# TODO 3: Seam Tracking Accuracy Metric Evaluation
# ==============================================================================
def evaluate_seam_tracking(tcp_positions: np.ndarray) -> dict:
    """
    Analyzes the TCP trajectory during the active welding portion:
      - The nominal seam line is located along X = 0.520 m, Z = 0.435 m, from Y = -0.12 m to +0.12 m.
      - Extract all points where Y is within [-0.10, +0.10] m (the active weld pass).
      - Compute cross-track error: e_cross = sqrt((X - 0.520)^2 + (Z - 0.435)^2).
      
    Returns:
        dict with:
          - "mean_error_mm": Mean cross-track deviation in millimeters.
          - "max_error_mm": Maximum cross-track deviation in millimeters.
          - "weld_points_count": Number of timesteps spent actively in the seam corridor.
    """
    # Filter points where arm is in the active welding region (Y between -0.10 and +0.10)
    # TODO 3: Compute mean and max cross-track error to (X=0.520, Z=0.435) in millimeters
    seam_x = 0.520
    seam_z = 0.435

    mask = (tcp_positions[:, 1] >= -0.155) & (tcp_positions[:, 1] <= 0.155) & (tcp_positions[:, 2] <= 0.455)
    weld_points = tcp_positions[mask]

    if len(weld_points) == 0:
        return {"mean_error_mm": 999.0, "max_error_mm": 999.0, "weld_points_count": 0}

    # Cross-track horizontal/vertical distance to seam
    dx = weld_points[:, 0] - seam_x
    dz = weld_points[:, 2] - seam_z
    errors_mm = np.sqrt(dx**2 + dz**2) * 1000.0

    return {
        "mean_error_mm": float(np.mean(errors_mm)),
        "max_error_mm": float(np.max(errors_mm)),
        "weld_points_count": int(len(weld_points)),
    }


# ==============================================================================
# AUTOMATED VERIFICATION SUITE
# ==============================================================================
if __name__ == "__main__":
    print("Testing Exercise 07: Closed-Loop Visuomotor Policy Evaluation...\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference on: {device}")

    # Load trained policy weights
    weights_path = "outputs/visuomotor_policy.pt"
    assert os.path.exists(weights_path), f"Policy weights not found at {weights_path}! Run Exercise 06 first."
    
    policy = VisuomotorPolicy().to(device)
    policy.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    policy.eval()
    print(f"Loaded trained policy from {weights_path}")

    # Initialize environment
    env = UR5eIndustrialWorkcell()

    # Unit Test TODO 1: predict_action
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_q = env.home_qpos.copy()
    act = predict_action(policy, dummy_frame, dummy_q, device)

    assert act is not None, "TODO 1: predict_action returned None!"
    assert isinstance(act, np.ndarray), f"Expected numpy array from predict_action, got {type(act)}"
    assert act.shape == (6,), f"Expected action shape (6,), got {act.shape}"
    print("PASS: TODO 1 (Observation Preprocessing & Policy Inference)")

    # Execute Autonomous Rollout
    rollout_data = run_closed_loop_rollout(env, policy, max_steps=160, device=device)
    assert rollout_data["qpos_history"].shape[0] == 160, "Rollout did not execute expected steps!"
    print("PASS: TODO 2 (Autonomous Closed-Loop Rollout Execution)")

    # Evaluate Tracking Accuracy
    metrics = evaluate_seam_tracking(rollout_data["tcp_positions"])
    print("\nAutonomous Policy Performance Metrics:")
    print(f"  Active Seam Steps:    {metrics['weld_points_count']}")
    print(f"  Mean Tracking Error:  {metrics['mean_error_mm']:.2f} mm")
    print(f"  Max Tracking Error:   {metrics['max_error_mm']:.2f} mm")

    # Render Rollout Video & GIF
    os.makedirs("outputs", exist_ok=True)
    frames = rollout_data["dashboard_frames"]
    h, w = frames[0].shape[:2]
    
    # Save video with OpenCV
    raw_video_path = "outputs/policy_autonomous_rollout_raw.mp4"
    out_writer = cv2.VideoWriter(raw_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))
    for f in frames:
        out_writer.write(f)
    out_writer.release()

    # Transcode to H.264 using ffmpeg for universal browser compatibility
    h264_video_path = "outputs/policy_autonomous_rollout.mp4"
    ffmpeg_bin = shutil.which("ffmpeg") or "/home/howley/anaconda3/envs/XCS224R_CUDA/bin/ffmpeg"
    if os.path.exists(ffmpeg_bin):
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", raw_video_path, "-vcodec", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", h264_video_path],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if os.path.exists(h264_video_path):
            os.remove(raw_video_path)
            print(f"Saved autonomous rollout video (H.264) to: {h264_video_path}")
        else:
            print(f"Saved autonomous rollout video to: {raw_video_path}")
    else:
        print(f"Saved autonomous rollout video to: {raw_video_path}")

    # Generate animated GIF preview
    gif_path = "outputs/policy_autonomous_rollout.gif"
    if os.path.exists(ffmpeg_bin):
        target_vid = h264_video_path if os.path.exists(h264_video_path) else raw_video_path
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", target_vid, "-vf", "fps=15,scale=720:-1:flags=lanczos", gif_path],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"Saved autonomous rollout GIF preview to: {gif_path}")

    assert metrics["weld_points_count"] > 30, f"Robot failed to track seam! Only {metrics['weld_points_count']} weld steps."
    assert metrics["mean_error_mm"] < 35.0, f"Tracking error too high: {metrics['mean_error_mm']:.2f} mm"

    print("\nCONGRATULATIONS! Your Visuomotor Policy successfully executed autonomous seam tracking!")

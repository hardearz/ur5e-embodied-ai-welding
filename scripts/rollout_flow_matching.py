"""
Executes an autonomous closed-loop rollout of the trained Flow Matching Trajectory Policy
in the UR5e industrial workcell simulation, evaluating seam tracking and saving video/GIF.

Usage:
    robots/bin/python scripts/rollout_flow_matching.py
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

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _import_file(mod_name, rel_path):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(base_dir, rel_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

robot_arm_mod = _import_file("robot_arm", "src/02_mujoco_sim/robot_arm.py")
UR5eIndustrialWorkcell = robot_arm_mod.UR5eIndustrialWorkcell

ex8_mod = _import_file("ex8", "exercises/08_flow_matching_policy.py")
FlowMatchingPolicy = ex8_mod.FlowMatchingPolicy
sample_action_chunk = ex8_mod.sample_action_chunk

IMG_HEIGHT, IMG_WIDTH = 120, 160
ACTION_HORIZON = 16


def run_flow_matching_rollout(
    env: UR5eIndustrialWorkcell,
    policy: FlowMatchingPolicy,
    stats: dict = None,
    max_steps: int = 135,
    replan_interval: int = 8,  # Receding horizon replanning every 8 control cycles (~0.3s)
    num_euler_steps: int = 8,
    device: str = "cpu"
) -> dict:
    env.reset()
    policy.eval()

    tcp_positions = []
    dashboard_frames = []
    qpos_history = []
    replan_count = 0
    current_chunk = None
    chunk_index = 0

    print(f"Executing Flow Matching autonomous rollout ({max_steps} steps, replan every {replan_interval} steps)...")

    for step in range(max_steps):
        # Time to sample a new action chunk via Flow Matching ODE integration
        if current_chunk is None or chunk_index >= replan_interval:
            wrist_raw = env.render_frame("wrist_cam")
            qpos = env.data.qpos[:6].copy()

            # Preprocess observation
            img = cv2.resize(wrist_raw, (IMG_WIDTH, IMG_HEIGHT))
            img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)
            qpos_tensor = torch.from_numpy(qpos).float().unsqueeze(0).to(device)

            # Generate 16-step trajectory chunk via 8-step Euler ODE
            current_chunk = sample_action_chunk(
                policy, img_tensor, qpos_tensor, num_euler_steps=num_euler_steps, stats=stats, device=device
            )
            chunk_index = 0
            replan_count += 1

        # Command next action from the planned chunk
        action = current_chunk[chunk_index]
        chunk_index += 1

        # Apply to robot actuators
        env.data.ctrl[:6] = action

        # Step physics (20 substeps to match 25 Hz control loop, dt = 0.04s)
        for _ in range(20):
            mujoco.mj_step(env.model, env.data)

        # Record telemetry
        tcp_pos, _ = env.get_tcp_pose()
        tcp_positions.append(tcp_pos.copy())
        qpos_history.append(env.data.qpos[:6].copy())

        # Render dashboard frame with live overlay
        dash_frame = env.render_dashboard()
        cv2.putText(
            dash_frame,
            f"FLOW MATCHING POLICY (CHUNK H={ACTION_HORIZON}) | STEP {step:03d}/{max_steps}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        dashboard_frames.append(dash_frame)

    return {
        "tcp_positions": np.array(tcp_positions),
        "dashboard_frames": dashboard_frames,
        "qpos_history": np.array(qpos_history),
    }


def evaluate_seam_tracking(tcp_positions: np.ndarray) -> dict:
    seam_x = 0.520
    seam_z = 0.435

    mask = (tcp_positions[:, 1] >= -0.155) & (tcp_positions[:, 1] <= 0.155) & (tcp_positions[:, 2] <= 0.455)
    weld_points = tcp_positions[mask]

    if len(weld_points) == 0:
        return {"mean_error_mm": 999.0, "max_error_mm": 999.0, "weld_points_count": 0}

    dx = weld_points[:, 0] - seam_x
    dz = weld_points[:, 2] - seam_z
    errors_mm = np.sqrt(dx**2 + dz**2) * 1000.0

    return {
        "mean_error_mm": float(np.mean(errors_mm)),
        "max_error_mm": float(np.max(errors_mm)),
        "weld_points_count": int(len(weld_points)),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Flow Matching rollout on: {device}")

    weights_path = "outputs/flow_matching_policy.pt"
    assert os.path.exists(weights_path), f"Weights not found at {weights_path}!"

    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    policy = FlowMatchingPolicy().to(device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        policy.load_state_dict(ckpt["model_state_dict"])
        stats = ckpt["action_stats"]
    else:
        policy.load_state_dict(ckpt)
        stats = None
    policy.eval()
    print(f"Loaded Flow Matching Policy and Action Stats from {weights_path}")

    env = UR5eIndustrialWorkcell()

    # Execute rollout
    rollout_data = run_flow_matching_rollout(
        env, policy, stats=stats, max_steps=175, replan_interval=4, num_euler_steps=8, device=device
    )
    metrics = evaluate_seam_tracking(rollout_data["tcp_positions"])

    print("\nFlow Matching Policy Metrics:")
    print(f"  Active Seam Steps:    {metrics['weld_points_count']}")
    print(f"  Mean Tracking Error:  {metrics['mean_error_mm']:.2f} mm")
    print(f"  Max Tracking Error:   {metrics['max_error_mm']:.2f} mm")

    # Save video and GIF
    os.makedirs("outputs", exist_ok=True)
    frames = rollout_data["dashboard_frames"]
    h, w = frames[0].shape[:2]

    raw_mp4 = "outputs/flow_matching_rollout_raw.mp4"
    out_writer = cv2.VideoWriter(raw_mp4, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))
    for f in frames:
        out_writer.write(f)
    out_writer.release()

    h264_mp4 = "outputs/flow_matching_rollout.mp4"
    ffmpeg_bin = shutil.which("ffmpeg") or "/home/howley/anaconda3/envs/XCS224R_CUDA/bin/ffmpeg"
    if os.path.exists(ffmpeg_bin):
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", raw_mp4, "-vcodec", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", h264_mp4],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if os.path.exists(h264_mp4):
            os.remove(raw_mp4)
            print(f"Saved Flow Matching rollout video (H.264) to: {h264_mp4}")
    else:
        print(f"Saved video to: {raw_mp4}")

    gif_path = "outputs/flow_matching_rollout.gif"
    if os.path.exists(ffmpeg_bin):
        target_vid = h264_mp4 if os.path.exists(h264_mp4) else raw_mp4
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", target_vid, "-vf", "fps=15,scale=720:-1:flags=lanczos", gif_path],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"Saved Flow Matching rollout GIF to: {gif_path}")


if __name__ == "__main__":
    main()

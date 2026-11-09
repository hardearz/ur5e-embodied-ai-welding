"""
EXERCISE 09: Action Ensembling & Trajectory Execution Benchmark

In this exercise, you will implement and benchmark the 4 ensembling strategies
that fuse absolute coordinates (global error bounding) and relative displacements
(forward momentum / anti-stall) to achieve true industrial seam welding:

  Method 1: Horizon-Weighted Blending (Near-Relative -> Far-Absolute)
  Method 2: Endpoint-Anchored Trajectory Bridge (Pinned at q_current -> q_target)
  Method 3: Direction-Decomposed Control (Tangent Relative Delta-Y + Normal Absolute X/Z)
  Method 4: Classical Feedforward Velocity + Feedback Position Control

Run this benchmark with:
    robots/bin/python exercises/09_action_ensembling_benchmark.py
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

# Dynamic module loaders
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


# ==============================================================================
# Part 1: Horizon-Weighted Blending (Near-Relative -> Far-Absolute)
# ==============================================================================
def ensemble_horizon_weighted(
    chunk_pred: np.ndarray,
    q_current: np.ndarray
) -> np.ndarray:
    """
    Blends relative displacement and absolute coordinates across the horizon:
      - Near steps (k close to 0): Dominated by relative displacement from q_current,
        preventing any initial jerk or lag.
      - Far steps (k close to H-1): Dominated by the predicted absolute joint targets,
        preventing accumulated drift.
        
    Formula for step k in [0, H-1]:
      alpha_k  = k / (H - 1)  in [0.0, 1.0]
      q_anchor = 0.25 * q_current + 0.75 * chunk_pred[0]
      q_rel_k  = q_anchor + 1.15 * (chunk_pred[k] - chunk_pred[0])
      q_abs_k  = chunk_pred[k]
      a_k      = (1.0 - alpha_k) * q_rel_k + alpha_k * q_abs_k
      
    Returns:
      ensembled_chunk of shape (H, 6)
    """
    H = len(chunk_pred)
    ensembled = np.zeros_like(chunk_pred)
    
    # Preserve 75% of the PD servo compliance lead at k=0 while anchoring to q_current,
    # and apply relative chunk displacements with 1.15x forward momentum gain.
    q_anchor = 0.25 * q_current + 0.75 * chunk_pred[0]
    for k in range(H):
        alpha_k = k / max(1, H - 1)
        q_rel_k = q_anchor + 1.15 * (chunk_pred[k] - chunk_pred[0])
        q_abs_k = chunk_pred[k]
        ensembled[k] = (1.0 - alpha_k) * q_rel_k + alpha_k * q_abs_k

    return ensembled


# ==============================================================================
# Part 2: Endpoint-Anchored Trajectory Bridge
# ==============================================================================
def ensemble_endpoint_anchored(
    chunk_pred: np.ndarray,
    q_current: np.ndarray
) -> np.ndarray:
    """
    Constructs a smooth trajectory bridge from the compliance-compensated current pose
    at step 0 to the policy's predicted endpoint target at step H-1:
    
    Formula for step k in [0, H-1]:
      w_k = k / (H - 1)
      q_start  = 0.20 * q_current + 0.80 * chunk_pred[0]
      q_target = chunk_pred[-1] + 0.50 * (chunk_pred[-1] - chunk_pred[-2])
      a_k = (1.0 - w_k) * q_start + w_k * q_target
      
    Returns:
      ensembled_chunk of shape (H, 6)
    """
    H = len(chunk_pred)
    ensembled = np.zeros_like(chunk_pred)
    q_start = 0.20 * q_current + 0.80 * chunk_pred[0]
    step_end = chunk_pred[-1] - chunk_pred[-2] if H > 1 else np.zeros(6)
    q_target = chunk_pred[-1] + 0.50 * step_end

    for k in range(H):
        w_k = k / max(1, H - 1)
        ensembled[k] = (1.0 - w_k) * q_start + w_k * q_target

    return ensembled


# ==============================================================================
# Part 3: Direction-Decomposed Cartesian Control (Tangent Relative + Normal Absolute)
# ==============================================================================
def ensemble_direction_decomposed(
    env: UR5eIndustrialWorkcell,
    chunk_pred: np.ndarray,
    q_current: np.ndarray
) -> np.ndarray:
    """
    Decomposes motion into Cartesian Task-Space corrections mapped via DLS Jacobian Pseudo-Inverse
    while preserving the policy's learned gravity/damping compliance offset:
      - Tangential axis (along seam Y): Relative feedforward step advancement (+Delta Y)
      - Normal axes (cross-track X and height Z): Absolute geometric alignment to the seam
      
    Returns:
      ensembled_chunk of shape (H, 6)
    """
    H = len(chunk_pred)
    ensembled = np.zeros_like(chunk_pred)

    temp_data = env._ik_data
    pred_tcp_positions = []
    jacobians = []
    for k in range(H):
        temp_data.qpos[:6] = chunk_pred[k]
        mujoco.mj_forward(env.model, temp_data)
        pred_tcp_positions.append(temp_data.site_xpos[env.tcp_id].copy())
        jacp, _ = env.compute_jacobian(temp_data)
        jacobians.append(jacp[:, :6].copy())
    pred_tcp_positions = np.array(pred_tcp_positions)

    # Use a local 4-step forward window velocity so Retract at step 15 does not contaminate mid-weld steps
    damping_sq = 0.02 ** 2
    for k in range(H):
        k_next = min(H - 1, k + 4)
        span = max(1, k_next - k)
        dy_local = (pred_tcp_positions[k_next, 1] - pred_tcp_positions[k, 1]) / span

        # When near welding height, project normal (X) tightly onto the seam line X=0.520
        pos_k = pred_tcp_positions[k]
        dx_corr = 0.35 * (0.520 - pos_k[0]) if pos_k[2] <= 0.460 else 0.0
        dy_corr = 0.85 * dy_local  # Local tangential feedforward along Y
        dz_corr = 0.0

        dp_task = np.array([dx_corr, dy_corr, dz_corr])
        J = jacobians[k]
        dq_task = J.T @ np.linalg.solve(J @ J.T + damping_sq * np.eye(3), dp_task)
        ensembled[k] = chunk_pred[k] + np.clip(dq_task, -0.05, 0.05)

    return ensembled


# ==============================================================================
# Part 4: Classical Feedforward Velocity + Feedback Position Compensation
# ==============================================================================
def ensemble_feedforward_velocity(
    chunk_pred: np.ndarray,
    q_current: np.ndarray,
    dt: float = 0.04,  # Control cycle dt (25 Hz)
    lead_gain: float = 0.85
) -> np.ndarray:
    """
    Augments the absolute position target with local 4-step reference velocity feedforward
    to eliminate steady-state actuator tracking lag without early Retract lookahead contamination:
    
      q_dot_local = (chunk_pred[min(H-1, k+4)] - chunk_pred[k]) / (span * dt)
      a_k = chunk_pred[k] + lead_gain * q_dot_local * dt
      
    Returns:
      ensembled_chunk of shape (H, 6)
    """
    H = len(chunk_pred)
    ensembled = np.zeros_like(chunk_pred)

    for k in range(H):
        k_next = min(H - 1, k + 4)
        span = max(1, k_next - k)
        q_dot_local = (chunk_pred[k_next] - chunk_pred[k]) / (span * dt)
        ensembled[k] = chunk_pred[k] + lead_gain * q_dot_local * dt

    return ensembled


# ==============================================================================
# BENCHMARK RUNNER & EVALUATION ENGINE (WITH CROSS-CHUNK TEMPORAL ENSEMBLING)
# ==============================================================================
def evaluate_strategy_rollout(
    strategy_name: str,
    ensemble_fn,
    env: UR5eIndustrialWorkcell,
    policy: FlowMatchingPolicy,
    stats: dict,
    max_steps: int = 160,
    replan_interval: int = 4,
    num_euler_steps: int = 8,
    device: str = "cpu"
) -> dict:
    env.reset()
    policy.eval()

    tcp_positions = []
    dashboard_frames = []
    replan_count = 0
    active_chunks = []  # List of tuples: (start_step, chunk_array)
    prev_cmd = env.data.qpos[:6].copy()

    print(f"\n--- Running Benchmark: [{strategy_name}] ---")

    for step in range(max_steps):
        if step % replan_interval == 0:
            wrist_raw = env.render_frame("wrist_cam")
            qpos = env.data.qpos[:6].copy()

            img = cv2.resize(wrist_raw, (IMG_WIDTH, IMG_HEIGHT))
            img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)
            qpos_tensor = torch.from_numpy(qpos).float().unsqueeze(0).to(device)

            # 1. Flow Matching generates trajectory chunk (with horizon smoothing)
            raw_chunk = sample_action_chunk(
                policy, img_tensor, qpos_tensor, num_euler_steps=num_euler_steps, stats=stats, device=device
            )

            # 2. Apply Intra-Chunk Ensembling Strategy
            if ensemble_fn is not None:
                if "Direction-Decomposed" in strategy_name:
                    new_chunk = ensemble_fn(env, raw_chunk, qpos)
                else:
                    new_chunk = ensemble_fn(raw_chunk, qpos)
            else:
                new_chunk = raw_chunk

            active_chunks.append((step, new_chunk))
            replan_count += 1

        # Prune expired chunks outside the current horizon window
        active_chunks = [(s, c) for s, c in active_chunks if step - s < len(c)]

        # 3. Cross-Chunk Temporal Ensembling (ACT-style exponential blending across overlapping chunks)
        valid_preds = []
        weights = []
        for idx, (start_s, chunk_arr) in enumerate(active_chunks):
            offset = step - start_s
            if 0 <= offset < len(chunk_arr):
                valid_preds.append(chunk_arr[offset])
                # Exponentially weight newer chunks slightly higher while blending out older chunks
                age = len(active_chunks) - 1 - idx
                weights.append(np.exp(-0.35 * age))

        weights = np.array(weights, dtype=np.float32)
        weights /= np.sum(weights)
        blended_action = np.sum([w * p for w, p in zip(weights, valid_preds)], axis=0)

        # First-order low-pass filter on actuator command (0.85/0.15 for minimal phase delay)
        action = 0.85 * blended_action + 0.15 * prev_cmd
        prev_cmd = action.copy()

        # Execute 20 physics substeps per control step (25 Hz control loop, dt = 0.04s)
        env.data.ctrl[:6] = action
        for _ in range(20):
            mujoco.mj_step(env.model, env.data)

        tcp_pos, _ = env.get_tcp_pose()
        tcp_positions.append(tcp_pos.copy())

        # Render visual dashboard
        dash_frame = env.render_dashboard()
        cv2.putText(
            dash_frame,
            f"{strategy_name.upper()} | STEP {step:03d}/{max_steps}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 200),
            2,
            cv2.LINE_AA,
        )
        dashboard_frames.append(dash_frame)

    tcp_positions = np.array(tcp_positions)
    
    # Calculate Metrics across the full 320 mm visible seam [-0.165, +0.165] at welding height (Z <= 0.455)
    seam_x = 0.520
    seam_z = 0.435
    weld_mask = (tcp_positions[:, 1] >= -0.165) & (tcp_positions[:, 1] <= 0.165) & (tcp_positions[:, 2] <= 0.455)
    weld_pts = tcp_positions[weld_mask]

    if len(weld_pts) > 0:
        cross_err = np.sqrt((weld_pts[:, 0] - seam_x)**2 + (weld_pts[:, 2] - seam_z)**2) * 1000.0
        mean_cross_mm = float(np.mean(cross_err))
        max_cross_mm = float(np.max(cross_err))
        weld_steps = int(len(weld_pts))
        y_min = float(weld_pts[:, 1].min())
        y_max = float(weld_pts[:, 1].max())
        y_span_mm = (y_max - y_min) * 1000.0
    else:
        mean_cross_mm = 999.0
        max_cross_mm = 999.0
        weld_steps = 0
        y_min = 0.0
        y_max = 0.0
        y_span_mm = 0.0

    print(f"  Active Seam Steps:    {weld_steps}")
    print(f"  Seam Range (Y):       [{y_min*1000:+.1f} mm, {y_max*1000:+.1f} mm] (Span: {y_span_mm:.1f} mm)")
    print(f"  Mean Cross-Track Err: {mean_cross_mm:.2f} mm")

    return {
        "name": strategy_name,
        "tcp_positions": tcp_positions,
        "dashboard_frames": dashboard_frames,
        "weld_steps": weld_steps,
        "y_span_mm": y_span_mm,
        "mean_cross_mm": mean_cross_mm,
        "max_cross_mm": max_cross_mm,
    }


def main():
    print("=" * 70)
    print("EXERCISE 09: ACTION ENSEMBLING & TRAJECTORY EXECUTION BENCHMARK")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    weights_path = "outputs/flow_matching_policy.pt"
    assert os.path.exists(weights_path), f"Policy weights not found at {weights_path}!"

    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    policy = FlowMatchingPolicy().to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    stats = ckpt["action_stats"]
    policy.eval()

    env = UR5eIndustrialWorkcell()

    strategies = [
        ("Raw Absolute Flow Matching", None, "raw_absolute"),
        ("Horizon-Weighted Blending", ensemble_horizon_weighted, "horizon_weighted"),
        ("Endpoint-Anchored Trajectory Bridge", ensemble_endpoint_anchored, "endpoint_bridge"),
        ("Direction-Decomposed Task-Space Control", ensemble_direction_decomposed, "direction_decomposed"),
        ("Velocity Feedforward Compensation", ensemble_feedforward_velocity, "velocity_feedforward"),
    ]

    results = []
    os.makedirs("outputs/ensembling_benchmark", exist_ok=True)
    ffmpeg_bin = shutil.which("ffmpeg") or "/home/howley/anaconda3/envs/XCS224R_CUDA/bin/ffmpeg"

    for name, fn, slug in strategies:
        res = evaluate_strategy_rollout(
            name, fn, env, policy, stats, max_steps=175, replan_interval=4, num_euler_steps=8, device=device
        )
        results.append(res)

        # Save MP4 & GIF for each strategy
        raw_mp4 = f"outputs/ensembling_benchmark/{slug}_raw.mp4"
        h264_mp4 = f"outputs/ensembling_benchmark/{slug}.mp4"
        gif_path = f"outputs/ensembling_benchmark/{slug}.gif"

        frames = res["dashboard_frames"]
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(raw_mp4, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()

        if os.path.exists(ffmpeg_bin):
            subprocess.run(
                [ffmpeg_bin, "-y", "-i", raw_mp4, "-vcodec", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", h264_mp4],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if os.path.exists(h264_mp4):
                os.remove(raw_mp4)
            target_vid = h264_mp4 if os.path.exists(h264_mp4) else raw_mp4
            subprocess.run(
                [ffmpeg_bin, "-y", "-i", target_vid, "-vf", "fps=15,scale=720:-1:flags=lanczos", gif_path],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

    # Print Comparative Benchmark Table
    print("\n" + "=" * 80)
    print("COMPARATIVE BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Strategy':<38} | {'Seam Span (mm)':<15} | {'Mean Error (mm)':<16} | {'Weld Steps':<10}")
    print("-" * 85)
    for r in results:
        print(f"{r['name']:<38} | {r['y_span_mm']:>11.1f} mm | {r['mean_cross_mm']:>12.2f} mm | {r['weld_steps']:>8d}")
    print("=" * 80)


if __name__ == "__main__":
    main()

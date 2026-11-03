"""
EXERCISE 05: Recording Real Industrial Demonstrations into HDF5

Run this script with:
    robots/bin/python exercises/05_record_welding_demos.py
"""

import os
import cv2
import h5py
import numpy as np

import importlib.util
import sys

# Dynamic import to handle directory/file names with leading digits
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _import_file(mod_name, rel_path):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(base_dir, rel_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

robot_arm_mod = _import_file("robot_arm", "src/02_mujoco_sim/robot_arm.py")
UR5eIndustrialWorkcell = robot_arm_mod.UR5eIndustrialWorkcell

ex4_mod = _import_file("ex4", "exercises/04_teleop_data_collection.py")
EpisodeBuffer = ex4_mod.EpisodeBuffer
save_episode_to_hdf5 = ex4_mod.save_episode_to_hdf5


# Target image resolution for ML policies (faster training, lower memory)
IMG_HEIGHT, IMG_WIDTH = 120, 160


# ==============================================================================
# TODO 1: Record a Single Episode
# ==============================================================================
def record_single_episode(env: UR5eIndustrialWorkcell, seam_start_y: float, seam_end_y: float) -> EpisodeBuffer:
    """
    Executes a seam welding demonstration and records observations into an EpisodeBuffer:
      1. Approach Seam: Safe descent from home to (0.52, seam_start_y, 0.435)
      2. Welding Seam: Linear travel from seam_start_y to seam_end_y
      3. Retract: Vertical lift back to inspection height (0.48, 0.0, 0.65)
      
    At every control step (every 4th physics cycle):
      - Render wrist_cam & workspace_cam and resize to (IMG_WIDTH, IMG_HEIGHT)
      - Read qpos and qvel from env.data
      - Read current action (commanded joint angles in env.data.ctrl[:6])
      - Call buffer.add_step(...)
    """
    buffer = EpisodeBuffer()
    env.reset()

    # Define non-intersecting industrial welding cycle waypoints:
    # 1. Transit at clearance height (Z=0.53m) to above seam start
    # 2. Vertical touchdown to seam start (Z=0.435m, Y=-0.15m)
    # 3. Linear weld pass along full 300mm seam (Y: -0.15m -> +0.15m at Z=0.435m)
    # 4. Vertical retract at seam end (Y=+0.15m, Z=0.60m)
    above_start_wp = np.array([0.52, seam_start_y, 0.530])
    start_wp       = np.array([0.52, seam_start_y, 0.435])
    end_wp         = np.array([0.52, seam_end_y,   0.435])
    retract        = np.array([0.50, seam_end_y,   0.600])

    phases = [
        ("Transit",      above_start_wp, 32),
        ("Touchdown",    start_wp,       18),
        ("Welding",      end_wp,         80),
        ("CraterDwell",  end_wp,          8),
        ("Retract",      retract,        22),
    ]

    curr_target = env.get_tcp_pose()[0]

    for phase_name, target_pos, num_steps in phases:
        p_start = curr_target.copy()
        for s in range(num_steps):
            # S-curve interpolation
            t = (s + 1) / num_steps
            alpha = 0.5 * (1 - np.cos(np.pi * t))
            curr_target = (1 - alpha) * p_start + alpha * target_pos

            # Step physics 20 times per control step (20 * 0.002s = 0.04s -> 25 Hz control rate)
            for _ in range(20):
                env.step_control(curr_target)

            # Record at every 25 Hz control step (160 steps per 6.4s episode)
            # 1. Render and resize cameras
            wrist_raw = env.render_frame("wrist_cam")
            work_raw = env.render_frame("workspace_cam")
            wrist_img = cv2.resize(wrist_raw, (IMG_WIDTH, IMG_HEIGHT))
            work_img = cv2.resize(work_raw, (IMG_WIDTH, IMG_HEIGHT))

            # 2. Read joint positions, velocities, and commanded action
            qpos = env.data.qpos[:6].copy().astype(np.float32)
            qvel = env.data.qvel[:6].copy().astype(np.float32)
            action = env.data.ctrl[:6].copy().astype(np.float32)

            # TODO: Add the step to `buffer` using buffer.add_step(...)
            buffer.add_step(wrist_img, work_img, qpos, qvel, action)

    return buffer


# ==============================================================================
# TODO 2: Collect Dataset with Initial Condition Perturbations
# ==============================================================================
def collect_demonstration_dataset(num_episodes: int = 3, output_dir: str = "data/demos"):
    """
    Collects `num_episodes` demonstrations with slight random seam variations,
    saving each episode as `data/demos/episode_0.h5`, `episode_1.h5`, etc.
    """
    os.makedirs(output_dir, exist_ok=True)
    env = UR5eIndustrialWorkcell()

    print(f"Collecting {num_episodes} demonstration episodes with task variations...")

    for ep in range(num_episodes):
        # Apply small random shift: +/- 3 mm across the full 320 mm visible yellow seam [-0.16, +0.16]
        shift_y = np.random.uniform(-0.003, 0.003)
        seam_start_y = -0.160 + shift_y
        seam_end_y   =  0.160 + shift_y

        print(f"\n>> Recording Episode {ep}: seam shifted by {shift_y*1000:+.1f} mm...")
        buffer = record_single_episode(env, seam_start_y, seam_end_y)

        # TODO: Save the buffer to f"{output_dir}/episode_{ep}.h5" using save_episode_to_hdf5
        save_episode_to_hdf5(buffer, f"{output_dir}/episode_{ep}.h5")
        print(f"   Episode {ep} recorded! Total steps: {len(buffer)}")


# ==============================================================================
# AUTOMATED TEST SUITE & PREVIEW
# ==============================================================================
if __name__ == "__main__":
    print("Testing Exercise 05: Real Demonstration Collection...\n")
    output_dir = "data/demos"
    collect_demonstration_dataset(num_episodes=3, output_dir=output_dir)

    # Verification: check that episode files exist and have non-zero data
    for ep in range(3):
        ep_path = f"{output_dir}/episode_{ep}.h5"
        assert os.path.exists(ep_path), f"Missing episode file: {ep_path}"
        with h5py.File(ep_path, "r") as f:
            n_steps = len(f["action"])
            assert n_steps > 100, f"Expected >100 steps in {ep_path}, got {n_steps}"
            assert f["observations/images/wrist_cam"].shape == (n_steps, IMG_HEIGHT, IMG_WIDTH, 3)
            assert f["action"].shape == (n_steps, 6)
        print(f"PASS: Verified {ep_path} ({n_steps} steps)")

    # Save a visual preview of the recorded wrist camera frames
    sample_ep = f"{output_dir}/episode_0.h5"
    with h5py.File(sample_ep, "r") as f:
        first_frame = f["observations/images/wrist_cam"][0]
        mid_frame = f["observations/images/wrist_cam"][len(f["action"]) // 2]
        last_frame = f["observations/images/wrist_cam"][-1]

        # Combine into a 3-frame progression strip: Start -> Welding -> Retract
        preview_strip = np.hstack([first_frame, mid_frame, last_frame])
        # Convert RGB to BGR for OpenCV saving
        bgr_strip = cv2.cvtColor(preview_strip, cv2.COLOR_RGB2BGR)
        os.makedirs("outputs", exist_ok=True)
        cv2.imwrite("outputs/recorded_demo_preview.png", bgr_strip)
        print("\nSaved visual demonstration preview to outputs/recorded_demo_preview.png")

    print("\nCONGRATULATIONS! You successfully recorded real robot visual demonstrations!")

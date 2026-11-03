"""
EXERCISE 04: Demonstration Data Collection & HDF5 Dataset for Imitation Learning

Run this script with:
    robots/bin/python exercises/04_teleop_data_collection.py
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# ==============================================================================
# TODO 1: In-Memory Episode Buffer
# ==============================================================================
class EpisodeBuffer:
    """
    Collects timesteps during a single demonstration trajectory.
    """
    def __init__(self):
        self.wrist_images = []
        self.workspace_images = []
        self.qpos_list = []
        self.qvel_list = []
        self.action_list = []

    def add_step(self, wrist_img: np.ndarray, workspace_img: np.ndarray, qpos: np.ndarray, qvel: np.ndarray, action: np.ndarray):
        """
        Appends the observations and action of a single timestep to the buffer.
        
        Args:
            wrist_img: (H, W, 3) uint8 RGB image
            workspace_img: (H, W, 3) uint8 RGB image
            qpos: (6,) float32 joint positions
            qvel: (6,) float32 joint velocities
            action: (6,) float32 commanded action
        """
        # TODO: Append each incoming array to its corresponding list above
        self.wrist_images.append(wrist_img)
        self.workspace_images.append(workspace_img)
        self.qpos_list.append(qpos)
        self.qvel_list.append(qvel)
        self.action_list.append(action)

    def __len__(self):
        return len(self.action_list)


# ==============================================================================
# TODO 2: Writing Episode Buffer to HDF5 (.h5) File
# ==============================================================================
def save_episode_to_hdf5(buffer: EpisodeBuffer, filepath: str):
    """
    Saves the buffered demonstration episode into an HDF5 file using standard
    Embodied AI layout (ALOHA / LeRobot format):
    
    File Structure:
      /observations/
          images/
              wrist_cam      -> shape (T, H, W, 3), dtype uint8
              workspace_cam  -> shape (T, H, W, 3), dtype uint8
          qpos               -> shape (T, 6), dtype float32
          qvel               -> shape (T, 6), dtype float32
      /action                -> shape (T, 6), dtype float32
      
    Args:
        buffer: EpisodeBuffer containing recorded steps
        filepath: path to save the .h5 file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # 1. Convert Python lists in buffer to NumPy arrays:
    wrist_arr = np.array(buffer.wrist_images, dtype=np.uint8)
    work_arr = np.array(buffer.workspace_images, dtype=np.uint8)
    qpos_arr = np.array(buffer.qpos_list, dtype=np.float32)
    qvel_arr = np.array(buffer.qvel_list, dtype=np.float32)
    action_arr = np.array(buffer.action_list, dtype=np.float32)

    # 2. Open HDF5 file in write mode: with h5py.File(filepath, "w") as f:
    with h5py.File(filepath, "w") as f:
        # 3. Create datasets (using fast LZF compression on image streams):
        obs_group = f.create_group("observations")
        img_group = obs_group.create_group("images")
        img_group.create_dataset("wrist_cam", data=wrist_arr, compression="lzf")
        img_group.create_dataset("workspace_cam", data=work_arr, compression="lzf")
        obs_group.create_dataset("qpos", data=qpos_arr)
        obs_group.create_dataset("qvel", data=qvel_arr)
        f.create_dataset("action", data=action_arr)




# ==============================================================================
# TODO 3: PyTorch Dataset for Training Imitation Learning Policies
# ==============================================================================
class RobotDemonstrationDataset(Dataset):
    """
    Reads demonstration data directly from HDF5 and formats it into PyTorch tensors
    ready to pass into a neural network policy (Diffusion Policy / ACT).
    """
    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        self.h5_file = None
        with h5py.File(self.hdf5_path, "r") as f:
            self.length = len(f["action"])

    def _ensure_open(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, "r")

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def __del__(self):
        self.close()

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        """
        Returns a single sample at timestep `idx`:
          - wrist_image: PyTorch float32 tensor of shape (3, H, W) normalized to [0, 1]
                         (Note: Vision networks expect Channels-first: C, H, W!)
          - qpos: PyTorch float32 tensor of shape (6,)
          - action: PyTorch float32 tensor of shape (6,)
        """
        self._ensure_open()
        # 1. Read raw numpy slice from HDF5:
        raw_img = self.h5_file["observations/images/wrist_cam"][idx] # (H, W, 3)
        raw_qpos = self.h5_file["observations/qpos"][idx]            # (6,)
        raw_action = self.h5_file["action"][idx]                     # (6,)

        # 2. Convert raw_img to Channels-First (H, W, C -> C, H, W) and normalize to [0.0, 1.0]:
        img_tensor = torch.from_numpy(raw_img).permute(2, 0, 1).float() / 255.0

        # 3. Convert qpos and action to torch.float32 tensors
        qpos_tensor = torch.from_numpy(raw_qpos).float()
        action_tensor = torch.from_numpy(raw_action).float()
        # 4. Return {"image": img_tensor, "qpos": qpos_tensor, "action": action_tensor}

        # TODO: Implement steps 1-4
        return {
            "image": img_tensor,
            "qpos": qpos_tensor,
            "action": action_tensor
        }


# ==============================================================================
# AUTOMATED TEST SUITE
# ==============================================================================
if __name__ == "__main__":
    print("Testing Exercise 04: Demonstration Data Pipeline...\n")
    test_h5_path = "data/test_demo.h5"

    # Test 1: EpisodeBuffer
    buffer = EpisodeBuffer()
    H, W = 120, 160
    # Simulate 10 steps of data
    for t in range(10):
        dummy_wrist = np.zeros((H, W, 3), dtype=np.uint8)
        dummy_work = np.zeros((H, W, 3), dtype=np.uint8)
        dummy_qpos = np.ones(6, dtype=np.float32) * t
        dummy_qvel = np.zeros(6, dtype=np.float32)
        dummy_act = np.ones(6, dtype=np.float32) * (t + 1)
        buffer.add_step(dummy_wrist, dummy_work, dummy_qpos, dummy_qvel, dummy_act)

    assert len(buffer) == 10, f"Expected buffer length 10, got {len(buffer)}"
    print("PASS: TODO 1 (EpisodeBuffer)")

    # Test 2: save_episode_to_hdf5
    save_episode_to_hdf5(buffer, test_h5_path)
    assert os.path.exists(test_h5_path), f"File {test_h5_path} was not created!"

    with h5py.File(test_h5_path, "r") as f:
        assert "observations/images/wrist_cam" in f, "Missing wrist_cam dataset"
        assert "observations/images/workspace_cam" in f, "Missing workspace_cam dataset"
        assert "observations/qpos" in f, "Missing qpos dataset"
        assert "observations/qvel" in f, "Missing qvel dataset"
        assert "action" in f, "Missing action dataset"

        wrist_shape = f["observations/images/wrist_cam"].shape
        assert wrist_shape == (10, H, W, 3), f"Expected shape (10, {H}, {W}, 3), got {wrist_shape}"
        action_shape = f["action"].shape
        assert action_shape == (10, 6), f"Expected shape (10, 6), got {action_shape}"
    print("PASS: TODO 2 (save_episode_to_hdf5)")

    # Test 3: PyTorch Dataset
    dataset = RobotDemonstrationDataset(test_h5_path)
    assert len(dataset) == 10, f"Expected dataset length 10, got {len(dataset)}"

    sample = dataset[3]
    assert "image" in sample and "qpos" in sample and "action" in sample, "Sample dictionary missing keys"
    # Check vision tensor shape: (C, H, W)
    assert sample["image"].shape == (3, H, W), f"Expected image tensor shape (3, {H}, {W}), got {sample['image'].shape}"
    assert sample["image"].dtype == torch.float32, "Image tensor must be float32"
    assert sample["qpos"].shape == (6,), f"Expected qpos shape (6,), got {sample['qpos'].shape}"
    assert sample["action"].shape == (6,), f"Expected action shape (6,), got {sample['action'].shape}"
    print("PASS: TODO 3 (RobotDemonstrationDataset)")

    # Clean up test file
    if os.path.exists(test_h5_path):
        os.remove(test_h5_path)

    print("\nCONGRATULATIONS! You built a complete Embodied AI demonstration data pipeline!")

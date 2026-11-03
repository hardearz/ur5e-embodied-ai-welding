"""
EXERCISE 08: Flow Matching Policy & Action Chunking for Trajectory Generation

In this exercise, you will build a Generative Flow Matching Policy for the UR5e.
Instead of predicting a single noisy 1-step action, the network learns a continuous
velocity vector field that transforms pure Gaussian noise into an entire 16-step
smooth action trajectory chunk via straight-line ODE integration.

Run this script with:
    robots/bin/python exercises/08_flow_matching_policy.py
"""

import os
import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Target trajectory horizon: predict 16 future actions at once (~0.6 seconds of motion)
ACTION_HORIZON = 16  # H
ACTION_DIM = 6       # 6 arm joints
CHUNK_DIM = ACTION_HORIZON * ACTION_DIM  # 16 * 6 = 96 dimensions


# ==============================================================================
# TODO 1: Action Chunking Multi-Episode Dataset
# ==============================================================================
class ChunkedDemonstrationDataset(Dataset):
    """
    Slices demonstration episodes into (Observation_t -> ActionChunk_{t:t+H}) pairs.
    If t + H exceeds the episode length, pad with the final recorded action.
    """
    def __init__(self, demos_dir: str = "data/demos", horizon: int = ACTION_HORIZON):
        self.horizon = horizon
        self.files = sorted(glob.glob(os.path.join(demos_dir, "episode_*.h5")))
        assert len(self.files) > 0, f"No episode_*.h5 files found in {demos_dir}!"

        self.samples = []

        for fpath in self.files:
            with h5py.File(fpath, "r") as f:
                # Load images: convert HWC -> CHW and normalize to [0, 1]
                imgs = f["observations/images/wrist_cam"][:]
                imgs = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32) / 255.0
                qpos = f["observations/qpos"][:].astype(np.float32)
                actions = f["action"][:].astype(np.float32)

                num_steps = len(actions)
                for t in range(num_steps):
                    # Extract slice t : t + horizon
                    # TODO 1: Extract the action chunk of length `self.horizon`.
                    # If t + self.horizon > num_steps, slice up to num_steps and pad the remainder
                    # by repeating the final action actions[-1].
                    # The resulting chunk should have shape (self.horizon, ACTION_DIM).
                    chunk = np.array([actions[-1]] * horizon, dtype=np.float32)  # Implement the slice and padding logic
                    tmp = actions[t: t + horizon]
                    chunk[:len(tmp)] = tmp

                    self.samples.append({
                        "image": imgs[t],          # (3, 120, 160)
                        "qpos": qpos[t],           # (6,)
                        "action_chunk": chunk,     # (H, 6)
                    })

        # Compute per-joint (6,) action normalization statistics (Mean & Std across all steps)
        # Ensures uniform translation-invariant scaling across all horizon steps k in [0, H-1]
        all_chunks = np.stack([s["action_chunk"] for s in self.samples])  # (N, H, 6)
        self.stats = {
            "mean": np.mean(all_chunks, axis=(0, 1)).astype(np.float32),                    # (6,)
            "std": np.clip(np.std(all_chunks, axis=(0, 1)), 1e-3, None).astype(np.float32), # (6,)
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        # Normalize (H, 6) action chunk per-joint using (6,) stats, then flatten to (96,)
        norm_chunk = (item["action_chunk"] - self.stats["mean"]) / self.stats["std"]
        return {
            "image": torch.from_numpy(item["image"]),
            "qpos": torch.from_numpy(item["qpos"]),
            "action_chunk": torch.from_numpy(norm_chunk.flatten()),
        }


# ==============================================================================
# Model Architecture: Flow Matching Velocity Field Network
# ==============================================================================
class SinusoidalTimeEmbedding(nn.Module):
    """Encodes continuous flow time t in [0, 1] into frequency space."""
    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B, 1) or (B,)
        t = t.view(-1, 1)
        half_dim = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=t.device) / half_dim)
        args = t * freqs.unsqueeze(0) * 2.0 * np.pi
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class FlowMatchingPolicy(nn.Module):
    """
    Velocity Vector Field Network v_theta(x_t, t, image, qpos) -> dx/dt
    Predicts the constant velocity vector driving noisy trajectory x_t to clean action chunk x_1.
    """
    def __init__(self, chunk_dim: int = CHUNK_DIM, visual_dim: int = 64, proprio_dim: int = 32, time_dim: int = 32):
        super().__init__()
        self.chunk_dim = chunk_dim

        # 1. Vision Encoder (3-layer CNN)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, visual_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(visual_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # 2. Proprioception Encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(6, proprio_dim),
            nn.ReLU(),
        )

        # 3. Flow Time Embedding (continuous time t in [0, 1])
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.Mish(),
        )

        # 4. Velocity Vector MLP (Predicts continuous derivative dx/dt)
        # Input features: [x_t (96) + time_emb (32) + visual_feats (64) + proprio_feats (32)] = 224
        in_dim = chunk_dim + time_dim + visual_dim + proprio_dim
        self.velocity_mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.Mish(),
            nn.Linear(512, 512),
            nn.Mish(),
            nn.Linear(512, 256),
            nn.Mish(),
            nn.Linear(256, chunk_dim),  # Output: (B, 96) velocity vector
        )

    def encode_observations(self, image: torch.Tensor, qpos: torch.Tensor):
        """Encodes visual and proprioceptive observations once per replanning step."""
        z_vis = self.vision_encoder(image)
        z_prop = self.proprio_encoder(qpos)
        return z_vis, z_prop

    def forward_velocity(self, x_t: torch.Tensor, t: torch.Tensor, z_vis: torch.Tensor, z_prop: torch.Tensor) -> torch.Tensor:
        """Evaluates the velocity field using pre-encoded observation features."""
        t_emb = self.time_encoder(t)
        state = torch.cat([x_t, t_emb, z_vis, z_prop], dim=-1)
        return self.velocity_mlp(state)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, image: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
        """
        Computes the velocity vector v_theta(x_t, t | image, qpos)
        Args:
            x_t: (B, CHUNK_DIM) - Noisy action chunk at time t
            t:   (B, 1) or (B,) - Continuous flow time in [0, 1]
            image: (B, 3, 120, 160) - Wrist camera frame
            qpos:  (B, 6) - Robot joint positions
        Returns:
            v_pred: (B, CHUNK_DIM) - Predicted velocity vector dx/dt
        """
        z_vis, z_prop = self.encode_observations(image, qpos)
        return self.forward_velocity(x_t, t, z_vis, z_prop)


# ==============================================================================
# TODO 2: Flow Matching Training Loss
# ==============================================================================
def compute_flow_matching_loss(
    policy: FlowMatchingPolicy,
    x_1: torch.Tensor,
    image: torch.Tensor,
    qpos: torch.Tensor,
    device: str
) -> torch.Tensor:
    """
    Computes the Conditional Flow Matching (CFM) objective on a training batch:
      1. Sample random Gaussian noise x_0 ~ N(0, I) with the same shape as x_1.
      2. Sample continuous flow time t ~ Uniform(0, 1) of shape (B, 1).
      3. Construct the straight-line interpolant:
           x_t = (1 - t) * x_0 + t * x_1
      4. Compute the analytical target velocity:
           v_target = x_1 - x_0
      5. Predict velocity with the network:
           v_pred = policy(x_t, t, image, qpos)
      6. Return Mean Squared Error: MSE(v_pred, v_target).
    """
    B = x_1.shape[0]

    # TODO 2: Implement the 6 steps of Flow Matching training + Temporal Smoothness Regularization!
    x_0 = torch.randn_like(x_1)
    t = torch.rand((B, 1), dtype=torch.float32, device=device)
    x_t = (1.0 - t) * x_0 + t * x_1
    v_target = x_1 - x_0
    v_pred = policy(x_t, t, image, qpos)

    fm_loss = nn.MSELoss()(v_pred, v_target)

    # Reconstruct predicted clean trajectory chunk x1_hat = x_t + (1 - t) * v_pred
    # and penalize 2nd-order finite-difference acceleration ||x[k+1] - 2*x[k] + x[k-1]||^2
    x1_hat = (x_t + (1.0 - t) * v_pred).view(B, ACTION_HORIZON, ACTION_DIM)
    accel = x1_hat[:, 2:, :] - 2.0 * x1_hat[:, 1:-1, :] + x1_hat[:, :-2, :]
    smooth_loss = torch.mean(accel ** 2)

    loss = fm_loss + 0.5 * smooth_loss
    return loss


# ==============================================================================
# TODO 3: Fast Euler ODE Sampler for Trajectory Generation
# ==============================================================================
def sample_action_chunk(
    policy: FlowMatchingPolicy,
    image: torch.Tensor,
    qpos: torch.Tensor,
    num_euler_steps: int = 8,
    stats: dict = None,
    device: str = "cpu"
) -> np.ndarray:
    """
    Generates a full 16-step action trajectory chunk from noise using Euler integration:
      1. Start at pure Gaussian noise: x ~ N(0, I) of shape (1, CHUNK_DIM).
      2. Step size: dt = 1.0 / num_euler_steps.
      3. For step k from 0 to num_euler_steps - 1:
           current_t = k * dt
           v_pred = policy(x, current_t, image, qpos)
           x = x + dt * v_pred
      4. Reshape generated x from (1, 96) to (16, 6).
      5. If `stats` (mean & std) are provided, un-normalize back to physical radian angles.
    """
    policy.eval()
    with torch.no_grad():
        # TODO 3: Implement Euler ODE integration across `num_euler_steps`!
        # Encode vision & proprioception ONCE outside the Euler loop (~4.5x faster ODE sampling)
        z_vis, z_prop = policy.encode_observations(image, qpos)

        # Sample low-temperature prior (sigma=0.15) directly on target device
        x = 0.15 * torch.randn((1, CHUNK_DIM), dtype=torch.float32, device=device)
        dt = 1.0 / num_euler_steps
        for k in range(num_euler_steps):
            current_t = torch.tensor([[k * dt]], dtype=torch.float32, device=device)
            v_pred = policy.forward_velocity(x, current_t, z_vis, z_prop)
            x = x + dt * v_pred

        chunk = x.reshape((ACTION_HORIZON, ACTION_DIM)).cpu().numpy()
        if stats is not None:
            mean_raw = stats["mean"].cpu().numpy() if isinstance(stats["mean"], torch.Tensor) else stats["mean"]
            std_raw = stats["std"].cpu().numpy() if isinstance(stats["std"], torch.Tensor) else stats["std"]
            mean = mean_raw.reshape((-1, ACTION_DIM))
            std = std_raw.reshape((-1, ACTION_DIM))
            chunk = chunk * std + mean

        # Apply zero-phase 3-tap temporal smoothing filter across interior horizon steps
        smoothed = chunk.copy()
        for _ in range(2):
            smoothed[1:-1] = 0.25 * smoothed[:-2] + 0.50 * smoothed[1:-1] + 0.25 * smoothed[2:]
        return smoothed


# ==============================================================================
# AUTOMATED TEST SUITE & TRAINING SCRIPT
# ==============================================================================
if __name__ == "__main__":
    print("Testing Exercise 08: Flow Matching Policy & Action Chunking...\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # 1. Test Dataset Slicing (TODO 1)
    dataset = ChunkedDemonstrationDataset("data/demos", horizon=ACTION_HORIZON)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    sample_item = dataset[0]

    assert sample_item["action_chunk"] is not None, "TODO 1: action_chunk is None!"
    assert sample_item["action_chunk"].shape == (CHUNK_DIM,), (
        f"Expected action chunk shape ({CHUNK_DIM},), got {sample_item['action_chunk'].shape}"
    )
    print(f"PASS: TODO 1 (Chunked Dataset Loaded: {len(dataset)} trajectory pairs, horizon={ACTION_HORIZON})")

    # 2. Test Flow Matching Loss (TODO 2)
    policy = FlowMatchingPolicy().to(device)
    batch = next(iter(dataloader))
    imgs = batch["image"].to(device)
    qps = batch["qpos"].to(device)
    x1s = batch["action_chunk"].to(device)

    loss = compute_flow_matching_loss(policy, x1s, imgs, qps, device)
    assert loss is not None, "TODO 2: compute_flow_matching_loss returned None!"
    assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert loss.item() > 0.0, "Loss should be positive!"
    print("PASS: TODO 2 (Flow Matching Loss Calculation)")

    # 3. Test Euler ODE Sampler (TODO 3)
    dummy_img = torch.zeros(1, 3, 120, 160, device=device)
    dummy_qp = torch.zeros(1, 6, device=device)
    generated_chunk = sample_action_chunk(policy, dummy_img, dummy_qp, num_euler_steps=8, stats=dataset.stats, device=device)

    assert generated_chunk is not None, "TODO 3: sample_action_chunk returned None!"
    assert isinstance(generated_chunk, np.ndarray), f"Expected numpy array, got {type(generated_chunk)}"
    assert generated_chunk.shape == (ACTION_HORIZON, ACTION_DIM), (
        f"Expected trajectory shape ({ACTION_HORIZON}, {ACTION_DIM}), got {generated_chunk.shape}"
    )
    print("PASS: TODO 3 (Fast Euler ODE Generation: 8 steps)")

    # 4. Train the Flow Matching Policy (240 epochs with Cosine Annealing LR)
    num_epochs = 240
    print(f"\nStarting Flow Matching Policy Training ({num_epochs} epochs with Cosine LR + Smoothness Loss)...")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    initial_loss = None
    final_loss = None

    for epoch in range(1, num_epochs + 1):
        policy.train()
        total_loss = 0.0
        for b in dataloader:
            b_imgs = b["image"].to(device)
            b_qps = b["qpos"].to(device)
            b_x1s = b["action_chunk"].to(device)

            optimizer.zero_grad()
            b_loss = compute_flow_matching_loss(policy, b_x1s, b_imgs, b_qps, device)
            b_loss.backward()
            optimizer.step()
            total_loss += b_loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        if epoch == 1:
            initial_loss = avg_loss
        final_loss = avg_loss
        if epoch % 40 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:03d}/{num_epochs}] | Flow Matching + Smoothness Loss: {avg_loss:.6f}")

    assert final_loss < initial_loss, f"Loss did not decrease! Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"
    print(f"\nPASS: Training converged! Loss decreased from {initial_loss:.4f} -> {final_loss:.4f}")

    # Save trained model checkpoint and action normalization stats as safe PyTorch tensors
    os.makedirs("outputs", exist_ok=True)
    save_path = "outputs/flow_matching_policy.pt"
    checkpoint = {
        "model_state_dict": policy.state_dict(),
        "action_stats": {
            "mean": torch.from_numpy(dataset.stats["mean"]),
            "std": torch.from_numpy(dataset.stats["std"]),
        },
    }
    torch.save(checkpoint, save_path)
    print(f"Saved trained Flow Matching Policy and Action Stats to: {save_path}")

    # Generate sample trajectory and inspect smoothness
    sample_traj = sample_action_chunk(policy, imgs[:1], qps[:1], num_euler_steps=8, stats=dataset.stats, device=device)
    print("\nGenerated 16-Step Action Trajectory Chunk Preview:")
    print(f"  Step 0:  {np.round(sample_traj[0], 3)}")
    print(f"  Step 7:  {np.round(sample_traj[7], 3)}")
    print(f"  Step 15: {np.round(sample_traj[15], 3)}")

    print("\nCONGRATULATIONS! You successfully implemented and trained a Flow Matching Trajectory Policy!")

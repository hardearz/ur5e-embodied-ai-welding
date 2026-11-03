"""
EXERCISE 06: Visuomotor Policy Network & Behavior Cloning in PyTorch

Run this script with:
    robots/bin/python exercises/06_visuomotor_policy_training.py
"""

import os
import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ==============================================================================
# Multi-Episode Dataset Loader
# ==============================================================================
class MultiEpisodeDataset(Dataset):
    """Aggregates all episode_*.h5 files in data/demos/ into a single dataset."""
    def __init__(self, demos_dir: str = "data/demos"):
        self.files = sorted(glob.glob(os.path.join(demos_dir, "episode_*.h5")))
        assert len(self.files) > 0, f"No episode_*.h5 files found in {demos_dir}!"
        
        # Preload datasets into memory for fast GPU training
        self.images = []
        self.qpos = []
        self.actions = []

        for fpath in self.files:
            with h5py.File(fpath, "r") as f:
                # Load images and convert HWC -> CHW and uint8 -> float32 [0, 1]
                imgs = f["observations/images/wrist_cam"][:]
                # Transpose from (T, H, W, C) to (T, C, H, W)
                imgs = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32) / 255.0
                qp = f["observations/qpos"][:].astype(np.float32)
                act = f["action"][:].astype(np.float32)

                self.images.append(imgs)
                self.qpos.append(qp)
                self.actions.append(act)

        self.images = np.concatenate(self.images, axis=0)
        self.qpos = np.concatenate(self.qpos, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        self.length = len(self.actions)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "image": torch.from_numpy(self.images[idx]),
            "qpos": torch.from_numpy(self.qpos[idx]),
            "action": torch.from_numpy(self.actions[idx]),
        }


# ==============================================================================
# TODO 1 & 2: Visuomotor Policy Architecture
# ==============================================================================
class VisuomotorPolicy(nn.Module):
    def __init__(self, visual_feature_dim: int = 64, proprio_dim: int = 32, action_dim: int = 6):
        super().__init__()

        # 1. Vision Backbone (CNN)
        # Input: (B, 3, 120, 160) -> Output: (B, visual_feature_dim)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # -> (32, 60, 80)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # -> (32, 30, 40)

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (64, 15, 20)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # -> (64, 7, 10)

            nn.Conv2d(64, visual_feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(visual_feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                          # -> (visual_feature_dim, 1, 1)
            nn.Flatten(),                                          # -> (visual_feature_dim,)
        )

        # 2. Proprioception Encoder (MLP)
        # Input: (B, 6) -> Output: (B, proprio_dim)
        self.proprio_encoder = nn.Sequential(
            nn.Linear(6, proprio_dim),
            nn.ReLU(),
        )

        # 3. Action Head (MLP)
        # Input: (B, visual_feature_dim + proprio_dim) -> Output: (B, action_dim)
        combined_dim = visual_feature_dim + proprio_dim
        self.action_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, image: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the policy:
          1. Extract visual features: z_vis = self.vision_encoder(image)
          2. Extract proprio features: z_prop = self.proprio_encoder(qpos)
          3. Concatenate: z = torch.cat([z_vis, z_prop], dim=1)
          4. Predict action: action_pred = self.action_head(z)
          
        Args:
            image: (B, 3, H, W)
            qpos: (B, 6)
            
        Expected Return:
            (B, 6) predicted action tensor
        """
        # TODO: Implement the forward pass
        z_vis = self.vision_encoder(image)
        z_prop = self.proprio_encoder(qpos)
        z = torch.cat([z_vis, z_prop], dim=1)
        action_pred = self.action_head(z)


        return action_pred


# ==============================================================================
# TODO 3: The Behavior Cloning Training Step
# ==============================================================================
def train_one_epoch(
    policy: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> float:
    """
    Executes one training epoch:
      1. Sets policy to train mode: policy.train()
      2. For each batch (image, qpos, action):
           a. Move batch tensors to device (GPU/CPU)
           b. Zero optimizer gradients: optimizer.zero_grad()
           c. Forward pass: pred_action = policy(images, qpos)
           d. Compute loss: loss = criterion(pred_action, target_action)
           e. Backward pass: loss.backward()
           f. Optimizer step: optimizer.step()
      3. Return average epoch loss
    """
    policy.train()
    total_loss = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        qpos = batch["qpos"].to(device)
        actions = batch["action"].to(device)

        # TODO: Implement the forward pass, loss calculation, backward pass, and optimizer step!
        optimizer.zero_grad()
        pred = policy(images, qpos)
        loss = criterion(pred, actions)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / max(1, len(dataloader))
    return avg_loss


# ==============================================================================
# AUTOMATED TRAINING PIPELINE
# ==============================================================================
if __name__ == "__main__":
    print("Testing Exercise 06: Visuomotor Policy Training...\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # Load multi-episode dataset
    dataset = MultiEpisodeDataset("data/demos")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"Loaded {len(dataset)} total timesteps across demonstration episodes.")

    # Initialize policy network
    policy = VisuomotorPolicy().to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # Test dummy forward pass
    dummy_img = torch.randn(4, 3, 120, 160, device=device)
    dummy_qpos = torch.randn(4, 6, device=device)
    dummy_out = policy(dummy_img, dummy_qpos)
    assert dummy_out.shape == (4, 6), f"Expected policy output shape (4, 6), got {dummy_out.shape}"
    assert not torch.allclose(dummy_out, torch.zeros_like(dummy_out)), "Policy forward pass should not return all zeros!"
    print("PASS: TODO 1 & 2 (Policy Architecture & Forward Pass)")

    # Run 60 training epochs with Cosine Annealing LR so 1-step BC resolves 25 Hz steps
    num_epochs = 60
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    print(f"\nStarting Behavior Cloning Training Loop ({num_epochs} epochs):")
    initial_loss = None
    final_loss = None

    for epoch in range(1, num_epochs + 1):
        loss = train_one_epoch(policy, dataloader, optimizer, criterion, device)
        scheduler.step()
        if epoch == 1:
            initial_loss = loss
        final_loss = loss
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:02d}/{num_epochs}] | MSE Loss: {loss:.6f}")

    assert final_loss < initial_loss, f"Loss did not decrease! Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"
    print(f"\nPASS: TODO 3 (Training Loop: Loss decreased from {initial_loss:.4f} -> {final_loss:.4f})")

    # Save trained policy weights
    os.makedirs("outputs", exist_ok=True)
    model_save_path = "outputs/visuomotor_policy.pt"
    torch.save(policy.state_dict(), model_save_path)
    print(f"Saved trained policy weights to: {model_save_path}")

    print("\nCONGRATULATIONS! You successfully trained an end-to-end Visuomotor Policy Network!")

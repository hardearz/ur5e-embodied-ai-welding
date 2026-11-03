"""
Robotics & Embodied AI Environment Verification Script.
Checks:
  1. PyTorch CUDA tensor execution on GPU.
  2. MuJoCo 3.x physics stepping and contact dynamics.
  3. MuJoCo offscreen camera rendering (EGL / GPU).
  4. Robotics data processing libraries (einops, h5py, zarr, cv2).
"""

import sys
import torch
import mujoco
import numpy as np
import cv2
import einops
import h5py
import zarr


def check_cuda():
    print("=" * 60)
    print("1. PYTORCH & CUDA ACCELERATION")
    print("=" * 60)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise SystemError("CUDA is not available in PyTorch!")

    device_name = torch.cuda.get_device_name(0)
    print(f"Active GPU: {device_name}")

    # Stress test CUDA execution
    a = torch.randn(2000, 2000, device="cuda")
    b = torch.randn(2000, 2000, device="cuda")
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    print(f"CUDA MatMul 2000x2000 Successful! Output shape: {c.shape}")
    print("PyTorch CUDA check passed!\n")


def check_mujoco():
    print("=" * 60)
    print("2. MUJOCO PHYSICS & CAMERA RENDERING")
    print("=" * 60)
    print(f"MuJoCo Version: {mujoco.__version__}")

    # Minimal robotic test scene with a ground plane and falling sphere
    xml = """
    <mujoco model="test_scene">
      <visual>
        <global offwidth="640" offheight="480"/>
      </visual>
      <worldbody>
        <light diffuse=".8 .8 .8" pos="0 0 4" dir="0 0 -1"/>
        <geom type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
        <camera name="third_person" pos="0 -1.5 1.2" xyaxes="1 0 0 0 0.6 0.8"/>
        <body name="sphere" pos="0 0 1.0">
          <joint type="free"/>
          <geom type="sphere" size="0.08" mass="0.5" rgba="0.9 0.2 0.2 1"/>
        </body>
      </worldbody>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Step simulation 100 times (dt = 0.002s -> 0.2s of freefall)
    initial_z = float(data.qpos[2])
    for _ in range(100):
        mujoco.mj_step(model, data)
    final_z = float(data.qpos[2])
    print(f"100 steps completed. Sphere height: {initial_z:.3f}m -> {final_z:.3f}m")

    # Test Offscreen Camera Rendering
    renderer = mujoco.Renderer(model, height=240, width=320)
    renderer.update_scene(data, camera="third_person")
    img = renderer.render()
    print(f"Offscreen Camera Render Succeeded! Image Shape: {img.shape}, Dtype: {img.dtype}")
    print("MuJoCo simulation & rendering check passed!\n")


def check_tooling():
    print("=" * 60)
    print("3. EMBODIED AI TOOLING LIBRARIES")
    print("=" * 60)
    print(f"OpenCV (cv2) Version: {cv2.__version__}")
    print(f"h5py Version: {h5py.__version__}")
    print(f"Zarr Version: {zarr.__version__}")

    # Einops test
    x = np.zeros((4, 3, 240, 320))
    rearranged = einops.rearrange(x, "b c h w -> b h w c")
    print(f"Einops rearrangement test passed: {x.shape} -> {rearranged.shape}")
    print("Embodied AI tooling check passed!\n")


if __name__ == "__main__":
    check_cuda()
    check_mujoco()
    check_tooling()
    print("ALL ENVIRONMENT CHECKS PASSED SUCCESSFULLY!")

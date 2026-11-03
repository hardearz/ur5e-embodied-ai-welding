"""
Renders a video (MP4) and animated GIF of the falling sphere scene from verify_env.py.

Usage:
    robots/bin/python scripts/render_falling_sphere.py
"""

import os
import shutil
import subprocess
import cv2
import mujoco
import numpy as np
from PIL import Image

# The exact test scene from verify_env.py
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

def main():
    os.makedirs("outputs", exist_ok=True)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    frames_rgb = []
    
    # 600 steps @ dt=0.002s = 1.2 seconds of physics
    # Render every 10 steps -> 50 FPS recording (60 frames total)
    total_steps = 600
    render_interval = 10
    
    print(f"Simulating {total_steps} steps (1.2s) and recording camera frames...")
    for step in range(total_steps):
        if step % render_interval == 0:
            renderer.update_scene(data, camera="third_person")
            frame = renderer.render()
            frames_rgb.append(frame)
        mujoco.mj_step(model, data)

    # 1. Export Animated GIF (playable in any image viewer / VS Code)
    gif_path = "outputs/falling_sphere.gif"
    pil_images = [Image.fromarray(f) for f in frames_rgb]
    pil_images[0].save(
        gif_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=20,  # 20ms per frame = 50 FPS
        loop=0
    )
    print(f"Saved GIF: {gif_path}")

    # 2. Export MP4 Video (H.264 encoded)
    raw_mp4 = "outputs/falling_sphere_raw.mp4"
    h264_mp4 = "outputs/falling_sphere.mp4"
    writer = cv2.VideoWriter(raw_mp4, cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (640, 480))
    for f in frames_rgb:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

    ffmpeg_bin = shutil.which("ffmpeg") or "/home/howley/anaconda3/envs/XCS224R_CUDA/bin/ffmpeg"
    if os.path.exists(ffmpeg_bin):
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", raw_mp4, "-vcodec", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", h264_mp4],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if os.path.exists(h264_mp4):
            os.remove(raw_mp4)
            print(f"Saved H.264 MP4: {h264_mp4}")
    else:
        os.rename(raw_mp4, h264_mp4)
        print(f"Saved MP4: {h264_mp4}")

if __name__ == "__main__":
    main()

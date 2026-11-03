"""
Core Spatial Math & SE(3) Transformations for Robotics.

Mastering coordinate frames, rotation representations, and transformation
chains is essential for Embodied AI and teleoperation pipelines.

Key Interview Concepts:
  1. SO(3) vs SE(3): Special Orthogonal group (3D rotations) vs Special Euclidean group (rigid body pose).
  2. Euler Angles vs Quaternions: Euler angles suffer from Gimbal Lock (loss of a degree of freedom when
     pitch approaches +/- 90 degrees). Quaternions provide smooth, singularity-free interpolation (SLERP).
  3. Homogeneous Transform Inversion:
     For T = [[R, p], [0, 1]], the inverse is [[R.T, -R.T @ p], [0, 1]].
     Never use generic matrix inversion `np.linalg.inv(T)` in real-time control loops!
"""

import numpy as np
from typing import Tuple


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Computes a 3x3 rotation matrix from Roll, Pitch, Yaw (ZYX convention / Tait-Bryan).
    Angles in radians.
    
    R = R_z(yaw) @ R_y(pitch) @ R_x(roll)
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R_x = np.array([
        [1,  0,   0],
        [0, cr, -sr],
        [0, sr,  cr]
    ])

    R_y = np.array([
        [ cp, 0, sp],
        [  0, 1,  0],
        [-sp, 0, cp]
    ])

    R_z = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [ 0,   0, 1]
    ])

    return R_z @ R_y @ R_x


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Converts a unit quaternion [w, x, y, z] to a 3x3 rotation matrix.
    """
    # Normalize to avoid numerical drift and guard against zero-norm input
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        raise ValueError("Quaternion norm must be non-zero.")
    q = q / norm
    w, x, y, z = q

    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    return R


def make_homogeneous_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Constructs a 4x4 SE(3) homogeneous transformation matrix:
        T = [[ R,  p ],
             [ 0,  1 ]]
    """
    assert R.shape == (3, 3), "Rotation matrix R must be 3x3"
    assert p.shape in ((3,), (3, 1)), "Position vector p must be 3-element"
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p.reshape(3)
    return T


def invert_homogeneous_transform(T: np.ndarray) -> np.ndarray:
    """
    Computes the exact inverse of a rigid SE(3) transformation.
    Using the algebraic identity:
        T^-1 = [[ R.T,  -R.T @ p ],
                [  0,       1    ]]
    O(1) complexity, perfectly numerically stable.
    """
    R = T[:3, :3]
    p = T[:3, 3]
    
    T_inv = np.eye(4)
    R_T = R.T
    T_inv[:3, :3] = R_T
    T_inv[:3, 3] = -R_T @ p
    return T_inv


def transform_point(T: np.ndarray, point_in_source: np.ndarray) -> np.ndarray:
    """
    Transforms a 3D point from frame A to frame B using T_B_A (pose of A in B).
    P_B = T_B_A @ [P_A, 1]^T
    """
    p_homogeneous = np.array([point_in_source[0], point_in_source[1], point_in_source[2], 1.0])
    p_transformed = T @ p_homogeneous
    return p_transformed[:3]


def demo_camera_to_base_transformation():
    """
    Practical Robotics Scenario:
    A wrist-mounted or overhead depth camera detects a target weld point or grasp target.
    Target coordinates are currently known in the CAMERA's local frame: P_cam.
    We need to compute the target coordinates in the ROBOT BASE frame: P_base.
    """
    print("\n--- Practical Scenario: Camera Target -> Robot Base Frame ---")
    
    # 1. Robot Base to Camera Extrinsics: T_base_cam
    # Say the camera is positioned at x=0.5m, y=-0.2m, z=0.8m relative to base,
    # and pitched 45 degrees downward facing the workspace.
    R_base_cam = euler_to_rotation_matrix(roll=0.0, pitch=np.radians(45), yaw=0.0)
    p_base_cam = np.array([0.5, -0.2, 0.8])
    T_base_cam = make_homogeneous_transform(R_base_cam, p_base_cam)
    
    # 2. Object detected by camera at local coordinates: [0.0, 0.0, 0.5] (0.5m directly along camera optical axis)
    p_cam = np.array([0.0, 0.0, 0.5])
    
    # 3. Compute position in robot base frame:
    p_base = transform_point(T_base_cam, p_cam)
    print(f"Object in Camera Frame: P_cam  = {p_cam}")
    print(f"Object in Robot Base Frame: P_base = {np.round(p_base, 4)}")
    
    # 4. Verify Inverse Transformation:
    T_cam_base = invert_homogeneous_transform(T_base_cam)
    p_cam_recovered = transform_point(T_cam_base, p_base)
    print(f"Recovered Object in Camera Frame   = {np.round(p_cam_recovered, 4)}")
    assert np.allclose(p_cam, p_cam_recovered), "Transformation recovery failed!"
    print("SE(3) transformation chain verified successfully!")


if __name__ == "__main__":
    print("Running Spatial Math & SE(3) Transformations Module...")
    
    # Check Gimbal Lock demonstration
    R1 = euler_to_rotation_matrix(np.radians(10), np.radians(89.9), np.radians(20))
    print(f"Rotation matrix with pitch ~90 deg:\n{np.round(R1, 3)}")
    
    # Check Quaternion conversion
    # 90 degree rotation around Z-axis: [cos(pi/4), 0, 0, sin(pi/4)]
    q_z90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    R_q = quaternion_to_rotation_matrix(q_z90)
    print(f"\nQuaternion 90-deg Z rotation matrix:\n{np.round(R_q, 3)}")
    
    # Run practical robotics frame transformation
    demo_camera_to_base_transformation()

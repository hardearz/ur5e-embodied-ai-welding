"""
EXERCISE 01: Spatial Math & Coordinate Frame Transformations

Run this script with:
    robots/bin/python exercises/01_spatial_math_exercise.py
"""

import numpy as np


# ==============================================================================
# Part 1: 2D/3D Rotation about the Z-axis
# ==============================================================================
def rot_z(theta_rad: float) -> np.ndarray:
    """
    Computes a 3x3 rotation matrix for a rotation around the Z-axis by theta radians.
    
    Recall the rotation equations for the X and Y axes:
        x' = x * cos(theta) - y * sin(theta)
        y' = x * sin(theta) + y * cos(theta)
        z' = z
        
    Expected Return:
        A 3x3 numpy array of dtype float64.
    """
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    
    R = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])
    return R


# ==============================================================================
# Part 2: Assemble a 4x4 Homogeneous Transformation Matrix
# ==============================================================================
def make_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Combines a 3x3 rotation matrix R and a 3-element translation vector p
    into a single 4x4 homogeneous transformation matrix:
    
        T = [[ R_11, R_12, R_13, p_x ],
             [ R_21, R_22, R_23, p_y ],
             [ R_31, R_32, R_33, p_z ],
             [    0,    0,    0,   1 ]]
             
    Args:
        R: (3, 3) numpy array
        p: (3,) or (3, 1) numpy array
        
    Expected Return:
        (4, 4) numpy array
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(p).reshape(3)
    return T


# ==============================================================================
# Part 3: Transform a 3D Point
# ==============================================================================
def transform_point(T: np.ndarray, point_in_source: np.ndarray) -> np.ndarray:
    """
    Applies the 4x4 transform T to a 3D point:
        P_dest = T * [point_in_source, 1]^T
        
    Args:
        T: (4, 4) transformation matrix
        point_in_source: (3,) numpy array representing [x, y, z]
        
    Expected Return:
        (3,) numpy array of the transformed [x, y, z] coordinates.
    """
    pis = np.hstack([point_in_source, 1.0])
    P = T @ pis
    return P[:3]


# ==============================================================================
# Part 4: Fast Matrix Inversion
# ==============================================================================
def fast_inverse(T: np.ndarray) -> np.ndarray:
    # 1. Extract R and p from T
    R = T[:3, :3]
    p = T[:3, 3]
    # 2. Compute R_inv and p_inv
    Rinv = R.T
    pinv = -Rinv @ p
    # 3. Assemble and return the inverted 4x4 matrix
    return make_transform(Rinv, pinv)


# ==============================================================================
# AUTOMATED TEST SUITE (Run this script to verify your solutions!)
# ==============================================================================
if __name__ == "__main__":
    print("Testing Exercise 01 Solutions...\n")
    
    # Test 1: rot_z by 90 degrees
    theta = np.pi / 2  # 90 degrees
    R_z90 = rot_z(theta)
    expected_R = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0]
    ])
    assert np.allclose(R_z90, expected_R, atol=1e-5), f"Part 1 Failed!\nExpected:\n{expected_R}\nGot:\n{R_z90}"
    print("PASS: Part 1 (rot_z)")

    # Test 2: make_transform
    p = np.array([0.5, 0.2, 0.8])
    T = make_transform(R_z90, p)
    assert T.shape == (4, 4), f"Part 2 Failed: Expected shape (4, 4), got {T.shape}"
    assert np.allclose(T[:3, :3], R_z90), "Part 2 Failed: R submatrix incorrect"
    assert np.allclose(T[:3, 3], p), "Part 2 Failed: p translation vector incorrect"
    assert np.allclose(T[3, :], [0, 0, 0, 1]), "Part 2 Failed: bottom row must be [0, 0, 0, 1]"
    print("PASS: Part 2 (make_transform)")

    # Test 3: transform_point
    # A point at [1, 0, 0] rotated 90 deg around Z becomes [0, 1, 0], then translated by [0.5, 0.2, 0.8] -> [0.5, 1.2, 0.8]
    pt_cam = np.array([1.0, 0.0, 0.0])
    pt_base = transform_point(T, pt_cam)
    expected_pt = np.array([0.5, 1.2, 0.8])
    assert np.allclose(pt_base, expected_pt, atol=1e-5), f"Part 3 Failed!\nExpected: {expected_pt}\nGot: {pt_base}"
    print("PASS: Part 3 (transform_point)")

    # Test 4: fast_inverse
    T_inv = fast_inverse(T)
    # Check that T @ T_inv is identity matrix
    assert np.allclose(T @ T_inv, np.eye(4), atol=1e-5), "Part 4 Failed: T @ T_inv is not identity!"
    # Check round-trip point transformation
    pt_recovered = transform_point(T_inv, pt_base)
    assert np.allclose(pt_recovered, pt_cam, atol=1e-5), f"Part 4 Failed: Round-trip point recovery failed!\nExpected: {pt_cam}\nGot: {pt_recovered}"
    print("PASS: Part 4 (fast_inverse)")

    print("\nCONGRATULATIONS! All spatial math unit tests passed!")

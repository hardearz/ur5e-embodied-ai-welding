"""
EXERCISE 02: 2-Link Arm Forward Kinematics, Analytical Jacobian & DLS-IK

Run this script with:
    robots/bin/python exercises/02_kinematics_jacobian_exercise.py
"""

import numpy as np


# ==============================================================================
# Part 1: Forward Kinematics (FK)
# ==============================================================================
def forward_kinematics(theta1: float, theta2: float, L1: float = 1.0, L2: float = 1.0) -> np.ndarray:
    """
    Computes the (x, y) Cartesian position of the hand (end-effector).
    
    Geometry reminder:
      - Elbow position:
          x1 = L1 * cos(theta1)
          y1 = L1 * sin(theta1)
      - Hand position is elbow position + Link 2 projection:
          Notice Link 2's absolute orientation in the world is (theta1 + theta2).
          
    Args:
        theta1: joint 1 angle (radians)
        theta2: joint 2 angle (radians)
        L1, L2: link lengths
        
    Expected Return:
        np.array([x, y]) of the hand position.
    """
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    return np.array([x, y], dtype=np.float64)


# ==============================================================================
# Part 2: The Analytical Jacobian Matrix J(theta)
# ==============================================================================
def compute_jacobian(theta1: float, theta2: float, L1: float = 1.0, L2: float = 1.0) -> np.ndarray:
    """
    Computes the 2x2 Jacobian matrix:
        J = [[ dx/d(theta1),  dx/d(theta2) ],
             [ dy/d(theta1),  dy/d(theta2) ]]
             
    The Jacobian maps joint velocities to hand velocity:
        [x_dot, y_dot]^T = J * [theta1_dot, theta2_dot]^T
        
    Take the partial derivatives of x and y with respect to theta1 and theta2!
    
    Expected Return:
        (2, 2) numpy array
    """
    J11 = -L1 * np.sin(theta1) - L2 * np.sin(theta1 + theta2)
    J12 = -L2 * np.sin(theta1 + theta2)
    J21 =  L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    J22 =  L2 * np.cos(theta1 + theta2)
    
    return np.array([
        [J11, J12],
        [J21, J22]
    ], dtype=np.float64)


# ==============================================================================
# Part 3: Damped Least Squares (DLS) Pseudo-Inverse
# ==============================================================================
def dls_inverse(J: np.ndarray, damping: float = 0.05) -> np.ndarray:
    """
    Computes the Damped Least Squares inverse of J:
        J_dls = J^T * (J * J^T + lambda^2 * I)^-1
    """
    JJT = J @ J.T + (damping ** 2) * np.eye(2)
    JJTinv = np.linalg.inv(JJT)
    return J.T @ JJTinv


# ==============================================================================
# Part 4: The Closed-Loop Inverse Kinematics (IK) Solver
# ==============================================================================
def solve_ik(
    target_xy: np.ndarray,
    q_init: np.ndarray,
    L1: float = 1.0,
    L2: float = 1.0,
    max_iters: int = 50,
    tol: float = 1e-4
) -> np.ndarray:
    """
    Iteratively finds joint angles [theta1, theta2] to reach target_xy:
        1. Compute current hand position: x_curr = FK(theta)
        2. Compute Cartesian error: e = target_xy - x_curr
        3. If ||e|| < tol: stop and return theta!
        4. Compute Jacobian J(theta)
        5. Compute delta_theta = J_dls * e
        6. Update: theta = theta + delta_theta
        7. Repeat!
        
    Expected Return:
        np.array([theta1, theta2])
    """
    q = q_init.copy().astype(np.float64)
    for i in range(max_iters):
        curr_pos = forward_kinematics(q[0], q[1], L1, L2)
        err = target_xy - curr_pos
        if np.linalg.norm(err) < tol:
            break
        J = compute_jacobian(q[0], q[1], L1, L2)
        J_inv = dls_inverse(J)
        dtheta = J_inv @ err
        q = q + dtheta

    return q


# ==============================================================================
# AUTOMATED TEST SUITE
# ==============================================================================
if __name__ == "__main__":
    print("Testing Exercise 02 Solutions...\n")

    # Link lengths
    L1, L2 = 1.0, 1.0

    # Test 1: Forward Kinematics
    # When theta1 = 0, theta2 = 0 -> Arm stretched along X-axis: x = L1 + L2 = 2.0, y = 0.0
    pt_stretched = forward_kinematics(0.0, 0.0, L1, L2)
    assert np.allclose(pt_stretched, [2.0, 0.0]), f"Part 1 Failed for [0, 0]: got {pt_stretched}"

    # When theta1 = 90 deg (pi/2), theta2 = 0 -> Arm pointing straight up along Y-axis: x = 0, y = 2.0
    pt_up = forward_kinematics(np.pi / 2, 0.0, L1, L2)
    assert np.allclose(pt_up, [0.0, 2.0]), f"Part 1 Failed for [pi/2, 0]: got {pt_up}"

    # When theta1 = 0, theta2 = 90 deg (pi/2) -> Elbow at (1, 0), hand at (1, 1)
    pt_bent = forward_kinematics(0.0, np.pi / 2, L1, L2)
    assert np.allclose(pt_bent, [1.0, 1.0]), f"Part 1 Failed for [0, pi/2]: got {pt_bent}"
    print("PASS: Part 1 (forward_kinematics)")

    # Test 2: Analytical Jacobian
    # At [0, pi/2], hand is at (1, 1).
    # If we rotate theta1 by small d_th, hand moves perpendicular to vector from base (1, 1) -> velocity (-1, 1).
    J_bent = compute_jacobian(0.0, np.pi / 2, L1, L2)
    expected_J = np.array([
        [-1.0, -1.0],
        [ 1.0,  0.0]
    ])
    assert np.allclose(J_bent, expected_J), f"Part 2 Failed!\nExpected:\n{expected_J}\nGot:\n{J_bent}"
    print("PASS: Part 2 (compute_jacobian)")

    # Test 3: Damped Least Squares
    # At [0, pi/2], J = [[-1, -1], [1, 0]]
    J_test = compute_jacobian(0.0, np.pi / 2, L1, L2)
    J_dls = dls_inverse(J_test, damping=0.05)
    expected_dls = np.array([
        [-0.0025,  0.9950],
        [-0.9950, -0.9925]
    ])
    assert np.allclose(J_dls, expected_dls, atol=1e-3), f"Part 3 Failed!\nExpected:\n{expected_dls}\nGot:\n{J_dls}"
    
    # Also verify singularity safety (arm fully stretched: theta2 = 0, det(J) == 0)
    J_singular = compute_jacobian(0.0, 0.0, L1, L2)
    assert np.isclose(np.linalg.det(J_singular), 0.0), "Sanity check: fully stretched arm must have det(J) == 0"
    J_dls_sing = dls_inverse(J_singular, damping=0.1)
    assert not np.any(np.isnan(J_dls_sing)) and not np.any(np.isinf(J_dls_sing)), "DLS returned NaN or Inf at singularity!"
    print("PASS: Part 3 (dls_inverse)")

    # Test 4: Inverse Kinematics Solver
    target = np.array([0.5, 1.2])
    q_seed = np.array([0.5, 0.5])
    q_sol = solve_ik(target, q_seed, L1, L2)
    final_pos = forward_kinematics(q_sol[0], q_sol[1], L1, L2)
    err = np.linalg.norm(target - final_pos)
    assert err < 1e-3, f"Part 4 Failed: Hand position {final_pos} did not reach target {target}! Error: {err:.4f}"
    print(f"PASS: Part 4 (solve_ik: reached target {target} with error {err*1000:.3f} mm)")

    print("\nCONGRATULATIONS! You built a complete kinematics and IK engine from scratch!")

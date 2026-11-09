"""
EXERCISE 03: MuJoCo Fundamentals & The UR5e Industrial Robot

Run this script with:
    robots/bin/python exercises/03_mujoco_fundamentals_exercise.py
"""

import os
import mujoco
import numpy as np


# Helper to locate models
def load_ur5e_workcell():
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "ur5e", "workcell.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    return model, data


# ==============================================================================
# Part 1: Inspecting MjModel & MjData
# ==============================================================================
def inspect_robot_properties(model: mujoco.MjModel, data: mujoco.MjData):
    """
    MuJoCo divides everything into MjModel (static) and MjData (dynamic).
    
    In this function, extract and return:
      - n_joints: Total number of joints (model.njnt)
      - n_actuators: Total number of actuators (model.nu)
      - n_qpos: Size of generalized position vector (model.nq)
      - n_qvel: Size of generalized velocity vector (model.nv)
      - timestep: Physics integration time step in seconds (model.opt.timestep)
    """
    n_joints = model.njnt 
    n_actuators = model.nu
    n_qpos = model.nq
    n_qvel = model.nv
    timestep = model.opt.timestep

    return {
        "n_joints": n_joints,
        "n_actuators": n_actuators,
        "n_qpos": n_qpos,
        "n_qvel": n_qvel,
        "timestep": timestep,
    }


# ==============================================================================
# Part 2: Reading the Tool Center Point (TCP) Site Pose
# ==============================================================================
def get_tcp_position_and_rotation(model: mujoco.MjModel, data: mujoco.MjData):
    """
    In MuJoCo, a "site" is a landmark placed on a robot link (like the tip of a tool).
    Our model has a site named "tcp".
    
    Steps:
      1. Find the site ID for "tcp":
         site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
      2. Read the 3D position vector:
         pos = data.site_xpos[site_id]
      3. Read the 3x3 rotation matrix:
         rot = data.site_xmat[site_id].reshape(3, 3)
         
    Expected Return:
        pos (3,), rot (3, 3)
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    pos = data.site_xpos[site_id].copy()
    rot = data.site_xmat[site_id].reshape(3, 3).copy()
    return pos, rot


# ==============================================================================
# Part 3: Extracting the Analytical Jacobian using MuJoCo
# ==============================================================================
def get_tcp_jacobian(model: mujoco.MjModel, data: mujoco.MjData):
    """
    Instead of calculating partial derivatives by hand like in Exercise 2,
    MuJoCo can compute the 6-DOF Jacobian for ANY site instantly using:
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        
    Where:
      - jacp is (3, model.nv): Linear velocity Jacobian (dx/dq, dy/dq, dz/dq)
      - jacr is (3, model.nv): Angular velocity Jacobian (wx/dq, wy/dq, wz/dq)
      
    Steps:
      1. Get site_id for "tcp".
      2. Allocate zero arrays for jacp and jacr with shape (3, model.nv).
      3. Call mujoco.mj_jacSite(model, data, jacp, jacr, site_id).
      4. Stack them vertically to form the full (6, model.nv) Jacobian:
         J = np.vstack([jacp, jacr])
         
    Expected Return:
        J (6, model.nv) numpy array
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    J = np.vstack([jacp, jacr])
    return J


# ==============================================================================
# Part 4: The Physics Loop - Forward vs Step
# ==============================================================================
def move_robot_with_step(model: mujoco.MjModel, data: mujoco.MjData, target_qpos: np.ndarray, num_steps: int = 200):
    """
    Difference between mj_forward and mj_step:
      - mujoco.mj_forward(model, data):
          Computes all kinematics, accelerations, and sensor readings at the CURRENT state.
          DOES NOT ADVANCE TIME. (Instantaneous computation).
      - mujoco.mj_step(model, data):
          Applies actuator control signals (data.ctrl), calculates contact forces,
          and INTEGRATES PHYSICS FORWARD in time by model.opt.timestep.
          
    Task:
      1. Set data.ctrl[:6] = target_qpos (command the position servos).
      2. Loop num_steps times:
             mujoco.mj_step(model, data)
      3. Return the final TCP position using get_tcp_position_and_rotation(model, data)[0].
    """
    data.ctrl[:6] = target_qpos
    for _ in range(num_steps):
        mujoco.mj_step(model, data)
    final_tcp_pos = get_tcp_position_and_rotation(model, data)[0]
    return final_tcp_pos


# ==============================================================================
# AUTOMATED TEST SUITE
# ==============================================================================
if __name__ == "__main__":
    print("Testing Exercise 03: MuJoCo Fundamentals...\n")
    model, data = load_ur5e_workcell()

    # Test 1: Model Properties
    props = inspect_robot_properties(model, data)
    assert props["n_joints"] == 6, f"Expected 6 joints for UR5e, got {props['n_joints']}"
    assert props["n_actuators"] == 6, f"Expected 6 actuators, got {props['n_actuators']}"
    assert props["n_qpos"] == 6, f"Expected nq=6, got {props['n_qpos']}"
    assert props["n_qvel"] == 6, f"Expected nv=6, got {props['n_qvel']}"
    assert np.isclose(props["timestep"], 0.002), f"Expected dt=0.002s, got {props['timestep']}"
    print("PASS: Part 1 (inspect_robot_properties)")

    # Test 2: TCP Site Position & Orientation
    # Reset to home keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    tcp_pos, tcp_rot = get_tcp_position_and_rotation(model, data)
    assert tcp_pos.shape == (3,), f"Expected shape (3,), got {tcp_pos.shape}"
    assert tcp_rot.shape == (3, 3), f"Expected shape (3, 3), got {tcp_rot.shape}"
    # In the home keyframe, the TCP should be roughly at [-0.134, 0.492, 0.768]
    assert np.allclose(tcp_pos, [-0.134, 0.492, 0.768], atol=0.01), f"Unexpected TCP pos: {tcp_pos}"
    # Verify that the rotation matrix is orthogonal (det == 1)
    assert np.isclose(np.linalg.det(tcp_rot), 1.0, atol=1e-4), "TCP rotation matrix must belong to SO(3)"
    print(f"PASS: Part 2 (get_tcp_position_and_rotation: TCP at {np.round(tcp_pos, 3)})")

    # Test 3: Analytical Jacobian
    J = get_tcp_jacobian(model, data)
    assert J.shape == (6, 6), f"Expected Jacobian shape (6, 6), got {J.shape}"
    assert not np.allclose(J, 0.0), "Jacobian should not be all zeros!"
    print(f"PASS: Part 3 (get_tcp_jacobian: shape {J.shape})")

    # Test 4: Stepping Physics
    target_angles = np.array([np.pi, -1.2, 1.6, -1.97, -1.57, 0.0])
    final_tcp = move_robot_with_step(model, data, target_angles, num_steps=300)
    assert not np.allclose(final_tcp, 0.0), "final_tcp should not be zeros"
    # When facing the table, TCP moves to x > 0.5m
    assert final_tcp[0] > 0.45, f"Robot should have reached over table (x > 0.45), got x = {final_tcp[0]:.3f}"
    print(f"PASS: Part 4 (move_robot_with_step: successfully stepped physics to {np.round(final_tcp, 3)})")

    print("\nCONGRATULATIONS! You now know how to program and control robots directly in MuJoCo!")

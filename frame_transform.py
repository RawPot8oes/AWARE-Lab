"""
frame_transform.py

Rotates a measured wrench (force/torque) from the load cell's own frame
-- which tips over with the arm every time roll/pitch/yaw changes -- into
the fixed tank/world frame, using the arm's logged pose at that instant.

Why this matters:
    Right now Fx/Fy/Fz in flap_test_log.csv are in the SENSOR's frame.
    That frame is bolted to the flange, so it rotates with pitch. Any
    force that's actually constant in the tank (airfoil weight, buoyancy,
    a steady thrust) gets smeared into a sinusoid at the flap frequency
    once expressed in a frame that's rotating at the flap frequency --
    it hasn't changed physically, you're just reading it off a tilting
    ruler. Rotating back into the world frame removes that artifact and
    gives you lift/thrust/side-force the way you'd define them
    physically in the tank.

Convention (confirmed against UFactory's own worked example in their
forum -- see https://forum.ufactory.cc, "Roll Pitch Yaw calculation
problem on xArm 6"):
    roll/pitch/yaw are fixed-axis (extrinsic) Euler angles, composed as

        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    and a vector in the local/tool frame maps to the world/base frame as

        v_world = R @ v_local

    get_position() already returns pose in this same world/base frame,
    which is why we can use roll/pitch/yaw straight out of the CSV with
    no extra bookkeeping.

IMPORTANT -- axis mapping assumption:
    This gives you forces in the xArm's BASE frame, not necessarily in
    "tank" axes labeled lift/thrust/lateral. If your base mount is
    aligned so base-Z is vertical and base-X runs along the tank's long
    axis (the way BASE_POSE's heave-along-x / TCP pointing down via
    roll=180 suggests), then:
        Fx_world -> thrust / longitudinal
        Fy_world -> lateral / side force
        Fz_world -> lift / vertical
    If the base is mounted at some angle to the tank, add one fixed
    rotation (tank-to-base) on top of this before trusting the labels --
    e.g. by jogging the arm to a few known orientations and checking
    which world axis lines up with "vertical in the tank" by eye.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def rotation_from_pose(pose):
    """
    pose: [x, y, z, roll, pitch, yaw] (mm, deg)
    Returns the 3x3 rotation matrix mapping local/sensor-frame vectors
    to the world/tank frame.
    """
    roll, pitch, yaw = pose[3], pose[4], pose[5]
    return R.from_euler('XYZ', [roll, pitch, yaw], degrees=True).as_matrix()


def wrench_to_world(pose, wrench):
    """
    pose:   [x, y, z, roll, pitch, yaw] -- arm pose at the moment this
            wrench was recorded (world/base frame, as logged)
    wrench: [Fx, Fy, Fz, Tx, Ty, Tz] -- as measured, i.e. still in the
            sensor's own rotating frame

    Returns [Fx, Fy, Fz, Tx, Ty, Tz] rotated into the world/tank frame.
    Only rotates (no lever-arm/moment correction) -- fine for forces;
    for torques this reports the torque about the sensor's own origin,
    just re-expressed in world axes, not re-referenced to some other
    point in the tank.
    """
    Rmat = rotation_from_pose(pose)
    F_local = np.asarray(wrench[0:3], dtype=float)
    T_local = np.asarray(wrench[3:6], dtype=float)
    F_world = Rmat @ F_local
    T_world = Rmat @ T_local
    return [*F_world, *T_world]


def batch_wrench_to_world(poses, wrenches):
    """
    Vectorized-ish convenience wrapper for a full log.

    poses:    Nx6 array-like, columns [x, y, z, roll, pitch, yaw]
    wrenches: Nx6 array-like, columns [Fx, Fy, Fz, Tx, Ty, Tz]

    Returns an Nx6 numpy array of world-frame wrenches, same column order.
    """
    poses = np.asarray(poses, dtype=float)
    wrenches = np.asarray(wrenches, dtype=float)
    out = np.zeros_like(wrenches)
    for i in range(len(poses)):
        out[i] = wrench_to_world(poses[i], wrenches[i])
    return out

"""
trajectory.py

Motion generators. Each function takes plain parameters and returns a
list of [x, y, z, roll, pitch, yaw] poses -- no hardware calls, no
classes, nothing to configure. Just call one and get poses back.
"""

import numpy as np


def generate_flap_trajectory(base_pose, n_cycles, heave_amp_mm, pitch_amp_deg,
                              freq_hz, phase_offset_deg=180.0, rate_hz=100):
    """
    Sinusoidal heave + pitch flapping motion.

    base_pose       : [x, y, z, roll, pitch, yaw] neutral pose to oscillate around
    n_cycles        : how many flap cycles to generate
    heave_amp_mm    : peak heave displacement (mm)
    pitch_amp_deg   : peak pitch angle (deg)
    freq_hz         : flap frequency (Hz)
    phase_offset_deg: phase of pitch relative to heave (deg). 180 = pitch
                       and heave move opposite each other.
    rate_hz         : poses generated per second (your command rate)

    Returns a list of poses.
    """
    duration_s = n_cycles / freq_hz
    n_samples = int(duration_s * rate_hz)
    t = np.linspace(0, duration_s, n_samples)

    omega = 2 * np.pi * freq_hz * t
    heave = heave_amp_mm * np.sin(omega)
    pitch = pitch_amp_deg * np.sin(omega + np.radians(phase_offset_deg))

    poses = []
    for h, p in zip(heave, pitch):
        pose = list(base_pose)
        pose[0] += h   # x = heave
        pose[4] += p   # pitch parameter = rotation about base Y
        poses.append(pose)

    return poses


def generate_straight_trajectory(start_pose, end_pose, duration_s, rate_hz=100):
    """
    Straight-line motion between two poses (linear interpolation of all
    6 values -- position and orientation).

    start_pose, end_pose: [x, y, z, roll, pitch, yaw]
    duration_s          : how long the move should take
    rate_hz             : poses generated per second (your command rate)

    Returns a list of poses, starting at start_pose and ending at end_pose.
    """
    n_samples = int(duration_s * rate_hz)
    start = np.array(start_pose)
    end = np.array(end_pose)

    poses = []
    for frac in np.linspace(0, 1, n_samples):
        pose = start + frac * (end - start)
        poses.append(list(pose))

    return poses
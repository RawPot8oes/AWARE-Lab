"""
trajectory.py

Motion generator. Takes plain parameters and returns a list of
[x, y, z, roll, pitch, yaw] poses -- no hardware calls, no classes,
nothing to configure. Just call it and get poses back.
"""

import numpy as np

# Index into a pose list for each named DOF, for readability.
DOF_INDEX = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}


def generate_oscillatory_trajectory(base_pose, n_cycles, ref_freq_hz,
                                     rate_hz=100, dofs=None):
    """
    General sinusoidal motion generator: any combination of DOFs, each
    with its own amplitude/frequency/phase, all oscillating simultaneously
    around base_pose. Covers flapping (heave+pitch), pure yaw, or any
    other combination -- just change what's in `dofs`.

    base_pose  : [x, y, z, roll, pitch, yaw] neutral pose to oscillate around
    n_cycles   : how many cycles of ref_freq_hz to run (sets total duration)
    ref_freq_hz: reference frequency (Hz) used with n_cycles to set
                 duration_s = n_cycles / ref_freq_hz. Usually just set
                 this equal to whatever your main oscillation frequency is.
    rate_hz    : poses generated per second (your command rate)
    dofs       : dict of {dof_name: (amplitude, freq_hz, phase_deg)}
                 dof_name is one of "x", "y", "z", "roll", "pitch", "yaw"
                 (mm for x/y/z, degrees for roll/pitch/yaw).
                 Any DOF not listed stays fixed at its base_pose value.

    Returns a list of poses.

    Example -- flap = heave + pitch, pitch lagging heave by 180 deg:
        generate_oscillatory_trajectory(
            base_pose, n_cycles=5, ref_freq_hz=1.5, rate_hz=100,
            dofs={
                "x":     (100, 1.5, 0),
                "pitch": (30,  1.5, 180),
            },
        )

    Example -- pure sinusoidal yaw:
        generate_oscillatory_trajectory(
            base_pose, n_cycles=8, ref_freq_hz=1.0, rate_hz=100,
            dofs={"yaw": (20, 1.0, 0)},
        )
    """
    dofs = dofs or {}
    duration_s = n_cycles / ref_freq_hz
    n_samples = int(duration_s * rate_hz)
    t = np.linspace(0, duration_s, n_samples)

    poses = np.tile(np.array(base_pose, dtype=float), (n_samples, 1))

    for dof_name, (amp, freq_hz, phase_deg) in dofs.items():
        idx = DOF_INDEX[dof_name]
        omega = 2 * np.pi * freq_hz * t + np.radians(phase_deg)
        poses[:, idx] += amp * np.sin(omega)

    return poses.tolist()


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

def generate_step_hold_poses(base_pose, dof, start, stop, step, hold_s,
                             ramp_s=1.0, lead_ramp_s=5.0, rate_hz=100):
    """Stepped sweep with holds, as a flat pose list. Smooth ramps between
    levels, plus a slow ramp in from base_pose and back out at the end."""
    idx = DOF_INDEX[dof]
    levels = np.arange(start, stop + step / 2, step)
    n_hold = int(hold_s * rate_hz)

    def ramp(a, b, secs):
        n = max(1, int(secs * rate_hz))
        s = np.linspace(0, 1, n + 1)[1:]
        s = s * s * (3 - 2 * s)          # smoothstep, zero velocity at ends
        return list(a + (b - a) * s)

    offsets = ramp(0.0, levels[0], lead_ramp_s)
    offsets += [levels[0]] * n_hold
    for prev, nxt in zip(levels[:-1], levels[1:]):
        offsets += ramp(prev, nxt, ramp_s)
        offsets += [nxt] * n_hold
    offsets += ramp(levels[-1], 0.0, lead_ramp_s)

    poses = np.tile(np.array(base_pose, dtype=float), (len(offsets), 1))
    poses[:, idx] += np.array(offsets)
    return poses.tolist()
# Flapping Flexible Airfoil Test Rig

Test rig code for running heave + pitch flapping motion on a UFactory xArm
while logging force/torque data from an ATI Mini40 F/T sensor (via an NI DAQ),
as part of the AWARE flapping flexible airfoil project.

## Hardware

- **UFactory xArm** — 6-DOF robot arm, drives the heave + pitch flapping motion
- **ATI Mini40** — 6-axis force/torque sensor, mounted between the arm and the airfoil
- **NI DAQ (6423)** — reads the Mini40's 6 raw gauge voltages

## Repo structure

| File | Purpose |
|---|---|
| `run_flap_test.py` | Main entry point — connects to the arm + sensor, runs a flap test, logs data, saves + plots results |
| `arm.py` | Wrapper around the xArm SDK (connect, move, TCP offset, abort/safety handling) |
| `ft_sensor.py` | Wrapper around the NI DAQ + Mini40 (raw voltage → calibrated wrench, zeroing, streaming, overload check) |
| `cal_parser.py` | Parses the ATI `.cal` calibration file into a 6×6 matrix + max ratings |
| `trajectory.py` | Generates flap (sinusoidal heave+pitch) and straight-line pose trajectories |
| `plotter.py` | Plots kinematics + sensor-frame forces/torques + tank-frame (world) forces from a logged CSV |
| `frame_transform.py` | Rotates measured forces/torques out of the sensor's (rotating) frame into the fixed tank/world frame |

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install numpy matplotlib scipy nidaqmx xarm-python-sdk
```

You'll also need:
- Your sensor's ATI `.cal` file (referenced by `CAL_FILE` in `run_flap_test.py`)
- The xArm controller reachable at the IP set in `ARM_IP`
- NI-DAQmx drivers installed for the DAQ channels in `DAQ_CHANNELS`

## Running a test

```bash
python run_flap_test.py
```

This connects to the arm and sensor, moves to `BASE_POSE`, zeros the sensor,
runs the flap trajectory defined at the top of `run_flap_test.py`
(cycles, heave amplitude, pitch amplitude, frequency, phase offset), logs
data while the arm is actually moving, and saves + plots the results.
Stop early with `Ctrl+C` — this triggers `arm.emergency_stop()` before
exiting.

**Before your first real run**, double-check:
- `TCP_OFFSET` matches where the sensor/airfoil actually sits relative to the flange
- `BASE_POSE` is a safe, collision-free neutral pose
- `CAL_FILE` points at the correct sensor serial's `.cal` file

## Plotting a saved log

```bash
python plotter.py flap_test_log.csv
```

Produces 4 panels: kinematics (heave/pitch), sensor-frame forces, sensor-frame
torques, and tank-frame (world) forces (thrust/lateral/lift), rotated out of
the sensor's own frame using the logged pose at each sample — see
`frame_transform.py` for the rotation convention and an important caveat
about how the arm's base frame is assumed to line up with the tank's axes.

## Safety notes

- `arm.py` tracks an abort flag that trips on any nonzero xArm error code;
  motion commands are refused once tripped until `reset_abort()` is called
  after manually confirming it's safe.
- `ft_sensor.check_overload()` flags any axis within 90% of its rated max —
  wire this into an `emergency_stop()` call before running near expected
  load limits.
- `move_to(..., wait=False)` queues commands on the controller — stopping
  the Python script does **not** stop the arm by itself; both the
  `KeyboardInterrupt` and generic exception handlers in `run_flap_test.py`
  explicitly call `arm.emergency_stop()` for this reason.

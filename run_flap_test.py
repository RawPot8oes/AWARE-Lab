"""
run_flap_test.py  (servo-mode version)

Oscillatory motion test on the xArm with F/T logging.

  * Motion is streamed in servo mode (mode 1) with absolute-deadline timing.
  * The wrench logger runs at LOG_RATE_HZ. Each sample is stamped with the
    most recently COMMANDED pose (updated at RATE_HZ), which is smooth and
    is what plotter.py / frame_transform.py use to rotate forces into the
    tank frame. Log format is unchanged: <run>_log.csv, 13 columns.
  * Optionally (MEASURE_RATE_HZ > 0) a second thread polls the arm's
    ACTUAL pose with get_position() and saves it to <run>_meas.csv, so
    you can compare commanded vs. measured motion. Set it to 0 to disable.

Run:  python run_flap_test.py      Stop any time with Ctrl+C.
"""

import os
import time
import csv
import threading
import traceback
from arm import Arm
from trajectory import generate_oscillatory_trajectory
from ft_sensor import FTSensor
from plotter import plot_results


# ======================================================================
# CONFIG
# ======================================================================

# --- Arm setup ---
ARM_IP = "192.168.1.235"
TCP_OFFSET = [0, 0, 200, 0, 0, -151.8]
BASE_POSE = [525.0, 0.0, -100.0, 180.0, 0.0, 0]

# --- Sensor setup ---
CAL_FILE = "FT75200/FT75200.cal"
DAQ_CHANNELS = "Dev1/ai0:5"

# --- Motion / timing ---
SPEED = 1000        # mm/s, point-to-point moves only
MVACC = 3000        # mm/s^2, same
HOME_SPEED = 100    # mm/s, gentle return to BASE_POSE after the run
RATE_HZ = 200       # servo command rate
LOG_RATE_HZ = 500   # wrench logging rate
MEASURE_RATE_HZ = 50  # measured-pose polling rate (0 = off). Each poll is a
                      # network round trip; if sends start missing deadlines,
                      # lower this or set it to 0.

# --- Motion definition ---
N_CYCLES = 30
REF_FREQ_HZ = 1

# {dof_name: (amplitude, freq_hz, phase_deg)}  (mm for x/y/z, deg otherwise)
DOFS = {
    "x":     (100, REF_FREQ_HZ, 0),
    "pitch": (20,  REF_FREQ_HZ, 180),
}

# --- Pure yaw example ---
# N_CYCLES = 8
# REF_FREQ_HZ = 1.0
# DOFS = {"yaw": (20, 1.0, 0)}

# Warn if the first waypoint is farther than this from the arm's current
# pose (servo mode faults on big jumps).
START_JUMP_WARN_MM = 2.0
START_JUMP_WARN_DEG = 1.0

# --- Output ---
# Every run writes its files into OUTPUT_DIR (created if missing), named
# from the motion settings plus the run's start time, so nothing is ever
# overwritten. Example:
#   data/x100_pitch30_0.5Hz_20261011_201153_log.csv
OUTPUT_DIR = "data"

LOG_HEADER = ["t", "x", "y", "z", "roll", "pitch", "yaw",
              "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
MEAS_HEADER = ["t", "x", "y", "z", "roll", "pitch", "yaw"]

# ======================================================================
# End of CONFIG
# ======================================================================


def wait_until(deadline):
    """
    Wait until perf_counter() >= deadline without hogging the GIL: short
    sleeps, then yield for the final millisecond so other threads run.
    """
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(0.0005 if remaining > 0.001 else 0)


def guarded(name, fn, *args):
    """Thread wrapper: print a traceback instead of dying silently."""
    try:
        fn(*args)
    except Exception:
        print(f"[{name}] CRASHED:")
        traceback.print_exc()


def log_wrench(sensor, state, t0, rows, start, stop, rate_hz):
    """Records (t, latest commanded pose, wrench) at rate_hz until stop."""
    start.wait()
    print("[log] Wrench logging started")
    dt = 1.0 / rate_hz
    next_t = time.perf_counter()

    while not stop.is_set():
        pose = state["cmd_pose"]          # replaced atomically by the sender
        wrench = sensor.read_wrench()
        rows.append([time.perf_counter() - t0, *pose, *wrench])
        next_t += dt
        wait_until(next_t)

    print("[log] Wrench logging stopped")


def poll_measured_pose(arm, t0, rows, start, stop, rate_hz):
    """Records (t, actual pose from the controller) at rate_hz until stop."""
    start.wait()
    print("[meas] Pose polling started")
    dt = 1.0 / rate_hz
    next_t = time.perf_counter()

    while not stop.is_set():
        pos = arm.get_position()
        if pos is not None:
            rows.append([time.perf_counter() - t0, *pos])
        next_t += dt
        wait_until(next_t)

    print("[meas] Pose polling stopped")


def make_run_prefix():
    """Unique file prefix: motion settings + start time, inside OUTPUT_DIR."""
    dof_tag = "_".join(f"{name}{amp:g}" for name, (amp, _f, _p) in DOFS.items())
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f"{dof_tag}_{REF_FREQ_HZ:g}Hz_{stamp}")


def save_rows(rows, header, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[run] Saved {len(rows)} rows to {path}")


def save_with_fallback(rows, header, path):
    try:
        save_rows(rows, header, path)
    except PermissionError:
        fallback = path.replace(".csv", f"_{int(time.time())}.csv")
        print(f"[run] {path} is locked -- saving to {fallback} instead.")
        save_rows(rows, header, fallback)


def check_start_pose(arm, first_pose):
    """Print how far the arm is from the first waypoint (servo jump check)."""
    cur = arm.get_position()
    if cur is None:
        print("[run] WARNING: could not read current pose for start check.")
        return
    dpos = max(abs(cur[i] - first_pose[i]) for i in range(3))
    dang = max(abs(cur[i] - first_pose[i]) for i in range(3, 6))
    print(f"[run] Start check: max pos offset {dpos:.2f} mm, "
          f"max angle offset {dang:.2f} deg")
    if dpos > START_JUMP_WARN_MM or dang > START_JUMP_WARN_DEG:
        print("[run] WARNING: first waypoint differs from current pose -- "
              "servo mode may fault on the initial jump.")


def main():
    run_prefix = make_run_prefix()
    arm = Arm(ARM_IP)
    sensor = FTSensor(cal_file_path=CAL_FILE, channels=DAQ_CHANNELS)

    # Created before the try block so cleanup can always reference them.
    log_rows = []
    meas_rows = []
    state = {"cmd_pose": list(BASE_POSE)}   # latest commanded pose
    start_logging = threading.Event()
    stop_logging = threading.Event()
    threads = []
    servo_active = False

    try:
        # --- Arm setup (mode 0, point-to-point) ---
        arm.connect()
        arm.set_tcp_offset(TCP_OFFSET)
        arm.move_to(*BASE_POSE, speed=SPEED, mvacc=MVACC, wait=True)
        if arm.is_aborted():
            print("[run] Arm is in an error state after setup -- not starting. "
                  "Check the error code printed above.")
            return

        # --- Sensor setup ---
        sensor.connect()
        sensor.zero()
        sensor.start_streaming()

        # --- Trajectory ---
        poses = generate_oscillatory_trajectory(
            base_pose=BASE_POSE,
            n_cycles=N_CYCLES,
            ref_freq_hz=REF_FREQ_HZ,
            rate_hz=RATE_HZ,
            dofs=DOFS,
        )
        dt = 1.0 / RATE_HZ
        print(f"[run] {len(poses)} poses at {RATE_HZ} Hz "
              f"(~{len(poses) * dt:.1f} s). Press Ctrl+C to stop.")

        check_start_pose(arm, poses[0])

        # --- Logger threads (wait for start_logging) ---
        run_start = time.perf_counter()

        threads.append(threading.Thread(
            target=guarded,
            args=("log", log_wrench, sensor, state, run_start, log_rows,
                  start_logging, stop_logging, LOG_RATE_HZ),
            daemon=True))
        if MEASURE_RATE_HZ > 0:
            threads.append(threading.Thread(
                target=guarded,
                args=("meas", poll_measured_pose, arm, run_start, meas_rows,
                      start_logging, stop_logging, MEASURE_RATE_HZ),
                daemon=True))
        for th in threads:
            th.start()

        # --- Stream the motion in servo mode ---
        arm.enter_servo_mode()
        servo_active = True

        start_logging.set()
        t_start = time.perf_counter()
        late_count = 0

        for i, pose in enumerate(poses):
            if arm.is_aborted():
                print(f"[run] Arm aborted at pose {i}/{len(poses)} -- stopping")
                break

            ret = arm.servo_to(pose)
            state["cmd_pose"] = pose
            if ret != 0:
                print(f"[run] servo_to returned {ret} at pose {i}")

            deadline = t_start + (i + 1) * dt
            if time.perf_counter() > deadline:
                late_count += 1
            wait_until(deadline)

        stop_logging.set()
        for th in threads:
            th.join(timeout=5.0)

        if late_count:
            print(f"[run] NOTE: {late_count} of {len(poses)} sends missed "
                  f"their deadline. If this is more than ~1-2%, lower "
                  f"MEASURE_RATE_HZ / LOG_RATE_HZ / RATE_HZ.")

        # --- Leave servo mode; return home unless the arm faulted ---
        arm.exit_servo_mode()
        servo_active = False
        if not arm.is_aborted():
            print("[run] Returning to BASE_POSE...")
            arm.move_to(*BASE_POSE, speed=HOME_SPEED, mvacc=MVACC, wait=True)
        else:
            print("[run] Arm is in an error state -- NOT returning home. "
                  "Check the error code, then reset_abort() when safe.")

        print(f"[run] Done. Wall-clock {time.perf_counter() - run_start:.2f}s, "
              f"{len(log_rows)} wrench samples, {len(meas_rows)} pose samples")

    except KeyboardInterrupt:
        print("\n[run] Ctrl+C received -- stopping arm.")
        arm.emergency_stop()

    except Exception as e:
        print(f"\n[run] Unexpected error: {e!r} -- stopping arm.")
        arm.emergency_stop()
        raise

    finally:
        print("[run] Cleaning up.")
        stop_logging.set()
        for th in threads:
            th.join(timeout=2.0)
        if servo_active:
            try:
                arm.exit_servo_mode()
            except Exception as e:
                print(f"[run] Could not exit servo mode: {e!r}")
        sensor.stop_streaming()
        sensor.disconnect()
        arm.disconnect()

    # --- Save data (outside the safety try block on purpose) ---
    if log_rows:
        save_with_fallback(log_rows, LOG_HEADER, f"{run_prefix}_log.csv")
        if meas_rows:
            save_with_fallback(meas_rows, MEAS_HEADER, f"{run_prefix}_meas.csv")
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            plot_results(log_rows, save_path=f"{run_prefix}_plot.png")
        except Exception as e:
            print(f"[run] Could not display plot: {e!r}")
    else:
        print("[run] No data was logged -- nothing to save.")


if __name__ == "__main__":
    main()
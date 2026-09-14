"""
run_flap_test.py

Main script for running a heave+pitch flap test on the xArm while
logging force/torque data from the ATI Mini40 sensor.

Data collection is now EVENT-DRIVEN rather than tied to a fixed pose
count: a separate logging thread watches the arm's actual motion state
(via get_cmdnum() -- how many queued commands are left to execute) and
only records while the arm is really moving, starting when motion
begins and stopping once the queue is confirmed drained. This means the
number of samples collected depends on how long the arm actually took
to move, not on how many poses we happened to pre-generate.

What this script does, in order:
  1. Connects to the arm and the F/T sensor.
  2. Moves the arm to a fixed starting pose (BASE_POSE).
  3. Zeros the F/T sensor while the arm is sitting still there.
  4. Generates a list of flap poses (the sine-wave motion).
  5. Starts a background logging thread that waits for real motion to
     begin, then records (time, wrench) continuously until the arm's
     queue is confirmed empty.
  6. Sends the poses to the arm.
  7. Waits for the arm to actually finish (forces the queue to drain).
  8. Saves the logged data to a CSV.
  9. Cleans up both devices whether the run finished normally, was
     stopped with Ctrl+C, or hit an error.

Run it with:
    python run_flap_test.py
Stop it at any time with Ctrl+C.
"""

import time
import csv
import threading
import math
from arm import Arm
from trajectory import generate_flap_trajectory
from ft_sensor import FTSensor
from plotter import plot_results

# ----------------------------------------------------------------------
# Arm setup
# ----------------------------------------------------------------------
ARM_IP = "192.168.1.235"

# TCP offset moves the arm's "tool point" from the bare flange out to
# wherever the sensor/airfoil actually sits. Set once, matches physical
# mounting -- see arm.py's set_tcp_offset() for details.
TCP_OFFSET = [0, 0, 200, 0, 0, 0]

# Neutral pose the flap motion oscillates around: [x, y, z, roll, pitch, yaw].
# Jog the arm to your desired starting position/orientation and read this
# back with arm.get_position() if you need to find your own values.
BASE_POSE = [400.0, 0.0, -300.0, 180.0, 0.0, 0.0]

# ----------------------------------------------------------------------
# Sensor setup
# ----------------------------------------------------------------------
CAL_FILE = "FT75200/FT75200.cal"   # path to your sensor's calibration file
DAQ_CHANNELS = "Dev1/ai0:5"         # NI channel string for the 6 gauge inputs

# ----------------------------------------------------------------------
# Motion / timing parameters
# ----------------------------------------------------------------------
SPEED = 500      # mm/s, arm default move speed
MVACC = 2000     # mm/s^2, arm default move acceleration
RATE_HZ = 100    # how many poses/sec we command to the arm. This is NOT
                 # the logging rate anymore (see LOG_RATE_HZ below), and
                 # NOT the DAQ's own internal sample_rate_hz (set in
                 # FTSensor) -- three separate rates, three separate jobs.
LOG_RATE_HZ = 500   # how often the logging thread records a sample while
                     # the arm is confirmed moving


def log_while_moving(arm, sensor, t0, log_rows, log_lock, done_sending,
                      log_rate_hz=LOG_RATE_HZ, poll_interval_s=0.02):
    """
    Runs in its own thread. Waits for the arm to actually start moving
    (get_cmdnum() > 0), then records (timestamp, wrench) continuously at
    log_rate_hz until the arm's command queue is confirmed drained
    (get_cmdnum() == 0) -- i.e. logging is bounded by REAL physical
    motion, not by how many poses we happened to generate ahead of time.
    """
    dt = 1.0 / log_rate_hz

    # Wait for motion to actually begin. Without this, we might check
    # get_cmdnum() before the very first command has been sent/
    # registered, see 0, and quit immediately having logged nothing.
    while arm.get_cmdnum() == 0 and not done_sending.is_set():
        time.sleep(poll_interval_s)

    print("[log] Motion detected -- starting to log")

    while arm.is_moving() or not done_sending.is_set():
        loop_start = time.perf_counter() # Record the start time of this loop iteration

        pos = arm.get_position()
        wrench = sensor.read_wrench()
        t = time.perf_counter() - t0
        with log_lock:
            log_rows.append([t, *pos, *wrench])

        elapsed = time.perf_counter() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)  # Sleep only for the remaining time to maintain the desired logging rate

    print("[log] Motion stopped -- ending log")


def save_log(rows, path="flap_test_log.csv"):
    """Writes collected rows to a CSV: time, position, then wrench."""
    header = ["t", "x", "y", "z", "roll", "pitch", "yaw",
              "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[run] Saved {len(rows)} rows to {path}")


def main():
    arm = Arm(ARM_IP)
    sensor = FTSensor(cal_file_path=CAL_FILE, channels=DAQ_CHANNELS)

    # Declared here (not inside the try block) so it's still accessible
    # afterward even if something goes wrong mid-run -- we don't want to
    # lose already-collected data just because of an error later on.
    log_rows = []


    try:
        # --- Arm setup ---
        arm.connect()
        arm.set_tcp_offset(TCP_OFFSET)
        # wait=True here: we want the arm to actually finish arriving at
        # BASE_POSE before we zero the sensor and start the fast loop.
        arm.move_to(*BASE_POSE, speed=SPEED, mvacc=MVACC, wait=True)

        # --- Sensor setup ---
        sensor.connect()
        sensor.zero()   # arm is at rest at BASE_POSE here -- good, repeatable
                         # moment to zero out any resting offset/tare
        sensor.start_streaming()   # starts the background DAQ read thread

        # --- Build the trajectory ---
        # All the sine-wave math lives in trajectory.py -- this just asks
        # for a list of poses given the parameters below.
        poses = generate_flap_trajectory(
            base_pose=BASE_POSE,
            n_cycles=5,
            heave_amp_mm=100,
            pitch_amp_deg=30,
            freq_hz=1.5,
            phase_offset_deg=180,
            rate_hz=RATE_HZ,
        )

        print(f"[run] Sending {len(poses)} poses. Press Ctrl+C to stop.")
        run_start = time.perf_counter()

        # Start the logger in its own thread -- it will wait for motion
        # to begin, log continuously while the arm is confirmed moving,
        # and stop once the queue is confirmed drained. This runs
        # concurrently with the pose-sending loop below.
        log_lock = threading.Lock()
        done_sending = threading.Event()

        logger_thread = threading.Thread(
            target=log_while_moving,
            args=(arm, sensor, run_start, log_rows, log_lock, done_sending),
            daemon=True,
        )
        logger_thread.start()

        # Send the poses. No sensor reading happens here anymore -- the
        # logger thread above handles that independently now.
        dt = 1.0 / RATE_HZ
        prev_pose = BASE_POSE # Keep track of the previous pose

        for pose in poses:
            if arm.is_aborted():
                print("[run] Arm aborted (error state) -- stopping")
                break

            # Calculate distance of next step
            dx = pose[0] - prev_pose[0]
            dy = pose[1] - prev_pose[1]
            dz = pose[2] - prev_pose[2]
            distance = math.sqrt(dx**2 + dy**2 + dz**2)

            # Calculate required speed to reach the next pose in the desired time
            dynamic_speed = max(0.1, distance / dt)  # Avoid zero speed

            t1 = time.perf_counter()

            arm.move_to(*pose, speed=dynamic_speed, mvacc=MVACC, radius=0, wait=False)
            elapsed = time.perf_counter() - t1
            if elapsed < dt:
                time.sleep(dt - elapsed)

            prev_pose = pose  # Update previous pose for the next iteration

        done_sending.set()  # signal the logger thread that we're done sending poses

        # Force-drain the queue and confirm the arm is really done. This
        # also gives the logger thread's is_moving() check a clean,
        # definite point at which to stop, and returns the arm to a
        # known, safe resting pose.
        print("[run] Waiting for arm to finish physically executing queued moves...")
        arm.move_to(*BASE_POSE, speed=SPEED, mvacc=MVACC, wait=True)
        print("[run] Arm confirmed finished.")

        logger_thread.join(timeout=5.0)   # let the logger thread notice and exit
        run_end = time.perf_counter()
        print(f"[run] Done. Wall-clock duration: {run_end - run_start:.2f}s, "
              f"{len(log_rows)} samples logged")

        save_log(log_rows)

    except KeyboardInterrupt:
        # Ctrl+C lands here. IMPORTANT: this only stops the Python loop
        # from sending new commands -- it does NOT stop the arm itself.
        # move_to(wait=False) queues commands on the controller, which can
        # keep executing already-queued moves even after we stop sending
        # more. We have to explicitly halt it.
        print("\n[run] Ctrl+C received -- stopping arm.")
        arm.emergency_stop()

    except Exception as e:
        # Anything else going wrong (network hiccup, sensor read error,
        # unexpected bug, etc.) lands here. Without this, an uncaught
        # exception would print a traceback and exit WITHOUT stopping the
        # arm -- and since move_to(wait=False) queues commands on the
        # controller, the arm keeps executing its queued moves even after
        # Python has crashed and disconnected. Always stop the arm first.
        print(f"\n[run] Unexpected error: {e!r} -- stopping arm.")
        arm.emergency_stop()
        raise  # re-raise so you still see the full traceback for debugging

    finally:
        # Always runs -- normal finish, Ctrl+C, or an unhandled exception
        # -- so cleanup happens no matter what.
        print("[run] Cleaning up.")
        sensor.stop_streaming()
        sensor.disconnect()
        arm.disconnect()

 
    # --- Save the data (outside the arm-safety try block on purpose) ---
    # If this fails (e.g. PermissionError because the file is open
    # elsewhere), we don't want that to look like an arm/motion failure,
    # and we don't want to lose the data we already collected. Try the
    # normal filename first, then fall back to a timestamped one so a
    # locked file doesn't cost you the whole run's data.
    if log_rows:
        try:
            save_log(log_rows)
        except PermissionError:
            fallback_path = f"flap_test_log_{int(time.time())}.csv"
            print(f"[run] flap_test_log.csv is locked/unwritable "
                  f"(probably open in another program) -- "
                  f"saving to {fallback_path} instead.")
            save_log(log_rows, path=fallback_path)
 
        try:
            plot_results(log_rows)
        except Exception as e:
            # A plotting problem (e.g. no display available, matplotlib
            # not installed) shouldn't be treated as a failed run -- the
            # data is already safely saved to CSV above regardless.
            print(f"[run] Could not display plot: {e!r}")
    else:
        print("[run] No data was logged -- nothing to save.")
 
 
if __name__ == "__main__":
    main()
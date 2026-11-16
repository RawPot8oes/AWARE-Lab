"""
run_step_test.py

Static step-hold test on the xArm with F/T logging.
Steps through a range of angles for one DOF, holding at each position
for HOLD_S seconds while recording wrench data.

All output files land in a timestamped folder under data/ — see
data_manager.py for the naming scheme.

Folder name example:
    data/20261116-154402_step-yaw_s-20_e20_by1_h30/
         (step test on yaw, -20→+20 deg in 1 deg steps, 30 s holds)

Run:  python run_step_test.py      Stop any time with Ctrl+C.
"""

import time
import math
import threading
from arm import Arm
from trajectory import generate_step_hold_poses
from ft_sensor import FTSensor
from plotter import plot_results
from data_manager import create_run

# ======================================================================
# CONFIG
# ======================================================================

# --- Comments (optional) -----------------------------------------------
# Fill this in before you run. It appears on line 1 of every CSV and in
# meta.json so future-you remembers the context of this run.
COMMENTS = ""   # e.g. "yaw sweep, calm water, no turbulence"

# --- Arm setup ---------------------------------------------------------
ARM_IP     = "192.168.1.235"
TCP_OFFSET = [0, 0, 200, 0, 0, 0]
BASE_POSE  = [558.4, 23.0, -314.7, 180.0, 0.0, -42.8]

# --- Sensor setup ------------------------------------------------------
CAL_FILE     = "FT75200/FT75200.cal"
DAQ_CHANNELS = "Dev1/ai0:5"

# --- Motion / timing ---------------------------------------------------
SPEED       = 100
MVACC       = 500
RATE_HZ     = 100
LOG_RATE_HZ = 500
MIN_SPEED   = 50     # floor so rotation-only moves don't get speed ≈ 0

# --- Step test parameters ----------------------------------------------
STEP_DOF  = "yaw"
START_DEG = -20
STOP_DEG  =  20
STEP_DEG  =   1
HOLD_S    =  30   # set to 2 for a dry run first!

# --- CSV column header -------------------------------------------------
LOG_HEADER = ["t", "x", "y", "z", "roll", "pitch", "yaw",
              "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

# ======================================================================
# End of CONFIG
# ======================================================================


def log_while_moving(arm, sensor, t0, log_rows, log_lock, done_sending,
                     log_rate_hz=LOG_RATE_HZ, poll_interval_s=0.02):
    dt = 1.0 / log_rate_hz
    while arm.get_cmdnum() == 0 and not done_sending.is_set():
        time.sleep(poll_interval_s)
    print("[log] Motion detected — starting to log")
    while arm.is_moving() or not done_sending.is_set():
        loop_start = time.perf_counter()
        pos    = arm.get_position()
        wrench = sensor.read_wrench()
        t      = time.perf_counter() - t0
        with log_lock:
            log_rows.append([t, *pos, *wrench])
        elapsed = time.perf_counter() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)
    print("[log] Motion stopped — ending log")


def main():
    # ------------------------------------------------------------------
    # Create the run folder via data_manager.
    # The DOF name is encoded in the test type ("step-yaw") so it appears
    # in the folder name without breaking the tag-number token format.
    # ------------------------------------------------------------------
    run = create_run(
        test=f"step-{STEP_DOF.lower()}",   # e.g. "step-yaw"
        params={
            "dof":         STEP_DOF,
            "start_deg":   START_DEG,
            "stop_deg":    STOP_DEG,
            "step_deg":    STEP_DEG,
            "hold_s":      HOLD_S,
            "rate_hz":     RATE_HZ,
            "log_rate_hz": LOG_RATE_HZ,
        },
        name_tags={
            "s":  START_DEG,   # start angle (deg)
            "e":  STOP_DEG,    # end angle   (deg)
            "by": STEP_DEG,    # step size   (deg)
            "h":  HOLD_S,      # hold time   (s)
        },
        comments=COMMENTS,
        config={
            "BASE_POSE":    BASE_POSE,
            "TCP_OFFSET":   TCP_OFFSET,
            "SPEED":        SPEED,
            "MVACC":        MVACC,
            "RATE_HZ":      RATE_HZ,
            "LOG_RATE_HZ":  LOG_RATE_HZ,
            "CAL_FILE":     CAL_FILE,
            "DAQ_CHANNELS": DAQ_CHANNELS,
        },
    )

    # ------------------------------------------------------------------
    arm      = Arm(ARM_IP)
    sensor   = FTSensor(cal_file_path=CAL_FILE, channels=DAQ_CHANNELS)
    log_rows = []
    run_start = time.perf_counter()

    try:
        arm.connect()
        arm.set_tcp_offset(TCP_OFFSET)
        print("[run] Position after TCP offset:", arm.get_position())

        arm.move_to(*BASE_POSE, speed=SPEED, mvacc=MVACC, wait=True)
        print("[run] At base pose:", arm.get_position())

        sensor.connect()
        sensor.zero()
        sensor.start_streaming()
        print("[run] Sensor streaming.")

        poses = generate_step_hold_poses(
            base_pose=BASE_POSE,
            dof=STEP_DOF,
            start=START_DEG,
            stop=STOP_DEG,
            step=STEP_DEG,
            hold_s=HOLD_S,
            rate_hz=RATE_HZ,
        )
        print(f"[run] {len(poses)} poses "
              f"(~{len(poses) / RATE_HZ / 60:.1f} min). Press Ctrl+C to stop.")

        run_start  = time.perf_counter()
        log_lock   = threading.Lock()
        done_flag  = threading.Event()
        logger     = threading.Thread(
            target=log_while_moving,
            args=(arm, sensor, run_start, log_rows, log_lock, done_flag),
            daemon=True,
        )
        logger.start()

        dt        = 1.0 / RATE_HZ
        prev_pose = list(BASE_POSE)
        for pose in poses:
            if arm.is_aborted():
                print(f"[run] Arm aborted — stopping. Position: {arm.get_position()}")
                break
            dx      = pose[0] - prev_pose[0]
            dy      = pose[1] - prev_pose[1]
            dz      = pose[2] - prev_pose[2]
            dist    = math.sqrt(dx**2 + dy**2 + dz**2)
            dyn_spd = max(MIN_SPEED, dist / dt)
            t1      = time.perf_counter()
            arm.move_to(*pose, speed=dyn_spd, mvacc=MVACC, radius=0, wait=False)
            elapsed = time.perf_counter() - t1
            if elapsed < dt:
                time.sleep(dt - elapsed)
            prev_pose = pose

        done_flag.set()
        print("[run] Waiting for arm to finish queued moves …")
        arm.move_to(*BASE_POSE, speed=SPEED, mvacc=MVACC, wait=True)
        print("[run] Arm confirmed finished.")

        logger.join(timeout=5.0)
        elapsed = time.perf_counter() - run_start
        print(f"[run] Done. {elapsed:.2f} s, {len(log_rows)} samples logged.")

    except KeyboardInterrupt:
        print("\n[run] Ctrl+C — stopping arm.")
        arm.emergency_stop()

    except Exception as e:
        print(f"\n[run] Unexpected error: {e!r} — stopping arm.")
        arm.emergency_stop()
        raise

    finally:
        print("[run] Cleaning up.")
        sensor.stop_streaming()
        sensor.disconnect()
        arm.disconnect()

    # ------------------------------------------------------------------
    # Save data
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - run_start
    run.update_meta(
        duration_s=round(elapsed, 2),
        n_log_samples=len(log_rows),
    )

    if log_rows:
        run.save_csv_with_fallback("log.csv", log_rows, LOG_HEADER)
        try:
            plot_results(log_rows, save_path=run.path("plot.png"), show=False)
        except Exception as e:
            print(f"[run] Could not save plot: {e!r}")
    else:
        print("[run] No data logged — nothing to save.")


if __name__ == "__main__":
    main()
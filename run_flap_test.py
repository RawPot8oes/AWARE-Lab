"""
run_flap_test.py  (servo-mode version)

Oscillatory motion test on the xArm with F/T logging.

  * Motion is streamed in servo mode (mode 1) with absolute-deadline timing.
  * The wrench logger runs at LOG_RATE_HZ.  Each sample is stamped with the
    most recently COMMANDED pose (updated at RATE_HZ).
  * Optionally (MEASURE_RATE_HZ > 0) a second thread polls the arm's ACTUAL
    pose and saves it to meas.csv.  Set to 0 to disable.
  * All output files land in a timestamped folder under data/ — see
    data_manager.py for the naming scheme.

Run:  python run_flap_test.py      Stop any time with Ctrl+C.
"""

import time
import threading
import traceback
from arm import Arm
from trajectory import generate_oscillatory_trajectory
from ft_sensor import FTSensor
from plotter import plot_results
from data_manager import create_run
from phase_average import generate_phase_avg

# ======================================================================
# CONFIG
# ======================================================================

# --- Comments (optional) -----------------------------------------------
# Fill this in before you run. It appears on line 1 of every CSV and in
# meta.json so future-you remembers the context of this run.
COMMENTS = ""   # e.g. "baseline, tank at rest, no turbulence grid"

# --- Arm setup ---------------------------------------------------------
ARM_IP     = "192.168.1.235"
TCP_OFFSET = [0, 0, 200, 0, 0, -151.8]
BASE_POSE  = [525.0, 0.0, -100.0, 180.0, 0.0, 0]

# --- Sensor setup ------------------------------------------------------
CAL_FILE     = "FT75200/FT75200.cal"
DAQ_CHANNELS = "Dev1/ai0:5"

# --- Motion / timing ---------------------------------------------------
SPEED           = 1000    # mm/s  (point-to-point moves only)
MVACC           = 3000    # mm/s²
HOME_SPEED      = 100     # mm/s, gentle return after the run
RATE_HZ         = 200     # servo command rate
LOG_RATE_HZ     = 500     # wrench logging rate
MEASURE_RATE_HZ = 50      # actual-pose polling rate (0 = off)

# --- Motion definition -------------------------------------------------
N_CYCLES    = 30
REF_FREQ_HZ = 1

# {dof_name: (amplitude, freq_hz, phase_deg)}  mm for x/y/z, deg otherwise
DOFS = {
    "x":     (100, REF_FREQ_HZ,   0),
    "pitch": ( 20, REF_FREQ_HZ, 180),
}

# Warn if the first waypoint is farther than this from current pose
START_JUMP_WARN_MM  = 2.0
START_JUMP_WARN_DEG = 1.0

# --- CSV column headers ------------------------------------------------
LOG_HEADER  = ["t", "x", "y", "z", "roll", "pitch", "yaw",
               "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
MEAS_HEADER = ["t", "x", "y", "z", "roll", "pitch", "yaw"]

# ======================================================================
# End of CONFIG
# ======================================================================


def wait_until(deadline):
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(0.0005 if remaining > 0.001 else 0)


def guarded(name, fn, *args):
    try:
        fn(*args)
    except Exception:
        print(f"[{name}] CRASHED:")
        traceback.print_exc()


def log_wrench(sensor, state, t0, rows, start, stop, rate_hz):
    """Records (t, latest commanded pose, wrench) at rate_hz until stop."""
    start.wait()
    print("[log] Wrench logging started")
    dt     = 1.0 / rate_hz
    next_t = time.perf_counter()
    while not stop.is_set():
        pose   = state["cmd_pose"]
        wrench = sensor.read_wrench()
        rows.append([time.perf_counter() - t0, *pose, *wrench])
        next_t += dt
        wait_until(next_t)
    print("[log] Wrench logging stopped")


def poll_measured_pose(arm, t0, rows, start, stop, rate_hz):
    """Records (t, actual pose from controller) at rate_hz until stop."""
    start.wait()
    print("[meas] Pose polling started")
    dt     = 1.0 / rate_hz
    next_t = time.perf_counter()
    while not stop.is_set():
        pos = arm.get_position()
        if pos is not None:
            rows.append([time.perf_counter() - t0, *pos])
        next_t += dt
        wait_until(next_t)
    print("[meas] Pose polling stopped")


def check_start_pose(arm, first_pose):
    cur = arm.get_position()
    if cur is None:
        print("[run] WARNING: could not read current pose for start check.")
        return
    dpos = max(abs(cur[i] - first_pose[i]) for i in range(3))
    dang = max(abs(cur[i] - first_pose[i]) for i in range(3, 6))
    print(f"[run] Start check: max pos offset {dpos:.2f} mm, "
          f"max angle offset {dang:.2f} deg")
    if dpos > START_JUMP_WARN_MM or dang > START_JUMP_WARN_DEG:
        print("[run] WARNING: first waypoint differs from current pose — "
              "servo mode may fault on the initial jump.")


def main():
    # ------------------------------------------------------------------
    # Create the run folder via data_manager
    # Pull amplitudes/phases out of DOFS safely so changing the DOFS
    # dict above never requires editing this block.
    # ------------------------------------------------------------------
    heave_amp = DOFS.get("x",     (0,)      )[0]
    pitch_amp = DOFS.get("pitch", (0,)      )[0]
    phase_off = DOFS.get("pitch", (0, 0, 0) )[2]

    run = create_run(
        test="flap",
        params={
            "freq_hz":          REF_FREQ_HZ,
            "n_cycles":         N_CYCLES,
            "heave_amp_mm":     heave_amp,
            "pitch_amp_deg":    pitch_amp,
            "phase_offset_deg": phase_off,
            "rate_hz":          RATE_HZ,
            "log_rate_hz":      LOG_RATE_HZ,
            "measure_rate_hz":  MEASURE_RATE_HZ,
        },
        name_tags={
            "f":  REF_FREQ_HZ,   # frequency  (Hz)
            "h":  heave_amp,     # heave amplitude (mm)
            "p":  pitch_amp,     # pitch amplitude (deg)
            "ph": phase_off,     # phase offset (deg)
            "n":  N_CYCLES,      # number of cycles
        },
        comments=COMMENTS,
        config={
            "BASE_POSE":    BASE_POSE,
            "TCP_OFFSET":   TCP_OFFSET,
            "SPEED":        SPEED,
            "MVACC":        MVACC,
            "HOME_SPEED":   HOME_SPEED,
            "RATE_HZ":      RATE_HZ,
            "LOG_RATE_HZ":  LOG_RATE_HZ,
            "CAL_FILE":     CAL_FILE,
            "DAQ_CHANNELS": DAQ_CHANNELS,
            "dofs":         {k: list(v) for k, v in DOFS.items()},
        },
    )

    # ------------------------------------------------------------------
    arm    = Arm(ARM_IP)
    sensor = FTSensor(cal_file_path=CAL_FILE, channels=DAQ_CHANNELS)

    log_rows      = []
    meas_rows     = []
    state         = {"cmd_pose": list(BASE_POSE)}
    start_logging = threading.Event()
    stop_logging  = threading.Event()
    threads       = []
    servo_active  = False
    run_start     = time.perf_counter()   # real value set once motion begins

    try:
        # --- Arm setup ---
        arm.connect()
        arm.set_tcp_offset(TCP_OFFSET)
        arm.move_to(*BASE_POSE, speed=SPEED, mvacc=MVACC, wait=True)
        if arm.is_aborted():
            print("[run] Arm in error state after setup — aborting.")
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

        # --- Logger threads ---
        run_start = time.perf_counter()
        threads.append(threading.Thread(
            target=guarded,
            args=("log", log_wrench, sensor, state, run_start,
                  log_rows, start_logging, stop_logging, LOG_RATE_HZ),
            daemon=True,
        ))
        if MEASURE_RATE_HZ > 0:
            threads.append(threading.Thread(
                target=guarded,
                args=("meas", poll_measured_pose, arm, run_start,
                      meas_rows, start_logging, stop_logging, MEASURE_RATE_HZ),
                daemon=True,
            ))
        for th in threads:
            th.start()

        # --- Stream motion in servo mode ---
        arm.enter_servo_mode()
        servo_active = True
        start_logging.set()

        t_start    = time.perf_counter()
        late_count = 0
        for i, pose in enumerate(poses):
            if arm.is_aborted():
                print(f"[run] Arm aborted at pose {i}/{len(poses)} — stopping.")
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
            print(f"[run] NOTE: {late_count}/{len(poses)} sends missed their "
                  "deadline. Consider lowering MEASURE_RATE_HZ / LOG_RATE_HZ.")

        arm.exit_servo_mode()
        servo_active = False
        if not arm.is_aborted():
            print("[run] Returning to BASE_POSE …")
            arm.move_to(*BASE_POSE, speed=HOME_SPEED, mvacc=MVACC, wait=True)
        else:
            print("[run] Arm in error state — NOT returning home.")

        elapsed = time.perf_counter() - run_start
        print(f"[run] Done. {elapsed:.2f} s, "
              f"{len(log_rows)} wrench samples, {len(meas_rows)} pose samples.")

    except KeyboardInterrupt:
        print("\n[run] Ctrl+C — stopping arm.")
        arm.emergency_stop()

    except Exception as e:
        print(f"\n[run] Unexpected error: {e!r} — stopping arm.")
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

    # ------------------------------------------------------------------
    # Save data (outside the safety try block on purpose)
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - run_start
    run.update_meta(
        duration_s=round(elapsed, 2),
        n_log_samples=len(log_rows),
        n_meas_samples=len(meas_rows),
    )

    if log_rows:
        run.save_csv_with_fallback("log.csv",  log_rows,  LOG_HEADER)
        if meas_rows:
            run.save_csv_with_fallback("meas.csv", meas_rows, MEAS_HEADER)

        # --- Overview plot (plotter.py) ---
        try:
            plot_results(log_rows, save_path=run.path("plot.png"), show=False)
        except Exception as e:
            print(f"[run] Could not save overview plot: {e!r}")

        # --- Phase-averaged plot (phase_average.py) ---
        try:
            generate_phase_avg(
                filepath=str(run.path("log.csv")),
                out_dir=str(run.dir),
                freq_hz=REF_FREQ_HZ,
            ) 
        except Exception as e:
            print(f"[run] Could not save phase-averaged plot: {e!r}")

    else:
        print("[run] No data logged — nothing to save.")


if __name__ == "__main__":
    main()
import time
import csv
import threading
import math
from arm import Arm
from trajectory import generate_step_hold_poses
from ft_sensor import FTSensor
from plotter import plot_results

# ======================= CONFIG =======================
ARM_IP = "192.168.1.235"
TCP_OFFSET = [0, 0, 200, 0, 0, 0]   # must match what the UI is using!
BASE_POSE = [558.4, 23.0, -314.7, 180.0, 0.0, -42.8]

CAL_FILE = "FT75200/FT75200.cal"
DAQ_CHANNELS = "Dev1/ai0:5"

SPEED = 100
MVACC = 500
RATE_HZ = 100
LOG_RATE_HZ = 500
MIN_SPEED = 50          # floor so rotation-only moves don't get speed ~0

# Step test: yaw offset from BASE_POSE
STEP_DOF = "yaw"
START_DEG = -20
STOP_DEG = 20
STEP_DEG = 1
HOLD_S = 30             # set to 2 for a dry run first!
# ======================================================


def log_while_moving(arm, sensor, t0, log_rows, log_lock, done_sending,
                     log_rate_hz=LOG_RATE_HZ, poll_interval_s=0.02):
    dt = 1.0 / log_rate_hz

    while arm.get_cmdnum() == 0 and not done_sending.is_set():
        time.sleep(poll_interval_s)

    print("[log] Motion detected -- starting to log")

    while arm.is_moving() or not done_sending.is_set():
        loop_start = time.perf_counter()

        pos = arm.get_position()
        wrench = sensor.read_wrench()
        t = time.perf_counter() - t0
        with log_lock:
            log_rows.append([t, *pos, *wrench])

        elapsed = time.perf_counter() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

    print("[log] Motion stopped -- ending log")


def save_log(rows, path="flap_test_log.csv"):
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
    log_rows = []

    try:
        arm.connect()
        arm.set_tcp_offset(TCP_OFFSET)
        print("[run] Position after TCP offset:", arm.get_position())

        arm.move_to(*BASE_POSE, speed=SPEED, mvacc=MVACC, wait=True)
        print("[run] At base pose:", arm.get_position())

        sensor.connect()
        sensor.zero()
        sensor.start_streaming()
        print("[run] Sensor streaming")

        poses = generate_step_hold_poses(
            base_pose=BASE_POSE,
            dof=STEP_DOF,
            start=START_DEG, stop=STOP_DEG, step=STEP_DEG,
            hold_s=HOLD_S,
            rate_hz=RATE_HZ,
        )

        print(f"[run] Sending {len(poses)} poses "
              f"(~{len(poses) / RATE_HZ / 60:.1f} min). Press Ctrl+C to stop.")
        run_start = time.perf_counter()

        log_lock = threading.Lock()
        done_sending = threading.Event()

        logger_thread = threading.Thread(
            target=log_while_moving,
            args=(arm, sensor, run_start, log_rows, log_lock, done_sending),
            daemon=True,
        )
        logger_thread.start()

        dt = 1.0 / RATE_HZ
        prev_pose = BASE_POSE

        for pose in poses:
            if arm.is_aborted():
                print(f"[run] Arm aborted -- stopping. Position: {arm.get_position()}")
                break

            dx = pose[0] - prev_pose[0]
            dy = pose[1] - prev_pose[1]
            dz = pose[2] - prev_pose[2]
            distance = math.sqrt(dx**2 + dy**2 + dz**2)
            dynamic_speed = max(MIN_SPEED, distance / dt)

            t1 = time.perf_counter()
            arm.move_to(*pose, speed=dynamic_speed, mvacc=MVACC,
                        radius=0, wait=False)
            elapsed = time.perf_counter() - t1
            if elapsed < dt:
                time.sleep(dt - elapsed)

            prev_pose = pose

        done_sending.set()

        print("[run] Waiting for arm to finish queued moves...")
        arm.move_to(*BASE_POSE, speed=SPEED, mvacc=MVACC, wait=True)
        print("[run] Arm confirmed finished.")

        logger_thread.join(timeout=5.0)
        run_end = time.perf_counter()
        print(f"[run] Done. Duration: {run_end - run_start:.2f}s, "
              f"{len(log_rows)} samples logged")

    except KeyboardInterrupt:
        print("\n[run] Ctrl+C received -- stopping arm.")
        arm.emergency_stop()

    except Exception as e:
        print(f"\n[run] Unexpected error: {e!r} -- stopping arm.")
        arm.emergency_stop()
        raise

    finally:
        print("[run] Cleaning up.")
        sensor.stop_streaming()
        sensor.disconnect()
        arm.disconnect()

    if log_rows:
        try:
            save_log(log_rows)
        except PermissionError:
            fallback_path = f"flap_test_log_{int(time.time())}.csv"
            print(f"[run] flap_test_log.csv locked -- saving to {fallback_path}")
            save_log(log_rows, path=fallback_path)

        try:
            plot_results(log_rows,
             min_hold_s=min(10.0, HOLD_S * 0.5),
             settle_s=min(5.0, HOLD_S * 0.25))
        except Exception as e:
            print(f"[run] Could not display plot: {e!r}")
    else:
        print("[run] No data was logged -- nothing to save.")


if __name__ == "__main__":
    main()
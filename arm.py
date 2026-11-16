"""
devices/arm.py

Wrapper around the UFactory xArm Python SDK (XArmAPI).

This class is the single point of contact between the rest of the test-rig
code and the physical robot arm. Nothing outside this file should import
`xarm` directly.

Responsibilities:
  1. Connect to the arm and put it in a known, safe state (clearing any
     stale fault left over from a previous session).
  2. Set the TCP offset so the tool point sits at the sensor/airfoil.
  3. Point-to-point moves (move_to, mode 0) for setup and homing.
  4. Servo streaming (mode 1) for oscillatory motion: enter_servo_mode(),
     then servo_to(pose) at a fixed rate, then exit_servo_mode().
  5. A shared "abort" flag, set the moment the arm reports an error, so the
     rest of the program stops instead of commanding a faulted arm.
"""

import time
import threading
from xarm.wrapper import XArmAPI


class Arm:
    def __init__(self, ip):
        """ip: IP address of the xArm controller box, e.g. '192.168.1.113'"""
        self.ip = ip
        self.arm = None  # XArmAPI instance, created in connect()

        # Thread-safe flag. The SDK's error callback runs on its own thread
        # and uses this to tell the main loop to stop.
        self.abort_event = threading.Event()

    # ------------------------------------------------------------------
    # Error callback (called by the SDK on its own thread)
    # ------------------------------------------------------------------
    def _on_error_warn(self, item):
        err = item['error_code']
        warn = item['warn_code']
        print(f"[Arm] ErrorCode: {err}, WarnCode: {warn}")

        if err != 0:
            # Every nonzero error trips the abort flag for now. Later you
            # may want to distinguish dangerous codes from informational ones.
            print("[Arm] Error detected -> setting abort flag")
            self.abort_event.set()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def connect(self):
        """
        Connects and puts the arm in a known-good state: errors cleared,
        motion enabled, mode 0 (position control), state 0 (ready).

        Does NOT set a TCP offset -- call set_tcp_offset() right after
        connect(), before recording or commanding any pose.
        """
        # do_not_open=True lets us register the error callback BEFORE any
        # traffic starts, so an early error isn't missed.
        self.arm = XArmAPI(self.ip, do_not_open=True)
        self.arm.register_error_warn_changed_callback(self._on_error_warn)
        self.arm.connect()

        print(f"[Arm] Connected. Firmware/version info: {self.arm.version}")

        self.arm.motion_enable(enable=True)
        self.arm.clean_error()
        # Re-enable: the first call can run while the arm is still faulted.
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        time.sleep(0.2)

        if self.arm.has_error:
            print(f"[Arm] Still in error after clean, code={self.arm.error_code}. "
                  f"Leaving abort flag set.")
        else:
            # Errors left over from a previous session were cleared above,
            # so drop the stale abort flag they raised.
            self.abort_event.clear()

    def disconnect(self):
        """Closes the connection to the arm controller."""
        if self.arm:
            self.arm.disconnect()

    # ------------------------------------------------------------------
    # Abort flag helpers
    # ------------------------------------------------------------------
    def is_aborted(self):
        """True if an error has been flagged and not yet cleared."""
        return self.abort_event.is_set()

    def reset_abort(self):
        """
        Call ONLY after confirming it's safe to continue. Clears both our
        abort flag and the arm's own error state.
        """
        self.abort_event.clear()
        self.arm.clean_error()

    # ------------------------------------------------------------------
    # Tool setup
    # ------------------------------------------------------------------
    def set_tcp_offset(self, offset, wait=True):
        """
        Moves the tool center point from the bare flange to the real tool.

        offset: [x, y, z, roll, pitch, yaw] from the flange, in mm / degrees.
        Call once after connect() and BEFORE recording a base pose.
        """
        ret = self.arm.set_tcp_offset(offset, wait=wait)
        if ret != 0:
            print(f"[Arm] set_tcp_offset failed, ret={ret}")
        self.arm.set_state(0)  # re-affirm ready state, per UFactory examples
        return ret

    # ------------------------------------------------------------------
    # Point-to-point motion (mode 0)
    # ------------------------------------------------------------------
    def move_to(self, x, y, z, roll, pitch, yaw, speed=50, mvacc=1000,
                radius=None, wait=True):
        """
        Moves to a single Cartesian pose (mm / degrees). Use for setup and
        homing, not for oscillatory streaming (use servo_to for that).

        speed : mm/s     mvacc : mm/s^2
        radius: None/<0 stops exactly at the point, >=0 blends through it
        wait  : True blocks until the move finishes

        Returns the SDK return code (0 = success).
        """
        if self.is_aborted():
            print("[Arm] Aborted, refusing move_to")
            return -1

        ret = self.arm.set_position(
            x=x, y=y, z=z,
            roll=roll, pitch=pitch, yaw=yaw,
            speed=speed, mvacc=mvacc, radius=radius,
            wait=wait
        )
        if ret != 0:
            print(f"[Arm] set_position failed, ret={ret}")
        return ret

    # ------------------------------------------------------------------
    # Servo streaming (mode 1)
    # ------------------------------------------------------------------
    def enter_servo_mode(self):
        """Switch to mode 1: streamed Cartesian setpoints, no command queue."""
        self.arm.set_mode(1)
        self.arm.set_state(0)
        time.sleep(0.1)

    def exit_servo_mode(self):
        """Back to mode 0 so move_to() works again."""
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.1)

    def servo_to(self, pose):
        """
        Streams one Cartesian setpoint [x, y, z, roll, pitch, yaw]
        (mm / degrees). Call at a fixed rate while in servo mode.
        Returns the SDK return code (0 = success), or -1 if aborted.
        """
        if self.is_aborted():
            return -1
        return self.arm.set_servo_cartesian(pose, is_radian=False)

    # ------------------------------------------------------------------
    # State readback
    # ------------------------------------------------------------------
    def get_position(self):
        """
        Current Cartesian pose [x, y, z, roll, pitch, yaw] queried from the
        controller (a network round trip, typically a few ms). Returns None
        if the query failed.

        Do NOT substitute arm.arm.position for this in a logger: that value
        is refreshed at the SDK's report rate (only a few Hz by default)
        and shows up as a staircase.
        """
        code, pos = self.arm.get_position()
        if code != 0:
            print(f"[Arm] get_position failed, code={code}")
            return None
        return pos

    def get_angles(self):
        """Current joint angles (degrees), or None if the query failed."""
        code, angles = self.arm.get_servo_angle()
        if code != 0:
            print(f"[Arm] get_servo_angle failed, code={code}")
            return None
        return angles

    def get_cmdnum(self):
        """
        Number of queued motion commands (mode 0 only; meaningless in servo
        mode, where there is no queue).
        """
        code, cmdnum = self.arm.get_cmdnum()
        if code != 0:
            print(f"[Arm] get_cmdnum failed, code={code}")
        return cmdnum

    def is_moving(self):
        """True if queued commands remain (mode 0 only)."""
        return self.get_cmdnum() > 0

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------
    def emergency_stop(self):
        """Immediately halts the arm and flags the abort state."""
        self.abort_event.set()
        self.arm.emergency_stop()

    # ------------------------------------------------------------------
    # Pass-throughs to the underlying SDK object
    # ------------------------------------------------------------------
    @property
    def has_error(self):
        return self.arm.has_error

    @property
    def state(self):
        return self.arm.state
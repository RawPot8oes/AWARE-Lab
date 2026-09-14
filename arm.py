"""
devices/arm.py

Wrapper around the UFactory xArm Python SDK (XArmAPI).

This class is the single point of contact between the rest of our test-rig
code (experiment_runner.py, main.py, etc.) and the physical robot arm.
Nothing outside this file should ever import `xarm` directly -- if we ever
need to change how we talk to the arm (different SDK version, different
robot), this is the only file that should need to change.

Key responsibilities:
  1. Connect to the arm and put it in a known, safe state.
  2. Provide a TCP offset setter, so the tool point can be moved from the
     bare flange out to wherever the sensor/airfoil actually sits.
  3. Provide move commands (move_to) for both discrete point-to-point
     moves and rapid non-blocking calls -- the pattern we're using for
     periodic motion like flapping trajectories (see motion/trajectory.py),
     following the approach validated in UFactory's own example code:
     mode 0, non-blocking set_position() calls in a tight timed loop,
     with radius set for smooth blending between points.
  4. Track a shared "abort" flag that gets set the moment the arm reports
     an error, so the rest of the program can stop cleanly instead of
     continuing to command a faulted arm.

NOTE: an earlier version of this class had servo-mode (mode 1) streaming
methods (enter_servo_mode / stream_joint_angles) for continuous motion.
Those are removed for now since the validated flapping pattern uses
mode 0 with non-blocking set_position() calls instead, which is simpler
and doesn't require the mode-1 servo dance. If mode-0 blending turns out
not to be smooth enough at your target flap frequency, that streaming
approach can be added back.
"""

import time
import threading
from xarm.wrapper import XArmAPI


class Arm:
    def __init__(self, ip):
        """
        ip: IP address of the xArm controller box, e.g. '192.168.1.113'
        """
        self.ip = ip
        self.arm = None  # will hold the XArmAPI instance once connected

        # threading.Event is a thread-safe on/off flag.
        # Any thread can call .set() to raise it, any thread can call
        # .is_set() to check it. We use this so the SDK's error callback
        # (which runs on its own internal thread) can safely signal our
        # main control loop to stop.
        self.abort_event = threading.Event()

    # ------------------------------------------------------------------
    # Internal callback -- the SDK calls this automatically whenever the
    # arm's error/warning state changes (e.g. joint limit hit, collision
    # detected, communication fault, etc.)
    # ------------------------------------------------------------------
    def _on_error_warn(self, item):
        err = item['error_code']
        warn = item['warn_code']
        print(f"[Arm] ErrorCode: {err}, WarnCode: {warn}")

        if err != 0:
            # Any nonzero error code trips the abort flag.
            # NOTE: this treats every error as equally serious for now.
            # Later you may want to look up which error codes are truly
            # dangerous vs. just informational, and only abort on the
            # dangerous ones.
            print("[Arm] Error detected -> setting abort flag")
            self.abort_event.set()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def connect(self):
        """
        Connects to the arm and puts it into a known-good starting state:
        motion enabled, no faults, position-control mode, running state.
        Call this once at the start of your program.

        NOTE: this does NOT set a TCP offset. If your sensor/airfoil sits
        away from the bare flange (it almost certainly does), call
        set_tcp_offset() once, right after connect(), before you record
        any base_pose or command any motion.
        """
        # do_not_open=True means "create the object but don't open the
        # socket yet" -- lets us register the error callback BEFORE any
        # traffic starts, so we don't miss an early error.
        self.arm = XArmAPI(self.ip, do_not_open=True)
        self.arm.register_error_warn_changed_callback(self._on_error_warn)

        self.arm.connect()                 # actually opens the connection

        # arm.version reads firmware/version info back from the controller,
        # which only works once the connection is actually live. Printing
        # it here doubles as confirmation that we're really talking to
        # the arm, not just that a socket object was created.
        print(f"[Arm] Connected. Firmware/version info: {self.arm.version}")

        self.arm.motion_enable(enable=True)  # powers/enables the motors
        self.arm.clean_error()             # clear any leftover fault state
        self.arm.set_mode(0)               # mode 0 = position control (discrete/queued moves)
        self.arm.set_state(state=0)        # state 0 = ready / sport state
        time.sleep(0.1)                    # brief pause to let state settle

        # Double check we didn't come up already in a fault state -- this
        # can happen silently after a prior hard e-stop, and we'd rather
        # catch it here than have the first move_to() mysteriously fail.
        if self.arm.has_error:
            print(f"[Arm] Warning: arm reports error state after connect, "
                  f"code={self.arm.error_code}")

    def disconnect(self):
        """Cleanly closes the connection to the arm controller."""
        if self.arm:
            self.arm.disconnect()

    # ------------------------------------------------------------------
    # Abort flag helpers
    # ------------------------------------------------------------------
    def is_aborted(self):
        """Returns True if an error has been flagged and not yet cleared."""
        return self.abort_event.is_set()

    def reset_abort(self):
        """
        Call this ONLY after you've manually confirmed it's safe to
        continue (e.g. checked the error code, moved the arm clear of an
        obstruction, etc.). Clears both our abort flag and the arm's own
        error state.
        """
        self.abort_event.clear()
        self.arm.clean_error()

    # ------------------------------------------------------------------
    # Tool setup
    # ------------------------------------------------------------------
    def set_tcp_offset(self, offset, wait=True):
        """
        Sets the tool center point (TCP) offset -- i.e. moves the point
        the arm considers "its hand" from the bare flange out to wherever
        your sensor/airfoil actually sits.

        offset: [x, y, z, roll, pitch, yaw] offset from the flange to the
                real tool point, in mm / degrees.

        Call this once during setup, right after connect(), and BEFORE
        recording a base_pose with get_position() or commanding any
        motion -- every pose you work with afterward is relative to
        wherever this offset places the TCP.
        """
        ret = self.arm.set_tcp_offset(offset, wait=wait)
        if ret != 0:
            print(f"[Arm] set_tcp_offset failed, ret={ret}")
        # Re-affirm ready state after changing the offset, matching the
        # pattern used in UFactory's own example code.
        self.arm.set_state(0)
        return ret

    # ------------------------------------------------------------------
    # Motion (position mode / mode 0)
    # ------------------------------------------------------------------
    def move_to(self, x, y, z, roll, pitch, yaw, speed=50, mvacc=1000,
                radius=None, wait=True):
        """
        Commands the arm to a single Cartesian pose.

        x, y, z         : target position in mm
        roll, pitch, yaw: target orientation in degrees
                           (per xArm's convention: roll = rotation about
                           base X, pitch = rotation about base Y,
                           yaw = rotation about base Z)
        speed           : max speed for this move (mm/s)
        mvacc           : max acceleration for this move (mm/s^2)
        radius          : blending radius for smooth multi-point paths.
                              None (or < 0) -> stop exactly at this point
                              >= 0          -> blend through this point
                                               (this is what we use for
                                               periodic/flapping motion:
                                               a rapid sequence of
                                               non-blocking calls with
                                               radius=0 produces smooth
                                               continuous motion without
                                               needing servo/mode-1)
        wait            : if True, blocks until the move finishes;
                           if False, returns immediately (command is
                           queued) -- use wait=False for the rapid-call
                           flapping pattern, wait=True for one-off
                           discrete moves like going to a start pose.

        Returns the SDK's return code (0 = success, nonzero = failure).
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
    # State readback
    # ------------------------------------------------------------------
    def get_position(self):
        """
        Returns the arm's current Cartesian pose as [x, y, z, roll, pitch, yaw],
        relative to whatever TCP offset is currently set.
        """
        code, pos = self.arm.get_position()
        if code != 0:
            print(f"[Arm] get_position failed, code={code}")
        return pos

    def get_angles(self):
        """
        Returns the arm's current joint angles (list of degrees, one per joint).
        """
        code, angles = self.arm.get_servo_angle()
        if code != 0:
            print(f"[Arm] get_servo_angle failed, code={code}")
        return angles

    def get_cmdnum(self):
        """
        Returns the number of motion commands still queued/executing on
        the controller. 0 means the arm has genuinely finished everything
        sent to it so far -- this is how we detect "motion actually
        stopped" from outside, rather than guessing based on how many
        commands our own Python loop has sent.
        """
        code, cmdnum = self.arm.get_cmdnum()
        if code != 0:
            print(f"[Arm] get_cmdnum failed, code={code}")
        return cmdnum

    def is_moving(self):
        """True if the arm still has queued commands left to execute."""
        return self.get_cmdnum() > 0

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------
    def emergency_stop(self):
        """Immediately halts the arm and flags the abort state."""
        self.abort_event.set()
        self.arm.emergency_stop()

    # ------------------------------------------------------------------
    # Convenience properties -- pass-throughs to the underlying SDK object
    # ------------------------------------------------------------------
    @property
    def has_error(self):
        return self.arm.has_error

    @property
    def state(self):
        return self.arm.state
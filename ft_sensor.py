"""
ft_sensor.py

Wrapper around the NI DAQ (6423) + ATI Mini40 F/T sensor.

Reads raw voltages from the DAQ and converts them to real forces/torques
using the sensor's actual calibration matrix (loaded from its .cal file
via cal_parser.py) -- no hand-typed placeholder matrix, no ATI .NET demo
required.
"""

import nidaqmx
import numpy as np
import threading
import time

from cal_parser import load_ati_calibration


class FTSensor:
    def __init__(self, cal_file_path, channels="Dev1/ai0:5", sample_rate_hz=500):
        """
        cal_file_path : path to your sensor's .cal file, e.g. "FT75200.cal"
        channels      : NI channel string for the 6 gauge voltage inputs,
                         e.g. "Dev1/ai0:5". Confirm ai0..ai5 are wired in
                         the same gauge order ATI used when calibrating
                         -- if the wiring order is scrambled, the matrix
                         multiply below will give wrong (rotated/mixed)
                         forces even though nothing "errors out".
        sample_rate_hz: streaming sample rate for start_streaming()
        """
        cal = load_ati_calibration(cal_file_path)
        self.cal_matrix = cal['matrix']       # 6x6, wrench = matrix @ voltage
        self.max_ratings = cal['max_ratings']  # per-axis max load, for safety checks
        self.force_units = cal['force_units']
        self.torque_units = cal['torque_units']
        print(f"[FTSensor] Loaded calibration for {cal['serial']} "
              f"({self.force_units}/{self.torque_units})")

        self.channels = channels
        self.sample_rate_hz = sample_rate_hz
        self.bias = np.zeros(6)

        self._latest_wrench = np.zeros(6)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._task = None

    def connect(self):
        """
        Opens the DAQ task once and confirms the channel is reachable.
        Must be called before zero()/start_streaming()/_read_raw().
        """
        self._task = nidaqmx.Task()
        self._task.ai_channels.add_ai_voltage_chan(
            self.channels, min_val=-10.0, max_val=10.0
        )

        self._task.timing.cfg_samp_clk_timing(
            rate=self.sample_rate_hz,
            sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS
        )

        self._read_raw(n_samples=1)

    def disconnect(self):
        self.stop_streaming()
        if self._task is not None:
            self._task.close()
            self._task = None

    def _read_raw(self, n_samples=1):
        if self._task is None:
            raise RuntimeError("FTSensor.connect() must be called before reading")

        if n_samples > 1:
            data = np.array(
                self._task.read(number_of_samples_per_channel=n_samples)
            )
            return data.T
        else:
            data = np.array(self._task.read(number_of_samples_per_channel=1))
            return data.reshape(1, -1)

    def zero(self, n_samples=200):
        """Call with no load on the sensor before a run to remove offset."""
        raw = self._read_raw(n_samples).mean(axis=0)
        self.bias = raw
        print(f"[FTSensor] Zeroed. Bias (V) = {self.bias}")

    def _stream_loop(self):
        # period = 1.0 / self.sample_rate_hz
        while self._running:
            raw = self._read_raw(n_samples=1)[0]
            wrench = self.cal_matrix @ (raw - self.bias)
            with self._lock:
                self._latest_wrench = wrench
            # time.sleep(period)

    def start_streaming(self):
        self._running = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()

    def stop_streaming(self):
        self._running = False
        if self._thread:
            self._thread.join()

    def read_wrench(self):
        """Returns the latest [Fx, Fy, Fz, Tx, Ty, Tz] in the sensor's own units."""
        with self._lock:
            return self._latest_wrench.copy()

    def check_overload(self, wrench=None, margin=0.9):
        """
        Returns True if any axis is at or above `margin` fraction of its
        max rating -- use this as a safety trip before calling
        arm.emergency_stop().
        """
        if wrench is None:
            wrench = self.read_wrench()
        axis_order = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
        for i, name in enumerate(axis_order):
            if abs(wrench[i]) >= margin * self.max_ratings[name]:
                print(f"[FTSensor] WARNING: {name}={wrench[i]:.3f} "
                      f"near max rating {self.max_ratings[name]}")
                return True
        return False
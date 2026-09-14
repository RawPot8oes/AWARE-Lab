"""
plotter.py

Visualization module for heave+pitch flap test data recorded by run_flap_test.py.
Can be imported directly into run_flap_test.py or executed standalone on saved CSV files.

Usage in run_flap_test.py:
    from plotter import plot_results
    plot_results(log_rows)

Standalone usage:
    python plotter.py flap_test_log.csv
"""

import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from frame_transform import batch_wrench_to_world


def load_data(data_input):
    """
    Normalizes input into a numpy array and extracts column headers.
    
    Parameters:
        data_input: List of rows, numpy array, or string path to a CSV file.
        
    Returns:
        tuple: (data_array, headers)
    """
    default_headers = ["t", "x", "y", "z", "roll", "pitch", "yaw",
                       "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

    if isinstance(data_input, str):
        if not os.path.exists(data_input):
            raise FileNotFoundError(f"CSV file not found: {data_input}")
        
        rows = []
        headers = default_headers
        with open(data_input, "r", newline="") as f:
            reader = csv.reader(f)
            first_row = next(reader, None)
            if first_row is None:
                raise ValueError("CSV file is empty.")
            
            # Check if first row is a header
            try:
                float(first_row[0])
                rows.append([float(val) for val in first_row])
            except ValueError:
                headers = first_row
                
            for row in reader:
                if row:
                    rows.append([float(val) for val in row])
        data_array = np.array(rows)

    elif isinstance(data_input, (list, tuple)):
        if len(data_input) == 0:
            raise ValueError("Data list is empty.")
        data_array = np.array(data_input, dtype=float)
        headers = default_headers

    elif isinstance(data_input, np.ndarray):
        data_array = data_input.astype(float)
        headers = default_headers

    else:
        raise TypeError(f"Unsupported data type: {type(data_input)}")

    return data_array, headers


def plot_results(data_input, save_path="flap_test_plot.png", show=True):
    """
    Plots motion kinematics and force/torque sensor data.

    Parameters:
        data_input: List of logged rows, numpy array, or path to CSV file.
        save_path: Path where the output figure will be saved (set None to skip saving).
        show: If True, calls plt.show() to display the plot window.
    """
    data, headers = load_data(data_input)

    if data.shape[1] < 13:
        raise ValueError(f"Expected at least 13 data columns, got {data.shape[1]}")

    t = data[:, 0]
    x, y, z = data[:, 1], data[:, 2], data[:, 3]
    roll, pitch, yaw = data[:, 4], data[:, 5], data[:, 6]
    Fx, Fy, Fz = data[:, 7], data[:, 8], data[:, 9]
    Tx, Ty, Tz = data[:, 10], data[:, 11], data[:, 12]

    # --- Rotate the sensor-frame wrench into the fixed tank/world frame ---
    # Fx/Fy/Fz above are still in the load cell's own frame, which tips
    # over with the arm every time roll/pitch/yaw changes. Using the
    # logged pose at each sample, rotate back into a frame that's fixed
    # in the tank -- see frame_transform.py for the convention/derivation
    # and the assumption about how the base frame lines up with the tank.
    poses = data[:, 1:7]
    wrenches = data[:, 7:13]
    world_wrench = batch_wrench_to_world(poses, wrenches)
    Fx_w, Fy_w, Fz_w = world_wrench[:, 0], world_wrench[:, 1], world_wrench[:, 2]
    Tx_w, Ty_w, Tz_w = world_wrench[:, 3], world_wrench[:, 4], world_wrench[:, 5]

    # Set up subplots layout
    fig, (ax_pos, ax_force, ax_torque, ax_world) = plt.subplots(
        4, 1, figsize=(11, 12), sharex=True
    )

    # --- Plot 1: Motion / Kinematics ---
    ax_pos.set_title("Heave + Pitch Flap Test Analysis", fontsize=14, fontweight="bold", pad=12)
    
    # Primary axis for Heave (X position)
    line_x = ax_pos.plot(t, x, label="Heave X (mm)", color="#1f77b4", linewidth=1.8)
    ax_pos.set_ylabel("X Position [mm]", color="#1f77b4", fontweight="bold")
    ax_pos.tick_params(axis="y", labelcolor="#1f77b4")
    ax_pos.grid(True, linestyle="--", alpha=0.6)

    # Secondary axis for Pitch angle
    ax_pitch = ax_pos.twinx()
    line_p = ax_pitch.plot(t, pitch, label="Pitch Angle (deg)", color="#ff7f0e", linestyle="--", linewidth=1.8)
    ax_pitch.set_ylabel("Pitch [deg]", color="#ff7f0e", fontweight="bold")
    ax_pitch.tick_params(axis="y", labelcolor="#ff7f0e")

    # Combined legend for top subplot
    lines_kin = line_x + line_p
    labels_kin = [l.get_label() for l in lines_kin]
    ax_pos.legend(lines_kin, labels_kin, loc="upper right", framealpha=0.9)

    # --- Plot 2: Forces ---
    ax_force.plot(t, Fx, label="Fx", color="#2ca02c", linewidth=1.5)
    ax_force.plot(t, Fy, label="Fy", color="#d62728", linewidth=1.5)
    ax_force.plot(t, Fz, label="Fz", color="#9467bd", linewidth=1.5)
    ax_force.set_ylabel("Force [N]", fontweight="bold")
    ax_force.set_title("Force Components", fontsize=11, fontweight="bold", loc="left")
    ax_force.grid(True, linestyle="--", alpha=0.6)
    ax_force.legend(loc="upper right", framealpha=0.9, ncol=3)

    # --- Plot 3: Torques ---
    ax_torque.plot(t, Tx, label="Tx", color="#8c564b", linewidth=1.5)
    ax_torque.plot(t, Ty, label="Ty", color="#e377c2", linewidth=1.5)
    ax_torque.plot(t, Tz, label="Tz", color="#17becf", linewidth=1.5)
    ax_torque.set_ylabel("Torque [N·m]", fontweight="bold")
    ax_torque.set_title("Torque Components", fontsize=11, fontweight="bold", loc="left")
    ax_torque.grid(True, linestyle="--", alpha=0.6)
    ax_torque.legend(loc="upper right", framealpha=0.9, ncol=3)

    # --- Plot 4: Tank-frame (world) forces -- lift / thrust / lateral ---
    # Same data as Plot 2, just rotated into a frame that doesn't rotate
    # with the arm. If a force is genuinely steady in the tank, it should
    # now show up as roughly flat here even though it oscillated in
    # Plot 2's sensor frame.
    ax_world.plot(t, Fx_w, label="Thrust (world Fx)", color="#2ca02c", linewidth=1.5)
    ax_world.plot(t, Fy_w, label="Lateral (world Fy)", color="#d62728", linewidth=1.5)
    ax_world.plot(t, Fz_w, label="Lift (world Fz)", color="#9467bd", linewidth=1.5)
    ax_world.set_xlabel("Time [s]", fontweight="bold")
    ax_world.set_ylabel("Force [N]", fontweight="bold")
    ax_world.set_title("Tank-Frame Forces (rotated out of sensor frame)",
                        fontsize=11, fontweight="bold", loc="left")
    ax_world.grid(True, linestyle="--", alpha=0.6)
    ax_world.legend(loc="upper right", framealpha=0.9, ncol=3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[plotter] Figure saved successfully to '{save_path}'")

    if show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        print(f"[plotter] Plotting data from '{csv_file}'...")
        plot_results(csv_file)
    else:
        default_csv = "flap_test_log.csv"
        if os.path.exists(default_csv):
            print(f"[plotter] Plotting default file '{default_csv}'...")
            plot_results(default_csv)
        else:
            print("Usage: python plotter.py <path_to_flap_test_log.csv>")
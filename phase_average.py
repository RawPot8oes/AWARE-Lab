import os
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
# Fallback file if no command-line argument is provided
DEFAULT_FILE = "data/Useful/Different Z/-150z_30cycles_20deg_1Hz_20261108_200215_log.csv"
OUT_DIR      = "./outputs/"
FREQ         = 1.0   # Flapping frequency (Hz)
DROP         = 1     # Cycles discarded at start
DROP_END     = 1     # Cycles discarded at end
NPH          = 400   # Number of phase points for interpolation
# ----------------------------------------

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 26, "axes.labelsize": 30,
    "axes.labelweight": "bold", "axes.titlesize": 32, "axes.titleweight": "bold",
    "xtick.labelsize": 26, "ytick.labelsize": 26, "legend.fontsize": 26,
    "lines.linewidth": 4, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.8, "xtick.major.width": 1.8, "ytick.major.width": 1.8,
    "xtick.major.size": 7, "ytick.major.size": 7, "axes.grid": True,
    "grid.color": "#dddddd", "grid.linewidth": 1.0,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.4,
})


def load_and_average(filepath, freq_hz, drop, drop_end, nph):
    """
    Load a CSV and return phase-averaged data.

    Parameters
    ----------
    filepath : str         Path to the CSV file (comment='#' line is skipped).
    freq_hz  : float       Flapping frequency in Hz.
    drop     : int         Cycles to discard at the start.
    drop_end : int         Cycles to discard at the end.
    nph      : int         Number of phase points for interpolation.

    Returns
    -------
    phase_time : np.ndarray  Shape (nph,) — time within one cycle (seconds).
    results    : dict        {channel: {"mean": array, "std": array}}
    """
    d = pd.read_csv(filepath, comment='#')
    t = d.t.values - d.t.values[0]
    T = 1.0 / freq_hz
    n_cyc = int(np.floor(t[-1] / T))

    ph   = np.linspace(0, 1, nph, endpoint=False)
    cols = [c for c in ["pitch", "Fy", "Tx", "Fz", "Fx"] if c in d.columns]

    stack = {c: [] for c in cols}
    for k in range(drop, n_cyc - drop_end):
        m = (t >= k * T) & (t < (k + 1) * T)
        if m.sum() < 100:
            continue
        p = (t[m] - k * T) / T
        for c in cols:
            stack[c].append(np.interp(ph, p, d[c].values[m]))

    results = {}
    for c in cols:
        a = np.array(stack[c])
        results[c] = {"mean": a.mean(axis=0), "std": a.std(axis=0, ddof=1)}

    return ph * T, results


def generate_phase_avg(filepath, out_dir,
                       freq_hz=FREQ, drop=DROP, drop_end=DROP_END, nph=NPH,
                       out_stem="phase_avg"):
    """
    Generate phase-averaged plots for one CSV file and save them to out_dir.

    Can be called from run_flap_test.py after a run, or from main() for
    standalone / batch use.

    Parameters
    ----------
    filepath : str    Path to the log CSV.
    out_dir  : str    Directory where output files are written.
    freq_hz  : float  Flapping frequency (Hz).  Default = CONFIG FREQ.
    drop     : int    Cycles dropped at start.  Default = CONFIG DROP.
    drop_end : int    Cycles dropped at end.    Default = CONFIG DROP_END.
    nph      : int    Phase interpolation points. Default = CONFIG NPH.
    out_stem : str    Output filename stem (no extension).
                      Default "phase_avg" → phase_avg.png / phase_avg.pdf.
                      Pass the CSV stem for standalone use to get
                      <csvname>_phase_avg.png.

    Returns
    -------
    out_path : str    Absolute path to the saved .png, or None on failure.
    """
    if not os.path.exists(filepath):
        print(f"[phase_avg] File not found: {filepath}")
        return None

    os.makedirs(out_dir, exist_ok=True)

    # Title uses the parent folder name when called from a run (more informative
    # than just "log"), falls back to the file stem for standalone use.
    folder_name = os.path.basename(os.path.dirname(os.path.abspath(filepath)))
    title_str   = folder_name if folder_name else os.path.splitext(os.path.basename(filepath))[0]

    phase_time, data = load_and_average(filepath, freq_hz, drop, drop_end, nph)
    channels = [c for c in ["pitch", "Fy", "Tx", "Fz", "Fx"] if c in data]

    if not channels:
        print(f"[phase_avg] No recognised channels found in {filepath}")
        return None

    col_map = {
        "pitch": "#8E44AD", "Fy": "#0072B2", "Tx": "#D55E00",
        "Fz": "#009E73",    "Fx": "#CC79A7",
    }
    ylabels = {
        "pitch": "Flap angle (°)",      "Fy": "Force $F_y$ (N)",
        "Tx":    "Torque $T_x$ (N·m)",  "Fz": "Force $F_z$ (N)",
        "Fx":    "Force $F_x$ (N)",
    }

    fig, axs = plt.subplots(len(channels), 1,
                             figsize=(12, 4 * len(channels)), sharex=True)
    if len(channels) == 1:
        axs = [axs]

    T = 1.0 / freq_hz
    for ax, c in zip(axs, channels):
        mu    = data[c]["mean"]
        sd    = data[c]["std"]
        color = col_map.get(c, "#333333")

        if c == "pitch":
            ax.plot(phase_time, mu, color=color)
            ax.set_yticks([-30, 0, 30])
            ax.set_ylim(-42, 42)
            ax.set_title(f"Phase-Averaged Loads: {title_str}", pad=15)
        else:
            ax.fill_between(phase_time, mu - sd, mu + sd,
                            color=color, alpha=0.25, lw=0)
            ax.plot(phase_time, mu, color=color)

        ax.axhline(0, color="k", lw=1)
        ax.set_ylabel(ylabels.get(c, c))

    axs[-1].set_xlim(-0.03 * T, 1.03 * T)
    axs[-1].set_xticks([0, 0.25 * T, 0.5 * T, 0.75 * T, T])
    axs[-1].set_xticklabels(["0", "25", "50", "75", "100"])
    axs[-1].set_xlabel("Flap cycle (%)")

    fig.align_ylabels(axs)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"{out_stem}.png")
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.replace(".png", ".pdf"))
    plt.close(fig)

    print(f"[phase_avg] Saved:\n  {out_path}\n  {out_path.replace('.png', '.pdf')}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate phase-averaged graphs for a single CSV file.")
    parser.add_argument("filepath", nargs="?", default=DEFAULT_FILE,
                        help="Path to the CSV file")
    args = parser.parse_args()

    # For standalone use, write into OUT_DIR and use the CSV stem in the filename
    # so multiple files don't overwrite each other.
    stem = os.path.splitext(os.path.basename(args.filepath))[0]
    generate_phase_avg(
        filepath=args.filepath,
        out_dir=OUT_DIR,
        freq_hz=FREQ,
        drop=DROP,
        drop_end=DROP_END,
        nph=NPH,
        out_stem=f"{stem}_phase_avg",   # e.g. log_phase_avg.png in outputs/
    )


if __name__ == "__main__":
    main()
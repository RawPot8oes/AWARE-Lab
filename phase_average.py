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

mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 26, "axes.labelsize": 30,
    "axes.labelweight": "bold", "axes.titlesize": 32, "axes.titleweight": "bold",
    "xtick.labelsize": 26, "ytick.labelsize": 26, "legend.fontsize": 26,
    "lines.linewidth": 4, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.8, "xtick.major.width": 1.8, "ytick.major.width": 1.8,
    "xtick.major.size": 7, "ytick.major.size": 7, "axes.grid": True,
    "grid.color": "#dddddd", "grid.linewidth": 1.0, "savefig.bbox": "tight", "savefig.pad_inches": 0.4})

def load_and_average(filepath):
    d = pd.read_csv(filepath)
    t = d.t.values - d.t.values[0]
    T = 1.0 / FREQ
    n_cyc = int(np.floor(t[-1] / T))
    
    ph = np.linspace(0, 1, NPH, endpoint=False)
    cols_to_process = ["pitch", "Fy", "Tx", "Fz", "Fx"]
    cols = [c for c in cols_to_process if c in d.columns]
    
    stack = {c: [] for c in cols}
    for k in range(DROP, n_cyc - DROP_END):
        m = (t >= k * T) & (t < (k + 1) * T)
        if m.sum() < 100: continue
        p = (t[m] - k * T) / T
        for c in cols:
            stack[c].append(np.interp(ph, p, d[c].values[m]))
            
    results = {}
    for c in cols:
        a = np.array(stack[c])
        results[c] = {"mean": a.mean(axis=0), "std": a.std(axis=0, ddof=1)}
        
    return ph * T, results

def main():
    parser = argparse.ArgumentParser(description="Generate phase-averaged graphs for a single CSV file.")
    parser.add_argument("filepath", nargs="?", default=DEFAULT_FILE, help="Path to the CSV file")
    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        print(f"Error: File not found -> {args.filepath}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    filename = os.path.splitext(os.path.basename(args.filepath))[0]
    
    ph, data = load_and_average(args.filepath)
    channels = [c for c in ["pitch", "Fy", "Tx", "Fz", "Fx"] if c in data]
    
    # Setup grid based on available channels
    fig, axs = plt.subplots(len(channels), 1, figsize=(12, 4 * len(channels)), sharex=True)
    if len(channels) == 1: axs = [axs]
    
    col_map = {"pitch": "#8E44AD", "Fy": "#0072B2", "Tx": "#D55E00", "Fz": "#009E73", "Fx": "#CC79A7"}
    ylabels = {"pitch": "Flap angle (°)", "Fy": "Force $F_y$ (N)", "Tx": "Torque $T_x$ (N·m)", 
               "Fz": "Force $F_z$ (N)", "Fx": "Force $F_x$ (N)"}
    
    for ax, c in zip(axs, channels):
        mu, sd = data[c]["mean"], data[c]["std"]
        color = col_map.get(c, "#333333")
        
        if c == "pitch":
            ax.plot(ph, mu, color=color)
            ax.set_yticks([-30, 0, 30])
            ax.set_ylim(-42, 42)
            ax.set_title(f"Phase-Averaged Loads: {filename}", pad=15)
        else:
            ax.fill_between(ph, mu - sd, mu + sd, color=color, alpha=0.25, lw=0)
            ax.plot(ph, mu, color=color)
            
        ax.axhline(0, color="k", lw=1)
        ax.set_ylabel(ylabels.get(c, c))
    
    # Format X-axis on the bottom plot
    T = 1.0 / FREQ
    axs[-1].set_xlim(-0.03 * T, 1.03 * T)
    axs[-1].set_xticks([0, 0.25 * T, 0.5 * T, 0.75 * T, 1.0 * T])
    axs[-1].set_xticklabels(["0", "25", "50", "75", "100"])
    axs[-1].set_xlabel("Flap cycle (%)")
    
    fig.align_ylabels(axs)
    fig.tight_layout()
    
    out_path = os.path.join(OUT_DIR, f"{filename}_phase_avg.png")
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.replace(".png", ".pdf"))
    plt.close(fig)
    
    print(f"Successfully generated graphs: \n - {out_path}\n - {out_path.replace('.png', '.pdf')}")

if __name__ == "__main__":
    main()
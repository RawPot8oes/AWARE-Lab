"""
compare_runs.py

Overlay phase-averaged data from multiple flap test runs on a single figure.

Runs are discovered automatically via data_manager.list_runs() (with optional
param filters), or supplied directly with --dirs.

Legend labels show ONLY the parameters that differ across the selected runs.
Everything constant across all runs goes into the figure title instead.

The x-axis is always normalised to 0–100% of one cycle so runs at different
frequencies can still be overlaid meaningfully.

Usage examples
--------------
  # All flap runs
  python compare_runs.py --test flap

  # All flap runs at 1 Hz
  python compare_runs.py --test flap --freq_hz 1

  # Sweep heave amplitude, fix pitch and frequency
  python compare_runs.py --test flap --freq_hz 1 --pitch_amp_deg 20

  # Specific folders only
  python compare_runs.py --dirs data/run1 data/run2 data/run3

  # Hide std shading, save to a custom path
  python compare_runs.py --test flap --no-std --out outputs/my_comparison.png
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from data_manager import list_runs, parse_run_name, DATA_ROOT
from phase_average import load_and_average, FREQ, DROP, DROP_END, NPH

# ============================================================
# CONFIG
# ============================================================
OUT_DIR  = "./outputs/"
CHANNELS = ["pitch", "Fy", "Tx", "Fz", "Fx"]  # preferred display order

YLABELS = {
    "pitch": "Flap angle (°)",
    "Fy":    "Force $F_y$ (N)",
    "Tx":    "Torque $T_x$ (N·m)",
    "Fz":    "Force $F_z$ (N)",
    "Fx":    "Force $F_x$ (N)",
}

mpl.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          14,
    "axes.labelsize":     15,
    "axes.labelweight":   "bold",
    "axes.titlesize":     15,
    "axes.titleweight":   "bold",
    "xtick.labelsize":    13,
    "ytick.labelsize":    13,
    "legend.fontsize":    12,
    "lines.linewidth":    2.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     1.5,
    "xtick.major.width":  1.5,
    "ytick.major.width":  1.5,
    "xtick.major.size":   6,
    "ytick.major.size":   6,
    "axes.grid":          True,
    "grid.color":         "#dddddd",
    "grid.linewidth":     1.0,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.3,
})
# ============================================================


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _fmt(v):
    """Clean number → string for labels.  1.0 → '1',  1.5 → '1.5'."""
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return f"{v:g}"


def _get_tags(meta):
    """
    Return a flat dict of numeric parameters for one run, merging:
      - meta["params"]  (full names, e.g. freq_hz=1.0)
      - folder name tags (short names, e.g. f=1.0)
    Short folder tags take precedence in the display dict so labels
    stay compact, but full param names are also present for filtering.
    """
    d = {}
    for k, v in meta.get("params", {}).items():
        if isinstance(v, (int, float)):
            d[k] = float(v)

    folder = meta.get("folder") or Path(meta["dir"]).name
    try:
        for k, v in parse_run_name(folder)["tags"].items():
            d[k] = v          # short tag wins if both exist
    except (ValueError, KeyError):
        pass

    return d


def _build_labels(run_metas):
    """
    Classify every numeric parameter as 'varying' or 'constant' across
    the selected runs, then build one short legend label per run that
    shows only the varying parameters.

    Returns
    -------
    labels    : list[str]   one label per run  (for the legend)
    constants : list[str]   "key=val" strings  (for the suptitle)
    """
    tag_dicts = [_get_tags(m) for m in run_metas]

    all_keys = []
    for d in tag_dicts:
        for k in d:
            if k not in all_keys:
                all_keys.append(k)

    varying   = []
    constants = []
    for k in all_keys:
        vals = [d[k] for d in tag_dicts if k in d]
        if len(set(vals)) > 1:
            varying.append(k)
        elif vals:
            constants.append(f"{k}={_fmt(vals[0])}")

    labels = []
    for d in tag_dicts:
        if varying:
            parts = [f"{k}={_fmt(d[k])}" for k in varying if k in d]
            labels.append(", ".join(parts) if parts else "?")
        else:
            # Nothing varies — number the runs so the legend is still useful
            labels.append(Path(run_metas[tag_dicts.index(d)]["dir"]).name)

    return labels, constants


# ------------------------------------------------------------
# Core comparison function
# ------------------------------------------------------------

def compare_runs(run_metas, out_path, show_std=True,
                 freq_hz=FREQ, drop=DROP, drop_end=DROP_END, nph=NPH):
    """
    Phase-average and overlay data from multiple runs.

    Parameters
    ----------
    run_metas : list[dict]   meta dicts (each must have a "dir" key)
    out_path  : str          where to save (.png; a .pdf is also saved)
    show_std  : bool         shade ±1 std around each mean
    freq_hz   : float        fallback frequency if not in meta["params"]
    drop      : int          cycles dropped at start of each run
    drop_end  : int          cycles dropped at end of each run
    nph       : int          number of phase interpolation points

    Returns
    -------
    out_path : str | None
    """
    if not run_metas:
        print("[compare] No runs to compare.")
        return None

    labels, constants = _build_labels(run_metas)

    # Colour cycle — tab10 gives 10 perceptually distinct colours
    try:
        cmap = mpl.colormaps.get_cmap("tab10")       # matplotlib ≥ 3.7
    except AttributeError:
        cmap = plt.cm.get_cmap("tab10")              # older matplotlib
    colors = [cmap(i % 10) for i in range(len(run_metas))]

    # ------------------------------------------------------------------
    # Load and phase-average each run
    # x-axis is normalised phase 0–1 (displayed as 0–100 %) so runs at
    # different frequencies can still be overlaid on the same axis.
    # ------------------------------------------------------------------
    ph_norm = np.linspace(0, 1, nph, endpoint=False)   # 0 … 1

    loaded        = []   # list of (results_dict | None)
    available_ch  = None

    for m in run_metas:
        log_path = Path(m["dir"]) / "log.csv"
        if not log_path.exists():
            print(f"[compare] WARNING: log.csv not found in {m['dir']} — skipping.")
            loaded.append(None)
            continue

        # Use the run's own frequency from meta if present
        f = float(m.get("params", {}).get("freq_hz", freq_hz))
        try:
            _, results = load_and_average(str(log_path), f, drop, drop_end, nph)
            loaded.append(results)
            present      = set(results.keys())
            available_ch = present if available_ch is None \
                           else available_ch & present
        except Exception as e:
            print(f"[compare] WARNING: could not process {log_path}: {e!r}")
            loaded.append(None)

    if not available_ch:
        print("[compare] No data could be loaded from any run.")
        return None

    channels = [c for c in CHANNELS if c in available_ch]
    n_ch     = len(channels)

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    fig, axs = plt.subplots(n_ch, 1,
                             figsize=(13, 4 * n_ch),
                             sharex=True)
    if n_ch == 1:
        axs = [axs]

    # Suptitle: constant parameters so the legend can stay short
    test_str   = run_metas[0].get("test", "flap")
    const_str  = " | ".join(constants) if constants else "all parameters vary"
    fig.suptitle(f"{test_str}  —  {const_str}",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, ch in zip(axs, channels):
        for m, results, label, color in zip(run_metas, loaded, labels, colors):
            if results is None or ch not in results:
                continue
            mu = results[ch]["mean"]
            sd = results[ch]["std"]

            # Plot on normalised phase axis (0–1)
            ax.plot(ph_norm, mu, color=color, label=label)
            if show_std:
                ax.fill_between(ph_norm, mu - sd, mu + sd,
                                color=color, alpha=0.15, lw=0)

        ax.axhline(0, color="k", lw=0.8, zorder=1)
        ax.set_ylabel(YLABELS.get(ch, ch))

        if ch == "pitch":
            ax.set_yticks([-30, 0, 30])
            ax.set_ylim(-42, 42)

    # X-axis ticks — normalised phase as percentage
    axs[-1].set_xlim(-0.02, 1.02)
    axs[-1].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axs[-1].set_xticklabels(["0", "25", "50", "75", "100"])
    axs[-1].set_xlabel("Flap cycle (%)")

    # Legend on the top subplot only
    axs[0].legend(loc="upper right", framealpha=0.9,
                  ncol=min(len(run_metas), 4))

    fig.align_ylabels(axs)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.replace(".png", ".pdf"))
    plt.close(fig)

    print(f"[compare] Saved:\n  {out_path}\n  {out_path.replace('.png', '.pdf')}")
    return out_path


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def _parse_extra_filters(unknown_args):
    """
    Turn leftover argparse tokens (--key value pairs) into a dict.
    Values are cast to int, then float, then left as str.

        ["--freq_hz", "1", "--n_cycles", "30"]
        → {"freq_hz": 1, "n_cycles": 30}
    """
    filters = {}
    it = iter(unknown_args)
    for tok in it:
        if not tok.startswith("--"):
            continue
        key = tok[2:].replace("-", "_")
        try:
            raw = next(it)
        except StopIteration:
            print(f"[compare] WARNING: flag {tok} has no value — ignored.")
            continue
        try:
            filters[key] = int(raw)
        except ValueError:
            try:
                filters[key] = float(raw)
            except ValueError:
                filters[key] = raw
    return filters


def main():
    parser = argparse.ArgumentParser(
        description="Compare phase-averaged flap test data across multiple runs.",
        epilog=(
            "Any extra --key value pairs are forwarded as parameter filters.\n"
            "Example:\n"
            "  python compare_runs.py --test flap --freq_hz 1 --heave_amp_mm 100"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test", default=None,
        help="Filter by test type, e.g. 'flap'.")
    parser.add_argument(
        "--dirs", nargs="+", default=None, metavar="DIR",
        help="Compare specific run folders instead of searching data/.")
    parser.add_argument(
        "--data-root", default=str(DATA_ROOT),
        help=f"Root folder to search for runs (default: {DATA_ROOT}).")
    parser.add_argument(
        "--out", default=None,
        help="Output file path (default: outputs/compare_<test>_<timestamp>.png).")
    parser.add_argument(
        "--no-std", action="store_true",
        help="Hide ±1 std shading.")
    parser.add_argument(
        "--freq-hz", type=float, default=None,
        help="Override flapping frequency for ALL runs. "
             "By default each run's own freq_hz from meta is used.")

    args, unknown = parser.parse_known_args()
    param_filters  = _parse_extra_filters(unknown)

    # ------------------------------------------------------------------
    # Collect run metadata
    # ------------------------------------------------------------------
    if args.dirs:
        run_metas = []
        for d in args.dirs:
            p = Path(d)
            meta_path = p / "meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    m = json.load(f)
                m["dir"] = str(p)
            else:
                print(f"[compare] WARNING: no meta.json in {p} — using folder name only.")
                m = {"dir": str(p), "folder": p.name, "params": {}}
            run_metas.append(m)
    else:
        run_metas = list_runs(root=args.data_root,
                              test=args.test,
                              **param_filters)

    if not run_metas:
        print("[compare] No matching runs found.")
        sys.exit(1)

    print(f"[compare] {len(run_metas)} run(s) selected:")
    for m in run_metas:
        print(f"  {m.get('folder') or Path(m['dir']).name}")

    # ------------------------------------------------------------------
    # Output path
    # ------------------------------------------------------------------
    if args.out:
        out_path = args.out
    else:
        stamp     = time.strftime("%Y%m%d-%H%M%S")
        test_slug = (args.test or "runs").replace("-", "")
        out_path  = os.path.join(OUT_DIR, f"compare_{test_slug}_{stamp}.png")

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    compare_runs(
        run_metas,
        out_path=out_path,
        show_std=not args.no_std,
        freq_hz=args.freq_hz or FREQ,
    )


if __name__ == "__main__":
    main()
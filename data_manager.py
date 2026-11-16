"""
data_manager.py
===============
Single module that decides where test data goes and how it is described.

Folder structure
----------------
    data/
        YYYYMMDD-HHMMSS_<testtype>_<tag><val>_<tag><val>.../
            meta.json      ← full run description (source of truth)
            log.csv        ← wrench log;  line 1 = human-readable metadata
            meas.csv       ← measured pose (optional, flap test only)
            plot.png       ← overview plot from plotter.py
            ...            ← anything else saved via run.path(name)
        index.csv          ← rebuilt by: python data_manager.py

Folder name format
------------------
    YYYYMMDD-HHMMSS _ <testtype> _ <tag><number> [ _ <tag><number> ... ]

    <testtype>  letters, digits, hyphens — encode DOF/variant here
                e.g.  "flap",  "step-yaw",  "step-pitch"
    <tag>       letters only  (e.g.  "f", "h", "p", "ph", "n")
    <number>    compact float (e.g.  1.5,  -20,  100)

    Examples
        20261116-154402_flap_f1_h100_p20_ph180_n30
        20261116-154402_step-yaw_s-20_e20_by1_h30

CSV comment line (line 1 of every saved CSV)
--------------------------------------------
    # test=flap | started=2026-11-16 15:44:02 | freq_hz=1 | heave_amp_mm=100 | ... | comments=baseline

    Skip it automatically:
        pandas  →  pd.read_csv(path, comment='#')
        numpy   →  np.genfromtxt(path, delimiter=',', comments='#', names=True)
        here    →  header, data, meta = data_manager.load_csv(path)

Quick-start
-----------
    from data_manager import create_run

    COMMENTS = ""   # fill in before running, e.g. "baseline, tank at rest"

    run = create_run(
        test="flap",
        params={"freq_hz": 1, "heave_amp_mm": 100, "pitch_amp_deg": 20,
                "phase_offset_deg": 180, "n_cycles": 30},
        name_tags={"f": 1, "h": 100, "p": 20, "ph": 180, "n": 30},
        comments=COMMENTS,
        config={"BASE_POSE": BASE_POSE, "TCP_OFFSET": TCP_OFFSET, ...},
    )
    # ... run the test ...
    run.update_meta(duration_s=12.3, n_log_samples=6000, n_meas_samples=1500)
    run.save_csv("log.csv",  log_rows,  LOG_HEADER)
    run.save_csv("meas.csv", meas_rows, MEAS_HEADER)
    plot_results(log_rows, save_path=run.path("plot.png"), show=False)
"""

import csv
import datetime
import json
import re
import subprocess
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
DATA_ROOT = Path("data")

_STAMP_FMT = "%Y%m%d-%H%M%S"
_STAMP_RE  = re.compile(r"^\d{8}-\d{6}$")
_TAG_RE    = re.compile(r"^[A-Za-z]+$")       # tag validation
_TOKEN_RE  = re.compile(                        # parses one name token
    r"^([A-Za-z]+)"                             #   tag  (letters only)
    r"(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)$"        #   number value
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _git_info():
    """Return 'abc1234' (or '...-dirty'), or None if git is unavailable."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def _fmt(v):
    """Compact number → string for folder name tokens: 100.0→'100', 1.5→'1.5', -20→'-20'."""
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)   # int passes through naturally


def _build_comment(meta: dict) -> str:
    """
    Build the human-readable comment written as line 1 of every CSV.

    Format:
        # test=flap | started=2026-11-16 15:44:02 | freq_hz=1 | heave_amp_mm=100 | ... | comments=baseline
    """
    parts = [f"test={meta['test']}", f"started={meta['started']}"]
    for k, v in meta.get("params", {}).items():
        parts.append(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}")
    if meta.get("comments"):
        parts.append(f"comments={meta['comments']}")
    return "# " + " | ".join(parts)


def _parse_comment(line: str) -> dict:
    """
    Inverse of _build_comment.

    '# test=flap | freq_hz=1 | n_cycles=30 | comments=baseline'
    →  {'test': 'flap', 'freq_hz': 1.0, 'n_cycles': 30, 'comments': 'baseline'}
    """
    result = {}
    for token in line.lstrip("# ").split(" | "):
        if "=" not in token:
            continue
        k, _, v = token.partition("=")
        k, v = k.strip(), v.strip()
        try:
            result[k] = int(v) if re.fullmatch(r"-?\d+", v) else float(v)
        except ValueError:
            result[k] = v   # string values (test name, comments, DOF)
    return result


# ---------------------------------------------------------------------------
# parse_run_name  (folder name → structured dict)
# ---------------------------------------------------------------------------

def parse_run_name(name: str) -> dict:
    """
    Parse a run folder name back into structured fields.

    '20261116-154402_flap_f1_h100_ph-90_n30'
    →  {
           'started': datetime(2026, 11, 16, 15, 44, 2),
           'test':    'flap',
           'tags':    {'f': 1.0, 'h': 100.0, 'ph': -90.0, 'n': 30.0},
       }

    '20261116-154402_step-yaw_s-20_e20_by1_h30'
    →  {
           'started': datetime(...),
           'test':    'step-yaw',
           'tags':    {'s': -20.0, 'e': 20.0, 'by': 1.0, 'h': 30.0},
       }

    Raises ValueError for malformed names.
    """
    parts = name.split("_")
    if len(parts) < 2 or not _STAMP_RE.match(parts[0]):
        raise ValueError(f"Not a valid run folder name: {name!r}")

    tags = {}
    for tok in parts[2:]:
        m = _TOKEN_RE.match(tok)
        if not m:
            raise ValueError(f"Unrecognised token '{tok}' in '{name}'")
        tags[m.group(1)] = float(m.group(2))

    return {
        "started": datetime.datetime.strptime(parts[0], _STAMP_FMT),
        "test":    parts[1],   # may contain hyphens, e.g. "step-yaw"
        "tags":    tags,
    }


# ---------------------------------------------------------------------------
# Run  (returned by create_run)
# ---------------------------------------------------------------------------

class Run:
    """
    Handle for a single test run.  Returned by create_run().

    Attributes
    ----------
    dir  : pathlib.Path   The run's folder.
    meta : dict           Live metadata (also mirrored in meta.json).
    """

    def __init__(self, directory: Path, meta: dict):
        self.dir  = Path(directory)
        self.meta = meta

    # ------------------------------------------------------------------ paths

    def path(self, filename: str) -> Path:
        """Return the full path to *filename* inside this run's folder."""
        return self.dir / filename

    # ---------------------------------------------------------- metadata I/O

    def update_meta(self, **kwargs):
        """
        Merge *kwargs* into self.meta and rewrite meta.json.

        Call after the run to record timing and sample counts:
            run.update_meta(duration_s=12.3, n_log_samples=6000, n_meas_samples=1500)
        """
        self.meta.update(kwargs)
        with open(self.dir / "meta.json", "w") as f:
            json.dump(self.meta, f, indent=2, default=str)

    # ---------------------------------------------------------------- CSV I/O

    def save_csv(self, filename: str, rows, header: list) -> Path:
        """
        Write *rows* to <run_folder>/<filename>.

        File layout
        -----------
        Line 1 : # test=... | started=... | param=val | ... | comments=...
        Line 2 : t,x,y,z,roll,pitch,yaw,Fx,Fy,Fz,Tx,Ty,Tz
        Line 3+: data rows

        pandas.read_csv(path, comment='#') and np.genfromtxt skip line 1.
        Call load_csv(path) to get the metadata back as well.
        """
        dest = self.dir / filename
        with open(dest, "w", newline="") as f:
            f.write(_build_comment(self.meta) + "\n")
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"[data] Saved {len(rows)} rows → {dest}")
        return dest

    def save_csv_with_fallback(self, filename: str, rows, header: list) -> Path:
        """
        Like save_csv(), but appends a Unix timestamp to the filename if the
        target is locked (PermissionError) so data is never lost.
        """
        try:
            return self.save_csv(filename, rows, header)
        except PermissionError:
            import time
            stem   = Path(filename).stem
            suffix = Path(filename).suffix
            fb     = f"{stem}_{int(time.time())}{suffix}"
            print(f"[data] {filename} is locked — saving to {fb} instead.")
            return self.save_csv(fb, rows, header)

    def __repr__(self):
        return f"Run(dir={self.dir!s}, test={self.meta.get('test')!r})"


# ---------------------------------------------------------------------------
# create_run  (main entry point for run scripts)
# ---------------------------------------------------------------------------

def create_run(
    test: str,
    params: dict,
    name_tags: dict,
    comments: str = "",
    config: dict = None,
    root: Path = DATA_ROOT,
) -> Run:
    """
    Create a timestamped run folder and return a Run object.

    Parameters
    ----------
    test
        Short test type: letters, digits, hyphens; NO underscores.
        Encode test variant / DOF in the name after a hyphen if useful:
            "flap", "step-yaw", "step-pitch"
    params
        Full parameter dict stored in meta.json and CSV line 1.
        Values should be numbers or short strings.
    name_tags
        Ordered dict {short_label: number} for the folder name.
        Labels must be letters only. Keep to 4–6 key parameters.
        Numbers only — put string info (DOF, variant) in *test* instead.
    comments
        Optional free-text note written into meta.json and CSV line 1.
        Set a COMMENTS = "" constant at the top of your run script and
        fill it in before each run.
    config
        Extra setup info (BASE_POSE, RATE_HZ, CAL_FILE …) stored only in
        meta.json, not in the folder name.
    root
        Parent directory; defaults to data/.
    """
    if "_" in test:
        raise ValueError(
            "test must not contain underscores — use hyphens for variants, "
            "e.g. 'step-yaw'."
        )
    for lbl in name_tags:
        if not _TAG_RE.match(lbl):
            raise ValueError(
                f"name_tag label '{lbl}' must be letters only (no digits / symbols)."
            )

    now    = datetime.datetime.now()
    tokens = [now.strftime(_STAMP_FMT), test] + \
             [f"{lbl}{_fmt(val)}" for lbl, val in name_tags.items()]
    name      = "_".join(tokens)
    directory = Path(root) / name
    directory.mkdir(parents=True, exist_ok=False)

    meta = {
        "folder":     name,
        "test":       test,
        "started":    now.isoformat(timespec="seconds"),
        "comments":   comments,
        "params":     params,
        "config":     config or {},
        "git_commit": _git_info(),
    }
    run = Run(directory, meta)
    run.update_meta()   # writes meta.json
    print(f"[data] Run folder: {directory}")
    return run


# ---------------------------------------------------------------------------
# Reading runs back
# ---------------------------------------------------------------------------

def load_csv(path):
    """
    Load a CSV produced by Run.save_csv().

    Returns
    -------
    header : list[str]
    data   : np.ndarray  (float64, shape N×C)
    meta   : dict        Parsed from the '#' comment line, or {} if absent.
    """
    path = Path(path)
    meta = {}
    with open(path, newline="") as f:
        first = f.readline()
        if first.startswith("#"):
            meta        = _parse_comment(first)
            header_line = f.readline()
        else:
            header_line = first
        header = next(csv.reader([header_line]))
        data   = np.loadtxt(f, delimiter=",", ndmin=2)
    return header, data, meta


def list_runs(root=DATA_ROOT, test=None, **param_filters):
    """
    Return meta dicts for all runs (oldest first), with optional filtering.

    Examples
    --------
    list_runs()                                 # every run
    list_runs(test="flap")                      # flap runs only
    list_runs(test="flap", freq_hz=1.0)         # flap at 1 Hz
    list_runs(test="step-yaw", hold_s=30)       # yaw steps, 30 s holds
    """
    runs = []
    for p in sorted(Path(root).glob("*/meta.json")):
        with open(p) as f:
            meta = json.load(f)
        meta["dir"] = str(p.parent)
        if test is not None and meta.get("test") != test:
            continue
        params = meta.get("params", {})
        if all(params.get(k) == v for k, v in param_filters.items()):
            runs.append(meta)
    return runs


def rebuild_index(root=DATA_ROOT):
    """
    Write (or overwrite) data/index.csv — one row per run, columns for every
    parameter ever seen across all runs.  Safe to delete and regenerate
    at any time (derived purely from meta.json files).

    Open in Excel / pandas to compare everything at once:
        import pandas as pd
        df = pd.read_csv("data/index.csv")
        df[df.test == "flap"].sort_values("freq_hz")
    """
    runs       = list_runs(root)
    param_keys: list[str] = []
    for r in runs:
        for k in r.get("params", {}):
            if k not in param_keys:
                param_keys.append(k)

    top_cols = [
        "folder", "started", "test",
        "duration_s", "n_log_samples", "n_meas_samples",
        "git_commit", "comments",
    ]
    all_cols = top_cols + param_keys
    out      = Path(root) / "index.csv"

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(all_cols)
        for r in runs:
            p = r.get("params", {})
            w.writerow([
                p.get(c, "") if c in param_keys else r.get(c, "")
                for c in all_cols
            ])

    print(f"[data] index.csv written → {out} ({len(runs)} runs)")
    return out


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rebuild_index()
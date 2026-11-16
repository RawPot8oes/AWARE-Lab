import csv, traceback
import numpy as np
import matplotlib.pyplot as plt

CSV = "flap_test_log_1791748166.csv"
MIN_S = 1.0      # min plateau length (use ~10 for the real 30 s holds)
SETTLE_S = 0.5   # seconds dropped at the start of each hold (use ~5 for real runs)
TOL = 0.3        # deg

try:
    with open(CSV) as f:
        r = csv.reader(f); next(r)
        d = np.array([[float(v) for v in row] for row in r if row])

    t, yaw, W = d[:, 0], d[:, 6], d[:, 7:13]
    rel = yaw - (-42.77)   # base yaw from your "At base pose" printout
    lvl = np.round(rel)
    ok = np.abs(rel - lvl) <= TOL

    levels, means, stds = [], [], []
    i, n = 0, len(t)
    while i < n:
        if not ok[i]:
            i += 1; continue
        j = i
        while j + 1 < n and ok[j + 1] and lvl[j + 1] == lvl[i]:
            j += 1
        if t[j] - t[i] >= MIN_S:
            idx = np.arange(i, j + 1)
            idx = idx[t[idx] >= t[i] + SETTLE_S]
            if len(idx) > 1:
                levels.append(lvl[i])
                means.append(W[idx].mean(0)); stds.append(W[idx].std(0))
        i = j + 1

    levels, means, stds = map(np.array, (levels, means, stds))
    o = np.argsort(levels)
    levels, means, stds = levels[o], means[o], stds[o]
    print(f"{len(levels)} holds, levels {levels.min():+.0f} to {levels.max():+.0f} deg")

    fig, ax = plt.subplots(3, 1, figsize=(11, 11))
    ax[0].plot(t, rel); ax[0].set_ylabel("Yaw offset [deg]"); ax[0].set_xlabel("Time [s]")
    for k, nme in enumerate(["Fx", "Fy", "Fz"]):
        ax[1].errorbar(levels, means[:, k], yerr=stds[:, k], marker="o", ms=3, capsize=2, label=nme)
    ax[1].set_ylabel("Force [N]"); ax[1].legend(ncol=3)
    for k, nme in enumerate(["Tx", "Ty", "Tz"]):
        ax[2].errorbar(levels, means[:, 3 + k], yerr=stds[:, 3 + k], marker="o", ms=3, capsize=2, label=nme)
    ax[2].set_ylabel("Torque [N·m]"); ax[2].set_xlabel("Yaw offset [deg]"); ax[2].legend(ncol=3)
    for a in ax: a.grid(True, ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("step_plot.png", dpi=150)
    print("saved step_plot.png")
    plt.show()
except Exception:
    traceback.print_exc()
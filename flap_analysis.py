"""Phase-averaged flapping-foil analysis (poster edition). Edit CONFIG and re-run."""
import os, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D, ScaledTranslation

# ---------------- CONFIG ----------------
DATA = "/mnt/user-data/uploads/"
OUT  = "/mnt/user-data/outputs/"
FREQ = 1.0; N_CYC = 30
DROP, DROP_END = 1, 1          # cycles discarded at start / end
NPH  = 400
FLOW_LABEL, STILL_LABEL = "Flow on", "Still water"   # e.g. "U = 0.25 m/s"
MODELS = ["Mount", "Rigid", "Flexible"]
LABELS = {"Mount": "Mount only", "Rigid": "Rigid foil", "Flexible": "Flexible foil"}
COL = {"Mount": "#7f7f7f", "Rigid": "#0072B2", "Flexible": "#D55E00"}   # Okabe-Ito, colorblind safe
ANGLE_COL = "#8E44AD"   # flap angle (called "pitch" in the CSV)
WITH_MOUNT_FIG = False   # fig2 (mount-only overlaid) -> optional/
SUMMARY = False         # per-cycle bar charts -> saved in optional/
# ----------------------------------------
CH = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 28, "axes.labelsize": 34,
    "axes.labelweight": "bold", "axes.titlesize": 38, "axes.titleweight": "bold",
    "xtick.labelsize": 28, "ytick.labelsize": 28, "legend.fontsize": 30,
    "lines.linewidth": 4, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.8, "xtick.major.width": 1.8, "ytick.major.width": 1.8,
    "xtick.major.size": 7, "ytick.major.size": 7, "axes.grid": True,
    "grid.color": "#dddddd", "grid.linewidth": 1.0, "savefig.bbox": "tight", "savefig.pad_inches": 0.6})
T = 1 / FREQ
XLAB = "Flap cycle (%)"
XT = [0, 0.25, 0.5, 0.75, 1.0]

def fx(ax, label=True):
    ax.set_xlim(-0.03 * T, 1.03 * T); ax.set_xticks([t * T for t in XT]); ax.set_xticklabels(["0", "25", "50", "75", "100"])
    if label: ax.set_xlabel(XLAB)

def save(fig, name):
    fig.savefig(OUT + name + ".png", dpi=300); fig.savefig(OUT + name + ".pdf"); plt.close(fig)

def load(model, flow):
    d = pd.read_csv(f"{DATA}_{model}_1Hz_30Cycles_{1000 if flow else 0}Flume.csv")
    t = d.t.values - d.t.values[0]
    ph = np.linspace(0, 1, NPH, endpoint=False)
    stack = {c: [] for c in CH + ["x", "pitch"]}
    for k in range(N_CYC):
        m = (t >= k * T) & (t < (k + 1) * T)
        if m.sum() < 100: continue
        p = (t[m] - k * T) / T
        for c in stack: stack[c].append(np.interp(ph, p, d[c].values[m]))
    S = {c: np.array(v) for c, v in stack.items()}
    S["ph"] = ph * T; S["t"], S["raw"] = t, d
    return S

DATAS = {(m, f): load(m, f) for m in MODELS for f in (0, 1)}
def use(S, c): return S[c][DROP:len(S[c]) - DROP_END]
def ms(S, c): a = use(S, c); return a.mean(0), a.std(0, ddof=1)
COND = [(0, STILL_LABEL), (1, FLOW_LABEL)]
def model_legend(fig, models, y=1.0):
    fig.legend(handles=[Line2D([], [], color=COL[m], lw=7, label=LABELS[m]) for m in models],
               loc="upper center", ncol=len(models), bbox_to_anchor=(0.5, y), frameon=False)

# ---- Fig 1: method ----
S, M = DATAS[("Flexible", 1)], DATAS[("Mount", 1)]
fig, ax = plt.subplots(1, 3, figsize=(28, 9), gridspec_kw={"width_ratios": [1.5, 1, 1]})
d, t = S["raw"], S["t"]
ax[0].plot(t, d.Fy, color=COL["Flexible"], lw=1.3)
for k in range(N_CYC + 1): ax[0].axvline(k, color="k", lw=0.7, alpha=0.35)
ax[0].set_ylim(-14, 14)
for x0, x1, ha in [(0, DROP, "left"), (N_CYC - DROP_END, N_CYC, "right")]:
    ax[0].axvspan(x0, x1, color="#777", alpha=0.35, lw=0)
    xc = (x0 + x1) / 2
    ax[0].annotate("discarded", xy=(xc, 1.0), xycoords=("data", "axes fraction"),
                   xytext=(x0 if ha == "left" else x1, 1.13), textcoords=("data", "axes fraction"),
                   ha=ha, va="bottom", fontsize=28, fontweight="bold", color="#333",
                   arrowprops=dict(arrowstyle="-|>", lw=3, color="#333", mutation_scale=28), annotation_clip=False)
ax[0].set(xlabel="Time (s)", ylabel="Force $F_y$ (N)", title="1. Raw signal", xlim=(0, 30))
ph = S["ph"]; a = use(S, "Fy"); mu, sd = a.mean(0), a.std(0, ddof=1)
am = use(M, "Fy"); mmu, msd = am.mean(0), am.std(0, ddof=1)
for row in a: ax[1].plot(ph, row, color="#999", lw=0.7, alpha=0.5)
ax[1].fill_between(ph, mu - sd, mu + sd, color=COL["Flexible"], alpha=0.3, lw=0)
ax[1].plot(ph, mu, color=COL["Flexible"], label="Flexible foil + mount")
ax[1].fill_between(ph, mmu - msd, mmu + msd, color=COL["Mount"], alpha=0.35, lw=0)
ax[1].plot(ph, mmu, color=COL["Mount"], label="Mount only")
ax[1].set(title="2. Average cycles", ylim=(-17, 28)); fx(ax[1])
ax[1].legend(loc="upper center", framealpha=0.95, fontsize=26)
net, err = mu - mmu, np.hypot(sd, msd)
ax[2].fill_between(ph, net - err, net + err, color=COL["Flexible"], alpha=0.3, lw=0)
ax[2].plot(ph, net, color=COL["Flexible"], label="Foil only (net)")
ax[2].plot(ph, mu, color=COL["Flexible"], lw=2, ls=":", label="Foil + mount")
ax[2].axhline(0, color="k", lw=1); ax[2].set(title="3. Mount removed", ylim=(-17, 28)); fx(ax[2])
ax[2].legend(loc="upper center", framealpha=0.95, fontsize=26)
fig.tight_layout(); save(fig, "fig1_method")

# ---- Fig 2: phase-averaged comparison incl. mount ----
if WITH_MOUNT_FIG:
    os.makedirs(OUT + "optional", exist_ok=True)
    rows = [("Fy", "Force $F_y$ (N)"), ("Tx", "Torque $T_x$ (N·m)"), ("Fz", "Force $F_z$ (N)"), ("Fx", "Force $F_x$ (N)")]
    fig, axs = plt.subplots(len(rows) + 1, 2, figsize=(22, 26), sharex=True,
                            gridspec_kw={"height_ratios": [0.75] + [1] * len(rows)})
    for j, (f, lab) in enumerate(COND):
        S0 = DATAS[("Rigid", f)]
        a1 = axs[0, j]
        a1.plot(S0["ph"], use(S0, "pitch").mean(0), color=ANGLE_COL); a1.axhline(0, color="k", lw=1)
        a1.set_ylabel("Flap angle (°)"); a1.set_yticks([-30, 0, 30]); a1.set_ylim(-42, 42); a1.set_title(lab, pad=14)
        for i, (c, yl) in enumerate(rows, start=1):
            ax = axs[i, j]
            for m in MODELS:
                mu, sd = ms(DATAS[(m, f)], c)
                ax.fill_between(S0["ph"], mu - sd, mu + sd, color=COL[m], alpha=0.25, lw=0)
                ax.plot(S0["ph"], mu, color=COL[m])
            ax.axhline(0, color="k", lw=1)
            if j == 0: ax.set_ylabel(yl)
            else: ax.sharey(axs[i, 0])
        fx(axs[-1, j])
    fig.align_ylabels(axs[:, 0]); fig.tight_layout(rect=(0, 0, 1, 0.965)); model_legend(fig, MODELS)
    save(fig, "optional/fig2_phase_averaged")


# ---- Fig 3: net foil loads ----
NETROWS = [("Fy", "Net $F_y$ (N)"), ("Tx", "Net $T_x$ (N·m)"), ("Fz", "Net $F_z$ (N)"), ("Fx", "Net $F_x$ (N)")]
fig, axs = plt.subplots(4, 2, figsize=(22, 22), sharex=True)
for j, (f, lab) in enumerate(COND):
    for i, (c, yl) in enumerate(NETROWS):
        ax = axs[i, j]; mm, ss = ms(DATAS[("Mount", f)], c)
        for m in ["Rigid", "Flexible"]:
            mu, sd = ms(DATAS[(m, f)], c); net, err = mu - mm, np.hypot(sd, ss)
            ax.fill_between(DATAS[(m, f)]["ph"], net - err, net + err, color=COL[m], alpha=0.25, lw=0)
            ax.plot(DATAS[(m, f)]["ph"], net, color=COL[m])
        ax.axhline(0, color="k", lw=1)
        if j == 0: ax.set_ylabel(yl)
        else: ax.sharey(axs[i, 0])
        if i == 0: ax.set_title(lab, pad=14)
    fx(axs[-1, j])
fig.align_ylabels(axs[:, 0]); fig.tight_layout(rect=(0, 0, 1, 0.96)); model_legend(fig, ["Rigid", "Flexible"])
save(fig, "fig3_net_foil_loads")

# ---- Fig 4 / 5: per-cycle metrics ----
panels = [("Fy", "rms", "$F_y$ RMS (N)"), ("Tx", "rms", "$T_x$ RMS (N·m)"),
          ("Fx", "mean", "Mean $F_x$ (N)"), ("Fz", "mean", "Mean $F_z$ (N)")]
w = 0.38
def bars(models, net, name, title):
    fig, axs = plt.subplots(2, 2, figsize=(17, 12)); out = []
    for ax, (c, kind, yl) in zip(axs.ravel(), panels):
        for j, (f, lab) in enumerate(COND):
            tare = use(DATAS[("Mount", f)], c).mean(0)
            for i, m in enumerate(models):
                a = use(DATAS[(m, f)], c) - (tare if net else 0)
                v = a.std(1) if kind == "rms" else a.mean(1)
                mu, sd = v.mean(), v.std(ddof=1)
                ax.bar(i + (j - 0.5) * w, mu, w, color=COL[m], alpha=1 if f else 0.45,
                       hatch="//" if f else None, edgecolor="k", lw=1.5)
                ax.errorbar(i + (j - 0.5) * w, mu, sd, color="k", capsize=8, lw=2.5, capthick=2.5)
                out.append(dict(metric=f"{'net_' if net else ''}{c}_{kind}", model=m, flow=lab, mean=mu, sd=sd, n=len(v)))
        ax.set_xticks(range(len(models))); ax.set_xticklabels([LABELS[m].replace(" foil", "") for m in models])
        ax.set_ylabel(("Net " if net else "") + yl); ax.axhline(0, color="k", lw=1); ax.grid(axis="x", visible=False)
    fig.legend(handles=[Patch(fc="#bbb", ec="k", label=STILL_LABEL), Patch(fc="#bbb", ec="k", hatch="//", label=FLOW_LABEL)],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, name)
    pd.DataFrame(out).round(4).to_csv(OUT + ("optional/summary_metrics_net.csv" if net else "optional/summary_metrics.csv"), index=False)
if SUMMARY:
    os.makedirs(OUT + "optional", exist_ok=True)
    bars(MODELS, False, "optional/fig4_summary_metrics", "")
    bars(["Rigid", "Flexible"], True, "optional/fig5_net_summary_metrics", "")

# ---- Fig 6: flap motion ----
S0 = DATAS[("Rigid", 0)]; ph = S0["ph"]
def sinfit(y):   # least-squares first harmonic -> smooth curve from the measured motion
    w = 2 * np.pi * ph / T; A = np.c_[np.sin(w), np.cos(w), np.ones_like(w)]
    return A @ np.linalg.lstsq(A, y, rcond=None)[0]
ang = sinfit(use(S0, "pitch").mean(0)); amp = (ang.max() - ang.min()) / 2
CA, CB, CC = "#1b9e77", "#e7298a", "#555555"
fig = plt.figure(figsize=(26, 11))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.12)
a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
a1.plot(ph, ang, color=ANGLE_COL); a1.axhline(0, color="k", lw=1)
a1.set(ylim=(-50, 50), ylabel="Flap angle (°)", title="Flap angle over one cycle"); a1.set_yticks([-30, 0, 30]); fx(a1)
for tt, c, lab in [(0.25, CA, "25% of cycle"), (0.75, CB, "75% of cycle")]:
    v = np.interp(tt, ph, ang); a1.plot(tt, v, "o", color=c, ms=24, zorder=5, mec="k", mew=2)
    a1.annotate(lab, (tt, v), xytext=(0, 34 if v > 0 else -46), textcoords="offset points", ha="center", fontsize=32, color=c, fontweight="bold")
def arm(theta_deg, L=1.0):
    th = np.radians(theta_deg); d = np.array([np.sin(th), -np.cos(th)]); n = np.array([np.cos(th), np.sin(th)])
    s_ = np.linspace(0, L, 50); w = 0.055 + 0.07 * s_ / L
    w = np.where(s_ > 0.88 * L, w * np.sqrt(np.clip(1 - ((s_ - 0.88 * L) / (0.12 * L)) ** 2, 0, 1)), w)
    ctr = s_[:, None] * d
    return np.vstack([ctr + w[:, None] * n, (ctr - w[:, None] * n)[::-1]])
from matplotlib.patches import Wedge
a2.add_patch(Wedge((0, 0), 1.0, -90 - amp, -90 + amp, fc="#eeeeee", ec="none", zorder=0))
a2.add_patch(Polygon(arm(0), closed=True, fc="none", ec=CC, lw=2.5, ls="--", zorder=2))
for th, c in [(-amp, CA), (amp, CB)]:
    a2.add_patch(Polygon(arm(th), closed=True, fc=c, ec="k", lw=2.5, alpha=0.9, zorder=3))
a2.plot(0, 0, "o", color="k", ms=20, zorder=6)
r = np.radians(np.linspace(-amp, amp, 60)); arc = 1.14 * np.c_[np.sin(r), -np.cos(r)]
a2.plot(arc[:, 0], arc[:, 1], color="k", lw=3)
for i0, i1 in [(2, 0), (-3, -1)]:
    a2.annotate("", xy=arc[i1], xytext=arc[i0], arrowprops=dict(arrowstyle="-|>", lw=3, color="k", mutation_scale=30))
a2.text(*(1.38 * np.array([np.sin(np.radians(-amp)), -np.cos(np.radians(-amp))])), f"\u2212{amp:.0f}°", color=CA, fontsize=38, fontweight="bold", ha="center", va="center")
a2.text(*(1.38 * np.array([np.sin(np.radians(amp)), -np.cos(np.radians(amp))])), f"+{amp:.0f}°", color=CB, fontsize=38, fontweight="bold", ha="center", va="center")
a2.text(0, -1.27, "0°", color=CC, fontsize=38, fontweight="bold", ha="center", va="center")
a2.set_aspect("equal"); a2.set_anchor("N"); a2.set_xlim(-1.08, 1.08); a2.set_ylim(-1.62, 0.12); a2.axis("off")
a2.set_title("Flapping range (schematic)")
fig.text(0.5, -0.04, f"Flap amplitude ±{amp:.0f}°  ·  1 Hz  ·  Schematic, not to scale", ha="center", fontsize=28, color="#444")
save(fig, "fig6_motion")
print("done")

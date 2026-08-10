#!/usr/bin/env python3
# Two-material bar, mixed u-p discretization: stacked setup/solve wall-clock time.
# Reads two_material_iters.csv (written by reproduce.sh --two-material) and draws
# one stacked bar per (mesh resolution, nu_top): preconditioner setup at the
# bottom, Krylov solve on top, so the bar height is the time to solution.
#
# Companion to plot_two_material.py, which plots the iteration counts from the
# same sweep. Both keep the same nu_top -> color mapping so a reader can follow a
# single ratio across the two figures.
import csv
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CSV = sys.argv[1] if len(sys.argv) > 1 else "two_material_iters.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "times_two_material_mixed.png"
DISC = sys.argv[3] if len(sys.argv) > 3 else "mixed"
RELTOL = 1.0e-6  # solver relative_tol; relres above this => did not converge

# Match scripts/analyze_statistics.py "docs" style.
plt.rcParams.update({
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.linewidth": 1.2, "axes.edgecolor": "#333333", "axes.axisbelow": True,
    "grid.color": "#b0b0b0", "grid.linestyle": "--", "grid.linewidth": 0.7,
    "grid.alpha": 0.7,
})

# nu -> color, identical to plot_two_material.py (Okabe-Ito; validated for CVD).
NU_ORDER = ["0.49", "0.499", "0.4999"]
NU_COLOR = {"0.49": "#0072B2", "0.499": "#E69F00", "0.4999": "#D55E00"}
NU_LABEL = {n: r"$\nu_{\mathrm{top}}=%s$" % n for n in NU_ORDER}


def tint(hex_color, amount=0.55):
    """Blend a hue toward white; used for the setup segment of each stack."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    mix = lambda c: int(round(c + (255 - c) * amount))
    return "#%02X%02X%02X" % (mix(r), mix(g), mix(b))


# The two stack segments share the ratio's hue: setup is a lighter, hatched tint
# and solve is the full-strength hue, so phase is encoded by lightness AND
# texture rather than by color alone.
SETUP_HATCH = "..."

required = ("disc", "nu", "dofs", "iters", "relres", "setup", "solve")
rows = [r for r in csv.DictReader(open(CSV))
        if r.get("disc") == DISC and all(r.get(key) for key in required)]
if not rows:
    sys.exit(f"no complete rows with disc={DISC!r} in {CSV}")

dof_list = sorted({int(r["dofs"]) for r in rows})


def dof_label(d):
    return f"{d/1e6:.1f}M" if d >= 1e6 else f"{round(d/1e3)}k"


xlabels = [dof_label(d) for d in dof_list]
xpos = list(range(len(dof_list)))


def lookup(nu, dofs):
    for r in rows:
        if r["nu"] == nu and int(r["dofs"]) == dofs:
            return (float(r["setup"]), float(r["solve"]), float(r["relres"]),
                    int(r["iters"]))
    return None


ymax = max(float(r["setup"]) + float(r["solve"]) for r in rows)
ytop = ymax * 1.20

fig, ax = plt.subplots(figsize=(10.0, 5.4))
w = 0.26
# Sub-second sweeps need a second decimal to say anything; second-scale ones read
# better without it.
tfmt = "%.1f" if ymax >= 10.0 else "%.2f"
for j, nu in enumerate(NU_ORDER):
    off = (j - 1) * w
    for xi, d in zip(xpos, dof_list):
        hit = lookup(nu, d)
        if hit is None:
            continue
        setup, solve, relres, iters = hit
        conv = relres <= RELTOL
        ax.bar(xi + off, setup, width=w, color=tint(NU_COLOR[nu]), zorder=3,
               edgecolor="#333333", linewidth=0.7, hatch=SETUP_HATCH,
               label=NU_LABEL[nu] if xi == 0 else None)
        ax.bar(xi + off, solve, width=w, bottom=setup, color=NU_COLOR[nu],
               zorder=3, edgecolor="#333333", linewidth=0.7)
        ax.text(xi + off, setup + solve + ytop * 0.012,
                (tfmt % (setup + solve)) + ("" if conv else "*"),
                ha="center", va="bottom", fontsize=12, rotation=90,
                fontweight="bold" if not conv else "normal", color="#333333")

ax.set_title(r"Mixed u–p  (FGMRES + MGR, mass-Schur):  time to solution",
             fontsize=17, fontweight="bold", pad=10)
ax.set_xticks(xpos)
ax.set_xticklabels(xlabels)
ax.set_xlabel("mesh resolution (displacement DOFs)", fontsize=15)
ax.set_ylabel("wall-clock time [s]", fontsize=15)
ax.set_ylim(0, ytop)
ax.grid(axis="y", zorder=0)
ax.tick_params(labelsize=13)

# Legend: the ratio hues, then the two stack segments shown neutrally so the
# setup/solve split is read from lightness + hatch rather than from a hue.
handles = [Patch(facecolor=NU_COLOR[n], edgecolor="#333333", label=NU_LABEL[n])
           for n in NU_ORDER]
handles.append(Patch(facecolor=tint("#666666"), edgecolor="#333333",
                     hatch=SETUP_HATCH, label="setup"))
handles.append(Patch(facecolor="#666666", edgecolor="#333333", label="solve"))
fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=12.5,
           frameon=True, framealpha=0.92, edgecolor="#cccccc",
           bbox_to_anchor=(0.5, 1.07))
fig.tight_layout()
fig.savefig(OUT, dpi=170)
print(f"wrote {OUT}")

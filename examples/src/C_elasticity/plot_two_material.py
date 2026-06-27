#!/usr/bin/env python3
# Two-material bar: grouped-bar iteration plots for the two LOCKING-FREE
# discretizations -- B-bar Q1-P0 (PCG+AMG) and mixed u-p (FGMRES+MGR with the
# scaled pressure-mass Schur as MGR's coarse operator).
# Reads two_material_iters.csv (written by reproduce.sh --two-material) and
# produces two side-by-side subplots: x = mesh resolution, y = Krylov
# iterations, one colored bar per top-material Poisson ratio.
import csv
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CSV = sys.argv[1] if len(sys.argv) > 1 else "two_material_iters.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "iters_two_material_bars.png"
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

# nu -> color (cooler = milder, hotter = closer to incompressible).
NU_ORDER = ["0.49", "0.499", "0.4999"]
NU_COLOR = {"0.49": "#0072B2", "0.499": "#E69F00", "0.4999": "#D55E00"}
NU_LABEL = {n: r"$\nu_{\mathrm{top}}=%s$" % n for n in NU_ORDER}

# Read CSV: disc,nu,nx,ny,nz,dofs,iters,relres
rows = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        rows.append(r)

# Mesh resolutions in run order (unique dofs, ascending).
dof_list = sorted({int(r["dofs"]) for r in rows})
def dof_label(d):
    return f"{d/1e6:.1f}M" if d >= 1e6 else f"{round(d/1e3)}k"
xlabels = [dof_label(d) for d in dof_list]
xpos = list(range(len(dof_list)))

def lookup(disc, nu, dofs):
    for r in rows:
        if r["disc"] == disc and r["nu"] == nu and int(r["dofs"]) == dofs:
            return int(r["iters"]), float(r["relres"])
    return None, None

ymax = max(int(r["iters"]) for r in rows)
ytop = ymax * 1.18

panels = [("bbar",  "B-bar Q1–P0  (PCG + AMG)"),
          ("mixed", "Mixed u–p  (FGMRES + MGR, mass-Schur)")]
fig, axes = plt.subplots(1, len(panels), figsize=(11.6, 5.4), sharey=True)
w = 0.26
for ax, (disc, title) in zip(axes, panels):
    for j, nu in enumerate(NU_ORDER):
        off = (j - 1) * w
        for xi, d in zip(xpos, dof_list):
            it, rr = lookup(disc, nu, d)
            if it is None:
                continue
            conv = rr is not None and rr <= RELTOL
            ax.bar(xi + off, it, width=w, color=NU_COLOR[nu], zorder=3,
                   edgecolor="#333333", linewidth=0.7,
                   hatch=None if conv else "////",
                   label=NU_LABEL[nu] if xi == 0 else None)
            ax.text(xi + off, it + ytop * 0.012, f"{it}" + ("" if conv else "*"),
                    ha="center", va="bottom", fontsize=12, rotation=90,
                    fontweight="bold" if not conv else "normal",
                    color="#333333")
    ax.set_title(title, fontsize=17, fontweight="bold", pad=10)
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("mesh resolution (displacement DOFs)", fontsize=15)
    ax.set_ylim(0, ytop)
    ax.grid(axis="y", zorder=0)
    ax.tick_params(labelsize=13)

axes[0].set_ylabel("Krylov iterations to rel. tol. $10^{-6}$", fontsize=15)

# Shared legend (nu colors) + non-converged note.
handles = [Patch(facecolor=NU_COLOR[n], edgecolor="#333333", label=NU_LABEL[n])
           for n in NU_ORDER]
handles.append(Patch(facecolor="white", edgecolor="#333333", hatch="////",
                     label="did not converge (100-iter cap)"))
fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=12.5,
           frameon=True, framealpha=0.92, edgecolor="#cccccc",
           bbox_to_anchor=(0.5, 1.07))
fig.tight_layout()
fig.savefig(OUT, dpi=170)
print(f"wrote {OUT}")

"""
Compare optimizer output between SiGe and SiIndent datasets.

Sections
--------
1. Raw homography statistics  — are the h values themselves different?
2. Convergence diagnostics    — iterations, residuals, dp_norms histograms
3. Parameter diff summary     — which processing/geometry settings differ
4. h-component histograms     — side-by-side per-component distributions
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ============================================================
# PATHS  — edit if needed
# ============================================================

SIGE = dict(
    label      = "SiGe",
    h_path     = "/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/SiGe/SiGe_2rows_partialInit_April_14_2026_npyfiles/SiGe_2rows_partialInit_homographies_April_14_2026.npy",
    iters_path = "/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/SiGe/SiGe_2rows_partialInit_April_14_2026_npyfiles/SiGe_2rows_partialInit_iterations_April_14_2026.npy",
    resid_path = "/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/SiGe/SiGe_2rows_partialInit_April_14_2026_npyfiles/SiGe_2rows_partialInit_residuals_April_14_2026.npy",
    dpnorm_path= "/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/SiGe/SiGe_2rows_partialInit_April_14_2026_npyfiles/SiGe_2rows_partialInit_dp_norms_April_14_2026.npy",
    hguess_path= "/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/SiGe/SiGe_2rows_partialInit_April_14_2026_npyfiles/SiGe_2rows_partialInit_h_guess_April_14_2026.npy",
    # post-processing geometry
    patshape      = (512, 512),
    pixel_size_um = 30.0,
    detector_tilt = 10.0,
    sample_tilt   = 70.0,
    pc_edax       = np.array([0.6871, 0.8929, 1.06971]),
    pc_convention = "standard",
    rows          = 2,
    cols          = 132,
)

SIINDENT = dict(
    label      = "SiIndent",
    h_path     = "/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/Si-Indent/SiIndent_April_14_2026_npyfiles/SiIndent_homographies_April_14_2026.npy",
    iters_path = "/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/Si-Indent/SiIndent_April_14_2026_npyfiles/SiIndent_iterations_April_14_2026.npy",
    resid_path = "/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/Si-Indent/SiIndent_April_14_2026_npyfiles/SiIndent_residuals_April_14_2026.npy",
    dpnorm_path= "/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/Si-Indent/SiIndent_April_14_2026_npyfiles/SiIndent_dp_norms_April_14_2026.npy",
    hguess_path= "/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/Si-Indent/SiIndent_April_14_2026_npyfiles/SiIndent_h_guess_April_14_2026.npy",
    # post-processing geometry
    patshape      = (516, 516),
    pixel_size_um = 55.0,
    detector_tilt = 2.0,
    sample_tilt   = 70.0,
    pc_edax       = np.array([0.508789, 0.765800, 0.624000]),
    pc_convention = "direct_electron",
    rows          = 41,
    cols          = 51,
)

SAVE_DIR = "/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/Optimizer_Comparison/"
os.makedirs(SAVE_DIR, exist_ok=True)

HLABELS = ["h11", "h12", "h13", "h21", "h22", "h23", "h31", "h32"]

# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(cfg):
    h = np.load(cfg["h_path"])
    if h.ndim != 2 or h.shape[1] != 8:
        h = h.reshape(-1, 8)
    iters  = np.load(cfg["iters_path"]).ravel()
    resids = np.load(cfg["resid_path"]).ravel()
    dpn    = np.load(cfg["dpnorm_path"]).ravel()
    hg = np.load(cfg["hguess_path"])
    if hg.ndim != 2 or hg.shape[1] != 8:
        hg = hg.reshape(-1, 8)
    return h, iters, resids, dpn, hg

print("Loading SiGe …")
h_sige,  iters_sige,  resids_sige,  dpn_sige,  hg_sige  = load_dataset(SIGE)
print("Loading SiIndent …")
h_si,    iters_si,    resids_si,    dpn_si,    hg_si    = load_dataset(SIINDENT)

# ============================================================
# SECTION 1 — RAW HOMOGRAPHY STATISTICS
# ============================================================

print("\n" + "="*60)
print("SECTION 1 — RAW HOMOGRAPHY STATISTICS")
print("="*60)
print(f"{'Component':<10}  {'SiGe mean':>14}  {'SiGe std':>14}  {'SiGe |max|':>14}  "
      f"{'SiIndent mean':>14}  {'SiIndent std':>14}  {'SiIndent |max|':>14}")
print("-"*100)
for i, lbl in enumerate(HLABELS):
    sg_mean = h_sige[:, i].mean();  sg_std = h_sige[:, i].std();  sg_max = np.abs(h_sige[:, i]).max()
    si_mean = h_si[:, i].mean();    si_std = h_si[:, i].std();    si_max = np.abs(h_si[:, i]).max()
    print(f"{lbl:<10}  {sg_mean:>14.4e}  {sg_std:>14.4e}  {sg_max:>14.4e}  "
          f"{si_mean:>14.4e}  {si_std:>14.4e}  {si_max:>14.4e}")

# ============================================================
# SECTION 2 — CONVERGENCE DIAGNOSTICS
# ============================================================

print("\n" + "="*60)
print("SECTION 2 — CONVERGENCE DIAGNOSTICS")
print("="*60)
for tag, iters, resids, dpn in [
    ("SiGe",    iters_sige, resids_sige, dpn_sige),
    ("SiIndent",iters_si,   resids_si,   dpn_si),
]:
    print(f"\n{tag}:")
    print(f"  iterations  — mean={iters.mean():.1f}  median={np.median(iters):.0f}  "
          f"max={iters.max():.0f}  (hit max: {(iters == iters.max()).sum()})")
    print(f"  residuals   — mean={resids.mean():.4e}  median={np.median(resids):.4e}  max={resids.max():.4e}")
    print(f"  dp_norms    — mean={dpn.mean():.4e}  median={np.median(dpn):.4e}  "
          f"min={dpn.min():.4e}  max={dpn.max():.4e}")

# ============================================================
# SECTION 3 — PARAMETER DIFF SUMMARY
# ============================================================

print("\n" + "="*60)
print("SECTION 3 — KEY PARAMETER DIFFERENCES")
print("="*60)

import conversions

def compute_xo(cfg):
    pc_bruker = conversions.Edax_to_Bruker_PC(cfg["pc_edax"])
    hc = np.array([0.5, 0.5])
    xo = conversions.Bruker_to_fractional_PC(pc_bruker, cfg["patshape"], cfg["pixel_size_um"], hc)
    return pc_bruker, xo

pc_sige, xo_sige = compute_xo(SIGE)
pc_si,   xo_si   = compute_xo(SIINDENT)

params = [
    ("patshape",       str(SIGE["patshape"]),          str(SIINDENT["patshape"])),
    ("pixel_size_um",  f"{SIGE['pixel_size_um']} µm",  f"{SIINDENT['pixel_size_um']} µm"),
    ("detector_tilt",  f"{SIGE['detector_tilt']}°",    f"{SIINDENT['detector_tilt']}°"),
    ("sample_tilt",    f"{SIGE['sample_tilt']}°",      f"{SIINDENT['sample_tilt']}°"),
    ("pc_convention",  SIGE["pc_convention"],           SIINDENT["pc_convention"]),
    ("PC (EDAX)",      str(SIGE["pc_edax"]),            str(SIINDENT["pc_edax"])),
    ("PC (Bruker)",    f"{pc_sige}",                    f"{pc_si}"),
    ("xo (h2F input)", f"{xo_sige}",                   f"{xo_si}"),
    ("xo[2] (z*)",     f"{xo_sige[2]:.4f}",            f"{xo_si[2]:.4f}"),
    ("n_patterns",     str(SIGE["rows"]*SIGE["cols"]),  str(SIINDENT["rows"]*SIINDENT["cols"])),
]
print(f"\n{'Parameter':<22}  {'SiGe':>40}  {'SiIndent':>40}")
print("-"*106)
for name, sg, si in params:
    marker = "  <<<" if sg != si else ""
    print(f"{name:<22}  {sg:>40}  {si:>40}{marker}")

# ============================================================
# SECTION 4 — h-COMPONENT HISTOGRAMS (side-by-side)
# ============================================================

fig, axes = plt.subplots(4, 2, figsize=(14, 14))
axes = axes.flatten()

for i, (ax, lbl) in enumerate(zip(axes, HLABELS)):
    sg = h_sige[:, i]
    si = h_si[:, i]

    # use common bin range for both
    all_vals = np.concatenate([sg, si])
    lo, hi = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
    bins = np.linspace(lo, hi, 60)

    ax.hist(sg, bins=bins, alpha=0.6, label=f"SiGe (σ={sg.std():.2e})", color="steelblue",   density=True)
    ax.hist(si, bins=bins, alpha=0.6, label=f"SiIndent (σ={si.std():.2e})", color="tomato", density=True)
    ax.axvline(sg.mean(), color="steelblue", linestyle="--", linewidth=1.2)
    ax.axvline(si.mean(), color="tomato",    linestyle="--", linewidth=1.2)
    ax.set_title(lbl, fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlabel("value")
    ax.set_ylabel("density")

plt.suptitle("Raw Homography Component Distributions\n(SiGe vs SiIndent)", fontsize=13)
plt.tight_layout()
save_path = f"{SAVE_DIR}/h_component_histograms.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show(block=False)
print(f"\nSaved: {save_path}")

# ============================================================
# SECTION 5 — CONVERGENCE COMPARISON FIGURE
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

datasets = [
    ("SiGe",     iters_sige,  resids_sige,  dpn_sige,  "steelblue"),
    ("SiIndent", iters_si,    resids_si,    dpn_si,    "tomato"),
]

# Compute shared bin edges and axis limits for each row before plotting
def _shared_bins(a, b, n):
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    return np.linspace(lo, hi, n + 1)

iters_bins  = _shared_bins(iters_sige,  iters_si,  30)
resids_bins = _shared_bins(resids_sige, resids_si, 40)
dpn_bins    = _shared_bins(dpn_sige,    dpn_si,    40)

row_bins = [iters_bins, resids_bins, dpn_bins]

for col, (label, iters, resids, dpn, color) in enumerate(datasets):
    # iterations histogram
    ax = axes[0, col]
    ax.hist(iters, bins=iters_bins, color=color, alpha=0.8)
    ax.set_title(f"{label} — iterations (mean={iters.mean():.1f})")
    ax.set_xlabel("iterations to converge")
    ax.set_ylabel("count")

    # residuals histogram
    ax = axes[1, col]
    ax.hist(resids, bins=resids_bins, color=color, alpha=0.8)
    ax.set_title(f"{label} — final residual (mean={resids.mean():.3e})")
    ax.set_xlabel("residual")
    ax.set_ylabel("count")

    # dp_norms histogram
    ax = axes[2, col]
    ax.hist(dpn, bins=dpn_bins, color=color, alpha=0.8)
    ax.set_title(f"{label} — dp_norms (mean={dpn.mean():.3e})")
    ax.set_xlabel("dp_norm")
    ax.set_ylabel("count")

# Enforce shared x and y limits across each row
for row in range(3):
    xlims = [axes[row, c].get_xlim() for c in range(2)]
    ylims = [axes[row, c].get_ylim() for c in range(2)]
    shared_x = (min(l[0] for l in xlims), max(l[1] for l in xlims))
    shared_y = (min(l[0] for l in ylims), max(l[1] for l in ylims))
    for c in range(2):
        axes[row, c].set_xlim(shared_x)
        axes[row, c].set_ylim(shared_y)

plt.suptitle("Optimizer Convergence: SiGe vs SiIndent", fontsize=13)
plt.tight_layout()
save_path = f"{SAVE_DIR}/convergence_comparison.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show(block=False)
print(f"Saved: {save_path}")

# ============================================================
# SECTION 6 — HOW MUCH DOES xo DIFFERENCE SCALE h → F?
# ============================================================
# The h13/h23 components feed into the translation part of F via xo[2] (z*).
# A larger z* means a smaller physical displacement per unit h13/h23.
# Print a rough scale factor.

print("\n" + "="*60)
print("SECTION 6 — xo[2] SCALE EFFECT ON h→F")
print("="*60)
print(f"  SiGe    xo[2] = {xo_sige[2]:.4f}   (z* — detector distance in pattern fractions)")
print(f"  SiIndent xo[2] = {xo_si[2]:.4f}")
ratio = xo_si[2] / xo_sige[2]
print(f"  Ratio SiIndent/SiGe = {ratio:.3f}  — h values need to be ~{ratio:.1f}x larger in")
print(f"  SiIndent to produce the same F (all else equal)")
print("\nNote: pixel_size_um also scales xo[0] and xo[1] (the in-plane PC offsets).")
print("      Detector tilt difference (10° vs 2°) changes the rotation matrix R,")
print("      rotating different components into the sample-frame strains.")

# ============================================================
# SECTION 7 — PARTIAL INIT DRIFT: first-row init vs final
# ============================================================
# For each dataset plot the init guess and the final h value
# along the first row for every component.  A growing gap
# between init and final as column index increases indicates
# the partial-init chain is accumulating error.

for cfg, h_final, h_init, color in [
    (SIGE,     h_sige, hg_sige, "steelblue"),
    (SIINDENT, h_si,   hg_si,   "tomato"),
]:
    label = cfg["label"]
    cols  = cfg["cols"]

    # slice out first row
    h_final_row = h_final[:cols, :]   # (cols, 8)
    h_init_row  = h_init[:cols, :]
    error_row   = h_final_row - h_init_row   # correction the optimizer applied
    x = np.arange(cols)

    fig, axes = plt.subplots(8, 1, figsize=(12, 20), sharex=True)
    fig.suptitle(
        f"{label} — partial init drift along first row\n"
        f"init (dashed) vs final (solid) vs correction (shaded)",
        fontsize=12,
    )

    for i, (ax, lbl) in enumerate(zip(axes, HLABELS)):
        ax.plot(x, h_final_row[:, i], color=color,      linewidth=1.5, label="final")
        ax.plot(x, h_init_row[:, i],  color=color,      linewidth=1.2, linestyle="--",
                alpha=0.6, label="init guess")
        ax.fill_between(x, h_init_row[:, i], h_final_row[:, i],
                         color=color, alpha=0.15, label="correction")
        ax.axhline(0, color="k", linewidth=0.5, linestyle=":")
        ax.set_ylabel(lbl, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Column index (first row)", fontsize=10)
    plt.tight_layout()
    save_path = f"{SAVE_DIR}/{label}_init_vs_final_first_row.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show(block=False)
    print(f"Saved: {save_path}")

    # Also print the RMS correction per component
    print(f"\n{label} — RMS(final − init) per component along first row:")
    for i, lbl in enumerate(HLABELS):
        rms = np.sqrt(np.mean(error_row[:, i]**2))
        print(f"  {lbl}: {rms:.4e}")

# ============================================================
# SECTION 8 — SiIndent h11: 3D surface of init and final
# ============================================================

rows_si = SIINDENT["rows"]   # 41
cols_si = SIINDENT["cols"]   # 51

col_grid, row_grid = np.meshgrid(np.arange(cols_si), np.arange(rows_si))

for fig_title, h_data in [("init guess", hg_si), ("final", h_si)]:
    fig = plt.figure(figsize=(22, 12))
    fig.suptitle(f"SiIndent — all h components ({fig_title})", fontsize=14)

    for i, lbl in enumerate(HLABELS):
        data = h_data[:, i].reshape(rows_si, cols_si)

        # per-component z limits shared across init and final figures
        z_all = np.concatenate([h_si[:, i], hg_si[:, i]])
        z_min, z_max = z_all.min(), z_all.max()
        # avoid degenerate range
        if np.isclose(z_min, z_max):
            z_min -= 1e-9; z_max += 1e-9

        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        surf = ax.plot_surface(
            col_grid, row_grid, data,
            cmap="coolwarm",
            vmin=z_min, vmax=z_max,
            linewidth=0, antialiased=True, alpha=0.9,
        )
        fig.colorbar(surf, ax=ax, shrink=0.4, pad=0.1)
        ax.set_xlabel("Col", fontsize=8, labelpad=2)
        ax.set_ylabel("Row", fontsize=8, labelpad=2)
        ax.set_zlim(z_min, z_max)
        ax.set_title(lbl, fontsize=10)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    save_path = f"{SAVE_DIR}/SiIndent_h_all_{fig_title.replace(' ', '_')}_3d.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show(block=False)
    print(f"Saved: {save_path}")

plt.show()
print("\nDone.")

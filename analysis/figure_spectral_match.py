"""
figure_spectral_match.py — publication figure for the effect of spectral
matching on simulated-pattern gradients.

This is a clean, publication-oriented distillation of the Figure-7b diagnostic
in ``debug_two_patterns.py``.  It loads a simulated reference and an
experimental target pattern (consecutive entries in one .up2 by default),
applies ``utilities.spectral_match_pattern`` to the sim using an experimental
amplitude target, and shows the effect on the spatial gradients that IC-GN
actually solves on (∇R via a 5th-order spline — same convention as
``get_homography_cpu.optimize``).

The science: a simulated pattern under-represents some Kikuchi-band
orientations, starving the corresponding gradient channels (notably the
y-gradients that drive h_22 / h_23 / h_32 in the IC-GN Hessian).  Spectral
matching reshapes the sim's amplitude spectrum to the experimental one while
preserving phase (band geometry), restoring those gradients.

Run
---
    python figure_spectral_match.py

Edit the USER INPUTS block to point at your data, then the figure is written
to OUT_PATH (PDF, vector) plus a PNG preview.  Set SHOW = True to also pop it
up interactively.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import interpolate

from fileio import Data
from core import utilities as _utils


# ───────── USER INPUTS ──────────────────────────────────────────────────────

up2_path = "/Users/crestiennedechaine/OriginalData/Si_Ge_Dataset/pats_newE_v6.up2"
sim_idx  = 7797       # pattern index of the simulated reference
exp_idx  = 7798       # pattern index of the experimental target

# Pattern-processing settings — mirror your runner / GUI Step 5 exactly so the
# figure represents the gradients the pipeline actually optimises.
low_pass_sigma   = 1.0
high_pass_sigma  = 10.0
mask_type        = "none"
gamma_correction = 0.80
flip_x           = False

crop_fraction = 0.80      # subset used for gradient diagnostics (matches IC-GN)

# Spectral target: "population" uses the average |FFT| over random exp patterns
# (this is what the pipeline's spectral_match_ref toggle does — recommended for
# publication), "single" uses just exp_idx's own |FFT|.
TARGET_MODE       = "population"
POP_N_SAMPLES     = 10        # patterns averaged for the population target

# Figure layout: "full" = 2×3 (gradient maps + flower + spectrum ratio + ZNCC
# bars); "compact" = 2×2 (gradient maps + flower only).
LAYOUT = "full"

# Which signed gradient channel to show in the top row.  "y" is the channel
# that is typically deficient in sim; "x" or "mag" also accepted.
GRAD_CHANNEL = "y"

OUT_PATH = "figures/spectral_match_effect.pdf"
SAVE_PNG = True               # also write a .png preview next to the PDF
SHOW     = False              # plt.show() at the end


# ───────── Publication style ─────────────────────────────────────────────────

plt.rcParams.update({
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "pdf.fonttype":     42,   # editable text in Illustrator/Inkscape
    "ps.fonttype":      42,
})

DIVERGING = "RdBu_r"
C_SIM, C_MATCH, C_EXP = "tab:blue", "tab:green", "tab:orange"


# ───────── Load patterns ─────────────────────────────────────────────────────

pat_obj = Data.UP2(up2_path)
pat_obj.set_processing(
    low_pass_sigma=low_pass_sigma,
    high_pass_sigma=high_pass_sigma,
    truncate_std_scale=3.0,
    mask_type=mask_type,
    center_cross_half_width=6,
    gamma=gamma_correction,
    flip_x=flip_x,
)

sim_pat = pat_obj.read_pattern(sim_idx, process=True).astype(np.float32)
exp_pat = pat_obj.read_pattern(exp_idx, process=True).astype(np.float32)

H, W = pat_obj.patshape
h0   = (W // 2, H // 2)
crop_row = int(H * (1 - crop_fraction) / 2)
crop_col = int(W * (1 - crop_fraction) / 2)
subset_slice = (slice(crop_row, H - crop_row), slice(crop_col, W - crop_col))


# ───────── Spectral matching ─────────────────────────────────────────────────

if TARGET_MODE == "population":
    target_amp = _utils.average_exp_amplitude_spectrum(
        pat_obj, n_samples=POP_N_SAMPLES, exclude_idx=[sim_idx, exp_idx]
    )
    target_label = f"population ⟨|FFT|⟩  (n={POP_N_SAMPLES})"
elif TARGET_MODE == "single":
    target_amp = np.abs(np.fft.fft2(exp_pat - exp_pat.mean()))
    target_label = "single exp |FFT|"
else:
    raise ValueError(f"TARGET_MODE must be 'population' or 'single', got {TARGET_MODE!r}")

sim_pat_matched = _utils.spectral_match_pattern(sim_pat, target_amp)


# ───────── Gradients (5th-order spline — IC-GN convention) ───────────────────

def _gradients(pat):
    x = np.arange(pat.shape[1]) - h0[0]
    y = np.arange(pat.shape[0]) - h0[1]
    spline = interpolate.RectBivariateSpline(x, y, pat.T, kx=5, ky=5)
    X, Y = np.meshgrid(x, y, indexing="xy")
    xi0, xi1 = X[subset_slice].flatten(), Y[subset_slice].flatten()
    out_shape = X[subset_slice].shape
    Gx = spline(xi0, xi1, dx=1, dy=0, grid=False).reshape(out_shape)
    Gy = spline(xi0, xi1, dx=0, dy=1, grid=False).reshape(out_shape)
    return Gx, Gy


sim_Gx,  sim_Gy  = _gradients(sim_pat)
simm_Gx, simm_Gy = _gradients(sim_pat_matched)
exp_Gx,  exp_Gy  = _gradients(exp_pat)


def _pick(channel, Gx, Gy):
    if channel == "x":   return Gx, r"$\partial R/\partial x$"
    if channel == "y":   return Gy, r"$\partial R/\partial y$"
    if channel == "mag": return np.sqrt(Gx**2 + Gy**2), r"$|\nabla R|$"
    raise ValueError("GRAD_CHANNEL must be 'x', 'y', or 'mag'")


sim_G,  glabel = _pick(GRAD_CHANNEL, sim_Gx,  sim_Gy)
simm_G, _      = _pick(GRAD_CHANNEL, simm_Gx, simm_Gy)
exp_G,  _      = _pick(GRAD_CHANNEL, exp_Gx,  exp_Gy)


# ───────── Metrics ───────────────────────────────────────────────────────────

def _zncc(a, b):
    a = a.ravel().astype(np.float64); a -= a.mean()
    b = b.ravel().astype(np.float64); b -= b.mean()
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def _sym_lim(*arrs, pct=99):
    return float(np.percentile(np.abs(np.concatenate([a.ravel() for a in arrs])), pct))


def _polar_hist(Gx, Gy, bins):
    theta = np.arctan2(Gy, Gx).ravel()
    w     = np.sqrt(Gx**2 + Gy**2).ravel()
    h, edges = np.histogram(theta, bins=bins, weights=w)
    centers  = 0.5 * (edges[:-1] + edges[1:])
    return h, centers


def _aniso(h):
    h = np.asarray(h, dtype=np.float64)
    return float(h.std() / max(h.mean(), 1e-12))


def _vert_frac(centers, h, hw_deg=20.0):
    hw = np.deg2rad(hw_deg)
    m  = np.abs(np.abs(centers) - np.pi / 2) <= hw
    return float(h[m].sum() / max(h.sum(), 1e-12))


bins = np.linspace(-np.pi, np.pi, 73)   # 5° bins
hist_sim,  centers = _polar_hist(sim_Gx,  sim_Gy,  bins)
hist_simm, _       = _polar_hist(simm_Gx, simm_Gy, bins)
hist_exp,  _       = _polar_hist(exp_Gx,  exp_Gy,  bins)

# shape-normalised (each curve sums to 1) → compares direction, not activity
n_sim  = hist_sim  / max(hist_sim.sum(),  1e-12)
n_simm = hist_simm / max(hist_simm.sum(), 1e-12)
n_exp  = hist_exp  / max(hist_exp.sum(),  1e-12)

zncc_x_before, zncc_x_after = _zncc(sim_Gx, exp_Gx), _zncc(simm_Gx, exp_Gx)
zncc_y_before, zncc_y_after = _zncc(sim_Gy, exp_Gy), _zncc(simm_Gy, exp_Gy)

metrics = {
    "anisotropy":  (_aniso(hist_sim),  _aniso(hist_simm),  _aniso(hist_exp)),
    "vert_frac":   (_vert_frac(centers, hist_sim),
                    _vert_frac(centers, hist_simm),
                    _vert_frac(centers, hist_exp)),
    "ZNCC_Gx":     (zncc_x_before, zncc_x_after, 1.0),
    "ZNCC_Gy":     (zncc_y_before, zncc_y_after, 1.0),
}


# ───────── Figure ────────────────────────────────────────────────────────────

def _panel_letter(ax, s):
    ax.text(0.02, 0.98, s, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left",
            color="black", bbox=dict(boxstyle="round,pad=0.15",
                                     fc="white", ec="none", alpha=0.7))


def _grad_row(axes, glim):
    titles = ["Simulated (original)", "Simulated (matched)", "Experimental (target)"]
    fields = [sim_G, simm_G, exp_G]
    im = None
    for ax, fld, ttl, let in zip(axes, fields, titles, "abc"):
        im = ax.imshow(fld, cmap=DIVERGING, vmin=-glim, vmax=glim)
        ax.set_title(ttl)
        ax.set_xticks([]); ax.set_yticks([])
        _panel_letter(ax, f"({let})")
    return im


def _polar_panel(ax):
    ax.plot(centers, n_sim,  color=C_SIM,   lw=1.2, ls="--", label="sim (orig.)")
    ax.plot(centers, n_simm, color=C_MATCH, lw=1.6,          label="sim (matched)")
    ax.plot(centers, n_exp,  color=C_EXP,   lw=1.6,          label="exp (target)")
    ax.set_title(f"Gradient orientation\n(|∇|-weighted, ∑=1)", pad=12)
    ax.set_yticklabels([])
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12))


def _spectrum_panel(ax):
    ratio = target_amp / (np.abs(np.fft.fft2(sim_pat - sim_pat.mean())) + 1e-6)
    log_ratio = np.log10(np.clip(np.fft.fftshift(ratio), 1e-2, 1e2))
    im = ax.imshow(log_ratio, cmap="magma", vmin=-1, vmax=1)
    ax.set_title(f"Amplitude deficit\nlog$_{{10}}$(target / |FFT$_{{sim}}$|)")
    ax.set_xticks([]); ax.set_yticks([])
    _panel_letter(ax, "(e)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _zncc_panel(ax):
    labels = [r"$G_x$", r"$G_y$"]
    before = [zncc_x_before, zncc_y_before]
    after  = [zncc_x_after,  zncc_y_after]
    xp = np.arange(len(labels)); w = 0.36
    ax.bar(xp - w/2, before, width=w, color=C_SIM,   label="original", alpha=0.8)
    ax.bar(xp + w/2, after,  width=w, color=C_MATCH, label="matched")
    ax.axhline(1.0, color="k", lw=0.7, ls=":")
    ax.set_xticks(xp); ax.set_xticklabels(labels)
    ax.set_ylabel("ZNCC vs exp")
    ax.set_ylim(0, 1.05)
    ax.set_title("Gradient agreement")
    ax.legend()
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    _panel_letter(ax, "(f)")


glim = _sym_lim(sim_G, simm_G, exp_G, pct=99)

if LAYOUT == "full":
    fig = plt.figure(figsize=(10.5, 6.6))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.22)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    im = _grad_row([ax_a, ax_b, ax_c], glim)
    cbar = fig.colorbar(im, ax=[ax_a, ax_b, ax_c], fraction=0.025, pad=0.01)
    cbar.set_label(glabel)
    _polar_panel(fig.add_subplot(gs[1, 0], projection="polar"))
    _spectrum_panel(fig.add_subplot(gs[1, 1]))
    _zncc_panel(fig.add_subplot(gs[1, 2]))
elif LAYOUT == "compact":
    fig = plt.figure(figsize=(9.5, 6.2))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.22,
                            height_ratios=[1, 1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    im = _grad_row([ax_a, ax_b, ax_c], glim)
    cbar = fig.colorbar(im, ax=[ax_a, ax_b, ax_c], fraction=0.025, pad=0.01)
    cbar.set_label(glabel)
    _polar_panel(fig.add_subplot(gs[1, :], projection="polar"))
else:
    raise ValueError(f"LAYOUT must be 'full' or 'compact', got {LAYOUT!r}")

fig.suptitle(
    f"Spectral matching of simulated reference → experiment  "
    f"(gradient channel: {glabel}, target: {target_label})",
    fontsize=10.5, fontweight="bold", y=0.99,
)


# ───────── Save / report ─────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved {OUT_PATH}")
if SAVE_PNG:
    png_path = os.path.splitext(OUT_PATH)[0] + ".png"
    fig.savefig(png_path, bbox_inches="tight")
    print(f"Saved {png_path}")

print()
print(f"{'metric':<14}{'original':>12}{'matched':>12}{'target':>12}")
print("-" * 50)
for name, (b, a, t) in metrics.items():
    print(f"{name:<14}{b:>12.4f}{a:>12.4f}{t:>12.4f}")

if SHOW:
    plt.show()
else:
    plt.close(fig)

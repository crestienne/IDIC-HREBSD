"""
spectral_match_walkthrough.py — an *intuition-building* visualization of what
spectral matching actually does, step by step.

This is NOT the publication figure (see ``figure_spectral_match.py`` for that).
The goal here is to make the mechanism obvious to the eye:

    1. Spectral matching reshapes the SIMULATED pattern's amplitude spectrum so
       it equals the EXPERIMENTAL amplitude template, while leaving the sim's
       PHASE (band geometry) untouched.  Row 2 shows the matched spectrum
       becoming the target; (g) shows *where* in frequency space amplitude is
       added vs removed.
    2. The downstream effect is in real space: the simulated reference's
       GRADIENTS are too directionally UNIFORM compared to experiment.  After
       matching, the sim's gradients take on the experimental directional
       ASYMMETRY — which is exactly what the IC-GN solve needs.  Panels (h)/(i)
       show the matched curve (green) leaving the sim (blue) and landing on
       experiment (orange).

NOTE on amplitude vs gradient anisotropy: the *raw* amplitude spectrum's
directional spread does NOT map one-to-one onto gradient directionality —
gradients weight each frequency by k² and rotate it 90°.  So the honest claim,
and the one this figure makes, lives in the GRADIENT panels (h)/(i), not in a
naive "amplitude is rounder" statement.  Panel (g) just shows the frequency-
domain edit; the gradient panels show its consequence.

The figure is a 3×3 story:

      row 1   spatial patterns ...... sim | matched | exp
      row 2   amplitude spectra ..... sim | matched | exp   (log|FFT|, fftshift)
      row 3   (g) amplitude change map  log10(target / |FFT_sim|)
              (h) gradient-orientation flower  (the money plot)
              (i) gradient anisotropy index bars

Run
---
    python -m analysis.spectral_match_walkthrough      # from repo root

Edit the USER INPUTS block to point at your data.
"""

import os
import sys

# Allow running directly (`python analysis/spectral_match_walkthrough.py`) as
# well as via `-m`: put the repo root (this file's parent's parent) on sys.path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

# Spectral target: "population" averages |FFT| over random exp patterns (what
# the pipeline's spectral_match_ref toggle does); "single" uses exp_idx alone.
TARGET_MODE   = "population"
POP_N_SAMPLES = 10

OUT_PATH = "figures/spectral_match_walkthrough.pdf"
SAVE_PNG = True
SHOW     = False


# ───────── Style ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

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
    target_label = f"population ⟨|FFT|⟩ (n={POP_N_SAMPLES})"
elif TARGET_MODE == "single":
    target_amp = np.abs(np.fft.fft2(exp_pat - exp_pat.mean()))
    target_label = "single exp |FFT|"
else:
    raise ValueError(f"TARGET_MODE must be 'population' or 'single', got {TARGET_MODE!r}")

sim_pat_matched = _utils.spectral_match_pattern(sim_pat, target_amp)


# ───────── Amplitude spectra (for display) ───────────────────────────────────

def _log_amp(pat):
    """fftshifted log-amplitude spectrum, for imshow."""
    F = np.abs(np.fft.fftshift(np.fft.fft2(pat - pat.mean())))
    return np.log10(F + 1.0)


amp_sim   = _log_amp(sim_pat)
amp_match = _log_amp(sim_pat_matched)
amp_exp   = np.log10(np.fft.fftshift(target_amp) + 1.0)   # the matching target


# ───────── Amplitude change map: what matching adds vs removes ───────────────
# This is exactly the per-frequency factor spectral_match_pattern applies (same
# eps floor and cap), so the map shows literally where the sim spectrum is
# boosted (>0 in log10) or suppressed (<0).  Bright = amplitude added.

_F_sim   = np.abs(np.fft.fft2(sim_pat - sim_pat.mean()))
_eps     = 1e-3 * _F_sim.max()
_ratio   = np.minimum(target_amp / (_F_sim + _eps), 10.0)
change_map = np.log10(np.fft.fftshift(np.clip(_ratio, 1e-2, 1e2)))


def _norm_mean(p):
    """normalise so the mean is 1 → curve shows ONLY directional shape."""
    return p / max(p.mean(), 1e-12)


def _mirror(centers, prof):
    """duplicate a [0,pi) profile to [0,2pi) for a full polar ring."""
    th = np.concatenate([centers, centers + np.pi, centers[:1]])
    pr = np.concatenate([prof, prof, prof[:1]])
    return th, pr


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


def _grad_orientation_profile(Gx, Gy, nbins=120):
    """|∇|-weighted distribution of gradient directions, folded to [0,pi)."""
    theta = np.mod(np.arctan2(Gy, Gx).ravel(), np.pi)
    w     = np.sqrt(Gx**2 + Gy**2).ravel()
    bins  = np.linspace(0.0, np.pi, nbins + 1)
    h, edges = np.histogram(theta, bins=bins, weights=w)
    centers  = 0.5 * (edges[:-1] + edges[1:])
    return centers, h


g_th, g_sim   = _grad_orientation_profile(sim_Gx,  sim_Gy)
_,    g_match = _grad_orientation_profile(simm_Gx, simm_Gy)
_,    g_exp   = _grad_orientation_profile(exp_Gx,  exp_Gy)


# ───────── Anisotropy index (coefficient of variation of the profile) ────────

def _aniso(p):
    p = np.asarray(p, dtype=np.float64)
    return float(p.std() / max(p.mean(), 1e-12))


aniso_grad = (_aniso(g_sim), _aniso(g_match), _aniso(g_exp))


# ───────── Figure ────────────────────────────────────────────────────────────

def _letter(ax, s, polar=False):
    x, y = (0.5, 1.18) if polar else (0.02, 0.98)
    ha   = "center" if polar else "left"
    ax.text(x, y, s, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="top", ha=ha,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))


fig = plt.figure(figsize=(11, 10))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.38, wspace=0.28)

# ── Row 1: spatial patterns ──────────────────────────────────────────────────
spatial = [(sim_pat, "Simulated (original)", "a"),
           (sim_pat_matched, "Simulated (matched)", "b"),
           (exp_pat, "Experimental (target)", "c")]
for col, (img, ttl, let) in enumerate(spatial):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(img, cmap="gray")
    ax.set_title(ttl)
    ax.set_xticks([]); ax.set_yticks([])
    _letter(ax, f"({let})")

# ── Row 2: amplitude spectra ─────────────────────────────────────────────────
amp_imgs = [(amp_sim, "log |FFT|  — sim", "d"),
            (amp_match, "log |FFT|  — matched", "e"),
            (amp_exp, "log |FFT|  — target", "f")]
vmax = max(a.max() for a, _, _ in amp_imgs)
for col, (img, ttl, let) in enumerate(amp_imgs):
    ax = fig.add_subplot(gs[1, col])
    ax.imshow(img, cmap="magma", vmin=0, vmax=vmax)
    ax.set_title(ttl)
    ax.set_xticks([]); ax.set_yticks([])
    _letter(ax, f"({let})")

# ── Row 3a: amplitude change map (what matching edits in frequency) ──────────
ax_g = fig.add_subplot(gs[2, 0])
im_g = ax_g.imshow(change_map, cmap="RdBu_r", vmin=-1, vmax=1)
ax_g.set_title("Amplitude change\nlog$_{10}$(target / |FFT$_{sim}$|)")
ax_g.set_xticks([]); ax_g.set_yticks([])
cb = plt.colorbar(im_g, ax=ax_g, fraction=0.046, pad=0.04,
                  ticks=[-1, 0, 1])
cb.ax.set_yticklabels(["÷10", "×1", "×10"])
_letter(ax_g, "(g)")

# ── Row 3b: gradient orientation flower (polar) ──────────────────────────────
ax_h = fig.add_subplot(gs[2, 1], projection="polar")
for prof, c, lbl, lw, ls in [
    (g_sim,   C_SIM,   "sim (orig.)",   1.3, "--"),
    (g_match, C_MATCH, "sim (matched)", 1.8, "-"),
    (g_exp,   C_EXP,   "exp (target)",  1.8, "-"),
]:
    th, pr = _mirror(g_th, _norm_mean(prof))
    ax_h.plot(th, pr, color=c, lw=lw, ls=ls, label=lbl)
ax_h.set_title("Gradient orientation\n(|∇|-weighted, mean=1)", pad=16)
ax_h.set_yticklabels([])
ax_h.legend(loc="upper right", bbox_to_anchor=(1.45, 1.18))
_letter(ax_h, "(h)", polar=True)

# ── Row 3c: gradient anisotropy index bars ───────────────────────────────────
ax_i = fig.add_subplot(gs[2, 2])
xp = np.arange(3)
ax_i.bar(xp, aniso_grad, color=[C_SIM, C_MATCH, C_EXP])
ax_i.set_xticks(xp)
ax_i.set_xticklabels(["sim\n(orig.)", "sim\n(matched)", "exp\n(target)"])
ax_i.set_ylabel("gradient anisotropy  (std/mean)")
ax_i.set_title("Gradient directional asymmetry")
ax_i.grid(True, axis="y", ls="--", alpha=0.4)
# annotate the "injection": sim → matched moves toward exp
ax_i.annotate("", xy=(1, aniso_grad[1]), xytext=(0, aniso_grad[0]),
              arrowprops=dict(arrowstyle="->", color="0.3", lw=1.2))
_letter(ax_i, "(i)")

fig.suptitle(
    "What spectral matching does: inject the experimental directional "
    f"anisotropy into the simulated reference's gradients\n(target: {target_label})",
    fontsize=11, fontweight="bold", y=0.995,
)


# ───────── Save / report ─────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved {OUT_PATH}")
if SAVE_PNG:
    png = os.path.splitext(OUT_PATH)[0] + ".png"
    fig.savefig(png, bbox_inches="tight")
    print(f"Saved {png}")

s, m, e = aniso_grad
print()
print("Gradient anisotropy index (std/mean of |∇|-weighted orientation):")
print(f"  sim (orig.)   {s:.4f}")
print(f"  sim (matched) {m:.4f}")
print(f"  exp (target)  {e:.4f}")
print()
print("Read it as: 'matched' should move OFF the sim value and TOWARD exp — "
      "that is the directional asymmetry being injected into the gradients.")

if SHOW:
    plt.show()
else:
    plt.close(fig)

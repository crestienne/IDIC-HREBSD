"""
Diagnostic tool for diagnosing strain error caused by using a simulated
reference pattern in IC-GN HREBSD.

Generates a set of figures in debug/simref_diagnostics/ covering:

  1. Pattern comparison      — real vs simulated, processed side by side
  2. Intensity histograms    — checks for distribution mismatch after preprocessing
  3. Power spectra           — checks for frequency-content mismatch
  4. Gradient comparison     — checks gradient magnitude / direction differences
                               (this is what directly biases the IC-GN Hessian)
  5. Hessian diagnostics     — diagonal of H and condition number for both
                               references; shows which parameters are poorly
                               constrained and whether the stiffness differs
  6. Radial profile of diff  — radial mean of (real - sim) in the subset; a
                               non-zero radial trend indicates residual PC error
  7. Pattern profile slices  — horizontal + vertical intensity slices through
                               the pattern centre; easy to spot blur differences
  8. Gradient line profiles  — Gx, Gy, and ‖G‖ traces along the mid-row and
                               mid-column of the subset; localises where the
                               gradient mismatch is worst across the detector

Point the file paths at the same inputs used in your runner script.

Usage
-----
    python debug_simulated_reference.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, linalg

import Data
import utilities
import conversions
import get_homography_cpu as core
from optimize_reference import (optimize_pc_and_euler,
                                optimize_preprocessing_params,
                                optimize_preprocessing_params_independent,
                                compute_h_landscape)
from PatternSimulation.SimPatGen import patternSimulation

# =============================================================================
# USER INPUTS  — match your runner script
# =============================================================================

ang_path            = '/Users/crestiennedechaine/OriginalData/Si_Ge_Dataset/dp_Si_new_refined.ang'
up2_path            = '/Users/crestiennedechaine/OriginalData/Si_Ge_Dataset/SiGe_updated_512x512.up2'
master_pattern_path = '/Users/crestiennedechaine/OriginalData/Si_Ge_Dataset/DI_largerRegion/MCoutput.h5'
ref_idx_yx          = (0, 0)          # (row, col) — must match GUI reference position
# TIP: GUI shows flat pattern index in the log. To convert:
#   import numpy as np; print(np.unravel_index(135, (n_rows, n_cols)))
sample_tilt_deg     = 70.0            # sample tilt (degrees)  — Step 2 in GUI
detector_tilt_deg   = 10.0            # detector tilt (degrees) — Step 2 in GUI
crop_fraction       = 0.8             # must match optimize() call

# Pattern processing — must match Step 3 settings in the GUI exactly
low_pass_sigma      = 1.0             # GUI: "Low-pass sigma"
high_pass_sigma     = 10.0            # GUI: "High-pass sigma"
mask_type           = "none"          # GUI: "Mask type"  ("none", "circular", "center_cross")
clahe_kernel        = (5, 5)          # GUI: "CLAHE kernel"  (enter as a square side, e.g. 5 → (5,5))
clahe_clip          = 0.01            # GUI: "CLAHE clip limit"
use_clahe           = False           # GUI: "Use CLAHE" checkbox
gamma_correction    = 0.80            # GUI: "Gamma"  (1.0 = off)
flip_x              = False           # GUI: "Flip patterns vertically" checkbox

# Sim high-pass sigma override — applied only to the simulated pattern during PC/Euler
# refinement.  Set higher than high_pass_sigma (e.g. 25–40) to strip the extra
# low-frequency background that simulations produce.  None = same as real pattern.
sim_high_pass_sigma_override = None   # e.g. 30.0

# Set True to search for the best high_pass_sigma, low_pass_sigma, and gamma
# before running PC/Euler refinement.  The optimised values replace the ones
# above for the rest of the script (diagnostics will also use them).
optimize_preprocessing   = False
# Search bounds: (min, max) for each parameter.
preproc_hp_bounds        = (3.0, 20.0)   # high_pass_sigma range
preproc_lp_bounds        = (0.0, 3.0)    # low_pass_sigma range
preproc_gamma_bounds     = (0.1, 1.5)    # gamma range
preproc_n_eval           = 400           # approx. optimiser evaluations

# Set True to run the INDEPENDENT preprocessing optimisation (6 parameters:
# hp/lp/gamma separately for real and sim).  Only one of the two optimisations
# should be True at a time; if both are True the independent one runs second
# and its real-pattern params overwrite the shared ones.
optimize_preprocessing_independent  = False
preproc_indep_real_hp_bounds        = (3.0, 80.0)
preproc_indep_real_lp_bounds        = (0.0, 5.0)
preproc_indep_real_gamma_bounds     = (0.1, 1.5)
preproc_indep_sim_hp_bounds         = (3.0, 80.0)
preproc_indep_sim_lp_bounds         = (0.0, 5.0)
preproc_indep_sim_gamma_bounds      = (0.1, 1.5)
preproc_indep_n_eval                = 600

# Set False to skip PC / Euler refinement entirely.  When False, the script
# uses the manual Euler angles / PC supplied above (or the .ang values) for
# every downstream step (diagnostics, simulated pattern generation, h
# landscape, …) without running optimize_pc_and_euler.
run_pc_euler_refinement   = True

# H-value landscape — for each (high_pass_sigma, gamma) combination, run IC-GN
# between the single reference pattern and its simulation and plot the 8
# h-component values as 2D intensity maps.  Set to True to enable.
compute_h_landscape_flag  = False
h_land_n_grid             = 8          # grid points along each axis
h_land_hp_bounds          = (3.0, 80.0)   # (min, max) high-pass sigma
h_land_gamma_bounds       = (0.1, 1.5)    # (min, max) gamma
h_land_max_iter           = 50         # IC-GN max iterations per grid point

# Euler angles in EDAX/TSL convention (phi1, Phi, phi2) in degrees.
# Set to None to read automatically from the .ang file at ref_idx_yx.
# Copy from GUI log: "[PC/Euler refinement] Euler init (deg): phi1=X  Phi=Y  phi2=Z"
euler_deg           = (143.467, 2.177, 172.148)

# Pattern center in EDAX/TSL convention (x*, y*, z*).
# Set to None to read automatically from the .ang file.
# Must match Step 2 spinbox values — copy from GUI log: "[PC/Euler refinement] PC init (EDAX)"
pc_manual = (0.65079, 0.87279, 1.06901)

OUT_DIR = "debug/simref_diagnostics"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# Setup
# =============================================================================

pat_obj = Data.UP2(up2_path)
pat_obj.set_processing(
    low_pass_sigma=low_pass_sigma,
    high_pass_sigma=high_pass_sigma,
    truncate_std_scale=3.0,
    mask_type=mask_type,
    center_cross_half_width=6,
    clahe_kernel=clahe_kernel,
    clahe_clip=clahe_clip,
    clahe_nbins=256,
    use_clahe=use_clahe,
    gamma=gamma_correction,
    flip_x=flip_x,
)

ang_data = utilities.read_ang(ang_path, pat_obj.patshape, segment_grain_threshold=None)
x0_flat  = np.ravel_multi_index(ref_idx_yx, ang_data.shape)

if euler_deg is not None:
    euler_ref = np.deg2rad(np.asarray(euler_deg, dtype=np.float64))
    print(f"Using manual Euler angles (deg): {euler_deg}")
else:
    euler_ref = ang_data.eulers[ref_idx_yx]
    print(f"Euler angles read from .ang (deg): {np.degrees(euler_ref)}")
if pc_manual is not None:
    pc_ref = tuple(pc_manual)
    print(f"Using manual PC (EDAX): {pc_ref}")
else:
    pc_ref = ang_data.pc
    print(f"PC read from .ang (EDAX): {pc_ref}")

patshape = pat_obj.patshape
H, W     = patshape

# =============================================================================
# Optional: preprocessing parameter optimisation
# Generates a raw sim pattern with the current euler_ref / pc_ref, then
# searches for the best (high_pass_sigma, low_pass_sigma, gamma) by minimising
# ZNSSD.  The optimised values are applied to pat_obj before anything else runs.
# =============================================================================

if optimize_preprocessing:
    import torch as _torch_pre

    print("\nGenerating raw simulated pattern for preprocessing optimisation …")
    _sim_pre = patternSimulation()
    _sim_pre.detector_height   = patshape[0]
    _sim_pre.detector_width    = patshape[1]
    _sim_pre.det_shape         = patshape
    _sim_pre.detector_tilt_deg = detector_tilt_deg
    _sim_pre.sample_tilt_deg   = sample_tilt_deg
    _sim_pre.mastersetup(master_pattern_path)
    _pc_bruker_pre = conversions.Edax_to_Bruker_PC(np.asarray(pc_ref))
    _sim_pre.EandPCSet(euler_ref, list(_pc_bruker_pre), verbose=False)
    with _torch_pre.no_grad():
        _pats_pre = _sim_pre.GenPattern()
    _sim_raw = _pats_pre[0].reshape(patshape).cpu().numpy().astype(np.float32)
    _sim_raw = np.fliplr(_sim_raw)
    _lo, _hi = _sim_raw.min(), _sim_raw.max()
    if _hi > _lo:
        _sim_raw = (_sim_raw - _lo) / (_hi - _lo)

    _real_raw = pat_obj.read_pattern(x0_flat, process=False).astype(np.float32)

    print("Running preprocessing parameter optimisation …")
    _best_pp = optimize_preprocessing_params(
        real_pat_raw    = _real_raw,
        sim_pat_raw     = _sim_raw,
        pat_obj         = pat_obj,
        high_pass_bounds= preproc_hp_bounds,
        low_pass_bounds = preproc_lp_bounds,
        gamma_bounds    = preproc_gamma_bounds,
        n_eval          = preproc_n_eval,
        save_dir        = OUT_DIR,
    )
    # Apply optimised params to pat_obj — all subsequent processing uses them
    pat_obj.high_pass_sigma = _best_pp["high_pass_sigma"]
    pat_obj.low_pass_sigma  = _best_pp["low_pass_sigma"]
    pat_obj.gamma           = _best_pp["gamma"]
    high_pass_sigma  = _best_pp["high_pass_sigma"]
    low_pass_sigma   = _best_pp["low_pass_sigma"]
    gamma_correction = _best_pp["gamma"]
    print(f"Updated preprocessing params: "
          f"hp={high_pass_sigma:.2f}  lp={low_pass_sigma:.3f}  gamma={gamma_correction:.4f}")

# These carry sim-specific overrides into optimize_pc_and_euler; populated by
# the independent optimisation below if it runs, otherwise fall back to the
# manual override values set at the top of this script.
_sim_hp_for_refine = sim_high_pass_sigma_override
_sim_lp_for_refine = None
_sim_g_for_refine  = None

# =============================================================================
# Optional: INDEPENDENT preprocessing optimisation
# Real and simulated patterns each get their own hp / lp / gamma.
# The optimised real params update pat_obj; the sim params are threaded into
# optimize_pc_and_euler so the same independent processing is used there too.
# =============================================================================

if optimize_preprocessing_independent:
    # Reuse the raw patterns generated for the shared optimisation if it ran,
    # otherwise generate them now.
    if not optimize_preprocessing:
        import torch as _torch_indep
        print("\nGenerating raw simulated pattern for independent optimisation …")
        _sim_indep = patternSimulation()
        _sim_indep.detector_height   = patshape[0]
        _sim_indep.detector_width    = patshape[1]
        _sim_indep.det_shape         = patshape
        _sim_indep.detector_tilt_deg = detector_tilt_deg
        _sim_indep.sample_tilt_deg   = sample_tilt_deg
        _sim_indep.mastersetup(master_pattern_path)
        _pc_bruker_indep = conversions.Edax_to_Bruker_PC(np.asarray(pc_ref))
        _sim_indep.EandPCSet(euler_ref, list(_pc_bruker_indep), verbose=False)
        with _torch_indep.no_grad():
            _pats_indep = _sim_indep.GenPattern()
        _sim_raw = _pats_indep[0].reshape(patshape).cpu().numpy().astype(np.float32)
        _sim_raw = np.fliplr(_sim_raw)
        _lo, _hi = _sim_raw.min(), _sim_raw.max()
        if _hi > _lo:
            _sim_raw = (_sim_raw - _lo) / (_hi - _lo)
        _real_raw = pat_obj.read_pattern(x0_flat, process=False).astype(np.float32)

    print("Running independent preprocessing optimisation …")
    _best_indep = optimize_preprocessing_params_independent(
        real_pat_raw          = _real_raw,
        sim_pat_raw           = _sim_raw,
        pat_obj               = pat_obj,
        real_high_pass_bounds = preproc_indep_real_hp_bounds,
        real_low_pass_bounds  = preproc_indep_real_lp_bounds,
        real_gamma_bounds     = preproc_indep_real_gamma_bounds,
        sim_high_pass_bounds  = preproc_indep_sim_hp_bounds,
        sim_low_pass_bounds   = preproc_indep_sim_lp_bounds,
        sim_gamma_bounds      = preproc_indep_sim_gamma_bounds,
        n_eval                = preproc_indep_n_eval,
        save_dir              = OUT_DIR,
    )
    # Apply optimised real params to pat_obj
    pat_obj.high_pass_sigma = _best_indep["real_high_pass_sigma"]
    pat_obj.low_pass_sigma  = _best_indep["real_low_pass_sigma"]
    pat_obj.gamma           = _best_indep["real_gamma"]
    high_pass_sigma  = _best_indep["real_high_pass_sigma"]
    low_pass_sigma   = _best_indep["real_low_pass_sigma"]
    gamma_correction = _best_indep["real_gamma"]
    # Store sim overrides — passed into optimize_pc_and_euler below
    _sim_hp_for_refine = _best_indep["sim_high_pass_sigma"]
    _sim_lp_for_refine = _best_indep["sim_low_pass_sigma"]
    _sim_g_for_refine  = _best_indep["sim_gamma"]
    print(f"Updated real params:  hp={high_pass_sigma:.2f}  "
          f"lp={low_pass_sigma:.3f}  gamma={gamma_correction:.4f}")
    print(f"Sim params for refine: hp={_sim_hp_for_refine:.2f}  "
          f"lp={_sim_lp_for_refine:.3f}  gamma={_sim_g_for_refine:.4f}")


# =============================================================================
# PC / Euler refinement — run before diagnostics so figures reflect best match
# =============================================================================

if run_pc_euler_refinement:
    print("Running PC / Euler refinement …")
    euler_ref, pc_ref = optimize_pc_and_euler(
        pat_obj=pat_obj,
        x0=x0_flat,
        master_pattern_path=master_pattern_path,
        euler_angles_init=euler_ref,
        pc_init=pc_ref,
        sample_tilt_deg=sample_tilt_deg,
        detector_tilt_deg=detector_tilt_deg,
        max_iter=300,
        save_dir=OUT_DIR,
        sim_high_pass_sigma=_sim_hp_for_refine,
        sim_low_pass_sigma=_sim_lp_for_refine,
        sim_gamma=_sim_g_for_refine,
    )
    print(f"Refined Euler angles (rad): {euler_ref}")
    print(f"Refined PC:                 {pc_ref}")
else:
    print("Skipping PC / Euler refinement (run_pc_euler_refinement=False).")
    print(f"  Using Euler angles (rad): {euler_ref}")
    print(f"  Using PC                : {pc_ref}")

# =============================================================================
# Optional: h-value landscape — IC-GN over (high_pass_sigma, gamma) grid
# The current pat_obj params (after any preprocessing optimisation) are marked
# with a red star in the output figure so you can see where you sit on the
# landscape relative to the full parameter space.
# =============================================================================

if compute_h_landscape_flag:
    print(f"\nRunning h-value landscape for reference pattern {ref_idx_yx} …")
    compute_h_landscape(
        pat_obj=pat_obj,
        ang_data=ang_data,
        ref_idx_yx=ref_idx_yx,
        euler_ref=euler_ref,
        pc_ref=pc_ref,
        master_pattern_path=master_pattern_path,
        sample_tilt_deg=sample_tilt_deg,
        detector_tilt_deg=detector_tilt_deg,
        hp_bounds=h_land_hp_bounds,
        gamma_bounds=h_land_gamma_bounds,
        n_grid=h_land_n_grid,
        crop_fraction=crop_fraction,
        max_iter_icgn=h_land_max_iter,
        mark_hp=pat_obj.high_pass_sigma,
        mark_gamma=pat_obj.gamma,
        save_dir=OUT_DIR,
    )

# =============================================================================
# Patterns
# =============================================================================

print("Loading real reference pattern …")
real_pat = pat_obj.read_pattern(x0_flat, process=True)

print("Generating simulated reference pattern (with refined params) …")
print(f"  sample_tilt={sample_tilt_deg}°  detector_tilt={detector_tilt_deg}°  "
      f"primary_tilt_arg={-(sample_tilt_deg - detector_tilt_deg):.1f}°")
_sim = patternSimulation()
_sim.detector_height   = patshape[0]
_sim.detector_width    = patshape[1]
_sim.det_shape         = patshape
_sim.detector_tilt_deg = detector_tilt_deg
_sim.sample_tilt_deg   = sample_tilt_deg
_sim.mastersetup(master_pattern_path)
_pc_bruker = conversions.Edax_to_Bruker_PC(np.asarray(pc_ref))
print(f"  PC (EDAX):   {pc_ref}")
print(f"  PC (Bruker): {_pc_bruker}")
_sim.EandPCSet(euler_ref, list(_pc_bruker), verbose=False)
import torch as _torch
with _torch.no_grad():
    _pats = _sim.GenPattern()
_pat_np = _pats[0].reshape(patshape).cpu().numpy().astype(np.float32)
_pat_np = np.fliplr(_pat_np)          # match GUI detector orientation convention
_lo, _hi = _pat_np.min(), _pat_np.max()
if _hi > _lo:
    _pat_np = (_pat_np - _lo) / (_hi - _lo)
# Apply the same preprocessing pipeline as real patterns (mask suppressed —
# detector artefacts have no physical equivalent in the simulation)
_orig_mask = pat_obj.mask_type
pat_obj.mask_type = None
sim_pat = pat_obj.process_pattern(_pat_np)
pat_obj.mask_type = _orig_mask

# =============================================================================
# Helper: subset coordinates (mirrors optimize())
# =============================================================================

h0 = (W // 2, H // 2)
crop_row = int(H * (1 - crop_fraction) / 2)
crop_col = int(W * (1 - crop_fraction) / 2)
subset_slice = (slice(crop_row, -crop_row), slice(crop_col, -crop_col))

mask = pat_obj.get_mask()
x = np.arange(W) - h0[0]
y = np.arange(H) - h0[1]
X2d, Y2d = np.meshgrid(x, y, indexing="xy")

xi_full = np.array([X2d[subset_slice].flatten(), Y2d[subset_slice].flatten()])
subset_shape = X2d[subset_slice].shape

valid = None
if mask is not None:
    valid = mask[subset_slice].flatten()
    xi = xi_full[:, valid]
else:
    xi = xi_full

def to_2d(arr):
    if valid is None:
        return arr.reshape(subset_shape)
    img = np.full(subset_shape[0] * subset_shape[1], np.nan)
    img[valid] = arr
    return img.reshape(subset_shape)


def _grad_pair(pat):
    spline = interpolate.RectBivariateSpline(x, y, pat.T, kx=5, ky=5)
    gx = spline(xi[0], xi[1], dx=1, dy=0, grid=False)
    gy = spline(xi[0], xi[1], dx=0, dy=1, grid=False)
    return gx, gy


def _hessian(gx, gy):
    _1 = np.ones(xi.shape[1])
    _0 = np.zeros(xi.shape[1])
    out0 = np.array([[xi[0], xi[1], _1, _0, _0, _0, -xi[0]**2, -xi[1]*xi[0]]])
    out1 = np.array([[_0, _0, _0, xi[0], xi[1], _1, -xi[0]*xi[1], -xi[1]**2]])
    Jac  = np.vstack((out0, out1))           # 2×8×N
    GR   = np.vstack((gx, gy)).reshape(2, 1, -1).transpose(1, 0, 2)
    NJac = np.einsum("ilk,ljk->ijk", GR, Jac)[0]   # 8×N
    vals = np.array([gx, gy])
    r_zmsv_sq = ((vals.ravel() - vals.mean())**2).sum()
    H = 2 / r_zmsv_sq * NJac.dot(NJac.T)
    return H, NJac


print("Computing gradients and Hessians …")
real_gx, real_gy = _grad_pair(real_pat)
sim_gx,  sim_gy  = _grad_pair(sim_pat)

H_real, NJac_real = _hessian(real_gx, real_gy)
H_sim,  NJac_sim  = _hessian(sim_gx,  sim_gy)

# =============================================================================
# Figure 1 — Pattern comparison
# =============================================================================

print("Saving Fig 1: pattern comparison …")
diff_pat = real_pat - sim_pat
vlim = np.percentile(np.abs(diff_pat), 98)

_shared_vmin = min(real_pat.min(), sim_pat.min())
_shared_vmax = max(real_pat.max(), sim_pat.max())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(real_pat, cmap="gray", vmin=_shared_vmin, vmax=_shared_vmax)
axes[0].set_title("Real (processed)", fontsize=12)
axes[0].axis("off")
axes[1].imshow(sim_pat,  cmap="gray", vmin=_shared_vmin, vmax=_shared_vmax)
axes[1].set_title("Simulated (processed)", fontsize=12)
axes[1].axis("off")
axes[2].imshow(diff_pat, cmap="RdBu", vmin=-vlim, vmax=vlim)
axes[2].set_title("Real − Simulated", fontsize=12)
axes[2].axis("off")
fig.suptitle("Pattern Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/1_pattern_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

# =============================================================================
# Figure 2 — Intensity histograms
# =============================================================================

print("Saving Fig 2: intensity histograms …")
bins = np.linspace(0, 1, 128)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(real_pat.ravel(), bins=bins, alpha=0.6, label="Real", color="steelblue", density=True)
ax.hist(sim_pat.ravel(),  bins=bins, alpha=0.6, label="Simulated", color="tomato", density=True)
ax.set_xlabel("Intensity (normalised)")
ax.set_ylabel("Density")
_clahe_note = "biased CLAHE / truncation" if use_clahe else "biased truncation (CLAHE off)"
ax.set_title(f"Intensity histogram after preprocessing\n"
             f"Large mismatch → different local contrast → {_clahe_note}",
             fontsize=10)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/2_intensity_histograms.png", dpi=200, bbox_inches="tight")
plt.close()

# =============================================================================
# Figure 3 — Power spectra (radially averaged)
# =============================================================================

print("Saving Fig 3: power spectra …")

def radial_power(pat):
    f = np.fft.fftshift(np.fft.fft2(pat - pat.mean()))
    power = np.abs(f)**2
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2).astype(int)
    r_max = min(cx, cy)
    radial = np.array([power[r == ri].mean() for ri in range(r_max)])
    freq   = np.arange(r_max) / max(H, W)
    return freq, radial

freq, rp_real = radial_power(real_pat)
_,    rp_sim  = radial_power(sim_pat)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].semilogy(freq, rp_real, label="Real",      color="steelblue")
axes[0].semilogy(freq, rp_sim,  label="Simulated", color="tomato")
axes[0].set_xlabel("Spatial frequency (cycles / pixel)")
axes[0].set_ylabel("Mean power (log scale)")
axes[0].set_title("Radially averaged power spectrum\n"
                  "Simulated higher at HF → steeper gradients → stiffer Hessian",
                  fontsize=10)
axes[0].legend()

ratio = rp_sim / (rp_real + 1e-30)
axes[1].plot(freq, ratio, color="purple")
axes[1].axhline(1, color="k", ls="--", lw=0.8)
axes[1].set_xlabel("Spatial frequency (cycles / pixel)")
axes[1].set_ylabel("Power ratio  sim / real")
axes[1].set_title("Power ratio  (>1 = simulated has more power at that frequency)",
                  fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/3_power_spectra.png", dpi=200, bbox_inches="tight")
plt.close()

# =============================================================================
# Figure 4 — Gradient comparison
# =============================================================================

print("Saving Fig 4: gradient maps …")
real_gmag = np.sqrt(real_gx**2 + real_gy**2)
sim_gmag  = np.sqrt(sim_gx**2  + sim_gy**2)

vx  = np.nanpercentile(np.abs(np.concatenate([real_gx, sim_gx])), 98)
vy  = np.nanpercentile(np.abs(np.concatenate([real_gy, sim_gy])), 98)
vmg = np.nanpercentile(np.concatenate([real_gmag, sim_gmag]), 98)

fig, axes = plt.subplots(3, 3, figsize=(13, 9))
rows = [("Gx", real_gx, sim_gx, "RdBu", vx, True),
        ("Gy", real_gy, sim_gy, "RdBu", vy, True),
        ("‖G‖", real_gmag, sim_gmag, "inferno", vmg, False)]

for row_i, (label, r_arr, s_arr, cmap, vlim, sym) in enumerate(rows):
    diff = r_arr - s_arr
    dl   = np.nanpercentile(np.abs(diff), 98)
    vmin = -vlim if sym else 0
    for col_i, (arr, vm, vM, cm) in enumerate([
        (r_arr, vmin, vlim,  cmap),
        (s_arr, vmin, vlim,  cmap),
        (diff, -dl,   dl,   "coolwarm"),
    ]):
        ax = axes[row_i, col_i]
        im = ax.imshow(to_2d(arr), cmap=cm, vmin=vm, vmax=vM)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")
        if row_i == 0:
            ax.set_title(["Real", "Simulated", "Difference"][col_i],
                         fontsize=11, fontweight="bold")
    axes[row_i, 0].set_ylabel(label, fontsize=11)

fig.suptitle("Gradient comparison — differences here directly bias the IC-GN Hessian",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/4_gradient_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

# =============================================================================
# Figure 5 — Hessian diagonal & condition numbers
# =============================================================================

print("Saving Fig 5: Hessian diagnostics …")
param_labels = ["h11", "h12", "h13", "h21", "h22", "h23", "h31", "h32"]

diag_real = np.diag(H_real)
diag_sim  = np.diag(H_sim)
cond_real = np.linalg.cond(H_real)
cond_sim  = np.linalg.cond(H_sim)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x_pos = np.arange(8)
w = 0.35
axes[0].bar(x_pos - w/2, diag_real / diag_real.max(), w, label="Real ref",      color="steelblue", alpha=0.8)
axes[0].bar(x_pos + w/2, diag_sim  / diag_sim.max(),  w, label="Simulated ref", color="tomato",    alpha=0.8)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(param_labels)
axes[0].set_ylabel("Normalised Hessian diagonal")
axes[0].set_title("Hessian diagonal (normalised)\n"
                  "Mismatch here → different sensitivity for each homography param",
                  fontsize=10)
axes[0].legend()

axes[1].bar(["Real ref\ncond={:.1e}".format(cond_real),
             "Sim ref\ncond={:.1e}".format(cond_sim)],
            [cond_real, cond_sim],
            color=["steelblue", "tomato"], alpha=0.8)
axes[1].set_ylabel("Condition number of H (log scale)")
axes[1].set_yscale("log")
axes[1].set_title("Condition number\n"
                  "Much larger for simulated → ill-conditioned Hessian → amplified noise",
                  fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/5_hessian_diagnostics.png", dpi=200, bbox_inches="tight")
plt.close()

print(f"  Hessian condition — real: {cond_real:.3e}   simulated: {cond_sim:.3e}")
print(f"  Hessian diagonal ratio (sim/real): {diag_sim / diag_real}")

# =============================================================================
# Figure 6 — Radial profile of (real - sim) in the subset
# A non-zero radial trend = residual PC error projecting onto the pattern
# =============================================================================

print("Saving Fig 6: radial residual profile …")

diff_flat = (real_pat - sim_pat)[subset_slice].ravel()
r_flat = np.sqrt(xi_full[0]**2 + xi_full[1]**2)

if valid is not None:
    diff_valid = diff_flat[valid]
    r_valid    = r_flat[valid]
else:
    diff_valid = diff_flat
    r_valid    = r_flat

r_bins  = np.linspace(0, r_valid.max(), 40)
r_mid   = 0.5 * (r_bins[:-1] + r_bins[1:])
r_mean  = np.array([diff_valid[(r_valid >= r_bins[i]) & (r_valid < r_bins[i+1])].mean()
                    for i in range(len(r_bins)-1)])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(r_mid, r_mean, "o-", color="purple", ms=4)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("Radial distance from pattern centre (pixels)")
ax.set_ylabel("Mean (real − simulated) intensity")
ax.set_title("Radial profile of intensity difference\n"
             "Non-zero slope / bowl shape → residual PC error (z* or x*/y*)",
             fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/6_radial_residual_profile.png", dpi=200, bbox_inches="tight")
plt.close()

# =============================================================================
# Figure 7 — Intensity slices through pattern centre
# =============================================================================

print("Saving Fig 7: intensity profile slices …")

mid_row = H // 2
mid_col = W // 2

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(real_pat[mid_row, :], label="Real",      color="steelblue")
axes[0].plot(sim_pat[mid_row,  :], label="Simulated", color="tomato", ls="--")
axes[0].set_title(f"Horizontal slice at row {mid_row}\n"
                  "Blur difference → PSF mismatch → gradient magnitude mismatch",
                  fontsize=10)
axes[0].set_xlabel("Column (px)")
axes[0].set_ylabel("Intensity")
axes[0].legend()

axes[1].plot(real_pat[:, mid_col], label="Real",      color="steelblue")
axes[1].plot(sim_pat[:,  mid_col], label="Simulated", color="tomato", ls="--")
axes[1].set_title(f"Vertical slice at col {mid_col}", fontsize=10)
axes[1].set_xlabel("Row (px)")
axes[1].set_ylabel("Intensity")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/7_intensity_slices.png", dpi=200, bbox_inches="tight")
plt.close()

# =============================================================================
# Figure 8 — Gradient line profiles across mid-row and mid-column of the subset
# Overlays Real vs Simulated for Gx, Gy, and ‖G‖ so you can see exactly where
# the gradient mismatch is worst across the detector face.
# =============================================================================

print("Saving Fig 8: gradient line profiles …")

real_gx_2d   = to_2d(real_gx)
real_gy_2d   = to_2d(real_gy)
real_gmag_2d = to_2d(real_gmag)
sim_gx_2d    = to_2d(sim_gx)
sim_gy_2d    = to_2d(sim_gy)
sim_gmag_2d  = to_2d(sim_gmag)

sub_h, sub_w = subset_shape
mid_r = sub_h // 2
mid_c = sub_w // 2

# Pixel coordinates relative to the pattern centre
col_coords = np.arange(sub_w) + crop_col - W // 2
row_coords = np.arange(sub_h) + crop_row - H // 2

fig, axes = plt.subplots(3, 2, figsize=(14, 11))
components = [
    ("Gx",   real_gx_2d,   sim_gx_2d),
    ("Gy",   real_gy_2d,   sim_gy_2d),
    ("‖G‖",  real_gmag_2d, sim_gmag_2d),
]

for row_i, (label, r_2d, s_2d) in enumerate(components):
    # ── Horizontal slice at the mid row of the subset ─────────────────────
    ax = axes[row_i, 0]
    r_h = r_2d[mid_r, :]
    s_h = s_2d[mid_r, :]
    ax.plot(col_coords, r_h,       color="steelblue", lw=1.2, label="Real")
    ax.plot(col_coords, s_h,       color="tomato",    lw=1.2, label="Simulated", ls="--")
    ax.plot(col_coords, r_h - s_h, color="gray",      lw=0.8, label="Real − Sim")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("Column offset from pattern centre (px)")
    ax.set_ylabel(label)
    if row_i == 0:
        ax.set_title("Horizontal slice (mid row of subset)",
                     fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    # ── Vertical slice at the mid column of the subset ────────────────────
    ax = axes[row_i, 1]
    r_v = r_2d[:, mid_c]
    s_v = s_2d[:, mid_c]
    ax.plot(row_coords, r_v,       color="steelblue", lw=1.2, label="Real")
    ax.plot(row_coords, s_v,       color="tomato",    lw=1.2, label="Simulated", ls="--")
    ax.plot(row_coords, r_v - s_v, color="gray",      lw=0.8, label="Real − Sim")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("Row offset from pattern centre (px)")
    ax.set_ylabel(label)
    if row_i == 0:
        ax.set_title("Vertical slice (mid column of subset)",
                     fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

fig.suptitle(
    "Gradient line profiles — persistent Real≠Simulated after refinement\n"
    "indicates frequency mismatch (see Fig 3) or PSF blur difference (see Fig 7)",
    fontsize=11, fontweight="bold",
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/8_gradient_line_profiles.png", dpi=200, bbox_inches="tight")
plt.close()

# =============================================================================
# Summary printout
# =============================================================================

_a = real_pat.ravel().astype(np.float64) - real_pat.ravel().mean()
_b = sim_pat.ravel().astype(np.float64)  - sim_pat.ravel().mean()
_a /= np.linalg.norm(_a)
_b /= np.linalg.norm(_b)
znssd_val = float(np.dot(_a - _b, _a - _b))

print("\n" + "="*60)
print("DIAGNOSTIC SUMMARY")
print("="*60)
print(f"  ZNSSD (real vs simulated, full pattern): {znssd_val:.4f}  "
      f"(0 = perfect match, 2 = uncorrelated, 4 = anti-correlated)")
print(f"  Mean gradient magnitude — real:         {np.sqrt(real_gx**2 + real_gy**2).mean():.4f}")
print(f"  Mean gradient magnitude — simulated:    {np.sqrt(sim_gx**2  + sim_gy**2).mean():.4f}")
print(f"  Gradient mag ratio (sim/real):          {np.sqrt(sim_gx**2+sim_gy**2).mean() / np.sqrt(real_gx**2+real_gy**2).mean():.3f}")
print(f"  Hessian cond — real:                    {cond_real:.3e}")
print(f"  Hessian cond — simulated:               {cond_sim:.3e}")
print(f"  RMS intensity diff (real−sim):          {np.sqrt(((real_pat-sim_pat)**2).mean()):.4f}")
print(f"\nAll figures saved to: {OUT_DIR}/")
print("\nWhat to look for:")
print("  Fig 3 (power spectrum):  Sim > Real at high freq → apply Gaussian blur to sim ref")
print("  Fig 4 (gradients):       Large diff → Hessian bias → systematic strain error")
print("  Fig 5 (Hessian cond):    Sim >> Real → ill-conditioned → amplified noise")
print("  Fig 6 (radial residual): Non-zero slope → residual PC error (tweak z*)")
print("  Fig 7 (slices):          Width of Kikuchi bands → PSF blur mismatch")
print("  Fig 8 (gradient lines):  Localises worst gradient mismatch spatially;"
      " peaks at band crossings → freq/PSF issue; smooth offset → DC bias")
if not use_clahe:
    print("\n  NOTE: CLAHE is OFF — intensity histogram mismatch is driven by high-pass")
    print("        filter and truncation only.  If sim/real histograms still diverge,")
    print("        adjust high_pass_sigma or sim_high_pass_sigma_override.")

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
import get_homography_cpu as core
from optimize_reference import optimize_pc_and_euler

# =============================================================================
# USER INPUTS  — match your runner script
# =============================================================================

up2_path            = '/Users/crestiennedechaine/OriginalData/Si-Indent/001_Si_spherical_indent_20kV.up2'
ang_path            = '/Users/crestiennedechaine/OriginalData/Si-Indent/dp2-refined.ang'
master_pattern_path = '/Users/crestiennedechaine/OriginalData/Si-Indent/Si-master-20kV.h5'
ref_idx_yx          = (0, 0)          # (row, col) of the reference pattern
tilt_deg            = 70.0
crop_fraction       = 0.9             # must match optimize() call

OUT_DIR = "debug/simref_diagnostics"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# Setup
# =============================================================================

pat_obj = Data.UP2(up2_path)
pat_obj.set_processing(
    low_pass_sigma=0.0,
    high_pass_sigma=15.0,
    truncate_std_scale=3.0,
    mask_type="center_cross",
    center_cross_half_width=6,
    clahe_kernel=(5, 5),
    clahe_clip=0.01,
    clahe_nbins=256,
)

ang_data = utilities.read_ang(ang_path, pat_obj.patshape, segment_grain_threshold=None)
x0_flat  = np.ravel_multi_index(ref_idx_yx, ang_data.shape)

euler_ref = ang_data.eulers[ref_idx_yx]
pc_ref    = ang_data.pc

patshape = pat_obj.patshape
H, W     = patshape

# =============================================================================
# PC / Euler refinement — run before diagnostics so figures reflect best match
# =============================================================================

print("Running PC / Euler refinement …")
euler_ref, pc_ref = optimize_pc_and_euler(
    pat_obj=pat_obj,
    x0=x0_flat,
    master_pattern_path=master_pattern_path,
    euler_angles_init=euler_ref,
    pc_init=pc_ref,
    tilt_deg=tilt_deg,
    max_iter=300,
    save_dir=OUT_DIR,
)
print(f"Refined Euler angles (rad): {euler_ref}")
print(f"Refined PC:                 {pc_ref}")

# =============================================================================
# Patterns
# =============================================================================

print("Loading real reference pattern …")
real_pat = pat_obj.read_pattern(x0_flat, process=True)

print("Generating simulated reference pattern (with refined params) …")
sim_pat = core.simulate_reference_pattern(
    master_pattern_path=master_pattern_path,
    euler_angles=euler_ref,
    PC=pc_ref,
    patshape=patshape,
    tilt_deg=tilt_deg,
    pat_obj=pat_obj,
)

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

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(real_pat, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Real (processed)", fontsize=12)
axes[0].axis("off")
axes[1].imshow(sim_pat,  cmap="gray", vmin=0, vmax=1)
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
ax.set_title("Intensity histogram after preprocessing\n"
             "Large mismatch → different local contrast → biased CLAHE / truncation",
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
# Summary printout
# =============================================================================

ncc_val = float(np.dot(
    (real_pat.ravel() - real_pat.mean()) / np.linalg.norm(real_pat.ravel() - real_pat.mean()),
    (sim_pat.ravel()  - sim_pat.mean())  / np.linalg.norm(sim_pat.ravel()  - sim_pat.mean()),
))

print("\n" + "="*60)
print("DIAGNOSTIC SUMMARY")
print("="*60)
print(f"  NCC (real vs simulated, full pattern):  {ncc_val:.4f}")
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

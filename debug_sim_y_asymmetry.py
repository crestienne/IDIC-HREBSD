"""
Diagnostic script: localise the y-direction gradient suppression in simulated
EBSD reference patterns.

The user observes that ∂I/∂y of a simulated reference pattern is systematically
smaller than ∂I/∂x at the same orientation/PC, while a real (experimental)
pattern at the same point has symmetric gradients.  Because IC-GN's Hessian
inherits the gradient deficit on the reference side, this attenuation shows
up downstream as a smaller-than-real ε_22 strain.

This script reproduces the simulation through the exact production pipeline
(`PatternSimulation.SimPatGen.patternSimulation`), then runs four targeted
isolation experiments and a battery of directional plots.  The output is a
folder of PNGs plus a printed summary table whose ratios point at the
responsible mechanism — see the verification section of the plan for how to
read the table.

Tests
-----
1. Marginal FFT power along each axis (P(kx) vs P(ky))
2. |Gx| / |Gy| bar chart
3. Polar histogram of gradient direction (weighted by |G|)
4. Line-averaged |∂I/∂y|(row) and |∂I/∂x|(col)

Isolation experiments
---------------------
5. Image-axis rotation sanity check (rot90 the output sim → should rotate)
6. Master-pattern rotation (rot90 mLPNH/mLPSH before projection)
7. No obliquity (replace per-pixel accum_e with uniform 1.0)
8. Synthetic isotropic master pattern (radial sinusoid)

Usage
-----
    python debug_sim_y_asymmetry.py
    # edit the CONFIG block below to point at your data
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # always non-interactive — no GUI window
import matplotlib.pyplot as plt
from scipy import interpolate

import Data
import conversions
import utilities
from PatternSimulation.SimPatGen import patternSimulation
from HREBSD import (
    accum_e_to_detector,
    project_HREBSD_pattern_energy_weighted,
)

# =============================================================================
# CONFIG  — edit these paths / values, then run.
# =============================================================================

# Required inputs ---------------------------------------------------------
MASTER_PATTERN_PATH = "/Users/crestiennedechaine/OriginalData/Si_Ge_Dataset/DI_largerRegion/MCoutput.h5"
EULER_DEG           = (143.467, 2.177, 172.148)        # (phi1, Phi, phi2)
PC_EDAX             = (0.65079, 0.87279, 1.06901)      # (x*, y*, z*)
SAMPLE_TILT_DEG     = 70.0
DET_TILT_DEG        = 10.0
DET_SHAPE           = (512, 512)                       # (rows, cols)

# Real-pattern comparison (optional — leave UP2_PATH = "" to skip) ---------
UP2_PATH            = "/Users/crestiennedechaine/OriginalData/Si_Ge_Dataset/SiGe_updated_512x512.up2"
PAT_IDX             = 0                                # flat index into UP2
# Step-3 processing applied to BOTH real and sim before gradient comparison.
# Match the GUI Step 3 settings the user actually runs at.
APPLY_STEP3         = True
LOW_PASS_SIGMA      = 1.0
HIGH_PASS_SIGMA     = 10.0
GAMMA               = 0.80
MASK_TYPE           = "none"
FLIP_X              = False

# Output -----------------------------------------------------------------
OUT_DIR             = f"debug/sim_y_asymmetry/{time.strftime('%Y%m%d_%H%M%S')}"

# Quality knobs ----------------------------------------------------------
POLAR_BINS          = 36
LINE_AVG_BAND_PX    = None         # None = full-width band; or set int (e.g. 64)
                                   # to average only the central band of rows/cols
ANALYSIS_CROP       = None         # None = use the whole pattern.  Set e.g. 0.8
                                   # to crop to the central 80 % before analysis.

# =============================================================================
# Helpers
# =============================================================================

def _make_sim(master_path: str, euler_deg: tuple, pc_edax: tuple,
              det_shape: tuple, det_tilt_deg: float, sample_tilt_deg: float
              ) -> tuple[patternSimulation, np.ndarray]:
    """Build a fully-loaded patternSimulation and return (sim, production_pattern).

    Mirrors what gui_workers.SimRefWorker does so the output matches what the
    GUI live preview / Compare button produces."""
    sim = patternSimulation()
    sim.detector_height   = det_shape[0]
    sim.detector_width    = det_shape[1]
    sim.det_shape         = det_shape
    sim.detector_tilt_deg = det_tilt_deg
    sim.sample_tilt_deg   = sample_tilt_deg
    sim.mastersetup(master_path)

    euler_rad = np.deg2rad(np.asarray(euler_deg, dtype=np.float64))
    pc_bruker = conversions.Edax_to_Bruker_PC(np.asarray(pc_edax))
    sim.EandPCSet(euler_rad, list(pc_bruker), verbose=False)

    pat = _gen(sim)
    return sim, pat


def _gen(sim: patternSimulation) -> np.ndarray:
    """Run GenPattern, reshape to (H, W), normalize to [0, 1]."""
    with torch.no_grad():
        pats = sim.GenPattern()
    H, W = sim.det_shape
    pat = pats[0].reshape(H, W).cpu().numpy().astype(np.float32)
    lo, hi = float(pat.min()), float(pat.max())
    if hi > lo:
        pat = (pat - lo) / (hi - lo)
    return pat


def _gen_with_interp_mode(sim: patternSimulation, mode: str) -> np.ndarray:
    """Re-run the projection step with an explicit `grid_sample` interp mode.
    Used to compare the new bicubic default against the old bilinear path —
    relevant for diagnosing high-freq y-direction under-resolution when the
    Lambert grid is coarser than the detector's angular density.
    """
    H, W = sim.det_shape
    quats = sim.quats / torch.norm(sim.quats, dim=1, keepdim=True)
    accum_e_det = accum_e_to_detector(
        sim.accum_e_mc,
        sim.pattern_centerInit,
        int(H), int(W),
        float(sim.detector_tilt_deg),
        float(sim.azimuthal_deg),
        float(sim.sample_tilt_deg),
    )
    with torch.no_grad():
        pats = project_HREBSD_pattern_energy_weighted(
            pcs                   = sim.pattern_centerInit,
            n_rows                = H,
            n_cols                = W,
            tilt                  = float(sim.detector_tilt_deg),
            azimuthal             = float(sim.azimuthal_deg),
            sample_tilt           = float(sim.sample_tilt_deg),
            quaternions           = quats,
            deformation_gradients = sim.F,
            master_pattern_MSLNH  = sim.mLPNH,
            master_pattern_MSLSH  = sim.mLPSH,
            accum_e               = accum_e_det,
            interp_mode           = mode,
        )
    pat = pats[0].reshape(H, W).cpu().numpy().astype(np.float32)
    lo, hi = float(pat.min()), float(pat.max())
    if hi > lo:
        pat = (pat - lo) / (hi - lo)
    return pat


def _gen_no_obliquity(sim: patternSimulation) -> np.ndarray:
    """Generate a sim with the per-pixel accum_e replaced by a uniform array
    (every pixel gets equal energy weight).  Isolates whether the obliquity
    factor `g` and the 90° coord rotation in accum_e_to_detector are the
    source of any axis bias."""
    H, W = sim.det_shape
    nE   = sim.mLPNH.shape[0]
    uniform = torch.ones((nE, H, W), dtype=sim.dtype, device=sim.device)
    quats = sim.quats / torch.norm(sim.quats, dim=1, keepdim=True)
    with torch.no_grad():
        pats = project_HREBSD_pattern_energy_weighted(
            pcs                  = sim.pattern_centerInit,
            n_rows               = H,
            n_cols               = W,
            tilt                 = float(sim.detector_tilt_deg),
            azimuthal            = float(sim.azimuthal_deg),
            sample_tilt          = float(sim.sample_tilt_deg),
            quaternions          = quats,
            deformation_gradients= sim.F,
            master_pattern_MSLNH = sim.mLPNH,
            master_pattern_MSLSH = sim.mLPSH,
            accum_e              = uniform,
        )
    pat = pats[0].reshape(H, W).cpu().numpy().astype(np.float32)
    lo, hi = float(pat.min()), float(pat.max())
    if hi > lo:
        pat = (pat - lo) / (hi - lo)
    return pat


def _gen_master_rotated(sim_template: patternSimulation, k: int = 1) -> np.ndarray:
    """Rotate mLPNH / mLPSH by k * 90° in the Lambert (H, W) axes, then project.
    If the y-suppression follows the master pattern (becomes x-suppression),
    the master-pattern data itself is anisotropic.  If it stays in y, the
    projection chain imprints the bias."""
    sim = patternSimulation()
    sim.detector_height   = sim_template.detector_height
    sim.detector_width    = sim_template.detector_width
    sim.det_shape         = sim_template.det_shape
    sim.detector_tilt_deg = sim_template.detector_tilt_deg
    sim.sample_tilt_deg   = sim_template.sample_tilt_deg
    sim.azimuthal_deg     = sim_template.azimuthal_deg
    # rot90 on the last two axes (Lambert H, W); accum_e and quats unchanged.
    sim.mLPNH      = torch.rot90(sim_template.mLPNH,      k=k, dims=(-2, -1)).contiguous()
    sim.mLPSH      = torch.rot90(sim_template.mLPSH,      k=k, dims=(-2, -1)).contiguous()
    sim.accum_e_mc = sim_template.accum_e_mc
    sim.quats              = sim_template.quats
    sim.pattern_centerInit = sim_template.pattern_centerInit
    return _gen(sim)


def _gen_synthetic_isotropic(sim_template: patternSimulation,
                             rings_k0: float = 6.0) -> np.ndarray:
    """Replace mLPNH and mLPSH with a perfectly radial test pattern (a radial
    sinusoid) and project.  If the output detector pattern is NOT isotropic,
    the projection chain itself imprints axis bias independent of master-pattern
    content."""
    nE, mH, mW = sim_template.mLPNH.shape
    # Build a normalised radius map on the Lambert grid in [0, 1].
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, mH),
        torch.linspace(-1, 1, mW),
        indexing="ij",
    )
    r = torch.sqrt(xx ** 2 + yy ** 2)
    pattern_2d = torch.sin(2.0 * np.pi * rings_k0 * r)
    # Broadcast over the energy axis — same intensity per energy bin.
    iso = pattern_2d.unsqueeze(0).expand(nE, -1, -1).to(
        dtype=sim_template.mLPNH.dtype, device=sim_template.mLPNH.device
    ).contiguous()

    sim = patternSimulation()
    sim.detector_height   = sim_template.detector_height
    sim.detector_width    = sim_template.detector_width
    sim.det_shape         = sim_template.det_shape
    sim.detector_tilt_deg = sim_template.detector_tilt_deg
    sim.sample_tilt_deg   = sim_template.sample_tilt_deg
    sim.azimuthal_deg     = sim_template.azimuthal_deg
    sim.mLPNH      = iso
    sim.mLPSH      = iso
    sim.accum_e_mc = sim_template.accum_e_mc
    sim.quats              = sim_template.quats
    sim.pattern_centerInit = sim_template.pattern_centerInit
    return _gen(sim)


def _step3_process(pat_raw_uint16: np.ndarray, pat_obj: Data.UP2) -> np.ndarray:
    """Apply the same Step-3 processing the optimizer sees, to either a real
    or simulated pattern.  `pat_obj` must already have its set_processing()
    invoked with the desired settings."""
    return pat_obj.process_pattern(pat_raw_uint16).astype(np.float32)


def _gradients(pat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (∂I/∂x, ∂I/∂y) of `pat` (rows × cols).  Uses RectBivariateSpline
    (order 5) for consistency with debug_simulated_reference._grad_pair."""
    H, W = pat.shape
    x = np.arange(W, dtype=np.float64)
    y = np.arange(H, dtype=np.float64)
    spline = interpolate.RectBivariateSpline(x, y, pat.T, kx=5, ky=5)
    xi_x, xi_y = np.meshgrid(x, y, indexing="xy")
    gx = spline(xi_x.ravel(), xi_y.ravel(), dx=1, dy=0, grid=False).reshape(H, W)
    gy = spline(xi_x.ravel(), xi_y.ravel(), dx=0, dy=1, grid=False).reshape(H, W)
    return gx, gy


def _crop_central(pat: np.ndarray, frac: float | None) -> np.ndarray:
    if frac is None or frac >= 1.0:
        return pat
    H, W = pat.shape
    h = int(H * frac)
    w = int(W * frac)
    r0 = (H - h) // 2
    c0 = (W - w) // 2
    return pat[r0:r0 + h, c0:c0 + w]


def _marginal_fft_power(pat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (freqs, P_kx, P_ky) where P_kx = Σ_ky |F|² and P_ky = Σ_kx |F|².
    Freqs are cycles/pixel; arrays cover the positive half (excluding DC)."""
    pat = pat - np.nanmean(pat)
    H, W = pat.shape
    F = np.fft.fftshift(np.fft.fft2(np.nan_to_num(pat, nan=0.0)))
    P = np.abs(F) ** 2                          # (H, W)
    p_kx = P.sum(axis=0)                        # marginal along rows → P as fn of kx
    p_ky = P.sum(axis=1)                        # marginal along cols → P as fn of ky
    # After fftshift on an even-length axis, DC sits at index W//2 and the
    # positive (non-DC) bins are W//2+1 … W-1 — that's W//2-1 entries.
    p_kx_pos = p_kx[W // 2 + 1:]
    p_ky_pos = p_ky[H // 2 + 1:]
    freqs_kx = np.arange(1, len(p_kx_pos) + 1) / W
    freqs_ky = np.arange(1, len(p_ky_pos) + 1) / H
    # Trim to a common length so plot helpers see matching shapes.
    n = min(len(freqs_kx), len(freqs_ky))
    return (freqs_kx[:n], p_kx_pos[:n], p_ky_pos[:n])


def _stats(gx: np.ndarray, gy: np.ndarray) -> dict:
    return {
        "mean_abs_gx": float(np.nanmean(np.abs(gx))),
        "mean_abs_gy": float(np.nanmean(np.abs(gy))),
        "med_abs_gx":  float(np.nanmedian(np.abs(gx))),
        "med_abs_gy":  float(np.nanmedian(np.abs(gy))),
        "ratio_mean":  float(np.nanmean(np.abs(gy)) / max(np.nanmean(np.abs(gx)), 1e-30)),
        "ratio_med":   float(np.nanmedian(np.abs(gy)) / max(np.nanmedian(np.abs(gx)), 1e-30)),
    }


def _hi_power_ratio(p_kx: np.ndarray, p_ky: np.ndarray, hi_frac: float = 0.5) -> float:
    """Return ΣP(ky) / ΣP(kx) in the top `hi_frac` of frequencies (where the
    sim's gradient deficit is most visible)."""
    n = len(p_kx)
    lo = int(n * (1.0 - hi_frac))
    sx = float(p_kx[lo:].sum())
    sy = float(p_ky[lo:].sum())
    return sy / max(sx, 1e-30)


def _line_average_grad(gx: np.ndarray, gy: np.ndarray,
                       band_px: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_|gx| as fn of col, mean_|gy| as fn of row).  If band_px is
    given, only average over a central horizontal band of `band_px` rows for
    |gx|, and a central vertical band of `band_px` cols for |gy|."""
    H, W = gx.shape
    if band_px is None:
        gx_col = np.nanmean(np.abs(gx), axis=0)
        gy_row = np.nanmean(np.abs(gy), axis=1)
    else:
        r0 = (H - band_px) // 2
        c0 = (W - band_px) // 2
        gx_col = np.nanmean(np.abs(gx[r0:r0 + band_px, :]), axis=0)
        gy_row = np.nanmean(np.abs(gy[:, c0:c0 + band_px]), axis=1)
    return gx_col, gy_row


def _polar_grad_hist(gx: np.ndarray, gy: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    theta = np.arctan2(gy.ravel(), gx.ravel())
    weight = np.sqrt(gx.ravel() ** 2 + gy.ravel() ** 2)
    finite = np.isfinite(theta) & np.isfinite(weight)
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    hist, _ = np.histogram(theta[finite], bins=edges, weights=weight[finite])
    return edges, hist


# =============================================================================
# Main
# =============================================================================

def run(
    master_pattern_path: str = MASTER_PATTERN_PATH,
    euler_deg: tuple = EULER_DEG,
    pc_edax: tuple = PC_EDAX,
    sample_tilt_deg: float = SAMPLE_TILT_DEG,
    det_tilt_deg: float = DET_TILT_DEG,
    det_shape: tuple = DET_SHAPE,
    up2_path: str = UP2_PATH,
    pat_idx: int = PAT_IDX,
    apply_step3: bool = APPLY_STEP3,
    low_pass_sigma: float = LOW_PASS_SIGMA,
    high_pass_sigma: float = HIGH_PASS_SIGMA,
    gamma: float = GAMMA,
    mask_type: str = MASK_TYPE,
    flip_x: bool = FLIP_X,
    out_dir: str | None = None,
):
    OUT = out_dir if out_dir is not None else OUT_DIR
    os.makedirs(OUT, exist_ok=True)
    print(f"Output directory: {OUT}")
    print(f"Master pattern:   {master_pattern_path}")
    print(f"Euler (deg):      {euler_deg}")
    print(f"PC (EDAX):        {pc_edax}")
    print(f"Sample tilt:      {sample_tilt_deg}°  Det tilt: {det_tilt_deg}°")

    # ── 1. Production sim ────────────────────────────────────────────────
    print("\n[1/4] Generating production sim …")
    sim, sim_prod = _make_sim(
        master_pattern_path, euler_deg, pc_edax,
        det_shape, det_tilt_deg, sample_tilt_deg,
    )

    # ── 2. Isolation variants ────────────────────────────────────────────
    print("[2/4] Generating bilinear-interp sim (old default) …")
    sim_bilinear = _gen_with_interp_mode(sim, mode="bilinear")

    print("[2/4] Generating master-rotated sim …")
    sim_master_rot = _gen_master_rotated(sim, k=1)

    print("[2/4] Generating no-obliquity sim …")
    sim_no_oblq = _gen_no_obliquity(sim)

    print("[2/4] Generating synthetic-isotropic-MP sim …")
    sim_synth = _gen_synthetic_isotropic(sim)

    # ── 3. Optional real pattern + Step-3 processing ─────────────────────
    real_pat = None
    pat_obj  = None
    if up2_path and os.path.exists(up2_path):
        print("[3/4] Loading and processing real pattern …")
        pat_obj = Data.UP2(up2_path)
        pat_obj.set_processing(
            low_pass_sigma          = low_pass_sigma,
            high_pass_sigma         = high_pass_sigma,
            truncate_std_scale      = 3.0,
            mask_type               = (None if mask_type == "none" else mask_type),
            center_cross_half_width = 6,
            flip_x                  = flip_x,
            gamma                   = gamma,
        )
        real_raw = pat_obj.read_pattern(pat_idx, process=False).astype(np.float32)
        real_pat = _step3_process(real_raw, pat_obj) if apply_step3 else real_raw
        lo, hi = float(real_pat.min()), float(real_pat.max())
        if hi > lo:
            real_pat = (real_pat - lo) / (hi - lo)
    else:
        print("[3/4] No UP2 path supplied — skipping real-pattern comparison.")

    # Step-3 process the sims too if requested so we compare like-for-like.
    if apply_step3 and pat_obj is not None:
        def _proc(pat):
            # process_pattern expects uint16-like in the same range as UP2 reads.
            pat_u16 = np.clip(pat * 65535.0, 0, 65535).astype(np.uint16)
            out = pat_obj.process_pattern(pat_u16).astype(np.float32)
            lo_, hi_ = float(out.min()), float(out.max())
            return (out - lo_) / (hi_ - lo_) if hi_ > lo_ else out
        sim_prod       = _proc(sim_prod)
        sim_bilinear   = _proc(sim_bilinear)
        sim_master_rot = _proc(sim_master_rot)
        sim_no_oblq    = _proc(sim_no_oblq)
        sim_synth      = _proc(sim_synth)

    # ── Spectral-matched sim (only when we have a UP2 to average over) ───
    # Mirrors the production path in get_homography_cpu.py:458-470 — build
    # the target amplitude from N=10 random experimental patterns (excluding
    # the reference index), then rescale the post-Step-3 sim's per-frequency
    # amplitude to match.  If this row's Gy/Gx and hi-P ratio climb toward
    # the real's, spectral_match_ref is already doing its job and the
    # remaining ε_22 deficit is elsewhere.
    sim_specmatch = None
    specmatch_err = None
    if pat_obj is not None:
        print("[4/4] Computing average exp amplitude spectrum (n=10) …")
        try:
            target_amp = utilities.average_exp_amplitude_spectrum(
                pat_obj, n_samples=10, exclude_idx=int(pat_idx),
            )
            sim_specmatch = utilities.spectral_match_pattern(
                sim_prod.astype(np.float32),
                target_amp.astype(np.float32),
            )
            # Re-normalise so the histogram-/gradient-stats stay on the
            # same intensity scale as the other rows.
            lo_, hi_ = float(sim_specmatch.min()), float(sim_specmatch.max())
            if hi_ > lo_:
                sim_specmatch = (sim_specmatch - lo_) / (hi_ - lo_)
        except Exception as exc:
            import traceback as _tb
            specmatch_err = (
                f"{type(exc).__name__}: {exc}\n"
                + "".join(_tb.format_tb(exc.__traceback__))
            )
            print(f"[4/4] Spectral match failed:\n{specmatch_err}",
                  file=sys.stderr)
            sim_specmatch = None

    # ── 4. Analyse + plot ────────────────────────────────────────────────
    print("[4/4] Analysing + plotting …")

    patterns = {
        "sim_prod":       sim_prod,
        "sim_bilinear":   sim_bilinear,
        "sim_master_rot": sim_master_rot,
        "sim_no_oblq":    sim_no_oblq,
        "sim_synth":      sim_synth,
    }
    if sim_specmatch is not None:
        # Insert right after sim_prod so it sits next to its un-matched twin
        # in the plots and the summary table.
        patterns = {"sim_prod": sim_prod, "sim_specmatch": sim_specmatch,
                    **{k: v for k, v in patterns.items() if k != "sim_prod"}}
    if real_pat is not None:
        patterns = {"real": real_pat, **patterns}

    pretty_label = {
        "real":           "real",
        "sim_prod":       "sim (production, bicubic)",
        "sim_specmatch":  "sim (spectral-matched to real)",
        "sim_bilinear":   "sim (bilinear, old default)",
        "sim_master_rot": "sim (master rot 90°)",
        "sim_no_oblq":    "sim (no obliquity)",
        "sim_synth":      "sim (synthetic isotropic MP)",
    }

    gradients = {}
    fft_data  = {}
    stats     = {}
    for k, pat in patterns.items():
        pat_a = _crop_central(pat, ANALYSIS_CROP)
        gx, gy = _gradients(pat_a)
        gradients[k] = (gx, gy)
        fft_data[k]  = _marginal_fft_power(pat_a)
        stats[k]     = _stats(gx, gy)
        freqs, p_kx, p_ky = fft_data[k]
        stats[k]["hi_power_ratio"] = _hi_power_ratio(p_kx, p_ky, hi_frac=0.5)

    # Figure 1 — pattern thumbnails
    fig, axes = plt.subplots(1, len(patterns), figsize=(3.2 * len(patterns), 3.6))
    if len(patterns) == 1:
        axes = [axes]
    for ax, (k, pat) in zip(axes, patterns.items()):
        ax.imshow(pat, cmap="gray", origin="upper")
        ax.set_title(pretty_label[k], fontsize=10)
        ax.axis("off")
    fig.suptitle("Pattern thumbnails", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/1_patterns.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Figure 2 — marginal FFT power per axis
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for k in patterns:
        freqs, p_kx, p_ky = fft_data[k]
        axes[0].semilogy(freqs, p_kx, label=pretty_label[k])
        axes[1].semilogy(freqs, p_ky, label=pretty_label[k])
    axes[0].set_title("P(kx)  —  marginal power along the horizontal frequency axis",
                      fontsize=10)
    axes[1].set_title("P(ky)  —  marginal power along the vertical frequency axis",
                      fontsize=10)
    for ax in axes:
        ax.set_xlabel("Spatial frequency (cycles/pixel)")
        ax.set_ylabel("Power (log)")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/2_marginal_fft.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Figure 3 — |Gx| / |Gy| bar chart with ratio annotations
    labels = list(patterns.keys())
    means_x = [stats[k]["mean_abs_gx"] for k in labels]
    means_y = [stats[k]["mean_abs_gy"] for k in labels]
    ratios  = [stats[k]["ratio_mean"]  for k in labels]
    x_pos   = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.5 * len(labels) + 3, 5))
    ax.bar(x_pos - w / 2, means_x, w, label="mean |Gx|", color="steelblue")
    ax.bar(x_pos + w / 2, means_y, w, label="mean |Gy|", color="tomato")
    for i, r in enumerate(ratios):
        ax.text(i, max(means_x[i], means_y[i]) * 1.04,
                f"Gy/Gx = {r:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([pretty_label[k] for k in labels], rotation=15, ha="right")
    ax.set_ylabel("Mean |gradient|")
    ax.set_title("Mean |∂I/∂x| vs |∂I/∂y|  (Gy / Gx ≈ 1 ⇒ isotropic)", fontsize=10)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/3_grad_means.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Figure 4 — polar gradient histograms
    n_cols_fig = len(patterns)
    fig, axes = plt.subplots(1, n_cols_fig, figsize=(3.6 * n_cols_fig, 4),
                             subplot_kw=dict(projection="polar"))
    if n_cols_fig == 1:
        axes = [axes]
    for ax, k in zip(axes, labels):
        gx, gy = gradients[k]
        edges, hist = _polar_grad_hist(gx, gy, POLAR_BINS)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width   = edges[1] - edges[0]
        ax.bar(centers, hist, width=width, color="steelblue", alpha=0.85,
               edgecolor="k", linewidth=0.4)
        ax.set_title(pretty_label[k], fontsize=10, pad=12)
        ax.set_yticklabels([])
    fig.suptitle("Gradient direction histogram (|G|-weighted) — uniform = isotropic",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/4_polar_hist.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Figure 5 — line-averaged |Gx|(col), |Gy|(row)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for k in patterns:
        gx, gy = gradients[k]
        gx_col, gy_row = _line_average_grad(gx, gy, LINE_AVG_BAND_PX)
        axes[0].plot(gx_col, label=pretty_label[k])
        axes[1].plot(gy_row, label=pretty_label[k])
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("mean |∂I/∂x| over rows")
    axes[0].set_title("|∂I/∂x| as a function of column", fontsize=10)
    axes[1].set_xlabel("Row")
    axes[1].set_ylabel("mean |∂I/∂y| over cols")
    axes[1].set_title("|∂I/∂y| as a function of row", fontsize=10)
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/5_line_averaged_grad.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Figure 6 — 2D FFT log-magnitudes side by side (quick visual)
    fig, axes = plt.subplots(1, len(patterns), figsize=(3.2 * len(patterns), 3.6))
    if len(patterns) == 1:
        axes = [axes]
    for ax, k in zip(axes, labels):
        pat_a = _crop_central(patterns[k], ANALYSIS_CROP)
        F = np.fft.fftshift(np.fft.fft2(pat_a - np.nanmean(pat_a)))
        ax.imshow(np.log10(np.abs(F) + 1e-3), cmap="inferno", origin="lower")
        ax.set_title(pretty_label[k], fontsize=10)
        ax.axis("off")
    fig.suptitle("2D FFT log-magnitude — anisotropy here = anisotropic frequency content",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/6_fft2d.png", dpi=180, bbox_inches="tight")
    plt.close()

    # ── Summary table (also written to disk) ─────────────────────────────
    lines = []
    header = f"{'pattern':<32} {'|Gx|':>10} {'|Gy|':>10} {'Gy/Gx':>8} {'hi-P(ky)/P(kx)':>16}"
    lines.append(header)
    lines.append("-" * len(header))
    for k in labels:
        s = stats[k]
        lines.append(
            f"{pretty_label[k]:<32} "
            f"{s['mean_abs_gx']:>10.4f} "
            f"{s['mean_abs_gy']:>10.4f} "
            f"{s['ratio_mean']:>8.3f} "
            f"{s['hi_power_ratio']:>16.3f}"
        )
    table = "\n".join(lines)
    print("\n" + table)
    with open(f"{OUT}/summary.txt", "w") as f:
        f.write(table + "\n")
        if specmatch_err is not None:
            f.write(
                "\n[spectral-match row was DROPPED — exception below]\n"
                + specmatch_err + "\n"
            )
    print(f"\nAll diagnostics saved under {OUT}/")
    return OUT


def _parse_triple(s: str) -> tuple:
    return tuple(float(x) for x in s.split(","))


def _main_cli():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--master-pattern", default=MASTER_PATTERN_PATH)
    p.add_argument("--euler-deg",  default=",".join(str(x) for x in EULER_DEG),
                   type=_parse_triple, help="phi1,Phi,phi2 in degrees")
    p.add_argument("--pc-edax",    default=",".join(str(x) for x in PC_EDAX),
                   type=_parse_triple, help="x*,y*,z* in EDAX convention")
    p.add_argument("--sample-tilt", type=float, default=SAMPLE_TILT_DEG)
    p.add_argument("--det-tilt",    type=float, default=DET_TILT_DEG)
    p.add_argument("--det-shape",   default=f"{DET_SHAPE[0]},{DET_SHAPE[1]}",
                   help="rows,cols")
    p.add_argument("--up2",         default=UP2_PATH)
    p.add_argument("--pat-idx",     type=int, default=PAT_IDX)
    p.add_argument("--no-step3",    action="store_true",
                   help="Skip Step-3 processing on real and sim patterns.")
    p.add_argument("--low-pass",    type=float, default=LOW_PASS_SIGMA)
    p.add_argument("--high-pass",   type=float, default=HIGH_PASS_SIGMA)
    p.add_argument("--gamma",       type=float, default=GAMMA)
    p.add_argument("--mask",        default=MASK_TYPE)
    p.add_argument("--flip-x",      action="store_true", default=FLIP_X)
    p.add_argument("--out",         default=None,
                   help="Output directory; defaults to a timestamped folder.")
    args = p.parse_args()
    det_shape = tuple(int(x) for x in args.det_shape.split(","))
    run(
        master_pattern_path = args.master_pattern,
        euler_deg           = args.euler_deg,
        pc_edax             = args.pc_edax,
        sample_tilt_deg     = args.sample_tilt,
        det_tilt_deg        = args.det_tilt,
        det_shape           = det_shape,
        up2_path            = args.up2,
        pat_idx             = args.pat_idx,
        apply_step3         = not args.no_step3,
        low_pass_sigma      = args.low_pass,
        high_pass_sigma     = args.high_pass,
        gamma               = args.gamma,
        mask_type           = args.mask,
        flip_x              = args.flip_x,
        out_dir             = args.out,
    )


if __name__ == "__main__":
    _main_cli()

"""
Optimize pattern center (PC) and Euler angles so that the simulated reference
pattern best matches the experimental one, measured via normalized cross-
correlation (ZNSSD).

Preprocessing (bandpass filter, CLAHE, etc.) is applied with the same mask
settings as the rest of the pipeline.  The ZNSSD is then computed over all
pixels so that the full detector area contributes to the metric.

Usage
-----
    from optimize_reference import optimize_pc_and_euler

    euler_opt, pc_opt = optimize_pc_and_euler(
        pat_obj=pat_obj,
        x0=x0,
        master_pattern_path=master_pattern_path,
        euler_angles_init=euler_angles_ref,
        pc_init=pc_ref,
        tilt_deg=tilt_deg,
    )
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PatternSimulation.SimPatGen import patternSimulation


# ---------------------------------------------------------------------------
# Quaternion helpers for gimbal-lock-free PC/Euler refinement
# ---------------------------------------------------------------------------
# These are exact pure-numpy mirrors of HREBSD.bu2qu_emsoft (and its inverse)
# plus the so(3) exponential and Hamilton product, so the Nelder-Mead loop
# can parameterise rotation perturbations in a singularity-free tangent
# space (rotation vector) and still hand a Bunge Euler triple to the
# existing _simulate(sim, euler, pc, ...) call.

def _bu2qu_emsoft(eu: np.ndarray) -> np.ndarray:
    """Bunge ZXZ Euler (radians, shape (3,)) → quaternion (w, x, y, z),
    matching HREBSD.bu2qu_emsoft term-by-term."""
    phi1, Phi, phi2 = float(eu[0]), float(eu[1]), float(eu[2])
    sigma = 0.5 * (phi1 + phi2)
    delta = 0.5 * (phi1 - phi2)
    c = np.cos(0.5 * Phi)
    s = np.sin(0.5 * Phi)
    qu = np.array([
         c * np.cos(sigma),
        -s * np.cos(delta),
        -s * np.sin(delta),
        -c * np.sin(sigma),
    ], dtype=np.float64)
    if qu[0] < 0.0:
        qu = -qu
    return qu


def _qu2bu_emsoft(qu: np.ndarray) -> np.ndarray:
    """Inverse of _bu2qu_emsoft: quaternion → Bunge ZXZ Euler (radians)."""
    w, x, y, z = float(qu[0]), float(qu[1]), float(qu[2]), float(qu[3])
    c = np.sqrt(w * w + z * z)
    s = np.sqrt(x * x + y * y)
    Phi = 2.0 * np.arctan2(s, c)
    if s < 1e-12:
        # Φ ≈ 0: only σ = (φ₁ + φ₂)/2 is determined.  Set φ₂ = 0 and absorb
        # the full rotation into φ₁.  This is the standard convention.
        sigma = np.arctan2(-z, w)
        return np.array([2.0 * sigma, 0.0, 0.0], dtype=np.float64)
    sigma = np.arctan2(-z, w)
    delta = np.arctan2(-y, -x)
    return np.array([sigma + delta, Phi, sigma - delta], dtype=np.float64)


def _rotvec_to_qu(rv: np.ndarray) -> np.ndarray:
    """so(3) exponential: axis·angle vector (radians, shape (3,)) → quaternion."""
    rv = np.asarray(rv, dtype=np.float64)
    angle = np.linalg.norm(rv)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    half = 0.5 * angle
    s = np.sin(half) / angle
    return np.array([np.cos(half), rv[0] * s, rv[1] * s, rv[2] * s], dtype=np.float64)


def _qu_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product (w, x, y, z) — same convention as HREBSD."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def _wrap_euler_to_branch(euler: np.ndarray, euler_ref: np.ndarray) -> np.ndarray:
    """Pick the Euler representative in the same 2π branch as `euler_ref`.

    bu2qu_emsoft canonicalises the quaternion to w ≥ 0 by sign-flipping it
    when needed.  After the flip, _qu2bu_emsoft returns φ₁ (or φ₂) shifted
    by 2π — same rotation, different representative.  The optimization
    sees this as a 360° "overshoot" even though the underlying orientation
    barely moved.  Wrapping φ₁ and φ₂ each to within π of `euler_ref`
    restores a sensible Δ Euler for display and downstream code.  Φ is
    already in [0, π] (no wrapping needed).
    """
    out = euler.copy()
    out[0] = euler_ref[0] + ((euler[0] - euler_ref[0] + np.pi) % (2.0 * np.pi)) - np.pi
    out[2] = euler_ref[2] + ((euler[2] - euler_ref[2] + np.pi) % (2.0 * np.pi)) - np.pi
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _simulate(sim: patternSimulation, euler: np.ndarray, pc: np.ndarray,
              pat_obj, patshape: tuple,
              high_pass_sigma_override: float = None,
              low_pass_sigma_override: float = None,
              gamma_override: float = None) -> np.ndarray:
    """Generate one simulated pattern, processed without mask.

    The mask is skipped because masked pixels in the real pattern are detector
    artefacts (direct beam, shadow) with no physical equivalent in the
    simulation.  Zero-filling them creates a boundary halo that biases the ZNSSD.
    """
    import torch

    # EDAX/TSL -> Bruker: flip y component
    pc_bruker = (float(pc[0]), 1.0 - float(pc[1]), float(pc[2]))
    sim.EandPCSet(euler, pc_bruker, verbose=False)

    with torch.no_grad():
        pats = sim.GenPattern()

    pat = pats[0].reshape(patshape).cpu().numpy().astype(np.float32)
    # PC sign convention fixed in HREBSD.detector_coords_to_ksphere_via_pc — no fliplr needed.
    lo, hi = pat.min(), pat.max()
    if hi > lo:
        pat = (pat - lo) / (hi - lo)

    if pat_obj is not None:
        orig_mask = pat_obj.mask_type
        orig_hp   = pat_obj.high_pass_sigma
        orig_lp   = pat_obj.low_pass_sigma
        orig_g    = pat_obj.gamma
        pat_obj.mask_type = None
        if high_pass_sigma_override is not None:
            pat_obj.high_pass_sigma = high_pass_sigma_override
        if low_pass_sigma_override is not None:
            pat_obj.low_pass_sigma = low_pass_sigma_override
        if gamma_override is not None:
            pat_obj.gamma = gamma_override
        try:
            pat = pat_obj.process_pattern(pat)
        finally:
            pat_obj.mask_type      = orig_mask
            pat_obj.high_pass_sigma = orig_hp
            pat_obj.low_pass_sigma  = orig_lp
            pat_obj.gamma           = orig_g

    return pat


def _znssd(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean Normalized Sum of Squared Differences.

    For zero-mean unit-norm a' and b':
        ZNSSD = Σ (a' − b')²  =  2 · (1 − ZNCC)

    Range:  0 (perfect match) → 2 (uncorrelated) → 4 (perfectly anti-correlated).
    Lower is better — minimised by IC-GN, so the optimisers below treat it as
    a loss to drive down."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 2.0
    a = a / na
    b = b / nb
    diff = a - b
    return float(np.dot(diff, diff))


def _build_initial_simplex(x0: np.ndarray,
                            euler_step_deg: float = 0.5,
                            pc_step: float = 0.005) -> np.ndarray:
    """
    Build an (n+1 x n) initial simplex with physically meaningful step sizes.
    First 3 params are *rotation-vector* deltas in radians (so(3) tangent
    space — one axis-component each); last 3 are PC deltas.  The
    rotation-vector step equals the per-axis step that would have been
    applied to Euler angles, so the simplex's perturbation magnitude
    stays consistent with the previous Euler-parameterised version.
    """
    n = len(x0)
    simplex = np.tile(x0, (n + 1, 1)).astype(np.float64)
    steps = np.array([
        np.deg2rad(euler_step_deg),
        np.deg2rad(euler_step_deg),
        np.deg2rad(euler_step_deg),
        pc_step,
        pc_step,
        pc_step,
    ])
    for i in range(n):
        simplex[i + 1, i] += steps[i]
    return simplex


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize_preprocessing_params(
    real_pat_raw: np.ndarray,
    sim_pat_raw: np.ndarray,
    pat_obj,
    high_pass_bounds: tuple = (3.0, 80.0),
    low_pass_bounds: tuple = (0.0, 5.0),
    gamma_bounds: tuple = (0.1, 1.5),
    n_eval: int = 400,
    seed: int = 42,
    save_dir: str = "debug",
) -> dict:
    """Find the high_pass_sigma, low_pass_sigma, and gamma that minimise ZNSSD
    between a real and simulated EBSD pattern.

    CLAHE is disabled; the mask is suppressed on both patterns so detector
    artefacts (direct beam, shadow) do not penalise the ZNSSD.  All other
    pat_obj pipeline settings (truncation, unsharp masking) stay fixed.

    The search uses differential evolution (global) followed by a local polish.
    Four diagnostic figures are saved to *save_dir*.

    Parameters
    ----------
    real_pat_raw : ndarray
        Raw unprocessed real pattern, any numeric dtype.
    sim_pat_raw : ndarray
        Raw simulated pattern normalised to [0, 1].
    pat_obj : Data.UP2
        Source of the process_pattern pipeline.  Attributes are temporarily
        overridden for each evaluation and fully restored afterwards.
    high_pass_bounds, low_pass_bounds, gamma_bounds : (min, max)
        Search bounds for each parameter.
    n_eval : int
        Approximate number of objective evaluations (default 400).
    seed : int
        Random seed for reproducibility.
    save_dir : str
        Directory for output figures.

    Returns
    -------
    dict
        Keys: high_pass_sigma, low_pass_sigma, gamma,
              znssd_init, znssd_opt,
              real_init, sim_init, real_opt, sim_opt
    """
    from scipy.optimize import differential_evolution

    os.makedirs(save_dir, exist_ok=True)

    init_hp    = pat_obj.high_pass_sigma
    init_lp    = pat_obj.low_pass_sigma
    init_gamma = pat_obj.gamma

    # ── helper: temporarily override hp / lp / gamma, restore on exit ────────
    def _proc(raw, hp, lp, g):
        saved = (pat_obj.high_pass_sigma, pat_obj.low_pass_sigma, pat_obj.gamma,
                 pat_obj.mask_type, pat_obj.use_clahe)
        pat_obj.high_pass_sigma = float(hp)
        pat_obj.low_pass_sigma  = float(lp)
        pat_obj.gamma           = float(g)
        pat_obj.mask_type       = None
        pat_obj.use_clahe       = False
        try:
            return pat_obj.process_pattern(raw.copy().astype(np.float32))
        finally:
            (pat_obj.high_pass_sigma, pat_obj.low_pass_sigma, pat_obj.gamma,
             pat_obj.mask_type, pat_obj.use_clahe) = saved

    # ── baseline ──────────────────────────────────────────────────────────────
    real_init = _proc(real_pat_raw, init_hp, init_lp, init_gamma)
    sim_init  = _proc(sim_pat_raw,  init_hp, init_lp, init_gamma)
    znssd_init  = _znssd(real_init, sim_init)
    print(f"\n[Preprocopt] initial: hp={init_hp:.1f}  lp={init_lp:.2f}  gamma={init_gamma:.3f}")
    print(f"[Preprocopt] initial ZNSSD: {znssd_init:.6f}")

    # ── optimisation ──────────────────────────────────────────────────────────
    nfev = [0]
    best = {"znssd": znssd_init, "hp": init_hp, "lp": init_lp, "g": init_gamma}

    def _objective(x):
        hp, lp, g = float(x[0]), float(x[1]), float(x[2])
        r = _proc(real_pat_raw, hp, lp, g)
        s = _proc(sim_pat_raw,  hp, lp, g)
        v = _znssd(r, s)
        nfev[0] += 1
        if v < best["znssd"]:
            best.update({"znssd": v, "hp": hp, "lp": lp, "g": g})
        if nfev[0] % 50 == 0:
            print(f"  [iter {nfev[0]:4d}]  ZNSSD={v:.5f}  best={best['znssd']:.5f}  "
                  f"hp={hp:.1f}  lp={lp:.2f}  g={g:.3f}")
        return v

    bounds  = [high_pass_bounds, low_pass_bounds, gamma_bounds]
    popsize = 12
    maxiter = max(5, n_eval // (popsize * len(bounds)))
    print(f"[Preprocopt] differential evolution: popsize={popsize}  maxiter={maxiter}  "
          f"~{popsize * len(bounds) * maxiter} evals  (polish=True) …")
    differential_evolution(
        _objective, bounds,
        maxiter=maxiter, popsize=popsize,
        tol=1e-5, seed=seed,
        mutation=(0.5, 1.0), recombination=0.7,
        polish=True,
    )

    opt_hp    = float(best["hp"])
    opt_lp    = float(best["lp"])
    opt_gamma = float(best["g"])
    znssd_opt   = best["znssd"]

    print(f"\n[Preprocopt] optimal: hp={opt_hp:.2f}  lp={opt_lp:.3f}  gamma={opt_gamma:.4f}")
    print(f"[Preprocopt] ZNSSD:     {znssd_init:.6f} → {znssd_opt:.6f}"
          f"  (Δ = {znssd_opt - znssd_init:+.6f})")

    real_opt = _proc(real_pat_raw, opt_hp, opt_lp, opt_gamma)
    sim_opt  = _proc(sim_pat_raw,  opt_hp, opt_lp, opt_gamma)

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 1 — pattern comparison: initial vs optimised
    # ─────────────────────────────────────────────────────────────────────────
    print("[Preprocopt] saving diagnostic figures …")
    _vabs = max(np.percentile(np.abs(real_init - sim_init), 98),
                np.percentile(np.abs(real_opt  - sim_opt),  98), 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row_i, (r, s, znssd_v, row_lbl) in enumerate([
        (real_init, sim_init, znssd_init,
         f"Initial   hp={init_hp:.1f}  lp={init_lp:.2f}  γ={init_gamma:.3f}"),
        (real_opt, sim_opt, znssd_opt,
         f"Optimised  hp={opt_hp:.1f}  lp={opt_lp:.2f}  γ={opt_gamma:.3f}"),
    ]):
        vlo, vhi = min(r.min(), s.min()), max(r.max(), s.max())
        diff = r - s
        axes[row_i, 0].imshow(r,    cmap="gray",  vmin=vlo,   vmax=vhi)
        axes[row_i, 0].set_title("Real", fontsize=11)
        axes[row_i, 0].axis("off")
        axes[row_i, 1].imshow(s,    cmap="gray",  vmin=vlo,   vmax=vhi)
        axes[row_i, 1].set_title(f"Simulated   ZNSSD={znssd_v:.4f}", fontsize=11)
        axes[row_i, 1].axis("off")
        axes[row_i, 2].imshow(diff, cmap="RdBu",  vmin=-_vabs, vmax=_vabs)
        axes[row_i, 2].set_title("Real − Simulated", fontsize=11)
        axes[row_i, 2].axis("off")
        axes[row_i, 0].set_ylabel(row_lbl, fontsize=9)
    fig.suptitle(f"Preprocessing optimisation — pattern comparison\n"
                 f"ZNSSD: {znssd_init:.4f}  →  {znssd_opt:.4f}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _p1 = os.path.join(save_dir, "preprocopt_1_patterns.png")
    plt.savefig(_p1, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved {_p1}")

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 2 — 1D ZNSSD sensitivity sweeps
    # ─────────────────────────────────────────────────────────────────────────
    n_sw = 25
    hp_sw  = np.linspace(*high_pass_bounds, n_sw)
    lp_sw  = np.linspace(*low_pass_bounds,  n_sw)
    gam_sw = np.linspace(*gamma_bounds,     n_sw)
    print("  Computing 1D sweeps …")
    znssd_hp  = [_znssd(_proc(real_pat_raw, h,      opt_lp, opt_gamma),
                    _proc(sim_pat_raw,  h,      opt_lp, opt_gamma)) for h in hp_sw]
    znssd_lp  = [_znssd(_proc(real_pat_raw, opt_hp, l,      opt_gamma),
                    _proc(sim_pat_raw,  opt_hp, l,      opt_gamma)) for l in lp_sw]
    znssd_gam = [_znssd(_proc(real_pat_raw, opt_hp, opt_lp, g),
                    _proc(sim_pat_raw,  opt_hp, opt_lp, g)) for g in gam_sw]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, xs, ys, xopt, xlbl in [
        (axes[0], hp_sw,  znssd_hp,  opt_hp,    "high_pass_sigma"),
        (axes[1], lp_sw,  znssd_lp,  opt_lp,    "low_pass_sigma"),
        (axes[2], gam_sw, znssd_gam, opt_gamma, "gamma"),
    ]:
        ax.plot(xs, ys, "o-", color="steelblue", ms=4)
        ax.axvline(xopt, color="tomato", ls="--", lw=1.5, label=f"opt = {xopt:.3f}")
        ax.set_xlabel(xlbl); ax.set_ylabel("ZNSSD")
        ax.legend(fontsize=9)
        ax.set_title(f"ZNSSD vs {xlbl}\n(other params fixed at optimal)", fontsize=10)
    fig.suptitle("Preprocessing optimisation: 1D ZNSSD sensitivity",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _p2 = os.path.join(save_dir, "preprocopt_2_sweeps.png")
    plt.savefig(_p2, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved {_p2}")

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 3 — ZNSSD landscape: high_pass_sigma × gamma (low_pass fixed)
    # ─────────────────────────────────────────────────────────────────────────
    n_land   = 18
    hp_land  = np.linspace(*high_pass_bounds, n_land)
    gam_land = np.linspace(*gamma_bounds,     n_land)
    print("  Computing ZNSSD landscape (hp × gamma) …")
    znssd_land = np.zeros((n_land, n_land))
    for i, hp in enumerate(hp_land):
        for j, g in enumerate(gam_land):
            r = _proc(real_pat_raw, hp, opt_lp, g)
            s = _proc(sim_pat_raw,  hp, opt_lp, g)
            znssd_land[i, j] = _znssd(r, s)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(gam_land, hp_land, znssd_land, cmap="viridis_r", shading="auto")
    plt.colorbar(im, ax=ax, label="ZNSSD")
    ax.plot(opt_gamma, opt_hp,   "r*",  ms=14, label=f"optimal ({opt_gamma:.3f}, {opt_hp:.1f})")
    ax.plot(init_gamma, init_hp, "w^",  ms=10, mec="gray",
            label=f"initial ({init_gamma:.3f}, {init_hp:.1f})")
    ax.set_xlabel("gamma"); ax.set_ylabel("high_pass_sigma")
    ax.set_title(f"ZNSSD landscape  (low_pass_sigma fixed at {opt_lp:.2f})", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _p3 = os.path.join(save_dir, "preprocopt_3_landscape.png")
    plt.savefig(_p3, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved {_p3}")

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 4 — intensity line scans: horizontal and vertical through centre
    # ─────────────────────────────────────────────────────────────────────────
    H, W  = real_opt.shape
    mid_r = H // 2
    mid_c = W // 2

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    slices_cfg = [
        ("Horizontal", np.arange(W), "Column (px)",
         real_init[mid_r, :], sim_init[mid_r, :],
         real_opt[mid_r,  :], sim_opt[mid_r,  :]),
        ("Vertical",   np.arange(H), "Row (px)",
         real_init[:, mid_c], sim_init[:, mid_c],
         real_opt[:,  mid_c], sim_opt[:,  mid_c]),
    ]
    for col, (direction, x_coord, xlabel, ri, si, ro, so) in enumerate(slices_cfg):
        for row, (r_sl, s_sl, znssd_v, suffix) in enumerate([
            (ri, si, znssd_init, f"initial   ZNSSD={znssd_init:.4f}"),
            (ro, so, znssd_opt,  f"optimised  ZNSSD={znssd_opt:.4f}"),
        ]):
            ax = axes[row, col]
            ax.plot(x_coord, r_sl, color="steelblue", label="Real",      lw=1.2)
            ax.plot(x_coord, s_sl, color="tomato",    label="Simulated", lw=1.2, ls="--")
            ax.set_title(f"{direction} slice — {suffix}", fontsize=10)
            ax.set_xlabel(xlabel); ax.set_ylabel("Intensity (z-score)")
            ax.legend(fontsize=9)
    fig.suptitle("Preprocessing optimisation: intensity line scans\n"
                 f"horizontal = row {mid_r},  vertical = col {mid_c}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _p4 = os.path.join(save_dir, "preprocopt_4_linescans.png")
    plt.savefig(_p4, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved {_p4}\n")

    return {
        "high_pass_sigma": opt_hp,
        "low_pass_sigma":  opt_lp,
        "gamma":           opt_gamma,
        "znssd_init":        znssd_init,
        "znssd_opt":         znssd_opt,
        "real_init":       real_init,
        "sim_init":        sim_init,
        "real_opt":        real_opt,
        "sim_opt":         sim_opt,
    }


def optimize_preprocessing_params_independent(
    real_pat_raw: np.ndarray,
    sim_pat_raw: np.ndarray,
    pat_obj,
    real_high_pass_bounds: tuple = (3.0, 80.0),
    real_low_pass_bounds: tuple = (0.0, 5.0),
    real_gamma_bounds: tuple = (0.1, 1.5),
    sim_high_pass_bounds: tuple = (3.0, 80.0),
    sim_low_pass_bounds: tuple = (0.0, 5.0),
    sim_gamma_bounds: tuple = (0.1, 1.5),
    n_eval: int = 600,
    seed: int = 42,
    save_dir: str = "debug",
) -> dict:
    """Like optimize_preprocessing_params but allows the real and simulated
    patterns to have *independent* high_pass_sigma, low_pass_sigma, and gamma.

    Searches over 6 parameters: (hp_r, lp_r, gamma_r, hp_s, lp_s, gamma_s).
    Both masks and CLAHE are suppressed during evaluation so detector artefacts
    do not bias the ZNSSD.  All other pat_obj pipeline settings stay fixed.

    Parameters
    ----------
    real_pat_raw : ndarray   — raw real pattern, any numeric dtype.
    sim_pat_raw  : ndarray   — raw simulated pattern normalised to [0, 1].
    pat_obj      : Data.UP2  — source of the processing pipeline.
    real_high_pass_bounds, real_low_pass_bounds, real_gamma_bounds : (min, max)
    sim_high_pass_bounds,  sim_low_pass_bounds,  sim_gamma_bounds  : (min, max)
    n_eval  : int  — approximate differential-evolution evaluations (default 600).
    seed    : int
    save_dir: str

    Returns
    -------
    dict
        real_high_pass_sigma, real_low_pass_sigma, real_gamma,
        sim_high_pass_sigma,  sim_low_pass_sigma,  sim_gamma,
        znssd_init, znssd_opt,
        real_init, sim_init, real_opt, sim_opt
    """
    from scipy.optimize import differential_evolution

    os.makedirs(save_dir, exist_ok=True)

    init_hp    = pat_obj.high_pass_sigma
    init_lp    = pat_obj.low_pass_sigma
    init_gamma = pat_obj.gamma

    # ── helpers: each pattern gets its own params; mask + CLAHE suppressed ────
    def _proc_real(raw, hp, lp, g):
        saved = (pat_obj.high_pass_sigma, pat_obj.low_pass_sigma, pat_obj.gamma,
                 pat_obj.mask_type, pat_obj.use_clahe)
        pat_obj.high_pass_sigma = float(hp)
        pat_obj.low_pass_sigma  = float(lp)
        pat_obj.gamma           = float(g)
        pat_obj.mask_type       = None
        pat_obj.use_clahe       = False
        try:
            return pat_obj.process_pattern(raw.copy().astype(np.float32))
        finally:
            (pat_obj.high_pass_sigma, pat_obj.low_pass_sigma, pat_obj.gamma,
             pat_obj.mask_type, pat_obj.use_clahe) = saved

    def _proc_sim(raw, hp, lp, g):
        return _proc_real(raw, hp, lp, g)   # same pipeline, separate param set

    # ── baseline ──────────────────────────────────────────────────────────────
    real_init = _proc_real(real_pat_raw, init_hp, init_lp, init_gamma)
    sim_init  = _proc_sim( sim_pat_raw,  init_hp, init_lp, init_gamma)
    znssd_init  = _znssd(real_init, sim_init)
    print(f"\n[Preprocopt-indep] initial (both): hp={init_hp:.1f}  lp={init_lp:.2f}  "
          f"gamma={init_gamma:.3f}")
    print(f"[Preprocopt-indep] initial ZNSSD: {znssd_init:.6f}")

    # ── optimisation ──────────────────────────────────────────────────────────
    # param vector: [hp_r, lp_r, g_r, hp_s, lp_s, g_s]
    nfev = [0]
    best = {"znssd": znssd_init,
            "hp_r": init_hp, "lp_r": init_lp, "g_r": init_gamma,
            "hp_s": init_hp, "lp_s": init_lp, "g_s": init_gamma}

    def _objective(x):
        hp_r, lp_r, g_r, hp_s, lp_s, g_s = (float(v) for v in x)
        r = _proc_real(real_pat_raw, hp_r, lp_r, g_r)
        s = _proc_sim( sim_pat_raw,  hp_s, lp_s, g_s)
        v = _znssd(r, s)
        nfev[0] += 1
        if v < best["znssd"]:
            best.update({"znssd": v,
                         "hp_r": hp_r, "lp_r": lp_r, "g_r": g_r,
                         "hp_s": hp_s, "lp_s": lp_s, "g_s": g_s})
        if nfev[0] % 50 == 0:
            print(f"  [iter {nfev[0]:4d}]  ZNSSD={v:.5f}  best={best['znssd']:.5f}  "
                  f"real(hp={hp_r:.1f} lp={lp_r:.2f} g={g_r:.3f})  "
                  f"sim(hp={hp_s:.1f} lp={lp_s:.2f} g={g_s:.3f})")
        return v

    bounds = [real_high_pass_bounds, real_low_pass_bounds, real_gamma_bounds,
              sim_high_pass_bounds,  sim_low_pass_bounds,  sim_gamma_bounds]
    popsize = 12
    maxiter = max(5, n_eval // (popsize * len(bounds)))
    print(f"[Preprocopt-indep] differential evolution: popsize={popsize}  "
          f"maxiter={maxiter}  ~{popsize * len(bounds) * maxiter} evals …")
    differential_evolution(
        _objective, bounds,
        maxiter=maxiter, popsize=popsize,
        tol=1e-5, seed=seed,
        mutation=(0.5, 1.0), recombination=0.7,
        polish=True,
    )

    opt_hp_r = float(best["hp_r"]); opt_lp_r = float(best["lp_r"]); opt_g_r = float(best["g_r"])
    opt_hp_s = float(best["hp_s"]); opt_lp_s = float(best["lp_s"]); opt_g_s = float(best["g_s"])
    znssd_opt  = best["znssd"]

    print(f"\n[Preprocopt-indep] optimal real: hp={opt_hp_r:.2f}  lp={opt_lp_r:.3f}  "
          f"gamma={opt_g_r:.4f}")
    print(f"[Preprocopt-indep] optimal sim:  hp={opt_hp_s:.2f}  lp={opt_lp_s:.3f}  "
          f"gamma={opt_g_s:.4f}")
    print(f"[Preprocopt-indep] ZNSSD:  {znssd_init:.6f} → {znssd_opt:.6f}"
          f"  (Δ = {znssd_opt - znssd_init:+.6f})")

    real_opt = _proc_real(real_pat_raw, opt_hp_r, opt_lp_r, opt_g_r)
    sim_opt  = _proc_sim( sim_pat_raw,  opt_hp_s, opt_lp_s, opt_g_s)

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 1 — pattern comparison: initial vs optimised
    # ─────────────────────────────────────────────────────────────────────────
    print("[Preprocopt-indep] saving diagnostic figures …")
    _vabs = max(np.percentile(np.abs(real_init - sim_init), 98),
                np.percentile(np.abs(real_opt  - sim_opt),  98), 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row_i, (r, s, znssd_v, row_lbl) in enumerate([
        (real_init, sim_init, znssd_init,
         f"Initial (shared)  hp={init_hp:.1f}  lp={init_lp:.2f}  γ={init_gamma:.3f}"),
        (real_opt, sim_opt, znssd_opt,
         f"Optimised — real(hp={opt_hp_r:.1f} lp={opt_lp_r:.2f} γ={opt_g_r:.3f})  "
         f"sim(hp={opt_hp_s:.1f} lp={opt_lp_s:.2f} γ={opt_g_s:.3f})"),
    ]):
        vlo, vhi = min(r.min(), s.min()), max(r.max(), s.max())
        diff = r - s
        axes[row_i, 0].imshow(r,    cmap="gray",  vmin=vlo,    vmax=vhi)
        axes[row_i, 0].set_title("Real",                     fontsize=11); axes[row_i, 0].axis("off")
        axes[row_i, 0].set_ylabel(row_lbl, fontsize=8)
        axes[row_i, 1].imshow(s,    cmap="gray",  vmin=vlo,    vmax=vhi)
        axes[row_i, 1].set_title(f"Simulated  ZNSSD={znssd_v:.4f}", fontsize=11); axes[row_i, 1].axis("off")
        axes[row_i, 2].imshow(diff, cmap="RdBu",  vmin=-_vabs, vmax=_vabs)
        axes[row_i, 2].set_title("Real − Simulated",          fontsize=11); axes[row_i, 2].axis("off")
    fig.suptitle(f"Preprocessing optimisation (independent) — pattern comparison\n"
                 f"ZNSSD: {znssd_init:.4f}  →  {znssd_opt:.4f}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _p1 = os.path.join(save_dir, "preprocopt_indep_1_patterns.png")
    plt.savefig(_p1, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved {_p1}")

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 2 — 1D sensitivity sweeps (2 rows × 3 cols: real / sim)
    # ─────────────────────────────────────────────────────────────────────────
    n_sw = 25
    print("  Computing 1D sweeps …")
    sweep_cfg = [
        # (label, x_arr, znssd_list_fn, xopt)
        ("real hp",    np.linspace(*real_high_pass_bounds, n_sw), opt_hp_r,
         lambda xs: [_znssd(_proc_real(real_pat_raw, h,       opt_lp_r, opt_g_r),
                          _proc_sim( sim_pat_raw,  opt_hp_s, opt_lp_s, opt_g_s)) for h in xs]),
        ("real lp",    np.linspace(*real_low_pass_bounds,  n_sw), opt_lp_r,
         lambda xs: [_znssd(_proc_real(real_pat_raw, opt_hp_r, l,       opt_g_r),
                          _proc_sim( sim_pat_raw,  opt_hp_s, opt_lp_s, opt_g_s)) for l in xs]),
        ("real gamma", np.linspace(*real_gamma_bounds,     n_sw), opt_g_r,
         lambda xs: [_znssd(_proc_real(real_pat_raw, opt_hp_r, opt_lp_r, g),
                          _proc_sim( sim_pat_raw,  opt_hp_s, opt_lp_s, opt_g_s)) for g in xs]),
        ("sim hp",     np.linspace(*sim_high_pass_bounds,  n_sw), opt_hp_s,
         lambda xs: [_znssd(_proc_real(real_pat_raw, opt_hp_r, opt_lp_r, opt_g_r),
                          _proc_sim( sim_pat_raw,  h,        opt_lp_s, opt_g_s)) for h in xs]),
        ("sim lp",     np.linspace(*sim_low_pass_bounds,   n_sw), opt_lp_s,
         lambda xs: [_znssd(_proc_real(real_pat_raw, opt_hp_r, opt_lp_r, opt_g_r),
                          _proc_sim( sim_pat_raw,  opt_hp_s, l,        opt_g_s)) for l in xs]),
        ("sim gamma",  np.linspace(*sim_gamma_bounds,      n_sw), opt_g_s,
         lambda xs: [_znssd(_proc_real(real_pat_raw, opt_hp_r, opt_lp_r, opt_g_r),
                          _proc_sim( sim_pat_raw,  opt_hp_s, opt_lp_s, g))       for g in xs]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for idx, (lbl, xs, xopt, znssd_fn) in enumerate(sweep_cfg):
        ax  = axes[idx // 3, idx % 3]
        ys  = znssd_fn(xs)
        ax.plot(xs, ys, "o-", color="steelblue" if idx < 3 else "tomato", ms=4)
        ax.axvline(xopt, color="black", ls="--", lw=1.5, label=f"opt = {xopt:.3f}")
        ax.set_xlabel(lbl); ax.set_ylabel("ZNSSD"); ax.legend(fontsize=9)
        ax.set_title(f"ZNSSD vs {lbl}\n(all others fixed at optimal)", fontsize=10)
    axes[0, 0].set_title("ZNSSD vs real hp\n(all others fixed)", fontsize=10)
    fig.suptitle("Preprocessing optimisation (independent): 1D ZNSSD sensitivity\n"
                 "Top row = real params  |  Bottom row = sim params",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _p2 = os.path.join(save_dir, "preprocopt_indep_2_sweeps.png")
    plt.savefig(_p2, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved {_p2}")

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 3 — ZNSSD landscapes: (hp_real × hp_sim) and (gamma_real × gamma_sim)
    # ─────────────────────────────────────────────────────────────────────────
    n_land = 15
    hp_r_land  = np.linspace(*real_high_pass_bounds, n_land)
    hp_s_land  = np.linspace(*sim_high_pass_bounds,  n_land)
    g_r_land   = np.linspace(*real_gamma_bounds,     n_land)
    g_s_land   = np.linspace(*sim_gamma_bounds,      n_land)
    print("  Computing ZNSSD landscapes (hp × hp,  gamma × gamma) …")

    znssd_hp_land = np.zeros((n_land, n_land))
    for i, hr in enumerate(hp_r_land):
        for j, hs in enumerate(hp_s_land):
            znssd_hp_land[i, j] = _znssd(
                _proc_real(real_pat_raw, hr,      opt_lp_r, opt_g_r),
                _proc_sim( sim_pat_raw,  hs,      opt_lp_s, opt_g_s))

    znssd_g_land = np.zeros((n_land, n_land))
    for i, gr in enumerate(g_r_land):
        for j, gs in enumerate(g_s_land):
            znssd_g_land[i, j] = _znssd(
                _proc_real(real_pat_raw, opt_hp_r, opt_lp_r, gr),
                _proc_sim( sim_pat_raw,  opt_hp_s, opt_lp_s, gs))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, land, xarr, yarr, xlabel, ylabel, xopt, yopt, title in [
        (axes[0], znssd_hp_land, hp_s_land, hp_r_land,
         "sim high_pass_sigma", "real high_pass_sigma",
         opt_hp_s, opt_hp_r, f"ZNSSD: hp_real × hp_sim\n(lp, gamma fixed at optimal)"),
        (axes[1], znssd_g_land,  g_s_land,  g_r_land,
         "sim gamma",           "real gamma",
         opt_g_s, opt_g_r, f"ZNSSD: gamma_real × gamma_sim\n(hp, lp fixed at optimal)"),
    ]:
        im = ax.pcolormesh(xarr, yarr, land, cmap="viridis_r", shading="auto")
        plt.colorbar(im, ax=ax, label="ZNSSD")
        ax.plot(xopt, yopt, "r*", ms=14, label=f"optimal ({xopt:.2f}, {yopt:.2f})")
        ax.plot(init_hp if "hp" in xlabel else init_gamma,
                init_hp if "hp" in ylabel else init_gamma,
                "w^", ms=10, mec="gray", label=f"initial")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10); ax.legend(fontsize=9)
    fig.suptitle("Preprocessing optimisation (independent): ZNSSD landscapes",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _p3 = os.path.join(save_dir, "preprocopt_indep_3_landscapes.png")
    plt.savefig(_p3, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved {_p3}")

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 4 — intensity line scans
    # ─────────────────────────────────────────────────────────────────────────
    H, W  = real_opt.shape
    mid_r = H // 2
    mid_c = W // 2

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    slices_cfg = [
        ("Horizontal", np.arange(W), "Column (px)",
         real_init[mid_r, :], sim_init[mid_r, :],
         real_opt[mid_r,  :], sim_opt[mid_r,  :]),
        ("Vertical",   np.arange(H), "Row (px)",
         real_init[:, mid_c], sim_init[:, mid_c],
         real_opt[:,  mid_c], sim_opt[:,  mid_c]),
    ]
    for col, (direction, x_coord, xlabel, ri, si, ro, so) in enumerate(slices_cfg):
        for row, (r_sl, s_sl, znssd_v, suffix) in enumerate([
            (ri, si, znssd_init, f"initial (shared)  ZNSSD={znssd_init:.4f}"),
            (ro, so, znssd_opt,  f"optimised (independent)  ZNSSD={znssd_opt:.4f}"),
        ]):
            ax = axes[row, col]
            ax.plot(x_coord, r_sl, color="steelblue", label="Real",      lw=1.2)
            ax.plot(x_coord, s_sl, color="tomato",    label="Simulated", lw=1.2, ls="--")
            ax.set_title(f"{direction} slice — {suffix}", fontsize=10)
            ax.set_xlabel(xlabel); ax.set_ylabel("Intensity (z-score)")
            ax.legend(fontsize=9)
    fig.suptitle("Preprocessing optimisation (independent): intensity line scans\n"
                 f"horizontal = row {mid_r},  vertical = col {mid_c}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _p4 = os.path.join(save_dir, "preprocopt_indep_4_linescans.png")
    plt.savefig(_p4, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved {_p4}\n")

    return {
        "real_high_pass_sigma": opt_hp_r,
        "real_low_pass_sigma":  opt_lp_r,
        "real_gamma":           opt_g_r,
        "sim_high_pass_sigma":  opt_hp_s,
        "sim_low_pass_sigma":   opt_lp_s,
        "sim_gamma":            opt_g_s,
        "znssd_init":             znssd_init,
        "znssd_opt":              znssd_opt,
        "real_init":            real_init,
        "sim_init":             sim_init,
        "real_opt":             real_opt,
        "sim_opt":              sim_opt,
    }


def optimize_pc_and_euler(
    pat_obj,
    x0: int,
    master_pattern_path: str,
    euler_angles_init: np.ndarray,
    pc_init: tuple,
    tilt_deg: float = 70.0,
    sample_tilt_deg: float = None,
    detector_tilt_deg: float = None,
    max_iter: int = 300,
    euler_step_deg: float = 0.5,
    pc_step: float = 0.005,
    sim_high_pass_sigma: float = None,
    sim_low_pass_sigma: float = None,
    sim_gamma: float = None,
    save_dir: str = "debug",
) -> tuple:
    """Refine PC and Euler angles by minimising ZNSSD.

    The simulated pattern is processed without mask (detector artefacts in the
    real pattern have no equivalent in the simulation) and optionally with a
    higher high-pass sigma to remove excess low-frequency content that
    simulations typically produce relative to experiment.

    The optimization perturbs a 6-element delta vector
    ``[dφ1, dΦ, dφ2, dxstar, dystar, dzstar]`` (radians / normalised detector
    units) from zero using the Nelder-Mead simplex method.

    Parameters
    ----------
    pat_obj :
        ``Data.UP2`` instance — provides the experimental pattern and the
        preprocessing pipeline used throughout the rest of the pipeline.
    x0 : int
        Flat pattern index of the reference pattern within the scan.
    master_pattern_path : str
        Path to the ``.h5`` master pattern file.
    euler_angles_init : array_like, shape (3,)
        Starting Bunge ZXZ Euler angles in **radians**.
    pc_init : tuple
        Starting pattern center ``(xstar, ystar, zstar)`` in EDAX/TSL
        convention.
    tilt_deg : float
        Legacy single-tilt parameter (default 70). Ignored when both
        sample_tilt_deg and detector_tilt_deg are provided.
    sample_tilt_deg : float, optional
        Sample tilt in degrees — Step 2 in GUI. If provided together with
        detector_tilt_deg, these override tilt_deg.
    detector_tilt_deg : float, optional
        Detector tilt in degrees — Step 2 in GUI.
    max_iter : int
        Maximum number of Nelder-Mead function evaluations (default 300).
    euler_step_deg : float
        Initial simplex step for Euler angles in degrees (default 0.5).
    pc_step : float
        Initial simplex step for each PC component (default 0.005).
    sim_high_pass_sigma : float, optional
        Override high_pass_sigma for the simulated pattern only.
        Set higher than the experimental value (e.g. 25–30) to remove the
        excess low-frequency content simulated patterns typically have.
        If None, uses pat_obj.high_pass_sigma unchanged.
    sim_low_pass_sigma : float, optional
        Override low_pass_sigma for the simulated pattern only.
        If None, uses pat_obj.low_pass_sigma unchanged.
    sim_gamma : float, optional
        Override gamma for the simulated pattern only.
        If None, uses pat_obj.gamma unchanged.
    save_dir : str
        Directory for debug output figures.

    Returns
    -------
    euler_opt : np.ndarray, shape (3,)
        Optimized Euler angles in radians.
    pc_opt : tuple
        Optimized pattern center ``(xstar, ystar, zstar)``.
    """
    os.makedirs(save_dir, exist_ok=True)

    patshape   = pat_obj.patshape
    euler_init = np.asarray(euler_angles_init, dtype=np.float64)
    pc_arr     = np.asarray(pc_init,           dtype=np.float64)

    print(f"[PC/Euler refinement] pattern index : {x0}")
    print(f"[PC/Euler refinement] Euler init (deg): "
          f"phi1={np.degrees(euler_init[0]):.3f}  "
          f"Phi={np.degrees(euler_init[1]):.3f}  "
          f"phi2={np.degrees(euler_init[2]):.3f}")
    print(f"[PC/Euler refinement] PC init (EDAX):   "
          f"x*={pc_arr[0]:.5f}  y*={pc_arr[1]:.5f}  z*={pc_arr[2]:.5f}")

    # ------------------------------------------------------------------
    # Experimental reference pattern — processed through the full pipeline
    # ------------------------------------------------------------------
    exp_pat = pat_obj.read_pattern(x0, process=True)
    print(f"[PC/Euler refinement] exp_pat  min={exp_pat.min():.4f}  "
          f"max={exp_pat.max():.4f}  mean={exp_pat.mean():.4f}  "
          f"std={exp_pat.std():.4f}")

    # ------------------------------------------------------------------
    # Load master pattern once; reuse the simulator object across calls
    # ------------------------------------------------------------------
    _sample_tilt   = sample_tilt_deg   if sample_tilt_deg   is not None else tilt_deg
    _detector_tilt = detector_tilt_deg if detector_tilt_deg is not None else 0.0

    sim = patternSimulation()
    sim.detector_height   = patshape[0]
    sim.detector_width    = patshape[1]
    sim.det_shape         = patshape
    sim.sample_tilt_deg   = _sample_tilt
    sim.detector_tilt_deg = _detector_tilt
    print(f"[PC/Euler refinement] sample_tilt={_sample_tilt}°  "
          f"detector_tilt={_detector_tilt}°  "
          f"primary_tilt_arg={-(_sample_tilt - _detector_tilt):.1f}°")
    sim.mastersetup(master_pattern_path)

    # ------------------------------------------------------------------
    # Baseline ZNSSD (initial params)
    # ------------------------------------------------------------------
    sim_init = _simulate(sim, euler_init, pc_arr, pat_obj, patshape,
                             sim_high_pass_sigma, sim_low_pass_sigma, sim_gamma)

    #save the initial simulated pattern for comparison
    out_path_init = os.path.join(save_dir, "pc_euler_refinement_sim_initial.png")
    plt.imsave(out_path_init, sim_init, cmap="gray")
    print(f"  Saved {out_path_init}")

    #save the initial reference pattern for comparison
    out_path_exp = os.path.join(save_dir, "pc_euler_refinement_exp_pattern.png")
    plt.imsave(out_path_exp, exp_pat, cmap="gray")
    print(f"  Saved {out_path_exp}")


    print(f"[PC/Euler refinement] sim_init min={sim_init.min():.4f}  "
          f"max={sim_init.max():.4f}  mean={sim_init.mean():.4f}  "
          f"std={sim_init.std():.4f}")
    znssd_init = _znssd(exp_pat, sim_init)
    print(f"\n[PC/Euler refinement] Initial ZNSSD: {znssd_init:.6f}")

    # ------------------------------------------------------------------
    # Objective — rotation perturbed in so(3) (rotation-vector / tangent
    # space) to avoid Bunge gimbal lock near Φ = 0 or Φ = π.
    # ------------------------------------------------------------------
    # Anchor quaternion: composing onto this gives the current orientation.
    q_init = _bu2qu_emsoft(euler_init)

    nfev       = [0]
    best_znssd = [znssd_init]
    best_state = [{"euler": euler_init.copy(),
                   "q":     q_init.copy(),
                   "pc":    pc_arr.copy(),
                   "sim":   sim_init.copy()}]

    def objective(delta: np.ndarray) -> float:
        # delta[:3] is a rotation vector in so(3); delta[3:] is the PC offset.
        # exp(δ_rotvec) gives a quaternion that's near-identity for small δ,
        # so the simplex steps in a smooth, singularity-free space.
        q_delta   = _rotvec_to_qu(delta[:3])
        q_current = _qu_mul(q_init, q_delta)
        # Normalise + canonical sign (w ≥ 0)
        q_current = q_current / (np.linalg.norm(q_current) + 1e-30)
        if q_current[0] < 0.0:
            q_current = -q_current

        # Convert back to Euler only as a transient step so _simulate's
        # existing API works; bu2qu_emsoft inside EandPCSet will round-trip
        # this back to q_current exactly (even at Φ = 0 — the (φ₁, φ₂)
        # split is arbitrary there, but the underlying quaternion is unique).
        euler = _qu2bu_emsoft(q_current)
        pc    = pc_arr + delta[3:]

        sim_pat = _simulate(sim, euler, pc, pat_obj, patshape,
                            sim_high_pass_sigma, sim_low_pass_sigma, sim_gamma)
        znssd_val = _znssd(exp_pat, sim_pat)
        nfev[0] += 1

        if znssd_val < best_znssd[0]:
            best_znssd[0] = znssd_val
            # Wrap φ₁/φ₂ to the same 2π branch as euler_init so the stored
            # representation can't accidentally look like a 360° "overshoot".
            euler_branched = _wrap_euler_to_branch(euler, euler_init)
            best_state[0] = {"euler": euler_branched,
                             "q":     q_current.copy(),
                             "pc":    pc.copy(),
                             "sim":   sim_pat.copy()}

        if nfev[0] % 20 == 0:
            print(f"  [iter {nfev[0]:4d}]  ZNSSD = {znssd_val:.6f}  "
                  f"best = {best_znssd[0]:.6f}")
        return znssd_val   # ZNSSD is a loss — minimise directly

    # ------------------------------------------------------------------
    # Nelder-Mead with physically tuned initial simplex
    # ------------------------------------------------------------------
    x0_delta = np.zeros(6)
    minimize(
        objective,
        x0_delta,
        method="Nelder-Mead",
        options={
            "maxiter":         max_iter,
            "maxfev":          max_iter,
            "xatol":           1e-5,
            "fatol":           1e-6,
            "adaptive":        True,
            "initial_simplex": _build_initial_simplex(
                x0_delta, euler_step_deg, pc_step
            ),
        },
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    euler_opt = best_state[0]["euler"]
    q_opt     = best_state[0]["q"]
    pc_opt    = tuple(best_state[0]["pc"])
    sim_opt   = best_state[0]["sim"]
    znssd_opt = best_znssd[0]

    # Misorientation angle between q_init and q_opt — gimbal-lock-free
    # equivalent of "how big was the rotational change?".
    q_rel       = _qu_mul(np.array([q_init[0], -q_init[1], -q_init[2], -q_init[3]]), q_opt)
    w_clamped   = float(np.clip(abs(q_rel[0]), -1.0, 1.0))
    misori_deg  = float(np.degrees(2.0 * np.arccos(w_clamped)))

    print(f"\n--- PC / Euler refinement summary ---")
    print(f"  Function evals:     {nfev[0]}")
    print(f"  ZNSSD (before):       {znssd_init:.6f}")
    print(f"  ZNSSD (after):        {znssd_opt:.6f}")
    print(f"  Misorientation:     {misori_deg:.4f}°   (gimbal-lock-free, q_init → q_opt)")
    print(f"  Δ Euler (deg):      {np.degrees(euler_opt - euler_init)}")
    print(f"  Euler opt (rad):    {euler_opt}")
    print(f"  ΔPC:                {np.array(pc_opt) - pc_arr}")
    print(f"  PC opt:             {pc_opt}")

    # ------------------------------------------------------------------
    # Save comparison figures
    # ------------------------------------------------------------------
    diff = exp_pat - sim_opt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(exp_pat, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Experimental", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(sim_opt, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Simulated (optimized)\nZNSSD = {znssd_opt:.4f}", fontsize=12)
    axes[1].axis("off")

    vabs = np.percentile(np.abs(diff), 98)
    axes[2].imshow(diff, cmap="RdBu", vmin=-vabs, vmax=vabs)
    axes[2].set_title("Difference (exp − sim)", fontsize=12)
    axes[2].axis("off")

    fig.suptitle(
        f"PC / Euler refinement  |  ZNSSD: {znssd_init:.4f} → {znssd_opt:.4f}",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(save_dir, "pc_euler_refinement.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
    axes2[0].imshow(sim_init, cmap="gray", vmin=0, vmax=1)
    axes2[0].set_title(f"Simulated (initial)\nZNSSD = {znssd_init:.4f}", fontsize=12)
    axes2[0].axis("off")
    axes2[1].imshow(sim_opt, cmap="gray", vmin=0, vmax=1)
    axes2[1].set_title(f"Simulated (optimized)\nZNSSD = {znssd_opt:.4f}", fontsize=12)
    axes2[1].axis("off")
    fig2.suptitle("Simulated pattern: before vs after refinement",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path2 = os.path.join(save_dir, "pc_euler_refinement_sim_comparison.png")
    plt.savefig(out_path2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path2}\n")

    return euler_opt, pc_opt


# ---------------------------------------------------------------------------
# H-value landscape
# ---------------------------------------------------------------------------

def compute_h_landscape(
    pat_obj,
    ang_data,
    ref_idx_yx: tuple,
    euler_ref: np.ndarray,
    pc_ref: tuple,
    master_pattern_path: str,
    sample_tilt_deg: float,
    detector_tilt_deg: float,
    hp_bounds: tuple = (3.0, 80.0),
    gamma_bounds: tuple = (0.1, 1.5),
    n_grid: int = 8,
    crop_fraction: float = 0.9,
    max_iter_icgn: int = 30,
    mark_hp: float = None,
    mark_gamma: float = None,
    save_dir: str = "debug",
) -> dict:
    """Run IC-GN between the single reference pattern and its simulation across
    a (high_pass_sigma, gamma) grid.

    The reference real pattern (at *ref_idx_yx*) is compared against the
    simulated pattern for each (hp, gamma) combination.  The 8 h-components
    and the residual are recorded and plotted as 2D intensity maps so you can
    see how preprocessing choices affect the recovered homography values.

    The same hp and gamma are applied to both the real and simulated pattern.
    No ROI selection is needed — only the single reference pattern is used.

    Parameters
    ----------
    pat_obj             : Data.UP2 — pattern file + preprocessing settings
    ang_data            : ang data object with .shape attribute (n_rows, n_cols)
    ref_idx_yx          : (row, col) position of the reference in the scan grid
    euler_ref           : Bunge Euler angles for the reference (radians)
    pc_ref              : pattern center in EDAX/TSL convention
    master_pattern_path : path to the .h5 master pattern
    sample_tilt_deg     : sample tilt in degrees
    detector_tilt_deg   : detector tilt in degrees
    hp_bounds           : (min, max) for high_pass_sigma search
    gamma_bounds        : (min, max) for gamma search
    n_grid              : number of grid points along each axis
    crop_fraction       : fraction of detector used by IC-GN
    max_iter_icgn       : IC-GN max iterations per grid point
    mark_hp             : hp value to mark with a red star in the plots
    mark_gamma          : gamma value to mark with a red star in the plots
    save_dir            : directory for the output figure

    Returns
    -------
    dict with keys: h_grid (n_grid, n_grid, 8), resid_grid (n_grid, n_grid),
                    iter_grid (n_grid, n_grid), hp_arr (n_grid,), gamma_arr (n_grid,)
    """
    import get_homography_cpu as _core
    from get_homography_cpu import InitType

    x0_flat  = int(np.ravel_multi_index(ref_idx_yx, ang_data.shape))
    ref_row, ref_col = int(ref_idx_yx[0]), int(ref_idx_yx[1])
    # 1×1 ROI — IC-GN on the single reference pattern vs its simulation
    ref_roi_slice = (slice(ref_row, ref_row + 1), slice(ref_col, ref_col + 1))

    hp_arr    = np.linspace(hp_bounds[0], hp_bounds[1], n_grid)
    gamma_arr = np.linspace(gamma_bounds[0], gamma_bounds[1], n_grid)

    h_grid     = np.full((n_grid, n_grid, 8), np.nan)
    resid_grid = np.full((n_grid, n_grid), np.nan)
    iter_grid  = np.full((n_grid, n_grid), np.nan)

    orig_hp = pat_obj.high_pass_sigma
    orig_g  = pat_obj.gamma

    n_total = n_grid * n_grid
    print(f"\n[compute_h_landscape] {n_grid}×{n_grid} grid  "
          f"hp {hp_bounds[0]}–{hp_bounds[1]}  gamma {gamma_bounds[0]}–{gamma_bounds[1]}")
    print(f"  Reference pattern: row={ref_row}, col={ref_col}  (flat idx {x0_flat})")

    try:
        for i, hp in enumerate(hp_arr):
            for j, gamma in enumerate(gamma_arr):
                pat_obj.high_pass_sigma = hp
                pat_obj.gamma = gamma
                try:
                    result = _core.optimize(
                        pats=pat_obj,
                        x0=x0_flat,
                        init_type=InitType.NONE,
                        crop_fraction=crop_fraction,
                        max_iter=max_iter_icgn,
                        conv_tol=1e-3,
                        n_jobs=1,
                        verbose=False,
                        roi_slice=ref_roi_slice,
                        scan_shape=ang_data.shape,
                        mask=None,
                        use_simulated_reference=True,
                        master_pattern_path=master_pattern_path,
                        euler_angles_ref=euler_ref,
                        pc_ref=pc_ref,
                        tilt_deg=sample_tilt_deg,
                        detector_tilt_deg=detector_tilt_deg,
                        sim_high_pass_sigma=None,
                    )
                except Exception as exc:
                    print(f"  [i={i}, j={j}] hp={hp:.1f}, gamma={gamma:.3f}: ERROR — {exc}")
                    continue

                h_vals, iters, resids, _ = result
                # h_vals.shape = (1, 1, 8) — single pattern
                h_grid[i, j]     = h_vals.ravel()
                resid_grid[i, j] = float(resids.ravel()[0])
                iter_grid[i, j]  = float(iters.ravel()[0])
                frac = (i * n_grid + j + 1) / n_total
                print(f"  [{frac*100:3.0f}%] hp={hp:6.1f}  gamma={gamma:.3f}  "
                      f"resid={resid_grid[i, j]:.5f}")
    finally:
        pat_obj.high_pass_sigma = orig_hp
        pat_obj.gamma = orig_g

    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Plot 3×3 heatmaps: 8 h components + residual
    # x-axis = gamma, y-axis = high_pass_sigma
    # ------------------------------------------------------------------
    h_labels = ["h₁₁", "h₁₂", "h₁₃", "h₂₁", "h₂₂", "h₂₃", "h₃₁", "h₃₂"]
    extent = [gamma_bounds[0], gamma_bounds[1], hp_bounds[0], hp_bounds[1]]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes_flat = axes.ravel()

    for k in range(8):
        ax = axes_flat[k]
        data = h_grid[:, :, k]
        vmax = np.nanpercentile(np.abs(data), 98)
        if vmax == 0:
            vmax = 1e-9
        im = ax.imshow(
            data, origin="lower", extent=extent, aspect="auto",
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        )
        ax.set_title(h_labels[k], fontsize=12, fontweight="bold")
        ax.set_xlabel("gamma")
        ax.set_ylabel("high-pass σ")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if mark_hp is not None and mark_gamma is not None:
            ax.plot(mark_gamma, mark_hp, "r*", markersize=10, zorder=5)

    ax = axes_flat[8]
    im = ax.imshow(
        resid_grid, origin="lower", extent=extent, aspect="auto",
        cmap="viridis_r",
    )
    ax.set_title("residual", fontsize=12, fontweight="bold")
    ax.set_xlabel("gamma")
    ax.set_ylabel("high-pass σ")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if mark_hp is not None and mark_gamma is not None:
        ax.plot(mark_gamma, mark_hp, "r*", markersize=10, zorder=5,
                label=f"current\nhp={mark_hp:.1f}\nγ={mark_gamma:.3f}")
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "IC-GN h-value landscape  ·  real vs simulated reference pattern\n"
        f"hp {hp_bounds[0]}–{hp_bounds[1]}  ·  gamma {gamma_bounds[0]}–{gamma_bounds[1]}"
        f"  ·  {n_grid}×{n_grid} grid  ·  max_iter={max_iter_icgn}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out_path = os.path.join(save_dir, "h_landscape.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")

    return {
        "h_grid":     h_grid,
        "resid_grid": resid_grid,
        "iter_grid":  iter_grid,
        "hp_arr":     hp_arr,
        "gamma_arr":  gamma_arr,
    }

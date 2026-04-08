"""
Optimize pattern center (PC) and Euler angles so that the simulated reference
pattern best matches the experimental one, measured via normalized cross-
correlation (NCC).

Preprocessing (bandpass filter, CLAHE, etc.) is applied with the same mask
settings as the rest of the pipeline.  The NCC is then computed over all
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
# Internal helpers
# ---------------------------------------------------------------------------

def _simulate(sim: patternSimulation, euler: np.ndarray, pc: np.ndarray,
              pat_obj, patshape: tuple,
              high_pass_sigma_override: float = None) -> np.ndarray:
    """Generate one simulated pattern, processed without mask.

    The mask is skipped because masked pixels in the real pattern are detector
    artefacts (direct beam, shadow) with no physical equivalent in the
    simulation.  Zero-filling them creates a boundary halo that biases the NCC.
    """
    import torch

    # EDAX/TSL -> Bruker: flip y component
    pc_bruker = (float(pc[0]), 1.0 - float(pc[1]), float(pc[2]))
    sim.EandPCSet(euler, pc_bruker, verbose=False)

    with torch.no_grad():
        pats = sim.GenPattern()

    pat = pats[0].reshape(patshape).cpu().numpy().astype(np.float32)
    lo, hi = pat.min(), pat.max()
    if hi > lo:
        pat = (pat - lo) / (hi - lo)

    if pat_obj is not None:
        orig_mask_type = pat_obj.mask_type
        orig_hp        = pat_obj.high_pass_sigma
        pat_obj.mask_type = None
        if high_pass_sigma_override is not None:
            pat_obj.high_pass_sigma = high_pass_sigma_override
        try:
            pat = pat_obj.process_pattern(pat)
        finally:
            pat_obj.mask_type      = orig_mask_type
            pat_obj.high_pass_sigma = orig_hp

    return pat


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation, result in [-1, 1]."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _build_initial_simplex(x0: np.ndarray,
                            euler_step_deg: float = 0.5,
                            pc_step: float = 0.005) -> np.ndarray:
    """
    Build an (n+1 x n) initial simplex with physically meaningful step sizes.
    First 3 params are Euler angle deltas (radians), last 3 are PC deltas.
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

def optimize_pc_and_euler(
    pat_obj,
    x0: int,
    master_pattern_path: str,
    euler_angles_init: np.ndarray,
    pc_init: tuple,
    tilt_deg: float = 70.0,
    max_iter: int = 300,
    euler_step_deg: float = 0.5,
    pc_step: float = 0.005,
    sim_high_pass_sigma: float = None,
    save_dir: str = "debug",
) -> tuple:
    """Refine PC and Euler angles by maximising NCC.

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
        Primary sample tilt in degrees (default 70).
    max_iter : int
        Maximum number of Nelder-Mead function evaluations (default 300).
    euler_step_deg : float
        Initial simplex step for Euler angles in degrees (default 0.5).
    pc_step : float
        Initial simplex step for each PC component (default 0.005).
    sim_high_pass_sigma : float, optional
        Override high_pass_sigma when processing the simulated pattern.
        Set higher than the experimental value (e.g. 25–30) to remove the
        excess low-frequency content simulated patterns typically have.
        If None, uses pat_obj.high_pass_sigma unchanged.
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

    # ------------------------------------------------------------------
    # Experimental reference pattern — processed through the full pipeline
    # ------------------------------------------------------------------
    exp_pat = pat_obj.read_pattern(x0, process=True)

    # ------------------------------------------------------------------
    # Load master pattern once; reuse the simulator object across calls
    # ------------------------------------------------------------------
    sim = patternSimulation()
    sim.detector_height   = patshape[0]
    sim.detector_width    = patshape[1]
    sim.det_shape         = patshape
    sim.detector_tilt_deg = tilt_deg
    sim.mastersetup(master_pattern_path)

    # ------------------------------------------------------------------
    # Baseline NCC (initial params)
    # ------------------------------------------------------------------
    sim_init = _simulate(sim, euler_init, pc_arr, pat_obj, patshape, sim_high_pass_sigma)
    ncc_init = _ncc(exp_pat, sim_init)
    print(f"\n[PC/Euler refinement] Initial NCC: {ncc_init:.6f}")

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    nfev       = [0]
    best_ncc   = [ncc_init]
    best_state = [{"euler": euler_init.copy(),
                   "pc":    pc_arr.copy(),
                   "sim":   sim_init.copy()}]

    def objective(delta: np.ndarray) -> float:
        euler   = euler_init + delta[:3]
        pc      = pc_arr     + delta[3:]
        sim_pat = _simulate(sim, euler, pc, pat_obj, patshape, sim_high_pass_sigma)
        ncc_val = _ncc(exp_pat, sim_pat)
        nfev[0] += 1

        if ncc_val > best_ncc[0]:
            best_ncc[0] = ncc_val
            best_state[0] = {"euler": euler.copy(),
                             "pc":    pc.copy(),
                             "sim":   sim_pat.copy()}

        if nfev[0] % 20 == 0:
            print(f"  [iter {nfev[0]:4d}]  NCC = {ncc_val:.6f}  "
                  f"best = {best_ncc[0]:.6f}")
        return -ncc_val   # minimise → maximise NCC

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
    pc_opt    = tuple(best_state[0]["pc"])
    sim_opt   = best_state[0]["sim"]
    ncc_opt   = best_ncc[0]

    print(f"\n--- PC / Euler refinement summary ---")
    print(f"  Function evals:     {nfev[0]}")
    print(f"  NCC (before):       {ncc_init:.6f}")
    print(f"  NCC (after):        {ncc_opt:.6f}")
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
    axes[1].set_title(f"Simulated (optimized)\nNCC = {ncc_opt:.4f}", fontsize=12)
    axes[1].axis("off")

    vabs = np.percentile(np.abs(diff), 98)
    axes[2].imshow(diff, cmap="RdBu", vmin=-vabs, vmax=vabs)
    axes[2].set_title("Difference (exp − sim)", fontsize=12)
    axes[2].axis("off")

    fig.suptitle(
        f"PC / Euler refinement  |  NCC: {ncc_init:.4f} → {ncc_opt:.4f}",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(save_dir, "pc_euler_refinement.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
    axes2[0].imshow(sim_init, cmap="gray", vmin=0, vmax=1)
    axes2[0].set_title(f"Simulated (initial)\nNCC = {ncc_init:.4f}", fontsize=12)
    axes2[0].axis("off")
    axes2[1].imshow(sim_opt, cmap="gray", vmin=0, vmax=1)
    axes2[1].set_title(f"Simulated (optimized)\nNCC = {ncc_opt:.4f}", fontsize=12)
    axes2[1].axis("off")
    fig2.suptitle("Simulated pattern: before vs after refinement",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path2 = os.path.join(save_dir, "pc_euler_refinement_sim_comparison.png")
    plt.savefig(out_path2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path2}\n")

    return euler_opt, pc_opt

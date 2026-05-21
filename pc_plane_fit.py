"""
pc_plane_fit.py — Sparse-grid PC refinement + plane fit.

Workflow:
  1. Pick a uniform N×N grid of test points across the scan.
  2. Load the master pattern ONCE (shared simulator across test points).
  3. For each test point:
     a. Predict the PC from the existing geometric model (scan-grid-to-PC),
        anchored at the user-supplied reference PC.
     b. Read the local orientation from the .ang file — FROZEN.
     c. Refine PC only (3-D Nelder-Mead) — orientation is held fixed at
        the .ang value, PC is perturbed around the geometric prediction.
     d. Capture the post-refinement ZNSSD as a per-point quality metric.
  4. Least-squares fit a plane (or bilinear surface) to each of pcx, pcy,
     pcz across the test points.  Optionally weight by 1/(ZNSSD + ε) so
     poorly-converged points contribute less.
  5. Build a diagnostic figure showing the refined PCs vs the fitted
     surface, plus a ZNSSD heatmap so the user can sanity-check the fit.

The result is a small dict of plane coefficients that downstream code can
evaluate at any (row, col) to get a per-pattern PC.  Pipeline wiring is
deliberately left out of this module — call pc_plane_fit_result["plane"]
directly when you're ready to use it.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Plane fitting
# ---------------------------------------------------------------------------

def _design_matrix(positions: np.ndarray, order: str) -> np.ndarray:
    """Build the LSQ design matrix for the chosen plane order.

    positions: (n, 2) — each row is (scan_row, scan_col) of a test point.
    order: 'linear' (3 coeffs/axis) or 'bilinear' (4 coeffs/axis).
    """
    r = positions[:, 0].astype(np.float64)
    c = positions[:, 1].astype(np.float64)
    if order == "linear":
        return np.column_stack([np.ones_like(r), r, c])
    if order == "bilinear":
        return np.column_stack([np.ones_like(r), r, c, r * c])
    raise ValueError(f"unknown plane order: {order!r} (expected 'linear' or 'bilinear')")


def _fit_plane(positions: np.ndarray,
               refined_pcs: np.ndarray,
               znssd: np.ndarray,
               order: str = "linear",
               weight_by_znssd: bool = True) -> dict:
    """Least-squares fit a plane per PC component.

    Returns dict with keys 'x', 'y', 'z' (each a 1-D ndarray of coefficients
    in the same order as the columns of the design matrix), plus 'order' and
    'rms_residual_um_equivalent' (only the dimensionless RMS for each axis).
    """
    A = _design_matrix(positions, order)

    if weight_by_znssd:
        # Soft inverse weighting: a point with ZNSSD = 0 gets weight ~1000;
        # a point with ZNSSD = 1 gets weight ~1.  Clamps the worst points
        # before they dominate.
        w = 1.0 / (np.asarray(znssd, dtype=np.float64) + 1e-3)
        w = np.minimum(w, 1000.0)
        Aw = A * w[:, None]
    else:
        w = np.ones(positions.shape[0])
        Aw = A

    plane: dict = {"order": order, "weights": w.copy()}
    rms = {}
    for i, axis in enumerate(("x", "y", "z")):
        y = refined_pcs[:, i].astype(np.float64)
        coeffs, *_ = np.linalg.lstsq(Aw, y * w if weight_by_znssd else y, rcond=None)
        plane[axis] = coeffs
        residual = y - A @ coeffs
        rms[axis] = float(np.sqrt(np.mean(residual ** 2)))
    plane["rms_residual"] = rms
    return plane


def evaluate_plane(plane: dict, row: int, col: int) -> np.ndarray:
    """Evaluate the fitted plane at a scan (row, col) and return (pcx, pcy, pcz)."""
    order = plane.get("order", "linear")
    if order == "linear":
        basis = np.array([1.0, float(row), float(col)])
    elif order == "bilinear":
        basis = np.array([1.0, float(row), float(col), float(row) * float(col)])
    else:
        raise ValueError(f"unknown plane order in plane dict: {order!r}")
    return np.array([
        float(np.dot(plane["x"], basis)),
        float(np.dot(plane["y"], basis)),
        float(np.dot(plane["z"], basis)),
    ])


def evaluate_plane_grid(plane: dict, scan_shape: tuple) -> np.ndarray:
    """Vectorised evaluation of the plane at every (row, col) in a scan.

    Returns:
        ndarray of shape (n_rows, n_cols, 3) — the (pcx, pcy, pcz) at every
        scan position.  Use this to feed the per-pattern PC drift correction
        in `pc_homography_correction.correct_homographies` via the
        `pc_grid_override` parameter.
    """
    n_r, n_c = int(scan_shape[0]), int(scan_shape[1])
    r_grid, c_grid = np.meshgrid(np.arange(n_r), np.arange(n_c), indexing="ij")
    r_flat = r_grid.ravel().astype(np.float64)
    c_flat = c_grid.ravel().astype(np.float64)

    order = plane.get("order", "linear")
    if order == "linear":
        basis = np.column_stack([np.ones_like(r_flat), r_flat, c_flat])
    elif order == "bilinear":
        basis = np.column_stack([np.ones_like(r_flat), r_flat, c_flat, r_flat * c_flat])
    else:
        raise ValueError(f"unknown plane order in plane dict: {order!r}")

    pc_grid = np.zeros((n_r * n_c, 3), dtype=np.float64)
    for i, axis in enumerate(("x", "y", "z")):
        pc_grid[:, i] = basis @ plane[axis]
    return pc_grid.reshape(n_r, n_c, 3)


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def _uniform_grid(scan_shape: tuple, n_grid: tuple,
                  edge_padding: int = 2) -> np.ndarray:
    """Pick (n_rows × n_cols) test points evenly across the scan, with
    `edge_padding` rows/cols of margin from each edge (so the corner test
    points are at `(edge_padding, edge_padding)` and
    `(rows - 1 - edge_padding, cols - 1 - edge_padding)`).

    Edge pixels often suffer from beam-edge artefacts, mask-area dropout,
    or low-signal patterns, so padding them out keeps the plane fit
    anchored on well-behaved interior points.

    If the scan is too small to accommodate the requested padding, the
    padding is silently clamped so a valid grid still gets produced.
    """
    n_r, n_c = int(n_grid[0]), int(n_grid[1])
    rows, cols = int(scan_shape[0]), int(scan_shape[1])
    if rows < 2 or cols < 2:
        raise ValueError(f"scan_shape {scan_shape} too small to grid over")

    # Clamp padding so we have at least 1 valid pixel to grid across.
    max_pad_r = (rows - 1) // 2
    max_pad_c = (cols - 1) // 2
    pad_r = max(0, min(int(edge_padding), max_pad_r))
    pad_c = max(0, min(int(edge_padding), max_pad_c))

    r_lo, r_hi = pad_r, rows - 1 - pad_r
    c_lo, c_hi = pad_c, cols - 1 - pad_c

    rs = np.round(np.linspace(r_lo, r_hi, n_r)).astype(int)
    cs = np.round(np.linspace(c_lo, c_hi, n_c)).astype(int)
    rr, cc = np.meshgrid(rs, cs, indexing="ij")
    return np.column_stack([rr.ravel(), cc.ravel()])


# ---------------------------------------------------------------------------
# Diagnostic figure
# ---------------------------------------------------------------------------

def _plot_diagnostics(positions: np.ndarray,
                      refined_pcs: np.ndarray,
                      predicted_pcs: np.ndarray,
                      znssd: np.ndarray,
                      plane: dict,
                      scan_shape: tuple) -> Figure:
    """Return a Figure with 4 panels: pcx/pcy/pcz scatter+plane + ZNSSD heatmap."""
    fig = Figure(figsize=(13, 9), tight_layout=True)
    axes = [fig.add_subplot(2, 2, i + 1, projection="3d" if i < 3 else None)
            for i in range(4)]

    # Surface grid for plane evaluation
    rr_surf, cc_surf = np.meshgrid(
        np.linspace(0, scan_shape[0] - 1, 20),
        np.linspace(0, scan_shape[1] - 1, 20),
        indexing="ij",
    )
    for i, (axis, label) in enumerate([("x", "pcx (x*)"),
                                       ("y", "pcy (y*)"),
                                       ("z", "pcz (z*)")]):
        ax = axes[i]
        # Evaluate plane on a smooth surface
        order = plane.get("order", "linear")
        if order == "linear":
            zz = (plane[axis][0]
                  + plane[axis][1] * rr_surf
                  + plane[axis][2] * cc_surf)
        else:
            zz = (plane[axis][0]
                  + plane[axis][1] * rr_surf
                  + plane[axis][2] * cc_surf
                  + plane[axis][3] * rr_surf * cc_surf)
        ax.plot_surface(rr_surf, cc_surf, zz, alpha=0.35, cmap="viridis",
                        edgecolor="none")
        ax.scatter(positions[:, 0], positions[:, 1], refined_pcs[:, i],
                   c="tomato", s=40, depthshade=False, label="refined")
        ax.scatter(positions[:, 0], positions[:, 1], predicted_pcs[:, i],
                   c="steelblue", marker="^", s=20, alpha=0.6,
                   depthshade=False, label="geometric prediction")
        ax.set_xlabel("row")
        ax.set_ylabel("col")
        ax.set_zlabel(label)
        rms = plane["rms_residual"][axis]
        ax.set_title(f"{label}\nplane RMS residual = {rms:.5f}", fontsize=10)
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)

    # ZNSSD heatmap
    ax = axes[3]
    # Reshape positions back into the (n_r, n_c) grid we sampled on
    znssd_grid = znssd.reshape(int(np.sqrt(len(znssd))), -1) \
        if int(np.sqrt(len(znssd))) ** 2 == len(znssd) else znssd[None, :]
    im = ax.imshow(znssd_grid, cmap="inferno", aspect="auto", origin="upper")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Per-point ZNSSD after refinement (lower = better)", fontsize=10)
    ax.set_xlabel("grid col index")
    ax.set_ylabel("grid row index")

    return fig


# ---------------------------------------------------------------------------
# PC-only refinement (3-DOF Nelder-Mead, orientation frozen at the .ang value).
# ---------------------------------------------------------------------------

def _refine_pc_only(sim, exp_pat: np.ndarray, pat_obj, patshape: tuple,
                    euler_fixed: np.ndarray, pc_init: np.ndarray,
                    pc_step: float = 0.005, max_iter: int = 200,
                    sim_hp_sigma=None, sim_lp_sigma=None,
                    sim_gamma=None) -> tuple:
    """PC-only Nelder-Mead at a single test point.

    Args:
        sim:          a pre-configured patternSimulation (master pattern
                      already loaded — shared across test points).
        exp_pat:      experimental pattern at this index, processed through
                      the same Step 3 pipeline used for the sim.
        pat_obj:      Data.UP2 — provides process_pattern for the sim.
        patshape:     (H, W) of the detector.
        euler_fixed:  (3,) Bunge ZXZ Euler angles in radians.  Held FIXED.
        pc_init:      (3,) initial PC (EDAX/TSL) at this test point.
        pc_step:      initial simplex perturbation per PC axis.
        max_iter:     max Nelder-Mead function evals.

    Returns:
        (pc_opt, znssd_opt) — refined PC (3-vector) and final ZNSSD scalar.
    """
    from scipy.optimize import minimize
    from optimize_reference import _simulate, _znssd

    pc_init = np.asarray(pc_init, dtype=np.float64)
    euler_fixed = np.asarray(euler_fixed, dtype=np.float64)

    def objective(pc_delta: np.ndarray) -> float:
        pc = pc_init + pc_delta
        sim_pat = _simulate(sim, euler_fixed, pc, pat_obj, patshape,
                            sim_hp_sigma, sim_lp_sigma, sim_gamma)
        return float(_znssd(exp_pat, sim_pat))

    # 3-D simplex centred on zero, one perturbation per PC axis.
    x0 = np.zeros(3, dtype=np.float64)
    simplex = np.tile(x0, (4, 1)).astype(np.float64)
    for i in range(3):
        simplex[i + 1, i] += pc_step

    result = minimize(
        objective, x0, method="Nelder-Mead",
        options={
            "maxiter":         max_iter,
            "maxfev":          max_iter,
            "xatol":           1e-6,
            "fatol":           1e-6,
            "adaptive":        True,
            "initial_simplex": simplex,
        },
    )
    pc_opt    = pc_init + result.x
    znssd_opt = float(result.fun)
    return pc_opt, znssd_opt


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def pc_plane_fit(
    pat_obj,
    ang_data,
    master_pattern_path: str,
    pc_ref,
    ref_position,
    sample_tilt_deg: float,
    detector_tilt_deg: float,
    step_size_um: float,
    pixel_size_um: float,
    scan_shape: tuple,
    n_grid: tuple = (5, 5),
    edge_padding: int = 2,
    scan_strategy: str = "standard",
    plane_order: str = "linear",
    weight_by_znssd: bool = True,
    refine_max_iter: int = 200,
    refine_euler_step_deg: float = 0.5,
    refine_pc_step: float = 0.005,
    sim_high_pass_sigma=None,
    sim_low_pass_sigma=None,
    sim_gamma=None,
    progress_cb=None,
    save_dir: str = "debug",
) -> dict:
    """Run the full PC-plane-fit workflow.

    Parameters
    ----------
    pat_obj
        ``Data.UP2`` instance, configured with the user's Step 3 processing.
    ang_data
        Parsed .ang result (must expose ``.eulers`` as an (n_rows, n_cols, 3)
        radians array).
    master_pattern_path
        Path to the EMsoft .h5 master pattern.
    pc_ref : array_like (3,)
        Reference PC in EDAX/TSL convention (xstar, ystar, zstar) at the
        scan position given by ``ref_position``.
    ref_position : (int, int)
        (row, col) where ``pc_ref`` applies.
    sample_tilt_deg, detector_tilt_deg : float
    step_size_um : float
        Scan step size in microns.
    pixel_size_um : float
        Detector pixel size in microns.
    scan_shape : (int, int)
        Full (n_rows, n_cols) of the scan.
    n_grid : (int, int)
        How many test points per axis (default 5×5 = 25 points).
    edge_padding : int
        Rows/cols of margin to leave free at each edge of the scan when
        placing test points.  Default 2 — edge pixels often suffer from
        beam-edge artefacts or low signal, so the grid is anchored on
        interior points only.  Clamped automatically if the scan is too
        small to accommodate the requested padding.
    scan_strategy : str
        Scan grid convention — must match Step 2's "Scan strategy" setting
        (e.g. "lower_left", "upper_left", "standard").  Controls which
        corner of the array is the physical (0, 0) and which way x/y point
        on the sample.  Forwarded verbatim to `make_scan_grid`.
    plane_order : 'linear' or 'bilinear'
    weight_by_znssd : bool
        If True, downweight badly-converged points in the LSQ fit.
    refine_max_iter : int
        Max Nelder-Mead function evals per test point.
    refine_pc_step : float
        Initial simplex perturbation per PC axis.
    refine_euler_step_deg : float
        UNUSED — kept for API compatibility.  Orientation is held FIXED at
        the .ang value, so this parameter has no effect.
    sim_high_pass_sigma, sim_low_pass_sigma, sim_gamma
        Optional Step 3 overrides applied only to the simulated patterns
        during refinement.
    progress_cb : callable
        Called as ``progress_cb(i, n_total, info_dict)`` after each test
        point.  ``info_dict`` has keys 'row', 'col', 'pc_predicted',
        'pc_refined', 'znssd'.
    save_dir : str
        Where to dump the diagnostic figure.  Use ``None`` to skip.

    Returns
    -------
    dict with:
        positions       : (n, 2) test grid (row, col)
        predicted_pcs   : (n, 3) geometric prediction at each point
        refined_pcs     : (n, 3) refinement output at each point
        znssd           : (n,)   post-refinement ZNSSD
        plane           : dict with 'order', 'x', 'y', 'z', 'rms_residual'
        fig             : matplotlib Figure (None if save_dir is None)
        saved_fig_path  : str or None
    """
    # Lazy-import these to keep this module's import cost low.
    from pc_homography_correction import make_scan_grid, scan_grid_to_pc_grid
    from PatternSimulation.SimPatGen import patternSimulation

    pc_ref       = np.asarray(pc_ref, dtype=np.float64)
    ref_position = (int(ref_position[0]), int(ref_position[1]))

    # Build the geometric prediction grid (per-(row,col) PC).  This is the
    # same model the apply_pc_correction path uses; we just sample from it.
    scan_grid = make_scan_grid(scan_shape, step_size_um, convention=scan_strategy)
    scan_grid.data = scan_grid.data - scan_grid.data[ref_position[0], ref_position[1]]
    pc_grid = scan_grid_to_pc_grid(scan_grid, pc_ref, pat_obj.patshape,
                                   pixel_size_um, sample_tilt_deg, detector_tilt_deg)

    # Sample test points (with edge padding so we don't sit on row 0 / col 0
    # or the opposite edges, where beam-edge artefacts are common).
    positions = _uniform_grid(scan_shape, n_grid, edge_padding=edge_padding)
    n_total   = positions.shape[0]

    predicted_pcs = np.zeros((n_total, 3), dtype=np.float64)
    refined_pcs   = np.zeros((n_total, 3), dtype=np.float64)
    refined_eus   = np.zeros((n_total, 3), dtype=np.float64)
    znssd_arr     = np.zeros(n_total, dtype=np.float64)

    print(f"\n[pc_plane_fit] {n_total} test points "
          f"({n_grid[0]}×{n_grid[1]} grid)  order={plane_order}  "
          f"weight_by_znssd={weight_by_znssd}")
    print(f"[pc_plane_fit] PC-only refinement (orientation FROZEN at .ang values)")

    # Shared simulator across all test points — master pattern is loaded
    # ONCE here instead of n_total times inside the loop.
    sim = patternSimulation()
    sim.detector_height   = pat_obj.patshape[0]
    sim.detector_width    = pat_obj.patshape[1]
    sim.det_shape         = pat_obj.patshape
    sim.sample_tilt_deg   = float(sample_tilt_deg)
    sim.detector_tilt_deg = float(detector_tilt_deg)
    sim.mastersetup(master_pattern_path)

    for i, (r, c) in enumerate(positions):
        pat_idx = int(r) * int(scan_shape[1]) + int(c)

        # Geometric prediction at this grid point.
        pc_pred = np.asarray(pc_grid[int(r), int(c)], dtype=np.float64)
        predicted_pcs[i] = pc_pred

        # Local orientation from .ang — FROZEN throughout the refinement.
        try:
            euler_init = np.asarray(ang_data.eulers[int(r), int(c)],
                                    dtype=np.float64)
        except Exception:
            euler_init = np.zeros(3, dtype=np.float64)

        # Experimental pattern at this position, processed through Step 3.
        try:
            exp_pat = pat_obj.read_pattern(pat_idx, process=True)
        except Exception as exc:
            print(f"  [pc_plane_fit] failed to read pattern {pat_idx}: {exc}")
            refined_pcs[i] = pc_pred
            refined_eus[i] = euler_init
            znssd_arr[i]   = float("inf")
            continue

        print(f"\n[pc_plane_fit]  point {i+1}/{n_total}  (row={r}, col={c})  "
              f"pc_predicted={tuple(pc_pred.round(5))}")

        try:
            pc_opt, znssd_val = _refine_pc_only(
                sim=sim, exp_pat=exp_pat, pat_obj=pat_obj,
                patshape=pat_obj.patshape,
                euler_fixed=euler_init,
                pc_init=pc_pred,
                pc_step=refine_pc_step,
                max_iter=refine_max_iter,
                sim_hp_sigma=sim_high_pass_sigma,
                sim_lp_sigma=sim_low_pass_sigma,
                sim_gamma=sim_gamma,
            )
        except Exception as exc:
            print(f"  [pc_plane_fit]  refinement FAILED at point {i+1}: {exc}")
            pc_opt    = pc_pred
            znssd_val = float("inf")        # huge ZNSSD → dropped by weighted LSQ

        refined_pcs[i] = np.asarray(pc_opt, dtype=np.float64)
        refined_eus[i] = euler_init        # orientation never moves
        znssd_arr[i]   = znssd_val

        print(f"  [pc_plane_fit]  refined PC={tuple(refined_pcs[i].round(5))}  "
              f"ZNSSD={znssd_val:.5f}")

        if progress_cb is not None:
            try:
                progress_cb(i + 1, n_total, {
                    "row": int(r), "col": int(c),
                    "pc_predicted": pc_pred,
                    "pc_refined":   refined_pcs[i].copy(),
                    "znssd":        float(znssd_val),
                })
            except Exception:
                pass

    # Drop failed points (znssd == inf) from the fit AND the diagnostic plot.
    keep = np.isfinite(znssd_arr)
    if not keep.all():
        print(f"[pc_plane_fit] dropping {(~keep).sum()} failed point(s) from the fit")

    plane = _fit_plane(
        positions[keep],
        refined_pcs[keep],
        znssd_arr[keep],
        order=plane_order,
        weight_by_znssd=weight_by_znssd,
    )

    print(f"\n[pc_plane_fit] plane coefficients ({plane_order}):")
    for axis in ("x", "y", "z"):
        print(f"  pc{axis}: {plane[axis].round(6)}  "
              f"(RMS residual = {plane['rms_residual'][axis]:.5f})")

    # Diagnostic figure.
    fig = _plot_diagnostics(positions[keep], refined_pcs[keep],
                            predicted_pcs[keep], znssd_arr[keep],
                            plane, scan_shape)
    saved_path = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        saved_path = os.path.join(save_dir, "pc_plane_fit_diagnostics.png")
        fig.savefig(saved_path, dpi=140, bbox_inches="tight")
        print(f"[pc_plane_fit] saved diagnostic figure → {saved_path}")

    return {
        "positions":      positions,
        "predicted_pcs":  predicted_pcs,
        "refined_pcs":    refined_pcs,
        "refined_eulers": refined_eus,
        "znssd":          znssd_arr,
        "plane":          plane,
        "fig":            fig,
        "saved_fig_path": saved_path,
    }

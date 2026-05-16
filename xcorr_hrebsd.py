"""
xcorr_hrebsd.py — Wilkinson-style cross-correlation HR-EBSD as an alternative
to the IC-GN homography approach in get_homography_cpu.py.

The two output formats and call signatures are kept compatible with
``get_homography_cpu.optimize`` so that gui_workers / Results_plotting
can consume xcorr-HREBSD results without modification.

Algorithm (per pattern in the scan):
    1. Extract N ROIs at fixed positions on both reference and target patterns.
    2. For each ROI:
       a. Apply a Hanning window.
       b. Compute the phase-correlation peak (FFT-based).
       c. Sub-pixel refine via 2D parabolic fit on the 3×3 around the peak.
    3. Build a linearised Wilkinson system  A · p = b  where p is the
       8-parameter homography and the rows of A use the same Jacobian as
       IC-GN (so downstream h2F / F2strain treat the result identically).
    4. Solve via numpy.linalg.lstsq.

Usage as a drop-in replacement:
    import xcorr_hrebsd as core
    h, iterations, residuals, dp_norms = core.optimize(
        pat_obj, x0,
        roi_grid=(7, 7),
        roi_size=64,
        ...
    )

Or as a script:
    python xcorr_hrebsd.py        # edit the USER INPUTS block at the bottom
"""

import os
import numpy as np
from scipy import signal as _signal
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import Data
import conversions
from get_homography_cpu import roi_indices_from_rect, tqdm_joblib


# ---------------------------------------------------------------------------
# ROI placement
# ---------------------------------------------------------------------------

def make_roi_grid(
    patshape: tuple,
    grid: tuple = (7, 7),
    roi_size: int = 64,
    border_frac: float = 0.05,
    mask: np.ndarray = None,
) -> list:
    """Return a list of (x_pix, y_pix) ROI-center coordinates on a regular grid.

    Parameters
    ----------
    patshape : (H, W)
    grid : (n_y, n_x) — number of ROIs along each axis
    roi_size : ROI side length in pixels (must be even)
    border_frac : leave this fraction of the pattern as a margin on each side
    mask : optional bool array of shape patshape; ROIs whose center sits on a
           False pixel are dropped
    """
    H, W = patshape
    n_y, n_x = grid
    half = roi_size // 2
    margin_y = int(border_frac * H) + half
    margin_x = int(border_frac * W) + half
    if n_x < 2 or n_y < 2:
        raise ValueError("grid must have at least 2 ROIs in each axis")
    xs = np.linspace(margin_x, W - margin_x - 1, n_x, dtype=int)
    ys = np.linspace(margin_y, H - margin_y - 1, n_y, dtype=int)
    positions = []
    for y in ys:
        for x in xs:
            if mask is not None and not bool(mask[y, x]):
                continue
            positions.append((int(x), int(y)))
    return positions


# ---------------------------------------------------------------------------
# Cross-correlation / sub-pixel peak
# ---------------------------------------------------------------------------

def _hann2d(shape: tuple) -> np.ndarray:
    """2-D separable Hanning window."""
    wy = np.hanning(shape[0])[:, None]
    wx = np.hanning(shape[1])[None, :]
    return wy * wx


def _phase_correlate(r_roi: np.ndarray, t_roi: np.ndarray) -> np.ndarray:
    """Return the centred phase-correlation surface for two ROIs."""
    win = _hann2d(r_roi.shape)
    Fr = np.fft.fft2((r_roi - r_roi.mean()) * win)
    Ft = np.fft.fft2((t_roi - t_roi.mean()) * win)
    cross = Fr * np.conj(Ft)
    cross /= np.maximum(np.abs(cross), 1e-12)
    cc = np.fft.ifft2(cross).real
    return np.fft.fftshift(cc)


def _parabolic_subpixel(cc: np.ndarray, py: int, px: int) -> tuple:
    """Sub-pixel offset (dx, dy) from a 2-D parabolic fit on the 3×3 around (py, px)."""
    if py <= 0 or py >= cc.shape[0] - 1 or px <= 0 or px >= cc.shape[1] - 1:
        return 0.0, 0.0
    cx_m = cc[py, px - 1]; cx_0 = cc[py, px]; cx_p = cc[py, px + 1]
    cy_m = cc[py - 1, px]; cy_p = cc[py + 1, px]
    den_x = (cx_m - 2 * cx_0 + cx_p)
    den_y = (cy_m - 2 * cx_0 + cy_p)
    dx = 0.5 * (cx_m - cx_p) / den_x if abs(den_x) > 1e-12 else 0.0
    dy = 0.5 * (cy_m - cy_p) / den_y if abs(den_y) > 1e-12 else 0.0
    return float(dx), float(dy)


def measure_shifts(
    R: np.ndarray,
    T: np.ndarray,
    positions: list,
    roi_size: int,
) -> tuple:
    """Cross-correlate every ROI in R vs T; return arrays of shifts and CC peaks.

    Returns
    -------
    shifts : (N, 2) float32 — (dx, dy) sub-pixel shift of T relative to R, in pixels
    peaks  : (N,) float32 — peak phase-correlation value (proxy for confidence)
    """
    half = roi_size // 2
    N = len(positions)
    shifts = np.zeros((N, 2), dtype=np.float32)
    peaks  = np.zeros(N, dtype=np.float32)
    for i, (x, y) in enumerate(positions):
        r_roi = R[y - half:y + half, x - half:x + half]
        t_roi = T[y - half:y + half, x - half:x + half]
        if r_roi.shape != (roi_size, roi_size) or t_roi.shape != (roi_size, roi_size):
            continue
        cc = _phase_correlate(r_roi, t_roi)
        py, px = np.unravel_index(np.argmax(cc), cc.shape)
        cy, cx = cc.shape[0] // 2, cc.shape[1] // 2
        dy_int = float(py - cy)
        dx_int = float(px - cx)
        dx_sub, dy_sub = _parabolic_subpixel(cc, py, px)
        # Convention: dx = horizontal shift (column direction) of T relative to R
        shifts[i, 0] = dx_int + dx_sub
        shifts[i, 1] = dy_int + dy_sub
        peaks[i]     = float(cc[py, px])
    return shifts, peaks


# ---------------------------------------------------------------------------
# Wilkinson linear system → 8-parameter homography
# ---------------------------------------------------------------------------

def solve_homography_lsq(
    positions: list,
    shifts: np.ndarray,
    h0: tuple,
) -> tuple:
    """Least-squares solve for the 8 homography parameters from the ROI shifts.

    The Jacobian rows match the IC-GN warp Jacobian in get_homography_cpu so
    the resulting `p` plugs into `h2F` exactly the same way IC-GN's output does.

    Parameters
    ----------
    positions : list of (x_pix, y_pix) ROI centres in the image coordinate system
    shifts    : (N, 2) sub-pixel shifts (dx, dy) measured by xcorr
    h0        : (cx, cy) homography reference centre (image-coord system)

    Returns
    -------
    p        : (8,) homography parameter vector  [h11_dev, h12, h13, h21, h22_dev, h23, h31, h32]
    residual : float — RMS of A·p − b  (in pixels)
    cond_A   : float — condition number of A  (poorly-conditioned design ⇒ unreliable)
    """
    cx, cy = float(h0[0]), float(h0[1])
    A_rows, b_vals = [], []
    for (x_img, y_img), (dx, dy) in zip(positions, shifts):
        x = float(x_img) - cx
        y = float(y_img) - cy
        # Same row layout as the IC-GN Jacobian (out0 / out1 in get_homography_cpu)
        A_rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -x * x, -x * y])
        A_rows.append([0.0, 0.0, 0.0, x, y, 1.0, -x * y, -y * y])
        b_vals.append(float(dx))
        b_vals.append(float(dy))
    A = np.asarray(A_rows, dtype=np.float64)
    b = np.asarray(b_vals,  dtype=np.float64)
    p, _, _, sv = np.linalg.lstsq(A, b, rcond=None)
    residual = float(np.sqrt(np.mean((A @ p - b) ** 2)))
    cond_A   = float(sv.max() / max(sv.min(), 1e-12))
    return p, residual, cond_A


# ---------------------------------------------------------------------------
# Per-pattern worker
# ---------------------------------------------------------------------------

def _process_one(idx, get_pat, R, positions, roi_size, h0):
    T = get_pat(idx)
    shifts, peaks = measure_shifts(R, T, positions, roi_size)
    h, residual, cond_A = solve_homography_lsq(positions, shifts, h0)
    # `iter` for xcorr is meaningless; report N_rois used as a stand-in.
    n_rois_used = int(np.sum(np.any(shifts != 0, axis=1)))
    # `dp_norm` proxy: median absolute shift across ROIs (px) — measure of motion magnitude
    dp_norm = float(np.median(np.linalg.norm(shifts, axis=1)))
    return h, n_rois_used, residual, dp_norm


# ---------------------------------------------------------------------------
# Public API — mirrors get_homography_cpu.optimize for InitType.NONE
# ---------------------------------------------------------------------------

def optimize(
    pats,
    x0,
    crop_fraction: float = 0.7,           # (unused — kept for API parity)
    n_jobs: int = -1,
    verbose: bool = False,
    roi_slice: tuple = None,
    scan_shape: tuple = None,
    mask: np.ndarray = None,
    progress_callback=None,
    # xcorr-specific:
    roi_grid: tuple = (7, 7),
    roi_size: int = 64,
    border_frac: float = 0.05,
    h_center: tuple = None,                # default = image centre
    **_ignored,
):
    """Cross-correlation HR-EBSD optimisation.

    Returns the same 4-tuple as ``get_homography_cpu.optimize(init_type=NONE)``:
        (homographies, iterations, residuals, dp_norms)

    `iterations` reports the number of ROIs successfully used for each pattern.
    `residuals` is the RMS pixel residual of the linear least-squares fit.
    `dp_norms`  is the median absolute ROI-shift magnitude in pixels.
    """
    if n_jobs == -1:
        n_jobs = max(os.cpu_count() - 1, 1)

    # ---- Resolve pattern source / shape ----
    if isinstance(pats, Data.UP2):
        if roi_slice is not None:
            if scan_shape is None:
                raise ValueError("roi_slice requires scan_shape=(nrows, ncols)")
            roi_indices = roi_indices_from_rect(roi_slice, scan_shape)
            roi_nrows = roi_slice[0].stop - roi_slice[0].start
            roi_ncols = roi_slice[1].stop - roi_slice[1].start
            N         = roi_nrows * roi_ncols
            out_shape = (roi_nrows, roi_ncols)
        else:
            roi_indices = None
            N           = pats.nPatterns
            out_shape   = (pats.nPatterns,)
        get_pat = lambda i: pats.read_pattern(i, process=True)
        patshape = pats.patshape
    elif isinstance(pats, np.ndarray):
        N         = int(np.prod(pats.shape[:-2]))
        out_shape = pats.shape[:-2]
        patshape  = pats.shape[-2:]
        flat      = pats.reshape(-1, *patshape)
        get_pat   = lambda i: flat[i]
        roi_indices = None
    else:
        raise TypeError("pats must be a Data.UP2 object or a numpy array")

    if h_center is None:
        h_center = (patshape[1] // 2, patshape[0] // 2)

    # ---- Reference pattern ----
    R = get_pat(int(x0))

    # ---- ROI placement ----
    positions = make_roi_grid(
        patshape, grid=roi_grid, roi_size=roi_size,
        border_frac=border_frac, mask=mask,
    )
    if len(positions) < 8:
        raise RuntimeError(
            f"xcorr-HREBSD needs at least 8 ROIs to solve for 8 parameters; "
            f"got {len(positions)}.  Increase roi_grid or reduce border_frac."
        )
    if verbose:
        print(f"[xcorr] {len(positions)} ROIs of size {roi_size}×{roi_size} "
              f"on a {patshape[0]}×{patshape[1]} pattern.")

    idx_list = roi_indices if roi_indices is not None else range(N)

    # ---- Per-pattern parallel solve ----
    if verbose:
        with tqdm_joblib(tqdm(total=N, desc="xcorr-HREBSD")):
            results = Parallel(n_jobs=n_jobs)(
                delayed(_process_one)(i, get_pat, R, positions, roi_size, h_center)
                for i in idx_list
            )
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_one)(i, get_pat, R, positions, roi_size, h_center)
            for i in idx_list
        )

    # ---- Pack into the same shape as core.optimize ----
    homographies = np.zeros((N, 8), dtype=float)
    iterations   = np.zeros(N, dtype=int)
    residuals    = np.zeros(N, dtype=float)
    dp_norms     = np.zeros(N, dtype=float)
    for k, (h, n_used, res, dpn) in enumerate(results):
        homographies[k] = h
        iterations[k]   = n_used
        residuals[k]    = res
        dp_norms[k]     = dpn
        if progress_callback is not None:
            progress_callback(k + 1, N)

    homographies = homographies.reshape(out_shape + (8,))
    iterations   = iterations.reshape(out_shape)
    residuals    = residuals.reshape(out_shape)
    dp_norms     = dp_norms.reshape(out_shape)

    return homographies, iterations, residuals, dp_norms


# ---------------------------------------------------------------------------
# Standalone runner — edit USER INPUTS and run with `python xcorr_hrebsd.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── USER INPUTS ─────────────────────────────────────────────────────────
    up2_path = "/Users/crestiennedechaine/OriginalData/Si_Ge_Dataset/SiGe_updated_512x512.up2"
    ang_path = "/Users/crestiennedechaine/OriginalData/Si_Ge_Dataset/dp_Si_new_refined.ang"
    save_dir = "debug/xcorr_hrebsd"
    ref_idx  = (0, 0)            # (row, col) reference position in the scan
    roi_grid_n = (7, 7)          # ROI grid (n_rows, n_cols)
    roi_size_n = 64              # ROI side in pixels
    sample_tilt_deg   = 70.0
    detector_tilt_deg = 10.0

    import utilities
    import datetime, time

    pat_obj = Data.UP2(up2_path)
    pat_obj.set_processing(
        low_pass_sigma=1.0, high_pass_sigma=10.0, truncate_std_scale=3.0,
        mask_type="none", use_clahe=False, gamma=0.8,
    )
    ang_data = utilities.read_ang(ang_path, pat_obj.patshape, segment_grain_threshold=None)
    x0 = int(np.ravel_multi_index(ref_idx, ang_data.shape))
    pc_edax   = np.asarray(ang_data.pc, dtype=float)
    pc_bruker = conversions.Edax_to_Bruker_PC(pc_edax)
    xo        = conversions.Bruker_to_fractional_PC(pc_bruker, pat_obj.patshape)

    print(f"xcorr-HREBSD on {ang_data.shape[0]}×{ang_data.shape[1]} = "
          f"{pat_obj.nPatterns} patterns, ref at flat idx {x0}.")
    t0 = time.perf_counter()
    h, iters, resid, dp_norm = optimize(
        pat_obj, x0,
        roi_grid=roi_grid_n, roi_size=roi_size_n,
        verbose=True,
    )
    t1 = time.perf_counter()
    print(f"Done in {t1 - t0:.1f} s.")

    # Strain / rotation via the same h2F / F2strain used by IC-GN
    h_flat = h.reshape(-1, 8).astype(np.float64)
    F      = conversions.h2F(h_flat, xo)
    epsilon, omega = conversions.F2strain(F)
    R = utilities.get_sample_to_detector_rotation(detector_tilt_deg, sample_tilt_deg)
    for i in range(epsilon.shape[0]):
        epsilon[i] = R @ epsilon[i] @ R.T
        omega[i]   = R @ omega[i]   @ R.T

    rows, cols = ang_data.shape
    def _2d(arr): return arr.reshape(rows, cols)

    os.makedirs(save_dir, exist_ok=True)
    date = datetime.date.today().strftime("%B_%d_%Y")
    out_path = os.path.join(save_dir, f"xcorr_hrebsd_results_{date}.npy")
    np.save(out_path, {
        "h11": _2d(h_flat[:, 0]), "h12": _2d(h_flat[:, 1]), "h13": _2d(h_flat[:, 2]),
        "h21": _2d(h_flat[:, 3]), "h22": _2d(h_flat[:, 4]), "h23": _2d(h_flat[:, 5]),
        "h31": _2d(h_flat[:, 6]), "h32": _2d(h_flat[:, 7]),
        "e11": _2d(epsilon[:, 0, 0]), "e12": _2d(epsilon[:, 0, 1]), "e13": _2d(epsilon[:, 0, 2]),
        "e22": _2d(epsilon[:, 1, 1]), "e23": _2d(epsilon[:, 1, 2]), "e33": _2d(epsilon[:, 2, 2]),
        "w13": _2d(np.degrees(omega[:, 0, 2])),
        "w21": _2d(np.degrees(omega[:, 1, 0])),
        "w32": _2d(np.degrees(omega[:, 2, 1])),
        "F":    F,
        "rows": np.array(rows),
        "cols": np.array(cols),
    })
    print(f"Saved results to {out_path}")
    print(f"You can load this in the visualization tab — the schema matches "
          f"PipelineWorker._save_npz so plot_all_results works unchanged.")

"""get_homography_cpu_reversed.py

Reversed-role IC-GN: same inverse-compositional Gauss-Newton math as
``get_homography_cpu.py``, but the roles of reference and target patterns are
swapped.

Role mapping (compared to ``get_homography_cpu.py``):

    +-----------------------------------+----------------------+----------------------+
    |                                   |  get_homography_cpu  | get_homography_cpu_  |
    |                                   |                      |      reversed        |
    +-----------------------------------+----------------------+----------------------+
    | Pattern providing GRADIENTS,      |  R (constant, from   |  get_pat(idx)        |
    | JACOBIAN, HESSIAN  →  the         |  x0 / sim / override)|  (per pattern)       |
    | "reference" in IC-GN              |                      |                      |
    +-----------------------------------+----------------------+----------------------+
    | Pattern that gets WARPED          |  get_pat(idx)        |  R (constant, from   |
    | (spline-sampled at W(xi; h))      |  (per pattern)       |  x0 / sim / override)|
    | each iteration  →  the "target"   |                      |                      |
    +-----------------------------------+----------------------+----------------------+
    | Built once before the parallel    |  Reference precompute|  Target spline +     |
    | loop                              |  (spline, GR, Jac,   |  FMT pieces of the   |
    |                                   |  H, Cholesky, FMT)   |  constant pattern    |
    +-----------------------------------+----------------------+----------------------+
    | Built per pattern in the worker   |  Target spline       |  Reference precompute|
    |                                   |                      |  (spline, GR, Jac,   |
    |                                   |                      |  H, Cholesky, FMT of |
    |                                   |                      |  the per-pattern)    |
    +-----------------------------------+----------------------+----------------------+

Output convention:
    The returned homography ``h`` warps the per-pattern image (the new
    reference, = former target) into the constant pattern (the new target,
    = former reference).  This is approximately the **inverse** of what
    ``get_homography_cpu.optimize`` returns for the same pair of patterns —
    invert each ``h`` (or compose with the original via ``warp.W``) before
    feeding into downstream code that assumes the original convention.

Notes:
    * All small helpers (``window_and_normalize_new``, ``FMT``,
      ``_select_peak_2d``, ``dp_norm``, ``simulate_reference_pattern``,
      ``compare_gradients``, ``roi_indices_from_rect``, ``tqdm_joblib``,
      ``InitType``) are imported from ``get_homography_cpu`` — they are
      role-agnostic.
    * The per-pattern reference precompute is the dominant cost of this
      variant: building a 5th-order RectBivariateSpline + Cholesky factor
      for every pattern is much more work than the original, where these
      were done once.  Expect the run time to scale roughly linearly with
      the number of patterns rather than being amortised.
"""

import os
from typing import Callable

import numpy as np
from scipy import linalg, interpolate, signal
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

import warp
import conversions
import Data

from get_homography_cpu import (
    InitType,
    PATS,
    ARRAY,
    tqdm_joblib,
    simulate_reference_pattern,
    compare_gradients,
    roi_indices_from_rect,
    window_and_normalize_new,
    FMT,
    _select_peak_2d,
    dp_norm,
)


def optimize_reversed(
    pats: PATS,
    x0: ARRAY,
    init_type: InitType = InitType.NONE,
    crop_fraction: float = 0.7,
    max_iter: int = 50,
    conv_tol: float = 1e-3,
    n_jobs: int = -1,
    verbose: bool = False,
    roi_slice: tuple[slice, slice] = None,
    scan_shape: tuple = None,
    mask: np.ndarray = None,
    use_simulated_reference: bool = False,
    master_pattern_path: str = None,
    euler_angles_ref: np.ndarray = None,
    pc_ref: tuple = None,
    tilt_deg: float = 70.0,
    detector_tilt_deg: float = 0.0,
    sim_high_pass_sigma: float = None,
    pc_xo: np.ndarray = None,
    ref_pat_override: np.ndarray = None,
    spectral_match_ref: bool = False,
    perspective_regularization: float = 0.0,
    rotate_patterns_90: bool = False,
    progress_callback: Callable = None,
    debug_gradients: bool = False,
    subset_shape_kind: str = "rect",
) -> np.ndarray:
    """Run IC-GN with reference and target roles swapped.

    Same signature as ``get_homography_cpu.optimize``.  The constant pattern
    designated by ``x0`` (or the simulated / override pattern) is now the
    TARGET; each ``get_pat(idx)`` is the new REFERENCE — its spline /
    gradients / Jacobian / Hessian are recomputed inside each worker.

    See module docstring for the output convention.
    """

    ### Prepare the inputs ###
    if subset_shape_kind not in ("rect", "circle"):
        raise ValueError(
            f"subset_shape_kind must be 'rect' or 'circle', got {subset_shape_kind!r}."
        )
    if crop_fraction <= 0:
        raise ValueError("Crop fraction must be greater than 0.")
    if subset_shape_kind == "rect" and crop_fraction >= 1:
        raise ValueError("Crop fraction must be < 1 in rect mode.")
    if subset_shape_kind == "circle" and crop_fraction > 1:
        raise ValueError("Crop fraction must be <= 1 in circle mode.")
    if max_iter <= 0:
        raise ValueError("Maximum number of iterations must be greater than 0.")
    if conv_tol <= 0:
        raise ValueError("Convergence tolerance must be greater than 0.")

    if not isinstance(init_type, InitType):
        if type(init_type) == str:
            init_type = InitType(init_type.lower())
        else:
            raise TypeError(
                "init_type must be an instance of the InitType enum or a string matching the values of the enum."
            )

    if n_jobs == -1:
        n_jobs = os.cpu_count() - 1

    if type(pats) == Data.UP2:
        if roi_slice is not None:
            if scan_shape is None:
                raise ValueError(
                    "roi_slice requires scan_shape=(nrows, ncols) — the grid dimensions of the full scan. "
                    "Pass it as scan_shape=ang_data.shape to optimize_reversed()."
                )
            roi_nrows = roi_slice[0].stop - roi_slice[0].start
            roi_ncols = roi_slice[1].stop - roi_slice[1].start
            N = roi_nrows * roi_ncols
            out_shape = (roi_nrows, roi_ncols)
            roi_indices = roi_indices_from_rect(roi_slice, scan_shape)
        else:
            roi_indices = None
            N = pats.nPatterns
            out_shape = (pats.nPatterns,)
        get_pat = lambda idx: pats.read_pattern(idx, process=True)
        patshape = pats.patshape
    elif type(pats) == np.ndarray:
        N = np.prod(pats.shape[:-2])
        out_shape = pats.shape[:-2]
        patshape = pats.shape[-2:]
        pats = pats.reshape(-1, pats.shape[-2], pats.shape[-1])
        get_pat = lambda idx: pats[idx]
        roi_indices = None
    else:
        raise TypeError("pats must be a Data.UP2 object or a numpy array.")

    # Diagnostic: rotate every pattern (constant target + per-pixel refs +
    # mask + override) by 90° CCW before the optimization runs.  The .up2
    # is untouched.  PC vectors are NOT rotated, so the geometry feeding
    # h2F stays in the original frame; this lets you compare strain output
    # with and without the 90° rotation to assess PC-sensitivity.
    _orig_patshape = patshape
    if rotate_patterns_90:
        print("[reversed][rotate_patterns_90] applying np.rot90(k=1) to all "
              "patterns (constant target, per-pixel refs, mask, override).  "
              ".up2 untouched.")
        _base_get_pat = get_pat
        get_pat = lambda idx: np.rot90(_base_get_pat(idx), k=1)
        # Rotation by 90° transposes the shape.
        patshape = (patshape[1], patshape[0])
        if mask is not None:
            mask = np.rot90(mask, k=1)
        if ref_pat_override is not None:
            ref_pat_override = np.rot90(ref_pat_override, k=1)

    h0 = (patshape[1] // 2, patshape[0] // 2)
    # Mirror of get_homography_cpu.optimize's subset-shape branch.
    subset_circle_mask = None
    if subset_shape_kind == "circle":
        _half  = min(patshape) / 2.0
        radius = int(round(crop_fraction * _half))
        radius = max(1, min(radius, patshape[0] // 2, patshape[1] // 2))
        cy = patshape[0] // 2
        cx = patshape[1] // 2
        crop_row = cy - radius
        crop_col = cx - radius
        subset_slice = (slice(crop_row, crop_row + 2 * radius),
                        slice(crop_col, crop_col + 2 * radius))
        yy, xx = np.ogrid[:2 * radius, :2 * radius]
        subset_circle_mask = ((yy - radius) ** 2 + (xx - radius) ** 2) <= radius ** 2
        if mask is not None:
            full = np.zeros(patshape, dtype=bool)
            full[subset_slice] = mask[subset_slice] & subset_circle_mask
            mask = full
        else:
            mask = np.zeros(patshape, dtype=bool)
            mask[subset_slice] = subset_circle_mask
        print(f"[reversed] Subset shape: circle (radius={radius} px)")
        # Skip the rect-style crop_row/crop_col reassignment below.
        _SKIP_RECT = True
    else:
        _SKIP_RECT = False

    if not _SKIP_RECT:
        crop_row = int(patshape[0] * (1 - crop_fraction) / 2)
        crop_col = int(patshape[1] * (1 - crop_fraction) / 2)
        subset_slice = (slice(crop_row, -crop_row), slice(crop_col, -crop_col))

    ### Constant target pattern (formerly the "reference") ###
    if use_simulated_reference:
        if master_pattern_path is None or euler_angles_ref is None or pc_ref is None:
            raise ValueError(
                "use_simulated_reference=True requires master_pattern_path, "
                "euler_angles_ref, and pc_ref."
            )
        T_const = simulate_reference_pattern(
            master_pattern_path=master_pattern_path,
            euler_angles=euler_angles_ref,
            PC=pc_ref,
            patshape=_orig_patshape,
            tilt_deg=tilt_deg,
            detector_tilt_deg=detector_tilt_deg,
            pat_obj=pats if isinstance(pats, Data.UP2) else None,
            high_pass_sigma_override=sim_high_pass_sigma,
        )
        if rotate_patterns_90:
            T_const = np.rot90(T_const, k=1)
        print(f"[reversed] Using simulated pattern as constant TARGET (shape {T_const.shape})")
        if debug_gradients:
            real_T = get_pat(x0)
            compare_gradients(
                real_pat=real_T,
                sim_pat=T_const,
                crop_fraction=crop_fraction,
                mask=mask,
                save_path="debug/gradient_comparison_reversed.png",
            )
    elif ref_pat_override is not None:
        T_const = np.asarray(ref_pat_override).astype(np.float32)
        if T_const.shape != tuple(patshape):
            raise ValueError(
                f"ref_pat_override shape {T_const.shape} does not match patshape "
                f"{tuple(patshape)}."
            )
        print(f"[reversed] Using caller-supplied pattern as constant TARGET "
              f"(shape {T_const.shape}, override preprocessing).")
    else:
        T_const = get_pat(x0)

    print(f"[reversed] Constant target pattern shape: {T_const.shape}")

    if spectral_match_ref:
        if isinstance(pats, Data.UP2):
            import utilities as _utils
            print("[reversed][spectral_match_ref] computing average exp amplitude spectrum…")
            target_amp = _utils.average_exp_amplitude_spectrum(
                pats, n_samples=10, exclude_idx=int(x0)
            )
            T_before = T_const.copy()
            T_const = _utils.spectral_match_pattern(
                T_const.astype(np.float32), target_amp.astype(np.float32)
            )
            print(f"[reversed][spectral_match_ref] T_const amplitude rescaled.  "
                  f"L2 change: {np.linalg.norm(T_const - T_before):.4f}")
        else:
            print("[reversed][spectral_match_ref] skipped — pats must be a Data.UP2.")

    ### Subset coordinates (same for every pattern) ###
    x = np.arange(T_const.shape[1]) - h0[0]
    y = np.arange(T_const.shape[0]) - h0[1]
    X, Y = np.meshgrid(x, y, indexing="xy")

    xi = np.array([X[subset_slice].flatten(), Y[subset_slice].flatten()])
    subset_shape = X[subset_slice].shape

    valid = None
    if mask is not None:
        valid = mask[subset_slice].flatten()
        xi = xi[:, valid]
        print(f"[reversed] Mask applied: {valid.sum()} / {valid.size} subset pixels used "
              f"({100 * valid.mean():.1f}%)")

    ### Constant target spline (built ONCE, shared across all workers) ###
    T_const_spline = interpolate.RectBivariateSpline(x, y, T_const.T, kx=5, ky=5)

    ### FMT precompute on the constant TARGET (former reference) ###
    init_subset_slice = None
    t_const_init = None
    t_const_fmt = None
    X_fmt = Y_fmt = x_fmt = y_fmt = None
    if init_type is not InitType.NONE:
        # Size the initial-guess subset to fit INSIDE the IC-GN crop region
        # (i.e. inside the same edge-excluded zone the main loop already uses)
        # AND be a power of 2 (FFT-friendly).  Mirrors the same change in
        # get_homography_cpu.py so reversed-roles inherits the edge protection.
        _ic_h = patshape[0] - 2 * crop_row     # IC-GN subset height
        _ic_w = patshape[1] - 2 * crop_col     # IC-GN subset width
        _s    = 2 ** (min(_ic_h, _ic_w).bit_length() - 1)
        row_start = (patshape[0] - _s) // 2
        col_start = (patshape[1] - _s) // 2
        init_subset_slice = (
            slice(row_start, row_start + _s),
            slice(col_start, col_start + _s),
        )

        # Window + normalise the constant target subset.  In the original code
        # this was r_init (because R was the reference); here the constant
        # pattern is the target, so we name it t_const_init.
        t_const_init = window_and_normalize_new(T_const[init_subset_slice])

        height, width = t_const_init.shape
        theta = np.linspace(0, np.pi, int(height), endpoint=False)
        radius = np.linspace(0, height / 2, int(height + 1), endpoint=False)[1:]
        radius_grid, theta_grid = np.meshgrid(radius, theta, indexing="ij")
        radius_grid = radius_grid.flatten()
        theta_grid = theta_grid.flatten()
        x_fmt = 2 ** (np.log2(height) - 1) + radius_grid * np.cos(theta_grid)
        y_fmt = 2 ** (np.log2(height) - 1) - radius_grid * np.sin(theta_grid)
        X_fmt = np.arange(width)
        Y_fmt = np.arange(height)

        t_const_fft = np.fft.fftshift(np.fft.fft2(t_const_init))
        t_const_fmt, _ = FMT(t_const_fft, X_fmt, Y_fmt, x_fmt, y_fmt)

    ### Debug folder for per-pattern saves ###
    debug_dir = os.path.join("debug", "pat_reversed")
    os.makedirs(debug_dir, exist_ok=True)
    for name in os.listdir(debug_dir):
        p = os.path.join(debug_dir, name)
        if os.path.isfile(p):
            os.remove(p)

    idx_list = roi_indices if roi_indices is not None else range(N)

    ### Run the optimization in parallel ###
    worker_kwargs = dict(
        get_pat=get_pat,
        init_type=init_type,
        init_subset_slice=init_subset_slice,
        t_const_init=t_const_init,
        t_const_fmt=t_const_fmt,
        X_fmt=X_fmt,
        Y_fmt=Y_fmt,
        x_fmt=x_fmt,
        y_fmt=y_fmt,
        T_const_spline=T_const_spline,
        xi=xi,
        h0=h0,
        max_iter=max_iter,
        conv_tol=conv_tol,
        pc_xo=pc_xo,
        perspective_regularization=perspective_regularization,
    )

    # tqdm subclass that pipes per-pattern progress to a callback — used by
    # the Qt GUI to drive a progress bar.  Mirrors get_homography_cpu.
    class _ProgressTqdm(tqdm):
        def __init__(self, *args, progress_cb=None, total=None, **kwargs):
            super().__init__(*args, total=total, **kwargs)
            self._progress_cb = progress_cb
            self._cb_total    = total
        def update(self, n=1):
            r = super().update(n)
            if self._progress_cb is not None and self._cb_total:
                try:
                    self._progress_cb(int(self.n), int(self._cb_total))
                except Exception:
                    pass
            return r

    with tqdm_joblib(_ProgressTqdm(
            total=N, desc="Patterns optimized (reversed)",
            progress_cb=progress_callback,
            disable=not verbose,
    )) as _:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_single_pattern_reversed)(idx, **worker_kwargs)
            for idx in idx_list
        )

    homographies = np.zeros((N, 8), dtype=float)
    homographies_guess = np.zeros((N, 8), dtype=float)
    iterations = np.zeros(N, dtype=int)
    residuals = np.zeros(N, dtype=float)
    dp_norms = np.zeros(N, dtype=float)

    for idx, (h, p_guess, num_iter, residual, dpn) in enumerate(results):
        homographies[idx] = h
        homographies_guess[idx] = p_guess
        iterations[idx] = num_iter
        residuals[idx] = float(residual)
        dp_norms[idx] = float(dpn)
        # progress_callback fires inside _ProgressTqdm.update() during the
        # joblib loop — no second emission needed after aggregation.

    homographies = homographies.reshape(out_shape + (8,))
    homographies_guess = homographies_guess.reshape(out_shape + (8,))
    iterations = iterations.reshape(out_shape)
    residuals = residuals.reshape(out_shape)
    dp_norms = dp_norms.reshape(out_shape)

    if init_type is not InitType.NONE:
        return homographies, homographies_guess, iterations, residuals, dp_norms
    else:
        return homographies, iterations, residuals, dp_norms


def _process_single_pattern_reversed(
    idx,
    get_pat,
    init_type,
    init_subset_slice,
    t_const_init,
    t_const_fmt,
    X_fmt,
    Y_fmt,
    x_fmt,
    y_fmt,
    T_const_spline,
    xi,
    h0,
    max_iter,
    conv_tol,
    pc_xo,
    perspective_regularization,
):
    """Process a single pattern under the reversed convention.

    Recomputes spline / gradient / Jacobian / Hessian / Cholesky for this
    pattern (which is the new reference), then runs IC-GN against the
    constant target spline.
    """
    new_R = get_pat(idx)

    savepat = True
    if savepat:
        plt.imsave(f'debug/pat_reversed/ref_pattern_{idx}_cpu.jpg', new_R, cmap='Greys_r')

    ### Build spline + gradients of the new reference at xi ###
    H_pat, W_pat = new_R.shape
    h_centre = (W_pat // 2, H_pat // 2)
    x_coords = np.arange(W_pat) - h_centre[0]
    y_coords = np.arange(H_pat) - h_centre[1]
    new_R_spline = interpolate.RectBivariateSpline(x_coords, y_coords, new_R.T, kx=5, ky=5)

    GRx = new_R_spline(xi[0], xi[1], dx=1, dy=0, grid=False)
    GRy = new_R_spline(xi[0], xi[1], dx=0, dy=1, grid=False)
    GR = np.vstack((GRx, GRy)).reshape(2, 1, -1).transpose(1, 0, 2)  # 2x1xN

    ### Build r and r_zmsv (reference subset values) ###
    r = new_R_spline(xi[0], xi[1], grid=False).flatten()
    r_zmsv = np.sqrt(((r - r.mean()) ** 2).sum())
    if r_zmsv == 0:
        # Degenerate (flat) pattern — bail out with a zero homography.
        return np.zeros(8), np.zeros(8), 0, 0.0, 0.0
    r = (r - r.mean()) / r_zmsv

    ### Jacobian of the shape function (depends only on xi) ###
    _1 = np.ones(xi.shape[1])
    _0 = np.zeros(xi.shape[1])
    out0 = np.array([[xi[0], xi[1], _1, _0, _0, _0, -xi[0] ** 2, -xi[1] * xi[0]]])
    out1 = np.array([[_0, _0, _0, xi[0], xi[1], _1, -xi[0] * xi[1], -xi[1] ** 2]])
    Jac = np.vstack((out0, out1))  # 2x8xN

    NablaR_dot_Jac = np.einsum("ilk,ljk->ijk", GR, Jac)[0]  # 8xN
    H = 2 / r_zmsv ** 2 * NablaR_dot_Jac.dot(NablaR_dot_Jac.T)

    if perspective_regularization and perspective_regularization > 0:
        factor = 1.0 + float(perspective_regularization)
        H[6, 6] *= factor
        H[7, 7] *= factor

    cho_params = linalg.cho_factor(H)

    ### Initial guess ###
    if init_type == InitType.NONE:
        h = np.zeros(8, dtype=float)
    else:
        measurement = _initial_guess_run_reversed(
            new_R, init_subset_slice,
            t_const_init, t_const_fmt,
            X_fmt, Y_fmt, x_fmt, y_fmt,
            idx,
        )
        if init_type == InitType.FULL:
            if pc_xo is None:
                raise ValueError(
                    "init_type='full' requires the pattern-center vector "
                    "pc_xo=(x01, x02, DD).  Pass it to optimize_reversed(..., pc_xo=...) "
                    "or use init_type='partial' / 'none'."
                )
            h = conversions.xyt2h(measurement, pc_xo)
        else:
            h = conversions.xyt2h_partial(measurement)

    initial_guess = h.copy()

    ### Run IC-GN against the constant target spline ###
    h, num_iter, residual, dpn = _optimize_run_reversed(
        T_const_spline,
        idx,
        h,
        r,
        r_zmsv,
        xi,
        NablaR_dot_Jac,
        cho_params,
        max_iter=max_iter,
        conv_tol=conv_tol,
        return_full=False,
    )

    return h, initial_guess, num_iter, residual, dpn


def _optimize_run_reversed(
    T_const_spline,
    idx: int,
    h: np.ndarray,
    r: np.ndarray,
    r_zmsv: float,
    xi: np.ndarray,
    NablaR_dot_Jac: np.ndarray,
    cho_params: tuple,
    max_iter: int = 50,
    conv_tol: float = 1e-3,
    return_full: bool = False,
) -> tuple:
    """IC-GN inner loop, reversed roles.

    The constant pattern's spline ``T_const_spline`` is sampled at the warped
    subset coordinates each iteration; the per-pattern reference values ``r``
    and gradient block ``NablaR_dot_Jac`` were precomputed by
    ``_process_single_pattern_reversed``.

    Identical structure to ``get_homography_cpu.optimize_run`` from line 814
    onwards — only the source of ``T_spline`` differs.
    """
    num_iter = 0
    norms = []
    residuals = []
    while num_iter < max_iter:
        num_iter += 1
        t_deformed = warp.deform(xi, T_const_spline, h)
        t_p1, t_p99 = np.percentile(t_deformed, [1, 99])
        t_deformed = np.clip(t_deformed, t_p1, t_p99)
        t_mean = t_deformed.mean()
        t_zmsv = np.sqrt(((t_deformed - t_mean) ** 2).sum())
        if t_zmsv > 0:
            t_deformed = (t_deformed - t_mean) / t_zmsv
        e = r - t_deformed
        residuals.append(np.abs(e).mean())
        dC_IC_ZNSSD = 2 / r_zmsv * np.matmul(e, NablaR_dot_Jac.T)
        dp = linalg.cho_solve(cho_params, -dC_IC_ZNSSD.reshape(-1, 1))[:, 0]
        norm = dp_norm(dp, xi)
        Wp = warp.W(h)
        Wdp = warp.W(dp)
        Wpdp = np.matmul(Wp, np.linalg.inv(Wdp))
        h = ((Wpdp / Wpdp[2, 2]) - np.eye(3)).reshape(9)[:8]
        norms.append(norm)
        if norm < conv_tol:
            break

    if return_full:
        return h, num_iter, residuals, norms
    else:
        return h, num_iter, residuals[-1], norms[-1]


def _initial_guess_run_reversed(
    new_R: np.ndarray,
    init_subset_slice: tuple,
    t_const_init: np.ndarray,
    t_const_fmt: np.ndarray,
    X_fmt: np.ndarray,
    Y_fmt: np.ndarray,
    x_fmt: np.ndarray,
    y_fmt: np.ndarray,
    idx: int,
) -> np.ndarray:
    """FMT-FCC initial guess with reference / target roles swapped.

    The per-pattern image ``new_R`` plays the role that ``R`` played in the
    original ``initial_guess_run`` (i.e. the "r" side of the cross-correlation).
    The constant pattern's precomputed FMT pieces (``t_const_init``,
    ``t_const_fmt``) play the role that the per-pattern target played in the
    original.  With this swap the formula
    ``measurement = [-shift[1], -shift[0], -theta]`` correctly encodes the
    rigid transform from new reference → constant target, which is exactly
    what the reversed IC-GN expects as its initial homography.
    """
    h0 = (new_R.shape[1] // 2, new_R.shape[0] // 2)
    new_R_init = window_and_normalize_new(
        new_R[init_subset_slice[0], init_subset_slice[1]], alpha=0.2
    )

    savepat = True
    if savepat:
        plt.imsave(f'debug/pat_reversed/ref_pattern_{idx}_init_guess_cpu.png',
                   new_R_init, cmap='Greys_r')

    new_R_fft = np.fft.fftshift(np.fft.fft2(new_R_init))
    new_R_fmt, _ = FMT(new_R_fft, X_fmt, Y_fmt, x_fmt, y_fmt)

    cc = signal.fftconvolve(new_R_fmt, t_const_fmt[::-1], mode="same").real
    theta = (np.argmax(cc) - len(cc) / 2) * np.pi / len(cc)

    h_init = conversions.xyt2h_partial(np.array([[0, 0, -theta]]))[0]
    t_const_init_rot = warp.deform_image(t_const_init, h_init, h0)

    cc = signal.fftconvolve(new_R_init, t_const_init_rot[::-1, ::-1], mode="same").real
    shift = _select_peak_2d(
        cc,
        max_shift=25.0,
        smooth_sigma=3.0,
        integration_radius=8,
        neighbourhood=5,
    )
    measurement = np.array([[-shift[1], -shift[0], -theta]])
    return measurement


def invert_homographies(h_arr: np.ndarray) -> np.ndarray:
    """Invert each 8-parameter homography in an (..., 8) array.

    Reversed-role IC-GN returns h that warps new_ref → new_target
    (= former target → former reference).  Inverting gives an h in the
    standard convention used everywhere else in this codebase
    (former reference → former target), so downstream PC drift correction
    and ``conversions.h2F`` / ``F2strain`` see the same convention as
    ``get_homography_cpu.optimize`` produces.

    NaN rows pass through as NaN; singular warps become NaN.
    """
    arr = np.asarray(h_arr, dtype=float)
    orig_shape = arr.shape
    flat = arr.reshape(-1, 8)
    out = np.empty_like(flat)
    I3 = np.eye(3)
    for i, h in enumerate(flat):
        if np.any(np.isnan(h)):
            out[i] = np.nan
            continue
        W = np.concatenate([h, [0.0]]).reshape(3, 3) + I3
        try:
            W_inv = np.linalg.inv(W)
            if W_inv[2, 2] == 0:
                raise np.linalg.LinAlgError("W_inv[2,2] == 0")
            W_inv = W_inv / W_inv[2, 2]
            out[i] = (W_inv - I3).flatten()[:8]
        except np.linalg.LinAlgError:
            out[i] = np.nan
    return out.reshape(orig_shape)


# Drop-in alias so callers that do `import get_homography_cpu_reversed as core`
# can call `core.optimize(...)` exactly the same way as the original module.
optimize = optimize_reversed

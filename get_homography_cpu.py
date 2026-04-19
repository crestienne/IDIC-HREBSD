import os
from enum import Enum
from typing import Union, Callable
import contextlib

import numpy as np
from scipy import linalg, interpolate, signal
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm
import joblib
from joblib import Parallel, delayed

import warp
import conversions
import Data
import matplotlib.pyplot as plt
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PatternSimulation.SimPatGen import patternSimulation


# Type hints
PATS = Union[Data.UP2, np.ndarray]
ARRAY = Union[np.ndarray, list, tuple]


# Make a Enum class for the different types of homography initialization
class InitType(Enum):
    NONE: str = "none"
    FULL: str = "full"
    PARTIAL: str = "partial"



# Context manager to patch joblib to report into tqdm progress bar given as argument
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


### Functions for the inverse composition gauss-newton algorithm


def compare_gradients(
    real_pat: np.ndarray,
    sim_pat: np.ndarray,
    crop_fraction: float = 0.9,
    mask: np.ndarray = None,
    save_path: str = "debug/gradient_comparison.png",
) -> None:
    """Compare the spline gradients of a real vs simulated reference pattern.

    Computes GRx and GRy for both patterns using the same spline/crop/mask
    settings used inside optimize(), then saves a side-by-side figure.

    Args:
        real_pat:      Preprocessed real reference pattern, shape (H, W).
        sim_pat:       Preprocessed simulated reference pattern, same shape.
        crop_fraction: Crop fraction used in optimize() (default 0.9).
        mask:          Boolean mask array (H, W) — same one passed to optimize().
        save_path:     Where to save the comparison figure.
    """
    assert real_pat.shape == sim_pat.shape, "Patterns must have the same shape"
    H, W = real_pat.shape
    h0 = (W // 2, H // 2)

    x = np.arange(W) - h0[0]
    y = np.arange(H) - h0[1]

    crop_row = int(H * (1 - crop_fraction) / 2)
    crop_col = int(W * (1 - crop_fraction) / 2)
    subset_slice = (slice(crop_row, -crop_row), slice(crop_col, -crop_col))

    X, Y = np.meshgrid(x, y, indexing="xy")
    xi = np.array([X[subset_slice].flatten(), Y[subset_slice].flatten()])
    subset_shape = X[subset_slice].shape

    valid = None
    if mask is not None:
        valid = mask[subset_slice].flatten()
        xi = xi[:, valid]

    def to_2d(arr):
        if valid is None:
            return arr.reshape(subset_shape)
        img = np.full(subset_shape[0] * subset_shape[1], np.nan)
        img[valid] = arr
        return img.reshape(subset_shape)

    def grad_pair(pat):
        spline = interpolate.RectBivariateSpline(x, y, pat.T, kx=5, ky=5)
        gx = spline(xi[0], xi[1], dx=1, dy=0, grid=False)
        gy = spline(xi[0], xi[1], dx=0, dy=1, grid=False)
        return to_2d(gx), to_2d(gy), to_2d(np.sqrt(gx**2 + gy**2))

    real_gx,  real_gy,  real_mag  = grad_pair(real_pat)
    sim_gx,   sim_gy,   sim_mag   = grad_pair(sim_pat)

    # shared colour limits per column
    vmax_x   = np.nanpercentile(np.abs(np.concatenate([real_gx[~np.isnan(real_gx)],  sim_gx[~np.isnan(sim_gx)]])),  98)
    vmax_y   = np.nanpercentile(np.abs(np.concatenate([real_gy[~np.isnan(real_gy)],  sim_gy[~np.isnan(sim_gy)]])),  98)
    vmax_mag = np.nanpercentile(np.concatenate([real_mag[~np.isnan(real_mag)], sim_mag[~np.isnan(sim_mag)]]), 98)

    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    titles_col = ["Real", "Simulated", "Difference"]
    row_labels  = ["Gradient X", "Gradient Y", "Gradient Magnitude"]

    data = [
        (real_gx,  sim_gx,  "RdBu",  vmax_x,   True),
        (real_gy,  sim_gy,  "RdBu",  vmax_y,   True),
        (real_mag, sim_mag, "inferno", vmax_mag, False),
    ]

    for row, (r, s, cmap, vmax, symmetric) in enumerate(data):
        vmin = -vmax if symmetric else 0
        diff = r - s
        diff_lim = np.nanpercentile(np.abs(diff[~np.isnan(diff)]), 98)

        for col, (img, vlim_min, vlim_max, cm) in enumerate([
            (r,    vmin,      vmax,      cmap),
            (s,    vmin,      vmax,      cmap),
            (diff, -diff_lim, diff_lim,  "coolwarm"),
        ]):
            ax = axes[row, col]
            im = ax.imshow(img, cmap=cm, vmin=vlim_min, vmax=vlim_max)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis("off")
            if row == 0:
                ax.set_title(titles_col[col], fontsize=12, fontweight="bold")
        axes[row, 0].set_ylabel(row_labels[row], fontsize=11)

    fig.suptitle("Gradient comparison: real vs simulated reference pattern", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gradient comparison saved to: {save_path}")


def simulate_reference_pattern(
    master_pattern_path: str,
    euler_angles: np.ndarray,
    PC: tuple,
    patshape: tuple,
    tilt_deg: float = 70.0,
    pat_obj: "Data.UP2" = None,
    high_pass_sigma_override: float = None,
) -> np.ndarray:
    """Simulate a single EBSD reference pattern using EBSDtorch / SimPatGen.

    Args:
        master_pattern_path (str): Path to the .h5 master pattern file.
        euler_angles (np.ndarray): Bunge ZXZ Euler angles in radians, shape (3,).
        PC (tuple): Pattern center (xstar, ystar, zstar) in EDAX/TSL convention.
        patshape (tuple): Detector shape in pixels (rows, cols).
        tilt_deg (float): Primary sample tilt in degrees (default 70).
        pat_obj (Data.UP2, optional): If provided, applies the same preprocessing
            pipeline (filters, CLAHE) as real patterns.  The mask is NOT applied
            to the simulated pattern — masked pixels in the real pattern correspond
            to detector artefacts that have no equivalent in the simulation, so
            zero-filling them creates a spurious centre-region intensity mismatch.
        high_pass_sigma_override (float, optional): If set, temporarily overrides
            pat_obj.high_pass_sigma when processing the simulated pattern.  Use a
            higher value (e.g. 25-30) to remove excess low-frequency content that
            the simulation produces relative to the experiment.

    Returns:
        np.ndarray: Simulated pattern as a float32 array, shape (rows, cols),
                    normalised to [0, 1].
    """
    sim = patternSimulation()
    sim.detector_height = patshape[0]
    sim.detector_width  = patshape[1]
    sim.det_shape       = patshape
    sim.detector_tilt_deg = tilt_deg

    sim.mastersetup(master_pattern_path)
    # EBSDtorch uses Bruker convention: pcy = 1 - ystar (EDAX/TSL flips y)
    pc_bruker = (PC[0], 1.0 - PC[1], PC[2])
    print(f'PC (EDAX):   {PC}')
    print(f'PC (Bruker): {pc_bruker}')
    sim.EandPCSet(euler_angles, pc_bruker)

    with __import__("torch").no_grad():
        pats = sim.GenPattern()

    pat = pats[0].reshape(patshape).cpu().numpy().astype(np.float32)
    pat_min, pat_max = pat.min(), pat.max()
    if pat_max > pat_min:
        pat = (pat - pat_min) / (pat_max - pat_min)

    if pat_obj is not None:
        # Process without mask: masked pixels in the real pattern are detector
        # artefacts with no physical equivalent in the simulation.  Zero-filling
        # them via masked_gaussian creates a boundary halo near the mask edge
        # that produces a spurious centre-region intensity mismatch.
        orig_mask_type = pat_obj.mask_type
        orig_hp_sigma  = pat_obj.high_pass_sigma
        pat_obj.mask_type = None
        if high_pass_sigma_override is not None:
            pat_obj.high_pass_sigma = high_pass_sigma_override
        try:
            pat = pat_obj.process_pattern(pat)
        finally:
            pat_obj.mask_type      = orig_mask_type
            pat_obj.high_pass_sigma = orig_hp_sigma

    return pat


def optimize(
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
    sim_high_pass_sigma: float = None,
    progress_callback: Callable = None,
    debug_gradients: bool = False,
) -> np.ndarray:
    """Routine for running the inverse composition gauss-newton algorithm.

    Args:
        pats (Data.UP2 or np.ndarray): The patterns to optimize. If array, last two dimensions should be the pattern. Shape is (..., H, W).
        x0 (array-like): The coordinate of the reference pattern. Can be an integer, list, tuple, or np.ndarray.
        init_type (InitType): The type of initial guess to use. Can be "none", "full", or "partial".
        crop_fraction (float): The fraction of the pattern to use for the homography optimization. Must be between 0 and 1.
        max_iter (int): The maximum number of iterations to run.
        conv_tol (float): The convergence tolerance.
        verbose (bool): Whether to print progress messages.

    Returns:
        homographies (np.ndarray): The optimized homography parameters. Shape matches pattern input.
        iterations (np.ndarray): The number of iterations taken to converge for each pattern. Shape matches pattern input.
        residuals (np.ndarray): The final residuals for each pattern. Shape matches pattern input.
        dp_norms (np.ndarray): The final deformation increment norms for each pattern. Shape matches pattern input.
    """

    ### Prepare the inputs ###
    # Check the crop fraction
    if crop_fraction <= 0 or crop_fraction >= 1:
        raise ValueError("Crop fraction must be between 0 and 1.")
    
    # Check convergence parameters
    if max_iter <= 0:
        raise ValueError("Maximum number of iterations must be greater than 0.")
    if conv_tol <= 0:
        raise ValueError("Convergence tolerance must be greater than 0.")

    # Check the init type, must be an instance of the InitType enum or a string matching the values of the enum
    if not isinstance(init_type, InitType):
        if type(init_type) == str:
            init_type = InitType(init_type.lower())
        else:
            raise TypeError(
                "init_type must be an instance of the InitType enum or a string matching the values of the enum."
            )

    # Check the number of jobs
    if n_jobs == -1:
        n_jobs = os.cpu_count() - 1

    # Prepare the patterns
    if type(pats) == Data.UP2:
        if roi_slice is not None:
            if scan_shape is None:
                raise ValueError(
                    "roi_slice requires scan_shape=(nrows, ncols) — the grid dimensions of the full scan. "
                    "Pass it as scan_shape=ang_data.shape to optimize()."
                )
            roi_nrows = roi_slice[0].stop - roi_slice[0].start
            roi_ncols = roi_slice[1].stop - roi_slice[1].start
            N = roi_nrows * roi_ncols
            out_shape = (roi_nrows, roi_ncols)   # results reshaped to (roi_rows, roi_cols)
            roi_indices = roi_indices_from_rect(roi_slice, scan_shape)

        else:
            roi_indices = None
            N = pats.nPatterns
            out_shape = (pats.nPatterns,)
        get_pat = lambda idx: pats.read_pattern(idx, process=True) #removed the process=True argument here
        patshape = pats.patshape
    elif type(pats) == np.ndarray:
        N = np.prod(pats.shape[:-2])
        out_shape = pats.shape[:-2]
        patshape = pats.shape[-2:]
        pats = pats.reshape(-1, pats.shape[-2], pats.shape[-1])
        get_pat = lambda idx: pats[idx]
    else:
        raise TypeError("pats must be a Data.UP2 object or a numpy array.")

    h0 = (patshape[1] // 2, patshape[0] // 2)
    crop_row = int(patshape[0] * (1 - crop_fraction) / 2)
    crop_col = int(patshape[1] * (1 - crop_fraction) / 2)
    subset_slice = (slice(crop_row, -crop_row), slice(crop_col, -crop_col)) #(y, x format)


    ### Reference precompute ###
    # Get the reference image
    if use_simulated_reference:
        if master_pattern_path is None or euler_angles_ref is None or pc_ref is None:
            raise ValueError(
                "use_simulated_reference=True requires master_pattern_path, "
                "euler_angles_ref, and pc_ref."
            )
        R = simulate_reference_pattern(
            master_pattern_path=master_pattern_path,
            euler_angles=euler_angles_ref,
            PC=pc_ref,
            patshape=patshape,
            tilt_deg=tilt_deg,
            pat_obj=pats if isinstance(pats, Data.UP2) else None,
            high_pass_sigma_override=sim_high_pass_sigma,
        )
        print(f"Using simulated reference pattern (shape {R.shape})")
        if debug_gradients:
            real_R = get_pat(x0)
            compare_gradients(
                real_pat=real_R,
                sim_pat=R,
                crop_fraction=crop_fraction,
                mask=mask,
                save_path="debug/gradient_comparison.png",
            )
    else:
        R = get_pat(x0)
    #add a small guassian blur to the reference pattern to smooth out interpolation artifacts and make the optimization landscape smoother, which can help with convergence
    #R = gaussian_filter(R, sigma= 0.8)

    print('the shape of the reference pattern is:', R.shape)

    
    # # Get coordinates
    x = np.arange(R.shape[1]) - h0[0] 
    y = np.arange(R.shape[0]) - h0[1]

    # # test: use pixel-center convention consistently
    # x = (np.arange(R.shape[1]) + 0.5) - h0[0]
    # y = (np.arange(R.shape[0]) + 0.5) - h0[1]

    X, Y = np.meshgrid(x, y, indexing="xy")

    #changing
    xi = np.array([X[subset_slice].flatten(), Y[subset_slice].flatten()]) #(y,x) ordering
    subset_shape = X[subset_slice].shape

    # Apply mask: exclude pixels that were zeroed out during preprocessing
    valid = None
    if mask is not None:
        valid = mask[subset_slice].flatten()
        xi = xi[:, valid]
        print(f"Mask applied: {valid.sum()} / {valid.size} subset pixels used ({100*valid.mean():.1f}%)")

    def to_2d(arr):
        """Reconstruct a 2D subset image from a (possibly masked) 1D array."""
        if valid is None:
            return arr.reshape(subset_shape)
        img = np.full(subset_shape[0] * subset_shape[1], np.nan)
        img[valid] = arr
        return img.reshape(subset_shape)

    # Compute the intensity gradients of the subset
    ref_spline = interpolate.RectBivariateSpline(x, y, R.T, kx=5, ky=5) #(y, x) ordering
    GRx = ref_spline(xi[0], xi[1], dx=1, dy=0, grid=False)
    GRy = ref_spline(xi[0], xi[1], dx=0, dy=1, grid=False)
    # GRy, GRx = np.gradient(R[subset_slice], axis=(0, 1))


    GR = np.vstack((GRx, GRy)).reshape(2, 1, -1).transpose(1, 0, 2)  # 2x1xN


    r = ref_spline(xi[0], xi[1], grid=False).flatten() #(y, x) ordering
    r_zmsv = np.sqrt(((r - r.mean()) ** 2).sum())
    r = (r - r.mean()) / r_zmsv

    # compute gradient magnitude
    grad_mag = np.sqrt(GRx**2 + GRy**2)

    if debug_gradients:
        os.makedirs("debug", exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        for a in ax.ravel():
            a.axis("off")
        ax[0].imshow(to_2d(GRx), cmap="Greys_r")
        ax[0].set_title("Gradient (x)")
        ax[1].imshow(to_2d(GRy), cmap="Greys_r")
        ax[1].set_title("Gradient (y)")
        plt.tight_layout()
        plt.savefig("debug/gradients_cpu.jpg")
        plt.close()

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        for a in ax.ravel():
            a.axis("off")
        ax[0].imshow(to_2d(GRx), cmap="RdBu")
        ax[0].set_title("Gradient X", fontweight="bold")
        ax[1].imshow(to_2d(GRy), cmap="RdBu")
        ax[1].set_title("Gradient Y", fontweight="bold")
        ax[2].imshow(to_2d(grad_mag), cmap="inferno")
        ax[2].set_title("Gradient Magnitude", fontweight="bold")
        plt.tight_layout()
        plt.savefig("debug/gradients_silicon.jpg")
        plt.close()

    # Compute the jacobian of the shape function
    _1 = np.ones(xi.shape[1])
    _0 = np.zeros(xi.shape[1])
    out0 = np.array([[xi[0], xi[1], _1, _0, _0, _0, -xi[0] ** 2, -xi[1] * xi[0]]])
    out1 = np.array([[_0, _0, _0, xi[0], xi[1], _1, -xi[0] * xi[1], -xi[1] ** 2]])
    Jac = np.vstack((out0, out1))  # 2x8xN
    #print(f"Jacobian - Min: {Jac.min():.5f}, Max: {Jac.max():.5f}, Mean: {Jac.mean():.5f}, Shape: {Jac.shape}")

    # Multiply the gradients by the jacobian
    NablaR_dot_Jac = np.einsum("ilk,ljk->ijk", GR, Jac)[0]  # 1x8xN -> 8xN
    H = 2 / r_zmsv**2 * NablaR_dot_Jac.dot(NablaR_dot_Jac.T)



    # Compute the Cholesky decomposition
    cho_params = linalg.cho_factor(H)

    # Store the precomputed values
    del GR, GRx, GRy, Jac, H, X, Y, ref_spline

    ### Create debug folder to save patterns

    debug_dir = os.path.join("debug", "pat")
    os.makedirs(debug_dir, exist_ok=True)

    for name in os.listdir(debug_dir):
        p = os.path.join(debug_dir, name)
        if os.path.isfile(p):
            os.remove(p)

    #### Precompute the FMT-FCC initial guess ###
    if init_type is not InitType.NONE:
        _s = 2 ** (min(patshape).bit_length() - 1)
        row_start = (patshape[0] - _s) // 2
        col_start = (patshape[1] - _s) // 2
        init_guess_subset_slice = (
            slice(row_start, row_start + _s), 
            slice(col_start, col_start + _s),
        )


    
        # Get the FMT-FCC initial guess precomputed items
        r_init = window_and_normalize_new(R[init_guess_subset_slice])
  
        # Get the dimensions of the image
        height, width = r_init.shape
        # Create a mesh grid of log-polar coordinates
        theta = np.linspace(0, np.pi, int(height), endpoint=False)
        radius = np.linspace(0, height / 2, int(height + 1), endpoint=False)[1:]
        radius_grid, theta_grid = np.meshgrid(radius, theta, indexing="ij")
        radius_grid = radius_grid.flatten()
        theta_grid = theta_grid.flatten()
        # Convert log-polar coordinates to Cartesian coordinates
        x_fmt = 2 ** (np.log2(height) - 1) + radius_grid * np.cos(theta_grid)
        y_fmt = 2 ** (np.log2(height) - 1) - radius_grid * np.sin(theta_grid)
        # Create a mesh grid of Cartesian coordinates
        X_fmt = np.arange(width)
        Y_fmt = np.arange(height)
        # FFT the reference and get the signal
        r_fft = np.fft.fftshift(np.fft.fft2(r_init))
        r_fmt, _ = FMT(r_fft, X_fmt, Y_fmt, x_fmt, y_fmt)
    idx_list = roi_indices if roi_indices is not None else range(N)
    ### Run the optimization in parallel ###
    if verbose:
        with tqdm_joblib(tqdm(total=N, desc="Patterns optimized")) as progress_bar:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_process_single_pattern)(
                    idx,
                    get_pat,
                    init_type,
                    init_guess_subset_slice if init_type is not InitType.NONE else None,
                    r_init if init_type is not InitType.NONE else None,
                    r_fmt if init_type is not InitType.NONE else None,
                    X_fmt if init_type is not InitType.NONE else None,
                    Y_fmt if init_type is not InitType.NONE else None,
                    x_fmt if init_type is not InitType.NONE else None,
                    y_fmt if init_type is not InitType.NONE else None,
                    r,
                    r_zmsv,
                    xi,
                    NablaR_dot_Jac,
                    cho_params,
                    h0,
                    max_iter,
                    conv_tol,
                )
                for idx in idx_list
            )
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_single_pattern)(
                idx,
                get_pat,
                init_type,
                init_guess_subset_slice,
                r_init if init_type is not InitType.NONE else None,
                r_fmt if init_type is not InitType.NONE else None,
                X_fmt if init_type is not InitType.NONE else None,
                Y_fmt if init_type is not InitType.NONE else None,
                x_fmt if init_type is not InitType.NONE else None,
                y_fmt if init_type is not InitType.NONE else None,
                r,
                r_zmsv,
                xi,
                NablaR_dot_Jac,
                cho_params,
                h0,
                max_iter,
                conv_tol,
            )
            for idx in idx_list
        )

    # Unpack results
    homographies = np.zeros((N, 8), dtype=float)
    homographies_guess = np.zeros((N, 8), dtype=float)
    iterations = np.zeros(N, dtype=int)
    residuals = np.zeros(N, dtype=float)
    dp_norms = np.zeros(N, dtype=float)

    for idx, (h, p_guess, num_iter, residual, dp_norm) in enumerate(results):
        homographies[idx] = h
        homographies_guess[idx] = p_guess
        iterations[idx] = num_iter
        residuals[idx] = float(residual)
        dp_norms[idx] = float(dp_norm)
        if progress_callback is not None:
            progress_callback(idx + 1, N)

    # Reshape the results to match the input pattern shape
    homographies = homographies.reshape(out_shape + (8,))
    homographies_guess = homographies_guess.reshape(out_shape + (8,))
    iterations = iterations.reshape(out_shape)
    residuals = residuals.reshape(out_shape)
    dp_norms = dp_norms.reshape(out_shape)

    # Return the results
    if init_type is not InitType.NONE:
        return homographies, homographies_guess, iterations, residuals, dp_norms
    else:
        return homographies, iterations, residuals, dp_norms


def _process_single_pattern(
    idx,
    get_pat,
    init_type,
    init_subset_slice,
    r_init,
    r_fmt,
    X_fmt,
    Y_fmt,
    x_fmt,
    y_fmt,
    r,
    r_zmsv,
    xi,
    NablaR_dot_Jac,
    cho_params,
    h0,
    max_iter,
    conv_tol,
):
    
    """Helper function to process a single pattern for parallel execution."""
    # Run initial guess
    if init_type == InitType.NONE:
         h = np.zeros(8, dtype=float)
         #h = 1e-5 * (2*np.random.rand(8) - 1)
         #print(f"Initial guess for pattern {idx}: {h}")
    else:
        measurement = initial_guess_run(
            get_pat, idx, init_subset_slice, r_init, r_fmt, X_fmt, Y_fmt, x_fmt, y_fmt
        )
        if init_type == InitType.FULL:
            h = conversions.xyt2h(measurement, h0)
        else:
            h = conversions.xyt2h_partial(measurement)

    initial_guess = h.copy()

    #clear the image saving folder

    # Run the optimization
    h, num_iter, residual, dp_norm = optimize_run(
        get_pat,
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

    return h, initial_guess, num_iter, residual, dp_norm

def roi_indices_from_rect(roi_slice, scan_shape):
    """
    Return a 1D array of flat pattern indices for a rectangular ROI.

    The .up2 file stores patterns in row-major order, so the flat index of
    pattern at scan position (row, col) is:  row * scan_ncols + col

    Parameters
    ----------
    roi_slice : tuple of slice
        (row_slice, col_slice) — row_slice selects scan rows (y),
                                  col_slice selects scan columns (x)
    scan_shape : tuple
        (scan_nrows, scan_ncols) — total rows and columns of the full scan grid
    """
    row_slice, col_slice = roi_slice
    scan_nrows, scan_ncols = scan_shape

    # Validate ROI is within the scan bounds
    if row_slice.stop > scan_nrows:
        raise ValueError(
            f"ROI row_slice {row_slice} exceeds scan rows ({scan_nrows})"
        )
    if col_slice.stop > scan_ncols:
        raise ValueError(
            f"ROI col_slice {col_slice} exceeds scan cols ({scan_ncols})"
        )

    roi_rows = np.arange(row_slice.start, row_slice.stop)   # e.g. [0,1,...,9]
    roi_cols = np.arange(col_slice.start, col_slice.stop)   # e.g. [0,1,...,131]

    # Build (n_roi_rows, n_roi_cols) grids of row and column indices
    row_grid, col_grid = np.meshgrid(roi_rows, roi_cols, indexing="ij")

    # Flat index in the .up2 file (row-major: row * scan_ncols + col)
    flat_indices = row_grid * scan_ncols + col_grid   # shape (n_roi_rows, n_roi_cols)

    print(
        f"ROI: rows {row_slice.start}–{row_slice.stop-1}, "
        f"cols {col_slice.start}–{col_slice.stop-1} "
        f"({len(roi_rows)} x {len(roi_cols)} = {flat_indices.size} patterns)\n"
        f"  Flat index range: {flat_indices.min()} – {flat_indices.max()} "
        f"(scan has {scan_nrows * scan_ncols} patterns total)"
    )

    return flat_indices.ravel()



def optimize_run(
    get_pat: Callable,
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
    """Run the homography optimization for a single point.

    Args:
        get_pat (Callable): Function to get the target image.
        idx (int): Index of the target image.
        h (np.ndarray): The initial guess of the homography.
        r (np.ndarray): The reference subset (the reference image flattened).
        r_zmsv (float): The zero mean, unit variance normalization factor for the reference subset.
        xi (np.ndarray): The coordinates of the reference subset.
        NablaR_dot_Jac (np.ndarray): The gradient of the reference subset.
        cho_params (tuple): The Cholesky decomposition parameters.
        max_iter (int): The maximum number of iterations to run.
        conv_tol (float): The convergence tolerance.
        return_full (bool): Whether to return the full optimization results.
    Returns:
        p (np.ndarray): The optimized homography parameters.
        num_iter (int): The number of iterations taken to converge.
        residuals (float or list): The final residuals for each iteration. If return_full is False, returns the final residual.
        norms (float or list): The final deformation increment norms for each iteration. If return_full is False, returns the final norm.
    """
    # Get the target image
    T = get_pat(idx)

    #add a small guassian blur to the target pattern to smooth out interpolation artifacts and make the optimization landscape smoother, which can help with convergence
    #T = gaussian_filter(T, sigma= 0.8)

    savepat = True
    if savepat:
        plt.imsave(f'debug/pat/target_pattern_{idx}_cpu.jpg', T, cmap='Greys_r')

    h0 = (T.shape[1] // 2, T.shape[0] // 2)
    #trying a different spline definition here
    # test: use pixel-center convention consistently
    # x = (np.arange(T.shape[1]) + 0.5) - h0[0]
    # y = (np.arange(T.shape[0]) + 0.5) - h0[1]

    x = np.arange(T.shape[1]) - h0[0]
    y = np.arange(T.shape[0]) - h0[1]
    T_spline = interpolate.RectBivariateSpline(x, y, T.T, kx=5, ky=5) #patterns read in (y, x) ordering

    # Run the optimization
    num_iter = 0
    norms = []
    residuals = []
    while num_iter < max_iter:
        # Warp the target subset
        num_iter += 1
        t_deformed = warp.deform(xi, T_spline, h)
        # Clip spline extrapolation outliers (5th-degree polynomials blow up outside
        # the image domain; this is especially important for zero-mean patterns where
        # even small out-of-bounds excursions can produce very large values that
        # dominate the ZMSV normalization and cause divergence).
        t_p1, t_p99 = np.percentile(t_deformed, [1, 99])
        t_deformed = np.clip(t_deformed, t_p1, t_p99)
        t_mean = t_deformed.mean()
        t_zmsv = np.sqrt(((t_deformed - t_mean) ** 2).sum())
        if t_zmsv > 0:
            t_deformed = (t_deformed - t_mean) / t_zmsv   # ZNSSD: each pattern normalised by its own variance
        # Compute the residuals
        e = r - t_deformed
        residuals.append(np.abs(e).mean())
        # Copmute the gradient of the correlation criterion
        dC_IC_ZNSSD = 2 / r_zmsv * np.matmul(e, NablaR_dot_Jac.T)  # 8x1
        # Find the deformation incriment, delta_p, by solving the linear system
        # H.dot(delta_p) = -dC_IC_ZNSSD using the Cholesky decomposition
        dp = linalg.cho_solve(cho_params, -dC_IC_ZNSSD.reshape(-1, 1))[:, 0]
        # Update the parameters
        norm = dp_norm(dp, xi)
        Wp = warp.W(h)
        Wdp = warp.W(dp)
        Wpdp = np.matmul(Wp, np.linalg.inv(Wdp))
        h = ((Wpdp / Wpdp[2, 2]) - np.eye(3)).reshape(9)[:8]
        # Store the update
        norms.append(norm)
        if norm < conv_tol:
            break

    if return_full:
        return h, num_iter, residuals, norms
    else:
        return h, num_iter, residuals[-1], norms[-1]


def dp_norm(dp, xi) -> float:
    """Compute the norm of the deformation increment.
    This is essentially a modified form of a homography magnitude.

    Args:
        dp (np.ndarray): The deformation increment. Shape is (8,).
        xi (np.ndarray): The subset coordinates. Shape is (2, N).

    Returns:
        float: The norm of the deformation increment."""
    xi1max = xi[0].max()
    xi2max = xi[1].max()
    ximax = np.array([xi1max, xi2max])
    dp_i0 = dp[0:2] * ximax
    dp_i1 = dp[3:5] * ximax
    dp_i2 = dp[6:8] * ximax
    out = np.sqrt(
        (dp_i0**2).sum()
        + (dp_i1**2).sum()
        + (dp_i2**2).sum()
        + (dp[2] ** 2 + dp[5] ** 2)
    )
    return out


### Functions for the global cross-correlation initial guess


def initial_guess_run(
    get_pat: Callable,
    idx: int,
    init_subset_slice: tuple[slice, slice],
    r_init: np.ndarray,
    r_fmt: np.ndarray,
    X_fmt: np.ndarray,
    Y_fmt: np.ndarray,
    x_fmt: np.ndarray,
    y_fmt: np.ndarray,
) -> np.ndarray:
    """Run the initial guess optimization for a single point."""
    # Get the target image
    T = get_pat(idx)

    h0 = (T.shape[1] // 2, T.shape[0] // 2)
    t_init = window_and_normalize_new(T[init_subset_slice[0], init_subset_slice[1]], alpha=0.2)

    savepat = True
    if savepat:
       plt.imsave(f'debug/pat/target_pattern_{idx}_init_guess_cpu.png', t_init, cmap='Greys_r')

    # Do the angle search first
    t_init_fft = np.fft.fftshift(np.fft.fft2(t_init))
    t_init_FMT, _ = FMT(t_init_fft, X_fmt, Y_fmt, x_fmt, y_fmt, )
    cc = signal.fftconvolve(r_fmt, t_init_FMT[::-1], mode="same").real
    theta = (np.argmax(cc) - len(cc) / 2) * np.pi / len(cc)
    # Apply the rotation
    h = conversions.xyt2h_partial(np.array([[0, 0, -theta]]))[0]
    t_init_rot = warp.deform_image(t_init, h, h0)
    # Do the translation search
    cc = signal.fftconvolve(r_init, t_init_rot[::-1, ::-1], mode="same").real
    shift = np.unravel_index(np.argmax(cc), cc.shape) - np.array(cc.shape) / 2
    # Store the homography
    measurement = np.array([[-shift[1], -shift[0], -theta]])
    

    return measurement


def Tukey_Hanning_window(sig, alpha=0.4, return_window=False):
    """Applies a Tukey-Hanning window to the input signal.
    Args:
        sig (np.ndarray): The input signal. The last two dimensions should be the image to be windowed. (N, M, H, W) or (N, H, W) or (H, W) for example.
        alpha (float): The alpha parameter of the Tukey-Hanning window.
    Returns:
        np.ndarray: The windowed signal."""
    if sig.ndim == 1:
        window = signal.windows.tukey(sig.shape[-1], alpha=alpha)
    else:
        window_row = signal.windows.tukey(sig.shape[-2], alpha=alpha)
        window_col = signal.windows.tukey(sig.shape[-1], alpha=alpha)
        window = np.outer(window_row, window_col)
        while sig.ndim > window.ndim:
            window = window[None, :]
    if return_window:
        return sig * window, window
    else:
        return sig * window


def window_and_normalize(images, alpha=0.4):
    """Applies a Tukey-Hanning window and normalizes the input images.
    Args:
        images (np.ndarray): The input images. The last two dimensions should be the image to be windowed. (N, M, H, W) or (N, H, W) or (H, W) for example.
        alpha (float): The alpha parameter of the Tukey-Hanning window.
    Returns:
        np.ndarray: The windowed and normalized images."""
    # Get axis to operate on
    if images.ndim >= 2:
        axis = (-2, -1)
    else:
        axis = -1
    # Apply the Tukey-Hanning window
    windowed, window = Tukey_Hanning_window(images, alpha, return_window=True)
    # Get the normalizing factors
    image_bar = images.mean(axis=axis)
    windowed_bar = (images * windowed).mean(axis=axis)
    bar = windowed_bar / image_bar
    del windowed, image_bar, windowed_bar
    while bar.ndim < images.ndim:
        bar = bar[..., None]
    # Window and normalize the image
    new_normalized_windowed = (images - bar) * window
    del window, bar
    variance = (new_normalized_windowed**2).sum(axis=axis) / (
        np.prod(images.shape[-2:]) - 1
    )
    while variance.ndim < images.ndim:
        variance = variance[..., None]
    out = new_normalized_windowed / np.sqrt(variance)
    return out

#C added new version of window and normalize that is ZNSSD style
def window_and_normalize_new(images, alpha=0.4):
    """
    Apply Tukey window and zero-mean, unit-variance normalization
    over the last two axes.
    """
    if images.ndim < 2:
        raise ValueError("images must be at least 2D")

    axis = (-2, -1)

    # get window only
    _, window = Tukey_Hanning_window(images, alpha, return_window=True)

    # apply window
    windowed = images * window

    # mean over windowed image
    mean = windowed.mean(axis=axis, keepdims=True)

    # zero-mean
    zm = windowed - mean

    # variance (ZNSSD-style, unbiased optional)
    num = np.prod(images.shape[-2:])
    var = (zm**2).sum(axis=axis, keepdims=True) / (num - 1)

    std = np.sqrt(var)
    std = np.where(std == 0, 1.0, std)

    return zm / std


def FMT(image, X, Y, x, y):
    """Fourier-Mellin Transform of an image in which polar resampling is applied first.
    Args:
        image (np.ndarray): The input image of shape (2**n, 2**n)
        X (np.ndarray): The x-coordinates of the input image. Should correspond to the x coordinate of the image.
        Y (np.ndarray): The y-coordinates of the input image. Should correspond to the y coordinate of the image.
        x (np.ndarray): The x-coordinates of the output image. Should correspond to the x coordinates of the polar image.
        y (np.ndarray): The y-coordinates of the output image. Should correspond to the y coordinates of the polar image.
    Returns:
        np.ndarray: The signal of the Fourier-Mellin Transform. (1D array of length 2**n)
    """

    spline = interpolate.RectBivariateSpline(X, Y, image.real, kx=2, ky=2)
    image_polar = np.abs(spline(x, y, grid=False).reshape(image.shape))
    sig = window_and_normalize(image_polar.mean(axis=1))
    return sig, image_polar


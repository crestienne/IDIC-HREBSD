"""
pc_homography_correction.py
===========================

PC / detector frame  (EDAX fractional)
    xstar  : measured from the LEFT  of the pattern, positive rightward
    ystar  : measured from the TOP   of the pattern, positive downward
    zstar  : detector distance / pattern_height,     positive toward sample

The sign with which a physical sample step maps onto a PC shift depends on
the scan convention (which direction x and y point on the sample).  Rather
than hard-coding signs into the conversion function, each ScanGrid carries
its own x_sign and y_sign so the conversion is always self-consistent:

    Δxstar =  x_sign * ΔX / (pattern_width  * pixel_size)
    Δystar =  y_sign * ΔY * cos(θ) / (pattern_height * pixel_size)
    Δzstar = -y_sign * ΔY * sin(θ) / (pattern_height * pixel_size)

where  θ = (90 − sample_tilt) + detector_tilt
"""

import numpy as np
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# ScanGrid dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScanGrid:
    """
    Physical scan coordinates plus the sign conventions needed for the PC
    conversion.

    Attributes
    ----------
    data        : ndarray (n_rows, n_cols, 2)  — [x_um, y_um] at each position
    x_sign      : +1 or -1  — sign for Δxstar  (+1 → positive x increases xstar,
                                                  -1 → positive x decreases xstar)
    y_sign      : +1 or -1  — sign for Δystar  (same logic, also flips Δzstar)
    convention  : str        — human-readable label
    """
    data:       np.ndarray
    x_sign:     int
    y_sign:     int
    convention: str


# ─────────────────────────────────────────────────────────────────────────────
# Supported conventions
# ─────────────────────────────────────────────────────────────────────────────

_CONVENTIONS = {
    #                          x_sign  y_sign   description
    "lower_right":     dict(x_sign=-1, y_sign=-1),  # origin lower-right, x←, y↑
    "direct_electron": dict(x_sign=-1, y_sign=+1),  # origin upper-right, x←, y↓
    "standard":      dict(x_sign=+1, y_sign=+1),  # origin upper-left,  x→, y↓
}


# ─────────────────────────────────────────────────────────────────────────────
# make_scan_grid
# ─────────────────────────────────────────────────────────────────────────────

def make_scan_grid(scan_shape, step_size_um, convention="standard"):
    """
    Build a ScanGrid of physical coordinates for each scan position (microns).

    The first pattern (array index [0, 0]) is always the origin (0, 0).
    The convention string controls what that origin corner is and which
    direction each axis points.

    Supported conventions
    ---------------------
    "standard"
        Origin : lower-right corner.  x increases leftward, y increases upward.
        grid[0,  0] = (0, 0)                               lower-right  ← origin
        grid[0, -1] = ((n_cols-1)*step, 0)                 lower-left
        grid[-1, 0] = (0, (n_rows-1)*step)                 upper-right
        grid[-1,-1] = ((n_cols-1)*step, (n_rows-1)*step)   upper-left

    "direct_electron"
        Origin : upper-right corner.  x increases leftward, y increases downward.
        grid[0,  0] = (0, 0)                               upper-right  ← origin
        grid[0, -1] = ((n_cols-1)*step, 0)                 upper-left
        grid[-1, 0] = (0, (n_rows-1)*step)                 lower-right
        grid[-1,-1] = ((n_cols-1)*step, (n_rows-1)*step)   lower-left

    Parameters
    ----------
    scan_shape   : (int, int)   (n_rows, n_cols)
    step_size_um : float        step size in microns
    convention   : str          one of the keys in _CONVENTIONS

    Returns
    -------
    ScanGrid
    """
    if convention not in _CONVENTIONS:
        raise ValueError(f"Unknown convention {convention!r}. "
                         f"Choose from {list(_CONVENTIONS)}")

    n_rows, n_cols = scan_shape
    col_coords = np.arange(n_cols) * step_size_um
    row_coords = np.arange(n_rows) * step_size_um
    xx, yy = np.meshgrid(col_coords, row_coords)
    data = np.stack([xx, yy], axis=-1)

    signs = _CONVENTIONS[convention]
    return ScanGrid(data=data, convention=convention, **signs)


# ─────────────────────────────────────────────────────────────────────────────
# scan_grid_to_pc_grid
# ─────────────────────────────────────────────────────────────────────────────

def scan_grid_to_pc_grid(scan_grid, pc_ref, patshape, pixel_size_um,
                         sample_tilt_deg, detector_tilt_deg):
    """
    Convert a ScanGrid to a per-position pattern centre (PC) grid.

    The sign conventions are read directly from scan_grid.x_sign and
    scan_grid.y_sign — no need to pass them separately.

    Physics
    -------
    A step Δy along the sample splits into two detector components via the
    effective tilt angle  θ = (90 − sample_tilt) + detector_tilt :

        parallel to detector face   →  cos(θ) * Δy   →  shifts ystar
        perpendicular to detector   →  sin(θ) * Δy   →  shifts zstar (changes DD)

    A step Δx (perpendicular to tilt axis) shifts only xstar.

    Conversion equations
        Δxstar =  x_sign * ΔX / (pattern_width  * pixel_size)
        Δystar =  y_sign * ΔY * cos(θ) / (pattern_height * pixel_size)
        Δzstar = -y_sign * ΔY * sin(θ) / (pattern_height * pixel_size)

    Parameters
    ----------
    scan_grid         : ScanGrid          from make_scan_grid
    pc_ref            : array-like (3,)   (xstar, ystar, zstar) at scan_grid[0, 0]
    patshape          : (int, int)        (pattern_height_px, pattern_width_px)
    pixel_size_um     : float             physical size of one detector pixel (µm)
    sample_tilt_deg   : float             sample tilt from horizontal (e.g. 70)
    detector_tilt_deg : float             detector tilt from vertical (e.g. 10)

    Returns
    -------
    pc_grid : ndarray (n_rows, n_cols, 3)
        pc_grid[r, c] = (xstar, ystar, zstar) at scan position (r, c)
    """
    pat_h, pat_w = patshape
    θ = np.radians((90 - sample_tilt_deg) + detector_tilt_deg)

    ΔX = scan_grid.data[..., 0]
    ΔY = scan_grid.data[..., 1]

    Δxstar =  scan_grid.x_sign * ΔX / (pat_w * pixel_size_um)
    Δystar =  scan_grid.y_sign * ΔY * np.cos(θ) / (pat_h * pixel_size_um)
    Δzstar =  -scan_grid.y_sign * ΔY * np.sin(θ) / (pat_h * pixel_size_um)

    xstar = pc_ref[0] + Δxstar
    ystar = pc_ref[1] + Δystar
    zstar = pc_ref[2] + Δzstar

    return np.stack([xstar, ystar, zstar], axis=-1)

# ─────────────────────────────────────────────────────────────────────────────
# Homography ↔ warp matrix conversions
# ─────────────────────────────────────────────────────────────────────────────

def h_to_warp(h):
    """
    Convert IC-GN 8-parameter homographies to 3×3 warp matrices.

        W = [[1+h11, h12,   h13],
             [h21,   1+h22, h23],
             [h31,   h32,    1 ]]

    Parameters
    ----------
    h : ndarray (..., 8)   [h11, h12, h13, h21, h22, h23, h31, h32]

    Returns
    -------
    W : ndarray (..., 3, 3)
    """
    W = np.zeros(h.shape[:-1] + (3, 3))
    W[..., 0, 0] = 1 + h[..., 0]
    W[..., 0, 1] =     h[..., 1]
    W[..., 0, 2] =     h[..., 2]
    W[..., 1, 0] =     h[..., 3]
    W[..., 1, 1] = 1 + h[..., 4]
    W[..., 1, 2] =     h[..., 5]
    W[..., 2, 0] =     h[..., 6]
    W[..., 2, 1] =     h[..., 7]
    W[..., 2, 2] = 1
    return W


def warp_to_h(W):
    """
    Convert 3×3 warp matrices back to IC-GN 8-parameter homographies.

    Parameters
    ----------
    W : ndarray (..., 3, 3)

    Returns
    -------
    h : ndarray (..., 8)   [h11, h12, h13, h21, h22, h23, h31, h32]
    """
    h = np.zeros(W.shape[:-2] + (8,))
    h[..., 0] = W[..., 0, 0] - 1
    h[..., 1] = W[..., 0, 1]
    h[..., 2] = W[..., 0, 2]
    h[..., 3] = W[..., 1, 0]
    h[..., 4] = W[..., 1, 1] - 1
    h[..., 5] = W[..., 1, 2]
    h[..., 6] = W[..., 2, 0]
    h[..., 7] = W[..., 2, 1]
    return h


# ─────────────────────────────────────────────────────────────────────────────
# Full correction pipeline
# ─────────────────────────────────────────────────────────────────────────────

def correct_homographies(h, scan_shape, step_size_um, pc_ref, patshape,
                         pixel_size_um, sample_tilt_deg, detector_tilt_deg,
                         convention="standard"):
    """
    Correct an array of homographies for pattern-centre drift across the scan.

    The measured homography at each scan position contains both the material
    deformation and a geometric contribution from the shifting PC.  This
    function removes the geometric part by building the per-position TS matrix
    (translation + scale) and pre-multiplying each warp by its inverse:

        W_corrected = TS_inv @ W_measured

    Pipeline
    --------
    1. h (N, 8) → W (n_rows, n_cols, 3, 3)        h_to_warp
    2. Build physical scan grid                     make_scan_grid
    3. Compute per-position PC                      scan_grid_to_pc_grid
    4. Δpc = pc_grid − pc_ref
    5. Build TS_inv matrices                        delta_pc_to_TS
    6. W_corrected = TS_inv @ W                     einsum
    7. W_corrected → h_corrected (N, 8)             warp_to_h

    Parameters
    ----------
    h                 : ndarray (N, 8)        measured homographies, row-major
    scan_shape        : (int, int)            (n_rows, n_cols)
    step_size_um      : float                 scan step size in microns
    pc_ref            : array-like (3,)       (xstar, ystar, zstar) at origin [0,0]
    patshape          : (int, int)            (pattern_height_px, pattern_width_px)
    pixel_size_um     : float                 detector pixel size in microns
    sample_tilt_deg   : float                 sample tilt from horizontal (e.g. 70)
    detector_tilt_deg : float                 detector tilt from vertical  (e.g. 10)
    convention        : str                   scan grid convention (default "standard")

    Returns
    -------
    h_corrected : ndarray (N, 8)
    TS_inv      : ndarray (n_rows, n_cols, 3, 3)
        The per-position correction matrices applied to each warp.
        At the scan origin [0,0] this is the 3×3 identity.
    """
    n_rows, n_cols = scan_shape
    N = n_rows * n_cols

    # 1. h → W, reshaped to (n_rows, n_cols, 3, 3)
    W = h_to_warp(h).reshape(n_rows, n_cols, 3, 3)

    # 2-4. scan grid → per-position PC shifts
    scan_grid = make_scan_grid(scan_shape, step_size_um, convention=convention)
    pc_grid   = scan_grid_to_pc_grid(scan_grid, pc_ref, patshape, pixel_size_um,
                                     sample_tilt_deg, detector_tilt_deg)
    Δpc = pc_grid - np.array(pc_ref)

    # 5. TS_inv per position
    TS_inv = delta_pc_to_TS(Δpc, pc_ref, patshape)   # (n_rows, n_cols, 3, 3)

    # ── sanity check: at the origin [0,0] Δpc=0 so TS_inv must be identity ──
    print("TS_inv at origin [0,0]:")
    print(np.round(TS_inv[0, 0], 6))
    print("(expected: 3×3 identity)")

    #print an example TS_inv matrix to check --- IGNORE ---
    print("Example TS_inv matrix at [2,2]:")
    print(np.round(TS_inv[1, 130], 6))

    W_corrected = TS_inv @ W
   #W_corrected = np.einsum('...ij,...jk->...ik', TS_inv, W)

    # 7. back to (N, 8)
    h_corrected = warp_to_h(W_corrected).reshape(N, 8)
    print(f"PC correction applied — convention: '{convention}'  "
          f"scan: {n_rows}×{n_cols}")
    return h_corrected, TS_inv


def delta_pc_to_TS(Δpc, pc_ref, patshape):
    """
    Convert a ΔPC shift (Δxstar, Δystar, Δzstar) to pixel-space shifts and
    build a per-position 3×3 translation matrix T.

    T encodes the shift of the pattern centre on the detector:

        T = [[1,  0,  Δx0],
             [0,  1,  Δy0],
             [0,  0,   1 ]]

    Parameters
    ----------
    Δpc      : ndarray (n_rows, n_cols, 3)  — (Δxstar, Δystar, Δzstar)
    pc_ref   : array-like (3,)              — (xstar, ystar, zstar) at origin
    patshape : (int, int)                   — (pattern_height_px, pattern_width_px)

    Returns
    -------
    T : ndarray (n_rows, n_cols, 3, 3)
    """
    pat_h, pat_w = patshape
    Δxstar, Δystar, Δzstar = Δpc[..., 0], Δpc[..., 1], Δpc[..., 2]

    # Reference PC in pixel units (h2F convention: measured from image centre)
    x01_ref = (0.5 - pc_ref[0]) * pat_w   # px, x offset of reference PC from centre
    x02_ref = (0.5 - pc_ref[1]) * pat_h   # px, y offset of reference PC from centre

    # PC shifts in pixel units (just the delta, no absolute offset)
    Δx0 = Δxstar * pat_w   # px
    Δy0 = Δystar * pat_h   # px

    # Scale factor: ratio of new detector distance to reference (dimensionless, ≈ 1)
    # Δzstar is fractional, pc_ref[2] is fractional → pure ratio, no unit issue
    alpha = (pc_ref[2] + Δzstar) / pc_ref[2]

    # Build the translation matrix T (n_rows, n_cols, 3, 3)
    n_rows, n_cols = Δpc.shape[:2]
    T = np.zeros((n_rows, n_cols, 3, 3))
    S = np.zeros((n_rows, n_cols, 3, 3))
    T[..., 0, 0] = 1       # identity diagonal
    T[..., 1, 1] = 1
    T[..., 2, 2] = 1
    T[..., 0, 2] = Δx0     # (1,3) position
    T[..., 1, 2] = Δy0     # (2,3) position

    S[..., 0, 0] = alpha   # scaling for perspective correction
    S[..., 1, 1] = alpha
    S[..., 2, 2] = 1
    S[..., 0, 2] = x01_ref * (1 - alpha)  # adjust for perspective shift
    S[..., 1, 2] = x02_ref * (1 - alpha)

    #multiply T by S to get the final homography correction matrix
    TS = np.einsum('...ij,...jk->...ik', S, T)
    #find the inverse of TS to get the correction that should be applied to the pattern
    TS_inv = np.linalg.inv(TS)
    #print the first TS_inv matrix to check --- IGNORE ---
    print("the shape of TS_inv is:", TS_inv.shape)

    return TS_inv



   

#─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ── inputs ────────────────────────────────────────────────────────────────
    scan_shape        = (10, 132)
    step_size_um      = 2.6
    pc_ref            = (0.6871, 0.8929, 1.06971)
    patshape          = (512, 512)
    pixel_size_um     = 30.0
    sample_tilt_deg   = 70.0
    detector_tilt_deg = 10.0

    # ── build both scan grids and their PC shifts ─────────────────────────────
    grid_std = make_scan_grid(scan_shape, step_size_um, convention="standard")
    grid_de  = make_scan_grid(scan_shape, step_size_um, convention="direct_electron")

    pc_std = scan_grid_to_pc_grid(grid_std, pc_ref, patshape, pixel_size_um,
                                   sample_tilt_deg, detector_tilt_deg)
    pc_de  = scan_grid_to_pc_grid(grid_de,  pc_ref, patshape, pixel_size_um,
                                   sample_tilt_deg, detector_tilt_deg)

    Δpc_std = pc_std - np.array(pc_ref)
    Δpc_de  = pc_de  - np.array(pc_ref)

    # ── plot helper ───────────────────────────────────────────────────────────
    def _plot(grid, Δpc, fig_title):
        y_up   = (grid.y_sign == -1)   # standard: y_sign=-1 means y increases up
        origin = "lower" if y_up else "upper"
        x_um   = grid.data[0, :, 0]
        y_um   = grid.data[:, 0, 1]
        extent = [x_um[0], x_um[-1], y_um[0], y_um[-1]]

        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        fig.suptitle(fig_title, fontsize=12)

        # row 0: physical scan coordinates
        for ax, data, label in zip(
            axes[0, :2],
            [grid.data[..., 0], grid.data[..., 1]],
            ["x  (µm, ← positive)", "y  (µm)"],
        ):
            im = ax.imshow(data, origin=origin, cmap="viridis", extent=extent)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(label)
            ax.set_xlabel("x (µm)")
            ax.set_ylabel("y (µm)")
            ax.invert_xaxis()

        axes[0, 2].axis("off")

        # row 1: PC shifts
        for ax, data, label in zip(
            axes[1],
            [Δpc[..., 0], Δpc[..., 1], Δpc[..., 2]],
            ["Δxstar", "Δystar", "Δzstar"],
        ):
            im = ax.imshow(data, origin=origin, cmap="coolwarm", extent=extent)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(label)
            ax.set_xlabel("x (µm)")
            ax.set_ylabel("y (µm)")
            ax.invert_xaxis()

        fig.tight_layout()

    _plot(grid_std, Δpc_std,
          f"Standard  (origin: lower-right, x←, y↑)  "
          f"x_sign={grid_std.x_sign}  y_sign={grid_std.y_sign}")
    _plot(grid_de,  Δpc_de,
          f"DirectElectron  (origin: upper-right, x←, y↓)  "
          f"x_sign={grid_de.x_sign}  y_sign={grid_de.y_sign}")

    plt.show()

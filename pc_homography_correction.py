"""
pc_homography_correction.py
===========================

Functions to account for the fact that the pattern center (PC) shifts as the electron beam steps across the sample.  That shift produces an apparent (geometric) homography even in a perfectly undeformed crystal.  The routines
here let you:

  1. Build a grid of physical scan coordinates (in EDAX scan framework).


  2. Compute the per-position PC (in the format expected by h2F).
  3. Derive the geometric homography that arises purely from the PC drift.
  4. Subtract that geometric contribution from measured homographies so that
     the remaining signal reflects only material deformation.

Coordinate conventions (matching the rest of the codebase)
-----------------------------------------------------------
* PC is stored in EDAX/TSL fractional units: (xstar, ystar, zstar).
* h2F / the IC-GN optimizer use "image-centre-relative" pixel coordinates:
      x01 = (0.5 - xstar ) * pattern_width    [pixels from image centre, x]
      x02 = (0.5 - ystar) * pattern_height   [pixels from image centre, y]
      DD  =  zstar * pattern_height           [detector distance in pixels]
  This is what the visualisation scripts pass to conversions.h2F().
* Homographies have 8 parameters [h11, h12, h13, h21, h22, h23, h31, h32]
  with shape function W = diag([1,1,1]) + h padded with a zero and reshaped.
* Scan indices: (row, col), with row increasing downward and col to the right.

Geometric homography derivation (brief)
----------------------------------------
With no deformation (F = I) the Kikuchi pattern at position (row, col) is a
scaled and translated version of the reference pattern.  In image-centre
coordinates xi the mapping is:

    xi_prime_x = x01_curr - scale * x01_ref  +  scale * xi_x
    xi_prime_y = x02_curr - scale * x02_ref  +  scale * xi_y

where  scale = DD_curr / DD_ref.

Comparing with the shape-function matrix this gives:
    h11 = h22 = scale - 1
    h13 = x01_curr - scale * x01_ref
    h23 = x02_curr - scale * x02_ref
    h12 = h21 = h31 = h32 = 0
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1.  Pattern grid
# ---------------------------------------------------------------------------

def make_pattern_grid(scan_shape: tuple, step_size_um: float):
    """
    Build integer index arrays and physical coordinate maps for every scan position. Map the scan grid (row, col) to physical coordinates (x_um, y_um) in the sample frame, assuming a simple orthogonal grid with a constant step size. 

    Note that the following functions defines the scan coordinates in the sample frame, 

    TODO: Need option to account for different step sizes in x and y, and/or a non-orthogonal grid (e.g. from stage calibration).  For now we assume a simple orthogonal grid with a constant step size.

    Args:
        scan_shape   : (nrows, ncols) – shape of the EBSD scan grid.
        step_size_um : Physical step size in microns (isotropic).

    Returns:
        row_idx  : (nrows, ncols) int array of row indices.
        col_idx  : (nrows, ncols) int array of column indices.
        x_um     : (nrows, ncols) float array of x positions in microns
                   (column direction, increases to the right).
        y_um     : (nrows, ncols) float array of y positions in microns
                   (row direction, increases downward).
    """
    row_idx, col_idx = np.indices(scan_shape)
    x_um = col_idx * step_size_um
    y_um = row_idx * step_size_um
    return row_idx, col_idx, x_um, y_um


# ---------------------------------------------------------------------------
# 2.  Per-position PC  (fractional EDAX units AND centred-pixel h2F format)
# ---------------------------------------------------------------------------

def fractional_to_h2f(PC_frac, patshape):
    """
    Convert a PC from Bruker fractional units to the centred-pixel format
    expected by conversions.h2F (and used in the visualisation scripts).

    Args:
        PC_frac  : (..., 3) array or (3,) tuple/array  (xstar, ystar, zstar).
        patshape : (nrows_pat, ncols_pat)  – detector pattern shape in pixels.

    Returns:
        PC_h2f : same leading shape + (3,)  –  (x01, x02, DD) in pixels
                 relative to the image centre.
    """
    PC = np.asarray(PC_frac, dtype=float)
    h, w = patshape
    out = np.empty_like(PC)
    out[..., 0] = (0.5 - PC[..., 0]) * w   # x01
    out[..., 1] = (0.5 - PC[..., 1]) * h   # x02
    out[..., 2] =  PC[..., 2]        * h   # DD
    return out


def h2f_to_fractional(PC_h2f, patshape):
    """
    Inverse of fractional_to_h2f: centred-pixel → Bruker fractional.

    fractional_to_h2f defines:
        x01 = (0.5 - xstar) * w     →  xstar = 0.5 - x01 / w
        x02 = (0.5 - ystar) * h     →  ystar = 0.5 - x02 / h
        DD  =  zstar * h            →  zstar = DD  / h

    Args:
        PC_h2f   : (..., 3) array  (x01, x02, DD) in centred pixels.
        patshape : (nrows_pat, ncols_pat).

    Returns:
        PC_frac : (..., 3)  (xstar, ystar, zstar).
    """
    PC = np.asarray(PC_h2f, dtype=float)
    h, w = patshape
    out = np.empty_like(PC)
    out[..., 0] = 0.5 - PC[..., 0] / w   # xstar = 0.5 - x01/w
    out[..., 1] = 0.5 - PC[..., 1] / h   # ystar = 0.5 - x02/h
    out[..., 2] =       PC[..., 2] / h   # zstar = DD/h
    return out


def compute_pc_grid(
    PC_ref_frac,
    ref_position,
    scan_shape,
    step_size_um: float,
    pixel_size_um: float,
    patshape: tuple,
    sample_tilt_deg: float = 70.0,
    detector_tilt_deg: float = 10.0,
):
    """
    Compute the pattern centre at every scan position.

    All geometry is computed in fractional (xstar, ystar, zstar) coordinates.
    The h2F-format array (x01, x02, DD) is produced only at the final step via
    fractional_to_h2f().

    Sign conventions:
        step right  (↑ col)  =>  xstar decreases
        step down   (↑ row)  =>  ystar increases
        step down   (↑ row)  =>  zstar increases 

    Args:
        PC_ref_frac      : (xstar, ystar, zstar) in Bruker fractional units
                           at the reference scan position.
        ref_position     : (row_ref, col_ref) – scan grid index of the
                           reference pattern (the one used in IC-GN).
        scan_shape       : (nrows, ncols) of the EBSD scan.
        step_size_um     : Physical step size in microns.
        pixel_size_um    : Detector pixel size in microns (after binning).
        patshape         : (nrows_pat, ncols_pat) – detector pattern shape.
        sample_tilt_deg  : Sample tilt from horizontal (degrees, default 70).
        detector_tilt_deg: Detector tilt (degrees, default 10).

    Returns:
        PC_grid_frac  : (nrows, ncols, 3)  PC in fractional units at each pos.
        PC_grid_h2f   : (nrows, ncols, 3)  PC in centred-pixel units (h2F fmt).
    """
    PC_ref = np.asarray(PC_ref_frac, dtype=float)
    xstar_ref, ystar_ref, zstar_ref = PC_ref[0], PC_ref[1], PC_ref[2]

    row_ref, col_ref = ref_position
    yi, xi = np.indices(scan_shape).astype(float)   # yi = row index, xi = col index

    delta_col = xi + col_ref   # positive = step right
    delta_row = yi + row_ref   # positive = step down

    h, w = patshape
    rel   = step_size_um / pixel_size_um   # one scan step in detector pixels

    theta = np.radians(90.0 - sample_tilt_deg)   # complement of sample tilt
    phi   = np.radians(detector_tilt_deg)

    # ── fractional PC at every scan position ─────────────────────────────────
    # Derived by expressing the h2f drift formulas in fractional units:
    #   x01 = (0.5 - xstar)*w   =>  xstar = 0.5 - x01/w
    #   x02 = (0.5 - ystar)*h   =>  ystar = 0.5 - x02/h
    #   DD  = zstar * h         =>  zstar = DD / h

    xstar = xstar_ref - delta_col * rel / w
    ystar = ystar_ref - delta_row * rel * np.cos(theta) / (h * np.cos(phi))
    zstar = zstar_ref - delta_row * rel * np.sin(theta + phi) / h

    PC_grid_frac = np.stack([xstar, ystar, zstar], axis=-1)          # (R, C, 3)
    PC_grid_h2f  = fractional_to_h2f(PC_grid_frac, patshape)         # (R, C, 3)

    return PC_grid_frac, PC_grid_h2f


# ---------------------------------------------------------------------------
# 3.  Geometric (PC-induced) homography
# ---------------------------------------------------------------------------

def compute_geometric_shape_function(PC_ref_h2f, PC_curr_h2f):
    """
    Compute the inverse of the geometric shape function T_S at each scan position.

    All inputs are in centred-pixel (h2F) coordinates: (x01, x02, DD).

    The geometric shape function T_S maps reference-pattern coordinates to
    current-pattern coordinates when there is no material deformation — purely
    from the PC shift.  Acting on centred-pixel coords (x01, x02, 1):

        T_S = [[scale,   0,   h13],
               [0,   scale,   h23],
               [0,       0,     1]]

    where  scale = DD_curr / DD_ref,
           h13   = x01_curr - scale * x01_ref,
           h23   = x02_curr - scale * x02_ref.

    The correction uses the inverse T_S_inv = T_S^{-1}:

        T_S_inv = [[1/scale,       0,   x01_ref - x01_curr/scale],
                   [0,       1/scale,   x02_ref - x02_curr/scale],
                   [0,             0,   1                        ]]

    Args:
        PC_ref_h2f  : (3,)           – reference PC as (x01, x02, DD) in pixels.
        PC_curr_h2f : (..., 3) array – per-position PC in centred-pixel units.
                      Can be (3,), (N, 3), or (nrows, ncols, 3).

    Returns:
        TS_inv : (..., 3, 3) array – inverse geometric shape function at each position.
    """
    ref  = np.asarray(PC_ref_h2f,  dtype=float)
    curr = np.asarray(PC_curr_h2f, dtype=float)

    x01_ref, x02_ref, DD_ref = ref[0], ref[1], ref[2]
    x01_curr = curr[..., 0]
    x02_curr = curr[..., 1]
    DD_curr  = curr[..., 2]

    # ── scale factor: ratio of working distances ──────────────────────────────
    scale     = DD_curr  / DD_ref    # DD_curr / DD_ref
    inv_scale = DD_ref   / DD_curr   # DD_ref  / DD_curr

    # ── translation entries (already in pixels) ───────────────────────────────
    TS13 = x01_ref - x01_curr * inv_scale
    TS23 = x02_ref - x02_curr * inv_scale

    # ── assemble (..., 3, 3) ─────────────────────────────────────────────────
    z = np.zeros_like(scale)
    o = np.ones_like(scale)

    TS_inv = np.stack([
        inv_scale, z,         TS13,
        z,         inv_scale, TS23,
        z,         z,         o,
    ], axis=-1).reshape(scale.shape + (3, 3))

    return TS_inv





# ---------------------------------------------------------------------------
# 4.  Correction routines
# ---------------------------------------------------------------------------

def _h_to_W(h: np.ndarray) -> np.ndarray:
    """Build a (3, 3) shape-function matrix from an (8,) homography vector."""
    W = np.eye(3)
    W[0, 0] += h[0];  W[0, 1] = h[1];  W[0, 2] = h[2]
    W[1, 0]  = h[3];  W[1, 1] += h[4]; W[1, 2] = h[5]
    W[2, 0]  = h[6];  W[2, 1] = h[7]
    return W


def _W_to_h(W: np.ndarray) -> np.ndarray:
    """Extract an (8,) homography vector from a (3, 3) shape-function matrix,
    normalising so that W[2, 2] = 1."""
    W = W / W[2, 2]
    return np.array([
        W[0, 0] - 1.0,  # h11
        W[0, 1],        # h12
        W[0, 2],        # h13
        W[1, 0],        # h21
        W[1, 1] - 1.0,  # h22
        W[1, 2],        # h23
        W[2, 0],        # h31
        W[2, 1],        # h32
    ])


def correct_homographies(H_measured, TS_inv):
    """
    Correct measured homographies for PC drift using the inverse geometric
    shape function.

    For each pattern:
      1. Convert the measured homography vector h → shape-function matrix W.
      2. Apply the correction:  W_corr = TS_inv @ W
         (TS_inv cancels the purely geometric warp introduced by the PC shift.)
      3. Convert W_corr back to an 8-parameter homography vector.

    Args:
        H_measured : (..., 8)    measured homographies from the IC-GN.
        TS_inv     : (..., 3, 3) inverse geometric shape function from
                     compute_geometric_shape_function(), same leading shape.

    Returns:
        H_corrected : (..., 8) PC-drift-corrected homographies.
    """
    H_m   = np.asarray(H_measured, dtype=float)
    TS    = np.asarray(TS_inv,      dtype=float)

    orig_shape = H_m.shape[:-1]
    H_flat  = H_m.reshape(-1, 8)
    TS_flat = TS.reshape(-1, 3, 3)
    H_c_flat = np.empty_like(H_flat)

    for i in range(len(H_flat)):
        W_meas      = _h_to_W(H_flat[i])
        W_corr      = TS_flat[i] @ W_meas
        H_c_flat[i] = _W_to_h(W_corr)

    return H_c_flat.reshape(orig_shape + (8,))


# ---------------------------------------------------------------------------
# 5.  Full pipeline
# ---------------------------------------------------------------------------

def correct_scan_homographies(
    H_measured,
    PC_ref_frac,
    ref_position,
    scan_shape,
    step_size_um: float,
    pixel_size_um: float,
    patshape: tuple,
    sample_tilt_deg: float = 70.0,
    detector_tilt_deg: float = 10.0,
):
    """
    End-to-end pipeline: compute the PC grid, build the inverse geometric
    shape function at every scan position, and return corrected homographies.

    For each pattern the correction is:
        W_corrected = TS_inv @ W_measured
    where TS_inv is the inverse of the purely geometric shape function that
    would be measured if the sample were undeformed.

    Args:
        H_measured       : (nrows, ncols, 8) or (N, 8) – measured homographies.
        PC_ref_frac      : (xstar, ystar, zstar) – reference PC in EDAX fractional units.
        ref_position     : (row_ref, col_ref) – scan-grid index of the reference pattern.
        scan_shape       : (nrows, ncols) of the full EBSD scan.
        step_size_um     : Physical step size in microns (from the .ang header).
        pixel_size_um    : Detector pixel size in microns.
        patshape         : (nrows_pat, ncols_pat) – detector pattern shape.
        sample_tilt_deg  : Sample tilt (default 70°).
        detector_tilt_deg: Detector tilt (default 10°).

    Returns:
        H_corrected  : (nrows, ncols, 8) – PC-drift-corrected homographies.
        PC_grid_frac : (nrows, ncols, 3) – PC in fractional units at each pos.
        PC_grid_h2f  : (nrows, ncols, 3) – PC in centred-pixel units (h2F fmt).
        TS_inv       : (nrows, ncols, 3, 3) – inverse geometric shape function.
    """
    PC_grid_frac, PC_grid_h2f = compute_pc_grid(
        PC_ref_frac, ref_position, scan_shape,
        step_size_um, pixel_size_um, patshape,
        sample_tilt_deg, detector_tilt_deg,
    )

    PC_ref_h2f = fractional_to_h2f(PC_ref_frac, patshape)
    TS_inv = compute_geometric_shape_function(PC_ref_h2f, PC_grid_h2f)

    H_corrected = correct_homographies(H_measured, TS_inv)

    return H_corrected, PC_grid_frac, PC_grid_h2f, TS_inv


# ---------------------------------------------------------------------------
# 6.  Diagnostic helper
# ---------------------------------------------------------------------------

def summarise_geometric_correction(TS_inv, scan_shape=None):
    """
    Print a quick summary of the geometric correction magnitude across the scan.

    Args:
        TS_inv     : (..., 3, 3) inverse geometric shape function from
                     compute_geometric_shape_function().
        scan_shape : optional (nrows, ncols) – if provided, shows corner values.
    """
    labels = ['h11', 'h12', 'h13', 'h21', 'h22', 'h23', 'h31', 'h32']
    TS = np.asarray(TS_inv).reshape(-1, 3, 3)
    H = np.array([_W_to_h(W) for W in TS])   # (N, 8)

    print("Geometric (PC-induced) shape-function correction summary")
    print("=========================================================")
    print(f"  {'component':<6}  {'min':>12}  {'max':>12}  {'|max|':>12}")
    for k, lbl in enumerate(labels):
        col = H[:, k]
        print(f"  {lbl:<6}  {col.min():>12.4e}  {col.max():>12.4e}  "
              f"{np.abs(col).max():>12.4e}")

    if scan_shape is not None:
        TS2d = np.asarray(TS_inv).reshape(scan_shape + (3, 3))
        print("\n  Corner values (h11, h13, h22, h23) [TL, TR, BL, BR]:")
        corners = [
            ('TL', (0,  0)),
            ('TR', (0,  scan_shape[1]-1)),
            ('BL', (scan_shape[0]-1, 0)),
            ('BR', (scan_shape[0]-1, scan_shape[1]-1)),
        ]
        for name, (r, c) in corners:
            h = _W_to_h(TS2d[r, c])
            print(f"    {name}: h11={h[0]:.4e}  h13={h[2]:.4e}  "
                  f"h22={h[4]:.4e}  h23={h[5]:.4e}")


# ---------------------------------------------------------------------------
# 7.  Step-by-step geometry visualisation
# ---------------------------------------------------------------------------


def plot_pc_geometry_steps(
    PC_ref_frac,
    ref_position,
    scan_shape,
    step_size_um: float,
    pixel_size_um: float,
    patshape: tuple,
    sample_tilt_deg: float = 70.0,
    detector_tilt_deg: float = 10.0,
    save_path: str = None,
):
    """
    Four-panel step-by-step figure showing exactly how the PC grid is built.

    Step 1 — Scan grid in the sample frame.
              Physical (x, y) positions of each beam position in microns.
              Scan (row=0, col=0) sits at the top-left corner.
              Colour encodes column index.

    Step 2 — Projection of a scan step onto the detector.
              Shows how a column step (Δcol, perpendicular to tilt axis) maps
              to a horizontal detector shift (Δx01), and how a row step (Δrow,
              along the tilt axis) projects via the tilt geometry to a vertical
              shift (Δx02) and a change in detector distance (ΔDD).

    Step 3 — Resulting PC drift (xstar, ystar, zstar) across the scan.
              Line plots (one per scan row).  Reference position marked and
              should sit exactly at the nominal PC value.
              Expected: xstar increases with col,
                        ystar / zstar increase with row.

    Step 4 — Geometric homography magnitude across the scan (h11, h13, h22, h23).
              Should be zero at the reference and grow monotonically away from it.

    Args:
        PC_ref_frac       : (xstar, ystar, zstar) at the reference (EDAX fractional).
        ref_position      : (row_ref, col_ref) scan-grid index of the reference pattern.
        scan_shape        : (nrows, ncols).
        step_size_um      : Physical step size in microns.
        pixel_size_um     : Detector pixel size in microns.
        patshape          : (height_px, width_px) of the detector.
        sample_tilt_deg   : Sample tilt in degrees (default 70).
        detector_tilt_deg : Detector tilt in degrees (default 10).
        save_path         : If given, save to this path.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    nrows, ncols = scan_shape
    ref_row, ref_col = ref_position
    PC_ref = np.asarray(PC_ref_frac, dtype=float)

    PC_frac, PC_h2f = compute_pc_grid(
        PC_ref, ref_position, scan_shape,
        step_size_um, pixel_size_um, patshape,
        sample_tilt_deg, detector_tilt_deg,
    )
    PC_ref_h2f = fractional_to_h2f(PC_ref, patshape)
    TS_inv = compute_geometric_shape_function(PC_ref_h2f, PC_h2f)
    # convert (nrows, ncols, 3, 3) → (nrows, ncols, 8) for plotting
    H_geom = np.array([[_W_to_h(TS_inv[r, c]) for c in range(ncols)]
                        for r in range(nrows)])

    theta = np.radians(90.0 - sample_tilt_deg)
    phi   = np.radians(detector_tilt_deg)
    rel   = step_size_um / pixel_size_um

    row_colors = plt.cm.tab10(np.linspace(0, 0.9, max(nrows, 1)))
    col_idx = np.arange(ncols)

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        f'PC drift geometry — step-by-step\n'
        f'scan {nrows}×{ncols},  step={step_size_um} µm,  pixel={pixel_size_um} µm,  '
        f'tilt={sample_tilt_deg}°/{detector_tilt_deg}°,  ref=({ref_row},{ref_col})',
        fontsize=12, fontweight='bold',
    )

    # ── Step 1: scan grid in sample frame ────────────────────────────────────
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.set_title(
        'Step 1: Scan grid\n(sample frame)',
        fontsize=10,
    )
    ri, ci = np.indices(scan_shape)
    x_phys = ci * step_size_um   # col=0 at left, increases right
    y_phys = ri * step_size_um   # row=0 at top, increases downward

    sc = ax1.scatter(x_phys.ravel(), y_phys.ravel(),
                     c=ci.ravel(), cmap='viridis', s=80, zorder=3)
    cb = fig.colorbar(sc, ax=ax1, shrink=0.85, pad=0.03)
    cb.set_label('col index', fontsize=8)
    x_ref_phys = ref_col * step_size_um
    y_ref_phys = ref_row * step_size_um
    ax1.plot(x_ref_phys, y_ref_phys, 'r*', ms=14, zorder=5,
             label=f'ref ({ref_row},{ref_col})')
    ax1.annotate('(0,0)\ntop-left', xy=(x_phys[0, 0], y_phys[0, 0]),
                 xytext=(x_phys[0, 0] + step_size_um, y_phys[0, 0] + step_size_um * 0.5),
                 fontsize=7, color='navy',
                 arrowprops=dict(arrowstyle='->', color='navy', lw=0.8))
    ax1.set_xlabel('x (µm, col=0 at left)', fontsize=8)
    ax1.set_ylabel('y (µm, row=0 at top)', fontsize=8)
    ax1.invert_yaxis()   # row 0 at top, matching image convention
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.grid(True, ls='--', alpha=0.4)

    # ── Step 2: projection geometry ──────────────────────────────────────────
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.set_title(
        'Step 2: Sample step → detector shift\n(centred-pixel coords)',
        fontsize=10,
    )
    n_demo = min(ncols, 25)
    demo_cols = np.round(np.linspace(0, ncols - 1, n_demo)).astype(int)
    demo_rows = np.arange(nrows)
    for r in demo_rows:
        dc = demo_cols - ref_col
        dr = r - ref_row
        delta_x01 = dc * rel
        delta_x02 = np.full_like(delta_x01, -dr * rel * np.cos(theta) / np.cos(phi))
        delta_DD  = float(dr * rel * np.sin(theta + phi))
        ax2.plot(delta_x01, delta_x02, color=row_colors[r], lw=2, marker='o', ms=4,
                 label=f'row {r}  (ΔDD={delta_DD:.2f} px)')
    ax2.axhline(0, color='k', lw=0.7, ls='--')
    ax2.axvline(0, color='k', lw=0.7, ls='--')
    ax2.plot(0, 0, 'r*', ms=14, zorder=5, label='ref')
    ax2.set_xlabel('Δx01 (px)  [−ve = step RIGHT = xstar↑]', fontsize=8)
    ax2.set_ylabel('Δx02 (px)  [+ve = step UP = ystar↓]', fontsize=8)
    ax2.legend(fontsize=8)
    ax2.grid(True, ls='--', alpha=0.4)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.text(
        0.97, 0.97,
        f'θ=90°−tilt={np.degrees(theta):.1f}°  φ={np.degrees(phi):.1f}°\n'
        f'rel={rel:.4f} px/step\n'
        f'Δx02/step={rel*np.cos(theta)/np.cos(phi):.4f} px\n'
        f'ΔDD/step ={rel*np.sin(theta+phi):.4f} px',
        transform=ax2.transAxes, fontsize=7.5, va='top', ha='right',
        bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.85),
    )

    # ── Step 3: PC components across the scan ────────────────────────────────
    pc_info = [
        (0, r'$x^*$ (xstar)', 'Increases with col\n(steps RIGHT = higher xstar)'),
        (1, r'$y^*$ (ystar)', 'Increases with row\n(steps UP = lower on detector)'),
        (2, r'$z^*$ (zstar)', 'Decreases with row\n(steps DOWN = closer to detector)'),
    ]
    for k, lbl, note in pc_info:
        ax = fig.add_subplot(2, 4, 3 + k)
        ax.set_title(f'Step 3: {lbl}\n{note}', fontsize=9)
        for r in range(nrows):
            ax.plot(col_idx, PC_frac[r, :, k], color=row_colors[r], lw=2,
                    label=f'row {r}' if nrows > 1 else None)
        ax.axhline(PC_ref[k], color='k', lw=1.0, ls='--', label='ref value')
        ax.axvline(ref_col, color='red', lw=1.2, ls=':', label='ref col' if k == 0 else None)
        ax.plot(ref_col, PC_ref[k], 'r*', ms=12, zorder=5)
        ax.set_xlabel('column index', fontsize=8)
        ax.set_ylabel(lbl, fontsize=9)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, ls='--', alpha=0.4)
        if k == 0 and nrows > 1:
            ax.legend(fontsize=8)

    # ── Step 4: geometric homography ─────────────────────────────────────────
    geom_info = [
        (0, r'$h_{11}$ = scale−1', 'Zero at ref; grows with row'),
        (2, r'$h_{13}$ (px)',       'Zero at ref; grows with col'),
        (4, r'$h_{22}$ = scale−1', 'Same as h11 by construction'),
        (5, r'$h_{23}$ (px)',       'Zero at ref; grows with row'),
    ]
    for idx, (hi, lbl, note) in enumerate(geom_info):
        ax = fig.add_subplot(2, 4, 5 + idx)
        ax.set_title(f'Step 4: {lbl}\n{note}', fontsize=9)
        for r in range(nrows):
            ax.plot(col_idx, H_geom[r, :, hi], color=row_colors[r], lw=2,
                    label=f'row {r}' if nrows > 1 else None)
        ax.axhline(0, color='k', lw=0.7, ls='--')
        ax.axvline(ref_col, color='red', lw=1.2, ls=':')
        ax.plot(ref_col, H_geom[ref_row, ref_col, hi], 'r*', ms=12, zorder=5)
        ax.set_xlabel('column index', fontsize=8)
        ax.set_ylabel(lbl, fontsize=9)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, ls='--', alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved {save_path}')

    plt.show()


# ---------------------------------------------------------------------------
# 8.  Summary visualisation (PC drift + geometric homography)
# ---------------------------------------------------------------------------


def plot_pc_correction_summary(
    PC_grid_frac,
    TS_inv,
    scan_shape: tuple,
    step_size_um: float,
    ref_position: tuple = (0, 0),
    save_path: str = None,
):
    """
    Two-row summary figure for the PC-drift correction.

    Top row    — PC deviation from the reference value (Δxstar, Δystar, Δzstar).
    Bottom row — The four non-zero geometric correction components extracted
                 from TS_inv: h11 / h22 (scale change) and h13 / h23 (translation).

    For scans with ≤ 5 rows the data are shown as line profiles (one line per
    scan row).  For larger scans 2-D colour maps are used instead.

    Args:
        PC_grid_frac : (nrows, ncols, 3)  PC in EDAX/TSL fractional units,
                       as returned by compute_pc_grid().
        TS_inv       : (nrows, ncols, 3, 3)  inverse geometric shape function from
                       compute_geometric_shape_function().
        scan_shape   : (nrows, ncols).
        step_size_um : Physical step size in microns (for axis labelling).
        ref_position : (row_ref, col_ref) — marked as a vertical dashed line.
        save_path    : If given, save the figure to this path.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    nrows, ncols = scan_shape
    ref_row, ref_col = ref_position
    x_um = np.arange(ncols) * step_size_um

    PC  = np.asarray(PC_grid_frac).reshape(nrows, ncols, 3)
    PC_ref = PC[ref_row, ref_col]
    delta_PC = PC - PC_ref

    TS = np.asarray(TS_inv).reshape(nrows, ncols, 3, 3)
    H = np.array([[_W_to_h(TS[r, c]) for c in range(ncols)] for r in range(nrows)])

    pc_data   = [delta_PC[..., 0], delta_PC[..., 1], delta_PC[..., 2]]
    pc_labels = [r'$\Delta x^*$', r'$\Delta y^*$', r'$\Delta z^*$']

    geom_data   = [H[..., 0], H[..., 2], H[..., 4], H[..., 5]]
    geom_labels = [r'$h_{11}$ (scale $-$ 1)', r'$h_{13}$ (px)',
                   r'$h_{22}$ (scale $-$ 1)', r'$h_{23}$ (px)']
    geom_cmaps  = ['RdBu_r', 'PiYG', 'RdBu_r', 'PiYG']

    use_lines  = nrows <= 5
    row_colors = plt.cm.tab10(np.linspace(0, 0.9, max(nrows, 1)))
    ref_x_um   = ref_col * step_size_um

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle('Pattern-centre drift and geometric correction summary',
                 fontsize=13, y=1.01)

    # ── top row: PC drift (3 panels; 4th hidden) ────────────────────────────
    for k in range(3):
        ax = axes[0, k]
        data = pc_data[k]

        if use_lines:
            for r in range(nrows):
                ax.plot(x_um, data[r], color=row_colors[r], lw=1.8,
                        label=f'row {r}' if nrows > 1 else None)
            ax.axhline(0, color='k', lw=0.7, ls='--')
            ax.axvline(ref_x_um, color='red', lw=1.2, ls=':',
                       label='ref' if k == 0 else None)
            ax.set_xlabel('x (µm)', fontsize=10)
            ax.set_ylabel(pc_labels[k], fontsize=11)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            if k == 0 and nrows > 1:
                ax.legend(fontsize=8, loc='best')
        else:
            extent = [0, (ncols - 1) * step_size_um,
                      (nrows - 1) * step_size_um, 0]
            vm = np.abs(data).max() or 1e-12
            im = ax.imshow(data, cmap='coolwarm', aspect='auto',
                           vmin=-vm, vmax=vm, extent=extent)
            fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
            ax.plot(ref_x_um, ref_row * step_size_um, 'r+', ms=10, mew=2)
            ax.set_xlabel('x (µm)', fontsize=10)
            ax.set_ylabel('y (µm)', fontsize=10)

        ax.set_title(pc_labels[k], fontsize=12)

    axes[0, 3].set_visible(False)

    # ── bottom row: geometric homography (4 panels) ─────────────────────────
    for k in range(4):
        ax = axes[1, k]
        data = geom_data[k]

        if use_lines:
            for r in range(nrows):
                ax.plot(x_um, data[r], color=row_colors[r], lw=1.8,
                        label=f'row {r}' if nrows > 1 else None)
            ax.axhline(0, color='k', lw=0.7, ls='--')
            ax.axvline(ref_x_um, color='red', lw=1.2, ls=':')
            ax.set_xlabel('x (µm)', fontsize=10)
            ax.set_ylabel(geom_labels[k], fontsize=10)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        else:
            extent = [0, (ncols - 1) * step_size_um,
                      (nrows - 1) * step_size_um, 0]
            vm = np.abs(data).max() or 1e-12
            im = ax.imshow(data, cmap=geom_cmaps[k], aspect='auto',
                           vmin=-vm, vmax=vm, extent=extent)
            fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
            ax.plot(ref_x_um, ref_row * step_size_um, 'r+', ms=10, mew=2)
            ax.set_xlabel('x (µm)', fontsize=10)
            ax.set_ylabel('y (µm)', fontsize=10)

        ax.set_title(geom_labels[k], fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved {save_path}')

    plt.show()

# ---------------------------------------------------------------------------
# Runner — execute directly to preview the PC drift for your scan geometry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    #  Edit these values to match your experiment                         #
    # ------------------------------------------------------------------ #
    PATTERN_CENTER  = np.array([0.6871, 1-0.8929, 1.06971])   # xstar, ystar, zstar  (EDAX fractional)
    PATSHAPE        = (512, 512)                              # (height_px, width_px)
    SCAN_SHAPE      = (10, 132)                               # (nrows, ncols)
    REF_POSITION    = (0, 0)                                  # (row_ref, col_ref)
    STEP_SIZE_UM    = 0.13                                    # physical step size in microns
    PIXEL_SIZE_UM   = 13.0                                    # detector pixel size in microns
    SAMPLE_TILT     = 70.0                                    # degrees
    DETECTOR_TILT   = 10.0                                    # degrees
    # ------------------------------------------------------------------ #

    print("Computing PC grid ...")
    PC_grid_frac, PC_grid_h2f = compute_pc_grid(
        PC_ref_frac       = PATTERN_CENTER,
        ref_position      = REF_POSITION,
        scan_shape        = SCAN_SHAPE,
        step_size_um      = STEP_SIZE_UM,
        pixel_size_um     = PIXEL_SIZE_UM,
        patshape          = PATSHAPE,
        sample_tilt_deg   = SAMPLE_TILT,
        detector_tilt_deg = DETECTOR_TILT,
    )

    PC_ref_h2f = fractional_to_h2f(PATTERN_CENTER, PATSHAPE)
    TS_inv = compute_geometric_shape_function(PC_ref_h2f, PC_grid_h2f)

    plot_pc_correction_summary(
        PC_grid_frac = PC_grid_frac,
        TS_inv       = TS_inv,
        scan_shape   = SCAN_SHAPE,
        step_size_um = STEP_SIZE_UM,
        ref_position = REF_POSITION,
        save_path    = "pc_correction_summary.png",
    )

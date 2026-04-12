"""
ipf_map.py  —  Inverse Pole Figure (IPF) map for EBSD .ang files
=================================================================
Supports cubic crystal symmetry (m-3m / Oh).

Color convention (matches TSL / OIM standard):
    [001] → blue
    [011] → green
    [111] → red

Rotation convention:
    Bunge ZXZ passive  —  matches rotations.eu2om used throughout this project.
    The crystal direction parallel to the chosen sample direction is:
        c_crystal = R^T @ d_sample

Usage (standalone):
    Edit the INPUTS block at the bottom and run:
        python ipf_map.py

Or import into another script:
    from ipf_map import compute_ipf_colors, plot_ipf_map, plot_ipf_triangle
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────────────────────────────────────
# Core color computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_ipf_colors(
    eulers: np.ndarray,
    sample_direction: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    """
    Compute IPF colors for cubic (m-3m) crystal symmetry.

    Parameters
    ----------
    eulers : array of shape (..., 3) in **radians**  [phi1, Phi, phi2]
        Any leading batch dimensions are supported —
        e.g. (N, 3) for a flat list or (rows, cols, 3) for a 2-D scan.
    sample_direction : (3,) unit vector of the sample direction to project onto.
        [0, 0, 1] = Normal  Direction (ND)  ← default
        [1, 0, 0] = Rolling Direction (RD)
        [0, 1, 0] = Transverse Direction (TD)

    Returns
    -------
    rgb : float array of shape (..., 3) with values in [0, 1].

    Algorithm
    ---------
    1.  Compute the rotation matrix R using the Bunge ZXZ convention
        (identical to rotations.eu2om in this project).
    2.  Find the crystal direction parallel to the sample direction:
            c_crystal = R^T @ d_sample
    3.  Reduce to the cubic fundamental zone by taking absolute values
        and sorting components in ascending order (x ≤ y ≤ z).
        This correctly applies all m-3m symmetry operations.
    4.  Apply the affine IPF colour map:
            xn = x / z,  yn = y / z           (normalised triangle coords)
            R  = xn                            → [111] corner (red)
            G  = yn − xn                       → [011] corner (green)
            B  = 1 − yn                        → [001] corner (blue)
        Then scale so that max(R, G, B) = 1 (full brightness).
        Corner verification:
            [001] (0,0,1) → xn=0, yn=0 → (0,0,1) = blue   ✓
            [011] (0,½,½) → xn=0, yn=1 → (0,1,0) = green  ✓
            [111] (⅓,⅓,⅓)→ xn=1, yn=1 → (1,0,0) = red    ✓

    Reference: formula derived from the standard TSL/OIM IPF triangle;
               symmetry reduction follows Nolze & Hielscher (2016).
    """
    from rotations import eu2om

    original_shape = eulers.shape  # (..., 3)
    batch_shape    = original_shape[:-1]
    eulers_flat    = eulers.reshape(-1, 3)
    N              = len(eulers_flat)

    # ── Rotation matrices (N, 3, 3) ─────────────────────────────────────────
    R = eu2om(eulers_flat)   # shape (N, 3, 3), same convention as project-wide

    # ── Crystal direction parallel to sample direction ───────────────────────
    # Passive convention: d_sample = R @ c_crystal  →  c_crystal = R^T @ d_sample
    d = np.asarray(sample_direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    c = np.einsum("nij,j->ni", R, d)   # = R @ d gives sample coords of crystal axis
    # Wait — we want the crystal direction that aligns with d_sample.
    # In passive convention g maps crystal→sample: d_sample = g c_crystal
    # so c_crystal = g^T d_sample = R^T d_sample
    c = np.einsum("nji,j->ni", R, d)   # R^T @ d, shape (N, 3)

    # ── Reduce to cubic fundamental zone ────────────────────────────────────
    c = np.abs(c)                         # apply mirror symmetries (m-3m)
    c = np.sort(c, axis=1)                # sort ascending: x ≤ y ≤ z
                                          # this covers 3- and 4-fold rotations

    # ── Normalise ────────────────────────────────────────────────────────────
    norms = np.linalg.norm(c, axis=1, keepdims=True)
    valid = (norms[:, 0] > 1e-10)
    c[valid] /= norms[valid]

    x, y, z = c[:, 0], c[:, 1], c[:, 2]

    # ── IPF colour map ───────────────────────────────────────────────────────
    z_safe = np.where(z > 1e-10, z, 1.0)
    xn = x / z_safe
    yn = y / z_safe

    r = xn
    g = yn - xn
    b = 1.0 - yn

    rgb = np.stack([r, g, b], axis=-1)   # shape (N, 3)

    # Scale to full brightness
    max_rgb = rgb.max(axis=-1, keepdims=True)
    max_rgb = np.where(max_rgb > 1e-10, max_rgb, 1.0)
    rgb /= max_rgb

    # Black out degenerate pixels
    rgb[~valid] = 0.0
    rgb = np.clip(rgb, 0.0, 1.0)

    return rgb.reshape(batch_shape + (3,))


# ─────────────────────────────────────────────────────────────────────────────
# IPF standard triangle (colour key)
# ─────────────────────────────────────────────────────────────────────────────

def plot_ipf_triangle(ax=None, n: int = 300) -> plt.Axes:
    """
    Draw the cubic IPF colour key (standard triangle).

    The triangle is plotted in the normalised (xn, yn) space where:
        [001] corner → (xn=0, yn=0) = bottom-left  = blue
        [011] corner → (xn=0, yn=1) = top-left     = green
        [111] corner → (xn=1, yn=1) = top-right    = red

    Parameters
    ----------
    ax  : matplotlib Axes to draw into (created if None)
    n   : grid resolution

    Returns
    -------
    ax  : the Axes used
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3.5, 3.5))

    xn_vals = np.linspace(0.0, 1.0, n)
    yn_vals = np.linspace(0.0, 1.0, n)
    XN, YN  = np.meshgrid(xn_vals, yn_vals)

    inside = (YN >= XN - 1e-6)   # valid region: yn ≥ xn

    R = XN
    G = YN - XN
    B = 1.0 - YN

    rgb = np.stack([R, G, B], axis=-1)
    mx  = rgb.max(axis=-1, keepdims=True)
    mx  = np.where(mx > 1e-10, mx, 1.0)
    rgb = np.clip(rgb / mx, 0.0, 1.0)
    rgb[~inside] = 1.0   # white outside

    ax.imshow(
        rgb, origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        interpolation="bilinear",
    )

    # Corner markers + labels
    corners = [
        (0.0, 0.0, "[001]", "blue"),
        (0.0, 1.0, "[011]", "green"),
        (1.0, 1.0, "[111]", "red"),
    ]
    for cx, cy, label, color in corners:
        ax.plot(cx, cy, "s", color=color, markersize=9, markeredgecolor="k", markeredgewidth=0.5)

    ax.text(0.0, -0.10, "[001]", ha="center", va="top",    fontsize=9,  color="blue",  fontweight="bold")
    ax.text(0.0,  1.10, "[011]", ha="center", va="bottom", fontsize=9,  color="darkgreen", fontweight="bold")
    ax.text(1.10, 1.00, "[111]", ha="left",   va="center", fontsize=9,  color="red",   fontweight="bold")

    ax.set_xlim(-0.15, 1.25)
    ax.set_ylim(-0.15, 1.20)
    ax.axis("off")
    ax.set_title("IPF colour key\n(cubic, m-3m)", fontsize=9)

    return ax


# ─────────────────────────────────────────────────────────────────────────────
# High-level plotting function
# ─────────────────────────────────────────────────────────────────────────────

_DIRECTION_VECTORS = {
    "ND": np.array([0.0, 0.0, 1.0]),
    "RD": np.array([1.0, 0.0, 0.0]),
    "TD": np.array([0.0, 1.0, 0.0]),
}


def plot_ipf_map(
    ang_path: str,
    sample_direction=None,
    direction_label: str = "ND",
    patshape: tuple = None,
    save_path: str = None,
    show: bool = True,
) -> np.ndarray:
    """
    Load a .ang file, compute the IPF map, and display it alongside the colour key.

    Parameters
    ----------
    ang_path         : path to the EDAX .ang file
    sample_direction : (3,) array or one of "ND", "RD", "TD".
                       Defaults to ND = [0, 0, 1].
    direction_label  : label shown in the figure title
    patshape         : (height_px, width_px) needed by read_ang.
                       Pass None to skip pattern-centre conversion.
    save_path        : if given, save the figure here (.png recommended)
    show             : whether to call plt.show(block=False)

    Returns
    -------
    rgb_map : float array of shape (rows, cols, 3) with values in [0, 1]
    """
    import utilities

    # ── Resolve sample direction ──────────────────────────────────────────────
    if sample_direction is None:
        sample_direction = _DIRECTION_VECTORS["ND"]
    elif isinstance(sample_direction, str):
        key = sample_direction.upper()
        if key not in _DIRECTION_VECTORS:
            raise ValueError(f"direction must be one of {list(_DIRECTION_VECTORS)}, got '{sample_direction}'")
        direction_label  = key
        sample_direction = _DIRECTION_VECTORS[key]
    sample_direction = np.asarray(sample_direction, dtype=np.float64)

    # ── Load .ang file ────────────────────────────────────────────────────────
    print(f"Reading {ang_path} …")
    ang_data = utilities.read_ang(ang_path, patshape, segment_grain_threshold=None)
    eulers   = ang_data.eulers          # shape (rows, cols, 3), radians
    rows, cols = ang_data.shape
    print(f"Scan shape: {rows} rows × {cols} cols")

    # ── Compute IPF colours ───────────────────────────────────────────────────
    print(f"Computing IPF colours  //  {direction_label} …")
    rgb_map = compute_ipf_colors(eulers, sample_direction)   # (rows, cols, 3)
    print("Done.")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 5))

    ax_map = fig.add_axes([0.04, 0.08, 0.68, 0.84])
    ax_map.imshow(rgb_map, origin="upper", interpolation="nearest")
    ax_map.set_title(f"IPF Map  //  {direction_label}  ({rows}×{cols})", fontsize=13)
    ax_map.set_xlabel("Column", fontsize=11)
    ax_map.set_ylabel("Row", fontsize=11)

    ax_key = fig.add_axes([0.76, 0.15, 0.22, 0.70])
    plot_ipf_triangle(ax_key)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved → {save_path}")

    if show:
        plt.show(block=False)

    return rgb_map


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # =========================================================================
    # INPUTS  — edit these lines then run:  python ipf_map.py
    # =========================================================================

    ang_file   = '/Users/crestiennedechaine/OriginalData/Si_Ge_Dataset/DI_largerRegion/SiGe_largerRegion_selectedArea_20260322_512x512.ang'
    patshape   = (512, 512)   # (height_px, width_px) of the detector
    output_dir = '/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/SiGe/'
    direction  = 'ND'         # 'ND', 'RD', or 'TD'

    # =========================================================================

    os.makedirs(output_dir, exist_ok=True)

    rgb_map = plot_ipf_map(
        ang_path         = ang_file,
        direction_label  = direction,
        patshape         = patshape,
        save_path        = os.path.join(output_dir, f"IPF_map_{direction}.png"),
        show             = True,
    )

    plt.show()
    print("Done.")

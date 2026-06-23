"""
Results_plotting.py — shared matplotlib helpers for DIC-HREBSD results.

  plot_component_grid  — render up to 9 data arrays in a 3×3 imshow grid
  compute_tfbc         — traction-free BC: solve for e33_abs and absolute strains
  plot_all_results     — generate all standard figures from VisWorker output
                         (includes TFBC figures when enabled in params)

Both functions are backend-agnostic: the caller is responsible for setting
the matplotlib backend before importing this module.
"""

import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Keep text in vector exports (PDF/SVG/EPS) as *real, editable text* rather than
# outlined paths, so figures can be relabelled later in Illustrator/Inkscape.
# Harmless for raster (PNG/JPG) output.
mpl.rcParams["pdf.fonttype"] = 42      # TrueType, editable in Illustrator
mpl.rcParams["ps.fonttype"]  = 42
mpl.rcParams["svg.fonttype"] = "none"  # emit <text>, not vector outlines

# Only paths that already carry one of these *figure* extensions get their
# format swapped — data exports (.csv, .tex, .txt, …) are left untouched.
_FIGURE_EXTS = (".png", ".jpg", ".jpeg", ".pdf", ".svg", ".eps", ".tif", ".tiff")


def _with_format(path, fmt):
    """Return `path` with its (figure) extension replaced by `fmt` (e.g. 'svg').

    Only swaps when the path already ends in a known figure/image extension, so
    CSV / TeX / other data files keep their own extension.  `fmt` of None/"" is
    a no-op, leaving existing callers and non-GUI scripts unaffected.
    """
    if not path or not fmt:
        return path
    root, ext = os.path.splitext(path)
    if ext.lower() not in _FIGURE_EXTS:
        return path
    return f"{root}.{str(fmt).lower().lstrip('.')}"


def _latex_sci(value: float, digits: int = 5) -> str:
    """Render a float as LaTeX math in scientific notation (inner math only)."""
    if not np.isfinite(value):
        return r"\text{NaN}"
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10 ** exponent
    return f"{mantissa:+.{digits}f} \\times 10^{{{exponent}}}"


# ─────────────────────────────────────────────────────────────────────────────
# Low-level grid helper
# ─────────────────────────────────────────────────────────────────────────────

def plot_component_grid(
    components,
    cmap="coolwarm",
    vmin=None,
    vmax=None,
    title="",
    save_path=None,
    show=True,
    figsize=(15, 10),
    fontsize=13,
    fontweight="normal",
    axis_off=True,
):
    """
    Plot up to 9 data components in a 3×3 grid.

    Parameters
    ----------
    components : list of dict
        Each dict must have:
            'data'  : 2-D array to display
            'label' : subplot title string
        Optional per-component overrides:
            'vmin'  : colour scale minimum  (falls back to function-level vmin)
            'vmax'  : colour scale maximum  (falls back to function-level vmax)
            'cmap'  : colormap              (falls back to function-level cmap)
    cmap : str
        Default colormap for all subplots.
    vmin, vmax : float or None
        Default colour limits for subplots that don't specify their own.
    title : str
        Figure suptitle.
    save_path : str or None
        If provided, saves the figure here (dpi=200, bbox_inches='tight').
    show : bool
        If True, calls plt.show(block=False) after rendering.
    figsize : tuple
        (width, height) in inches.
    fontsize : int
        Font size for subplot titles.
    axis_off : bool
        If True, hides axis ticks/labels on all subplots.
    """
    if len(components) > 9:
        raise ValueError("plot_component_grid supports at most 9 components (3×3 grid).")

    fig, axes = plt.subplots(3, 3, figsize=figsize)
    axes_flat = axes.flatten()

    for i, ax in enumerate(axes_flat):
        if i >= len(components):
            ax.axis("off")
            continue

        comp   = components[i]
        data   = comp["data"]
        label  = comp.get("label", "")
        c_vmin = comp.get("vmin", vmin)
        c_vmax = comp.get("vmax", vmax)
        c_cmap = comp.get("cmap", cmap)

        im = ax.imshow(data, cmap=c_cmap, vmin=c_vmin, vmax=c_vmax)
        fig.colorbar(im, ax=ax)
        ax.set_title(label, fontsize=fontsize, fontweight=fontweight)
        if axis_off:
            ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=fontsize + 2, fontweight=fontweight)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved {save_path}")

    if show:
        plt.show(block=False)


# ─────────────────────────────────────────────────────────────────────────────
# Traction-free boundary condition
# ─────────────────────────────────────────────────────────────────────────────

def compute_tfbc(results: dict, params: dict) -> dict:
    """
    Apply the traction-free surface boundary condition (σ₃₃ = 0) to recover
    the absolute out-of-plane strain e33_abs and, from it, the absolute
    in-plane strains e11_abs / e22_abs and the tetragonal strain ε_T.

    Tetragonal strain definition (Chpt 3, Eq. for lattice tetragonality):
        ε_T = ε_33 − (ε_11 + ε_22) / 2

    Positive ε_T → c-axis stretched relative to in-plane (tetragonal with c > a).
    Zero ε_T → cubic.

    Requires ``results`` to contain:
        'base_quats' : (N, 4) quaternions from the .ang file
        'e11'..'e23' : 2-D relative strain maps (rows × cols)

    Requires ``params`` to contain:
        'crystal_C11', 'crystal_C12', 'crystal_C44' : elastic constants in GPa
        'crystal_structure' : 'cubic' (only cubic is supported currently)

    Returns a dict with keys:
        'e11_abs', 'e22_abs', 'e33_abs' : 2-D absolute strain maps
        'tetragonal_strain'              : 2-D ε_T map  (ε_33 − (ε_11+ε_22)/2)
        'tetragonality_ca'               : 2-D c/a ratio map  (1+ε_33)/(1+(ε_11+ε_22)/2)
    """
    from core import utilities, rotations

    rows = results["rows"]
    cols = results["cols"]
    N    = int(rows) * int(cols)

    # Single-orientation override: replace per-pattern .ang quats with one
    # fixed Bunge orientation (degrees) tiled across the whole scan.
    if params.get("tfbc_use_single_euler", False):
        eu_deg = params.get("tfbc_euler_deg", (0.0, 0.0, 0.0))
        eu_rad = np.deg2rad(np.asarray(eu_deg, dtype=np.float64))
        single_q = rotations.eu2qu(eu_rad)
        base_quats = np.tile(np.asarray(single_q, dtype=np.float64), (N, 1))
        print(f"[TFBC] Using single Euler override (Bunge, °): "
              f"φ₁={eu_deg[0]:.3f}  Φ={eu_deg[1]:.3f}  φ₂={eu_deg[2]:.3f}")
    else:
        base_quats = np.asarray(results["base_quats"])
        # Strain fields are stored at ROI bbox resolution (rows × cols), but
        # base_quats may still be the full scan if the visualizer wasn't
        # given a roi_slice or used a key-mismatched ang_path lookup.  Align
        # the two so the broadcast against C_rot[:, ...] works.
        if base_quats.shape[0] != N:
            roi_slice = params.get("roi_slice", None)
            sliced = False
            if roi_slice is not None:
                full_r = params.get("full_rows", 0)
                full_c = params.get("full_cols", 0)
                if full_r and full_c and base_quats.shape[0] == full_r * full_c:
                    bq2d = base_quats.reshape(int(full_r), int(full_c), -1)
                    base_quats = bq2d[roi_slice[0], roi_slice[1], :].reshape(-1, 4)
                    sliced = True
            if not sliced:
                # Last-resort: clip to N (only correct if the strain field
                # is the first rows×cols chunk of the scan, which is true
                # only when ROI starts at (0, 0)).  Logged so the user
                # knows the alignment may be off.
                if base_quats.shape[0] >= N:
                    print(
                        f"[TFBC] WARNING: base_quats has {base_quats.shape[0]} entries "
                        f"but strain field has {N}.  No roi_slice in params — "
                        f"clipping quats to the first {N} entries.  Pass "
                        f"`roi_slice` (and `full_rows`/`full_cols`) into the "
                        f"visualizer params for a correct slice."
                    )
                    base_quats = base_quats[:N]
                else:
                    raise ValueError(
                        f"base_quats has {base_quats.shape[0]} entries but the "
                        f"strain field requires {N}.  Pass `roi_slice` + "
                        f"`full_rows`/`full_cols` so compute_tfbc can align them."
                    )

    # ── Rotate stiffness tensor to sample frame for each pattern ──────────────
    # Use the .ang quaternion directly: the IC-GN rotation is the difference
    # between target and reference patterns, which is already represented in
    # the per-point .ang orientations.
    C11 = params["crystal_C11"]
    C12 = params["crystal_C12"]
    C44 = params["crystal_C44"]
    structure = params.get("crystal_structure", "cubic")

    C_crystal = utilities.get_stiffness_tensor(C11, C12, C44, structure=structure)
    C_rot     = utilities.rotate_stiffness_to_sample_frame(C_crystal, base_quats)  # (N, 6, 6)

    # # ── Post-rotation 180° rotation about z (negate sample-frame x and y) ────
    # # R_z(180°) = diag(-1, -1, +1).  For a rank-4 tensor C_ijkl this flips
    # # entries with an odd number of x or y indices — in Voigt notation, the
    # # cross-block between {1,2,3,6} and {4,5}: C₁₄, C₁₅, C₂₄, C₂₅, C₃₄, C₃₅,
    # # C₄₆, C₅₆ (and symmetric counterparts).
    # _voigt_xy_parity = np.array([0, 0, 0, 1, 1, 0])
    # _flip_z = (-1.0) ** (_voigt_xy_parity[:, None] + _voigt_xy_parity[None, :])
    # C_rot   = C_rot * _flip_z


    # ── Voigt index: 0=11, 1=22, 2=33, 3=23, 4=13, 5=12 ─────────────────────
    C31 = C_rot[:, 2, 0]; C32 = C_rot[:, 2, 1]; C33 = C_rot[:, 2, 2]
    C34 = C_rot[:, 2, 3]; C35 = C_rot[:, 2, 4]; C36 = C_rot[:, 2, 5]

    e11_f = results["e11"].ravel()
    e22_f = results["e22"].ravel()
    e33_f = results["e33"].ravel()
    e12_f = results["e12"].ravel()
    e13_f = results["e13"].ravel()
    e23_f = results["e23"].ravel()

    # Traction-free surface: σ_33 = 0 in the sample frame.
    #   C31·ε_11 + C32·ε_22 + C33·ε_33 + 2·C34·ε_23 + 2·C35·ε_13 + 2·C36·ε_12 = 0
    # The measured strains use the HR-EBSD "ε_33 = 0" convention (h2F normalises
    # F_33→1, F2strain divides V by V_33), so ε_11/ε_22 are already the true
    # in-plane strains and only ε_33 is unknown.  Solving σ_33 = 0 for ε_33:
    #   ε_33_abs = -(C31·ε_11 + C32·ε_22 + 2·C34·ε_23 + 2·C35·ε_13 + 2·C36·ε_12) / C33
    numerator   = (C31*e11_f + C32*e22_f
                   + 2*C34*e23_f + 2*C35*e13_f + 2*C36*e12_f)
    denominator = C33
    e33_abs_f   = -numerator / denominator
    e11_abs_f   = e11_f
    e22_abs_f   = e22_f




    e33_abs = e33_abs_f.reshape(rows, cols)
    e11_abs = e11_abs_f.reshape(rows, cols)
    e22_abs = e22_abs_f.reshape(rows, cols)

    # Tetragonal strain: ε_T = ε_33 − (ε_11 + ε_22) / 2
    # (Chpt 3 lattice-tetragonality definition; 0 for cubic, > 0 for c-axis tension)
    tetragonal_strain = e33_abs - 0.5 * (e11_abs + e22_abs)

    # Lattice tetragonality c/a from deformed-lattice spacings:
    #   c/a = (1 + ε_33) / (1 + (ε_11 + ε_22)/2)
    e_in_plane       = 0.5 * (e11_abs + e22_abs)
    tetragonality_ca = (1.0 + e33_abs) / (1.0 + e_in_plane)

    return {
        "e11_abs":           e11_abs,
        "e22_abs":           e22_abs,
        "e33_abs":           e33_abs,
        "tetragonal_strain": tetragonal_strain,
        "tetragonality_ca":  tetragonality_ca,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reference + neighbor pattern export
# ─────────────────────────────────────────────────────────────────────────────

def _generate_sim_reference(params: dict, pat_obj):
    """Generate the *simulated* reference pattern from the master pattern +
    Euler angles + PC (mirrors gui_workers.SimRefWorker), normalise it, and run
    it through the same Step-3 processing as the experimental patterns so the
    two are directly comparable.  Returns a float32 (H, W) array, or None if it
    can't be generated (missing master pattern, sim failure, etc.).
    """
    master = (params.get("master_pattern_path") or "").strip()
    if not master or not os.path.exists(master):
        print(f"[ref/neighbor export] Simulated reference requested but master "
              f"pattern not found ({master!r}) — falling back to experimental.")
        return None
    try:
        from PatternSimulation.SimPatGen import patternSimulation
        from core import conversions

        euler_deg = params.get("euler_deg", (0.0, 0.0, 0.0))
        pc_edax   = tuple(params.get("pc_edax", (0.5, 0.5, 0.5)))
        # Generate at the experimental detector shape so the Step-3 mask lines
        # up and the histograms compare like-for-like.
        H, W = pat_obj.patshape

        sim = patternSimulation()
        sim.detector_height   = H
        sim.detector_width    = W
        sim.det_shape         = (H, W)
        sim.detector_tilt_deg = float(params.get("det_tilt", 10.0))
        sim.sample_tilt_deg   = float(params.get("tilt", 70.0))
        sim.mastersetup(master)

        euler_rad = np.deg2rad(euler_deg)
        pc_bruker = conversions.Edax_to_Bruker_PC(pc_edax)
        sim.EandPCSet(euler_rad, list(pc_bruker), verbose=False)

        pat = sim.GenPattern()
        pat_np = pat.detach().cpu().numpy().astype(np.float32).reshape(H, W)
        lo, hi = pat_np.min(), pat_np.max()
        pat_np = (pat_np - lo) / (hi - lo + 1e-9)
        # Step-3 processing so it matches the processed experimental neighbor.
        return pat_obj.process_pattern(pat_np).astype(np.float32)
    except Exception as exc:
        print(f"[ref/neighbor export] Simulated reference generation failed "
              f"({exc}) — falling back to experimental.")
        return None


def _export_ref_and_neighbor(params: dict):
    """Save a JPG of the reference pattern, a JPG of the pattern one scan
    column to its right (col+1), and a single figure overlaying the two
    intensity distributions.

    The reference image follows what the run actually used:
      • real / single reference  → the experimental pattern at ref_position;
      • simulated reference (``ref_mode == "simulated"``)  → the simulated
        pattern regenerated from the master pattern + Euler + PC.
    The neighbor (col+1) is always the experimental pattern from the .up2.

    Patterns are read/processed with the run's Step-3 settings (background
    removal / band-pass / gamma / mask), matching what the pipeline and the
    Step-4 preview show — not the raw detector counts.  Needs ``up2_path`` (or
    ``up2``), ``ref_position`` = (ref_row, ref_col), ``full_cols`` (full-scan
    column count, for the flat pattern index), and ``save_folder``.  The
    Step-3 settings (``low_pass_sigma``, ``high_pass_sigma``, ``mask_type``,
    ``flip_x``, ``gamma``) and — for simulated refs — ``ref_mode``,
    ``master_pattern_path`` and ``euler_deg`` are forwarded from the run;
    sensible defaults are used when the viewer is opened standalone.
    """
    from fileio import Data

    up2_path  = (params.get("up2_path") or params.get("up2") or "").strip()
    ref_pos   = params.get("ref_position", None)
    full_cols = int(params.get("full_cols") or params.get("cols") or 0)
    save      = params.get("save_folder", "")

    if not save:
        print("[ref/neighbor export] No save folder set — nothing written.")
        return
    if not up2_path or not os.path.exists(up2_path):
        print(f"[ref/neighbor export] .up2 not found ({up2_path!r}) — skipped.")
        return
    if ref_pos is None:
        print("[ref/neighbor export] No reference position available — skipped.")
        return
    if full_cols <= 0:
        print("[ref/neighbor export] Unknown scan column count — skipped.")
        return

    ref_row, ref_col = int(ref_pos[0]), int(ref_pos[1])
    nbr_col = ref_col + 1
    if nbr_col >= full_cols:
        print(f"[ref/neighbor export] Reference is in the last column "
              f"(col={ref_col}); no neighbor to the right — skipped.")
        return

    pat_obj = Data.UP2(up2_path)
    # Apply the same Step-3 processing the run used so the saved patterns and
    # their histograms match the pipeline / Step-4 preview (not raw counts).
    pat_obj.set_processing(
        low_pass_sigma  = float(params.get("low_pass_sigma", 1.0)),
        high_pass_sigma = float(params.get("high_pass_sigma", 10.0)),
        truncate_std_scale = 3.0,
        mask_type       = params.get("mask_type", "none"),
        center_cross_half_width = 6,
        flip_x          = bool(params.get("flip_x", False)),
        gamma           = float(params.get("gamma", 0.8)),
    )
    n_pat   = pat_obj.nPatterns
    ref_idx = ref_row * full_cols + ref_col
    nbr_idx = ref_row * full_cols + nbr_col
    if ref_idx >= n_pat or nbr_idx >= n_pat:
        print(f"[ref/neighbor export] Pattern index out of range "
              f"(ref={ref_idx}, neighbor={nbr_idx}, nPatterns={n_pat}) — skipped.")
        return

    # Reference image: simulated (regenerated) if the run used a simulated
    # reference, otherwise the experimental pattern at ref_position.  Fall back
    # to experimental if sim generation isn't possible.
    ref_mode = str(params.get("ref_mode", "single")).lower()
    ref_src  = "experimental"
    ref_pat  = None
    if ref_mode == "simulated":
        ref_pat = _generate_sim_reference(params, pat_obj)
        if ref_pat is not None:
            ref_src = "simulated"
    if ref_pat is None:
        ref_pat = pat_obj.read_pattern(ref_idx, process=True).astype(np.float32)

    # Neighbor is always the experimental pattern one column to the right.
    nbr_pat = pat_obj.read_pattern(nbr_idx, process=True).astype(np.float32)
    print(f"[ref/neighbor export] reference source: {ref_src}")

    # Grayscale JPGs of the Step-3-processed patterns.
    ref_jpg = os.path.join(save, "Reference_pattern.jpg")
    nbr_jpg = os.path.join(save, "Neighbor_pattern_colplus1.jpg")
    plt.imsave(ref_jpg, ref_pat, cmap="gray")
    plt.imsave(nbr_jpg, nbr_pat, cmap="gray")
    print(f"Saved {ref_jpg}")
    print(f"Saved {nbr_jpg}")

    # Combined intensity-distribution histogram (shared bins, density-normalised
    # so the two patterns are comparable regardless of any pixel-count mismatch).
    lo    = float(min(ref_pat.min(), nbr_pat.min()))
    hi    = float(max(ref_pat.max(), nbr_pat.max()))
    edges = np.linspace(lo, hi, 256)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ref_pat.ravel(), bins=edges, color="tab:blue", alpha=0.55, density=True,
            label=f"Reference [{ref_src}] (row {ref_row + 1}, col {ref_col + 1})")
    ax.hist(nbr_pat.ravel(), bins=edges, color="tab:orange", alpha=0.55, density=True,
            label=f"Neighbor +1 col (row {ref_row + 1}, col {nbr_col + 1})")
    ax.set_xlabel("Pixel intensity (Step-3 processed)")
    ax.set_ylabel("Probability density")
    ax.set_title("Intensity distribution — reference vs right-neighbor pattern")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    hist_path = _with_format(
        os.path.join(save, "Reference_neighbor_intensity_hist.png"),
        params.get("figure_format", "png"),
    )
    fig.savefig(hist_path, dpi=200, bbox_inches="tight")
    plt.show(block=False)
    print(f"Saved {hist_path}")


# ─────────────────────────────────────────────────────────────────────────────
# High-level: generate all standard figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_results(results: dict, params: dict):
    """
    Generate all standard visualization figures from VisWorker output.

    Parameters
    ----------
    results : dict
        Output of VisWorker._compute():  h11..h32, e11..e33, w13/w21/w32,
        plus 'rows' and 'cols'.
    params : dict
        Must contain:
            'strain_lim'   — ± colour limit for coolwarm strain plots
            'rot_lim'      — ± colour limit (degrees) for rotation plots
            'linemap_row'  — row index for the strain line-map figure
            'save_folder'  — directory to save PNGs (empty string = don't save)
    """
    rows = results["rows"]
    cols = results["cols"]
    sv   = params["strain_lim"]
    rv   = params["rot_lim"]
    r    = min(params["linemap_row"], rows - 1)
    save = params.get("save_folder", "")
    # "png" (default), "pdf", "svg", "eps".  Vector formats keep editable text
    # (see rcParams at module top).  Filenames below are written with ".png"
    # literals; _sp swaps the extension to the requested format.
    fig_fmt = params.get("figure_format", "png")

    h11, h12, h13 = results["h11"], results["h12"], results["h13"]
    h21, h22, h23 = results["h21"], results["h22"], results["h23"]
    h31, h32       = results["h31"], results["h32"]
    e11, e12, e13 = results["e11"], results["e12"], results["e13"]
    e22, e23, e33 = results["e22"], results["e23"], results["e33"]
    w13, w21, w32 = results["w13"], results["w21"], results["w32"]

    def _sp(name):
        return _with_format(os.path.join(save, name), fig_fmt) if save else None

    # The visualization dialog drives which optional plots run; the only
    # figure that is *always* rendered is the post-TFBC strain & rotation
    # grid further down.  Everything else is gated by params["plot_options"].
    plot_opts = params.get("plot_options", {}) or {}

    # ── Reference + neighbor pattern export (JPG + intensity histogram) ──────
    if plot_opts.get("export_ref_neighbor", False):
        try:
            _export_ref_and_neighbor(params)
        except Exception as exc:
            print(f"[plot_all_results] Reference/neighbor export skipped: {exc}")

    # ── Homography components ─────────────────────────────────────────────────
    if plot_opts.get("plot_homography_grid", False):
        plot_component_grid(
            [
                {"data": h11, "label": r"$h_{11}$"}, {"data": h12, "label": r"$h_{12}$"},
                {"data": h13, "label": r"$h_{13}$"}, {"data": h21, "label": r"$h_{21}$"},
                {"data": h22, "label": r"$h_{22}$"}, {"data": h23, "label": r"$h_{23}$"},
                {"data": h31, "label": r"$h_{31}$"}, {"data": h32, "label": r"$h_{32}$"},
            ],
            cmap="coolwarm",
            title="Homography Components",
            save_path=_sp("Homography_Components.png"),
            fontsize=16,
            fontweight="bold",
        )

    # ── Strain + rotation (coolwarm) — pre-TFBC ───────────────────────────────
    if plot_opts.get("plot_relative_strain_grid", False):
        plot_component_grid(
            [
                {"data": e11, "label": r"$\epsilon_{11}$",   "vmin": -sv, "vmax": sv},
                {"data": e12, "label": r"$\epsilon_{12}$",   "vmin": -sv, "vmax": sv},
                {"data": e13, "label": r"$\epsilon_{13}$",   "vmin": -sv, "vmax": sv},
                {"data": w21, "label": r"$\omega_{21}$ (°)", "vmin": -rv, "vmax": rv},
                {"data": e22, "label": r"$\epsilon_{22}$",   "vmin": -sv, "vmax": sv},
                {"data": e23, "label": r"$\epsilon_{23}$",   "vmin": -sv, "vmax": sv},
                {"data": w13, "label": r"$\omega_{13}$ (°)", "vmin": -rv, "vmax": rv},
                {"data": w32, "label": r"$\omega_{32}$ (°)", "vmin": -rv, "vmax": rv},
            ],
            cmap="coolwarm",
            title="Strain and Rotation Components (relative)",
            save_path=None,
        )

    # Euler-angle plotting (relative and absolute) was removed from the
    # results viewer — the fields are still saved in the .npy result file
    # for downstream tooling that wants them.

    # ── Strain line maps ──────────────────────────────────────────────────────
    # Use physical distance (µm) along the line if step_size is provided;
    # otherwise fall back to column index.
    step_um = float(params.get("step_size", 0.0) or 0.0)
    if step_um > 0.0:
        x_pos = np.arange(cols) * step_um
        x_axis_label = "Distance along line (µm)"
    else:
        x_pos = np.arange(cols)
        x_axis_label = "Column (pattern index)"

    def _sci_fmt(ax):
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((-3, -3))
        ax.yaxis.set_major_formatter(fmt)

    if plot_opts.get("plot_strain_lines_before_tfbc", False):
        fig_lm, axes_lm = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        normal_strains = [
            (e11[r], r"$\epsilon_{11}$", "tab:blue"),
            (e22[r], r"$\epsilon_{22}$", "tab:orange"),
            (e33[r], r"$\epsilon_{33}$", "tab:green"),
        ]
        shear_strains = [
            (e12[r], r"$\epsilon_{12}$", "tab:red"),
            (e13[r], r"$\epsilon_{13}$", "tab:purple"),
            (e23[r], r"$\epsilon_{23}$", "tab:brown"),
        ]
        rot_lines = [
            (w13[r], r"$\omega_{13}$ (°)", "tab:blue"),
            (w21[r], r"$\omega_{21}$ (°)", "tab:orange"),
            (w32[r], r"$\omega_{32}$ (°)", "tab:green"),
        ]

        for data, label, color in normal_strains:
            axes_lm[0].plot(x_pos, data, label=label, linewidth=1.5, color=color)
        axes_lm[0].set_ylim(-20e-3, 20e-3)
        _sci_fmt(axes_lm[0])
        axes_lm[0].set_ylabel("Strain")
        axes_lm[0].set_title(f"Normal strains — sample frame, deviatoric (before TFBC) — row {r}")
        axes_lm[0].axhline(0, color="k", linewidth=0.6, linestyle="--")
        axes_lm[0].legend(fontsize=11)
        axes_lm[0].grid(True, linestyle="--", alpha=0.4)

        for data, label, color in shear_strains:
            axes_lm[1].plot(x_pos, data, label=label, linewidth=1.5, color=color)
        axes_lm[1].set_ylim(-20e-3, 20e-3)
        _sci_fmt(axes_lm[1])
        axes_lm[1].set_ylabel("Shear strain")
        axes_lm[1].set_title(f"Shear strains — sample frame, deviatoric (before TFBC) — row {r}")
        axes_lm[1].axhline(0, color="k", linewidth=0.6, linestyle="--")
        axes_lm[1].legend(fontsize=11)
        axes_lm[1].grid(True, linestyle="--", alpha=0.4)

        for data, label, color in rot_lines:
            axes_lm[2].plot(x_pos, data, label=label, linewidth=1.5, color=color)
        axes_lm[2].set_ylabel("Rotation (°)")
        axes_lm[2].set_xlabel(x_axis_label)
        axes_lm[2].set_title(f"Lattice rotations — sample frame — row {r}")
        axes_lm[2].axhline(0, color="k", linewidth=0.6, linestyle="--")
        axes_lm[2].legend(fontsize=11)
        axes_lm[2].grid(True, linestyle="--", alpha=0.4)

        fig_lm.suptitle(
            f"Strain line scan — sample frame, deviatoric (before TFBC) — row {r}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        sp = _sp("Strain_linemap.png")
        if sp:
            plt.savefig(sp, dpi=200, bbox_inches="tight")
        plt.show(block=False)

    # ── Detector-frame line scan (always shown, before any other processing) ─
    # Recompute ε / ω in the detector frame *directly* from the saved
    # homographies (h2F → F2strain) along the chosen row.  No rotation, no
    # round-trip — guaranteed to give ε_33_det = 0 by IC-GN's F_33 = 1
    # normalization, regardless of which sample-frame rotation was used.
    from core import conversions as _conv
    h_row = np.stack([
        results["h11"][r], results["h12"][r], results["h13"][r],
        results["h21"][r], results["h22"][r], results["h23"][r],
        results["h31"][r], results["h32"][r],
    ], axis=1).astype(np.float64)              # (cols, 8)

    # PC vector required by h2F: EDAX → Bruker → fractional offset
    pc_edax_pre   = np.asarray(params.get("pc_edax", (0.5, 0.5, 0.5)), dtype=float)
    pc_bruker_pre = _conv.Edax_to_Bruker_PC(pc_edax_pre)
    patshape_pre  = (params.get("pat_h", 512), params.get("pat_w", 512))
    xo_pre        = _conv.Bruker_to_fractional_PC(pc_bruker_pre, patshape_pre)

    # ── Diagnostic prints (detector-frame line scan) ─────────────────────────
    print()
    print("=" * 78)
    print(f"[detector-frame line scan @ row {r}]")
    print(f"  pc_edax (from params)  : {tuple(np.round(pc_edax_pre, 4))}")
    if params.get("pc_edax") is None:
        print("  ⚠  params['pc_edax'] is missing — using fallback (0.5, 0.5, 0.5)")
        print("     This sets x01 = x02 = 0, which kills the h_32·x_02 → ε_22 coupling.")
    print(f"  pc_bruker              : {tuple(np.round(pc_bruker_pre, 4))}")
    print(f"  patshape (H, W)        : {patshape_pre}")
    print(f"  xo (x01, x02, DD), px  : {tuple(np.round(xo_pre, 4))}")
    print()
    print(f"  h_row component ranges along row {r}:")
    print(f"    h11  min={results['h11'][r].min():+.4e}  max={results['h11'][r].max():+.4e}  std={results['h11'][r].std():.4e}")
    print(f"    h22  min={results['h22'][r].min():+.4e}  max={results['h22'][r].max():+.4e}  std={results['h22'][r].std():.4e}")
    print(f"    h23  min={results['h23'][r].min():+.4e}  max={results['h23'][r].max():+.4e}  std={results['h23'][r].std():.4e}")
    print(f"    h32  min={results['h32'][r].min():+.4e}  max={results['h32'][r].max():+.4e}  std={results['h32'][r].std():.4e}")
    print()
    # Mid-row F_22 contribution breakdown
    i_mid       = h_row.shape[0] // 2
    hh_mid      = h_row[i_mid]
    x01_, x02_, DD_ = float(xo_pre[0]), float(xo_pre[1]), float(xo_pre[2])
    beta0_mid   = 1.0 - hh_mid[6]*x01_ - hh_mid[7]*x02_
    F22_unnorm  = 1.0 + hh_mid[4] + hh_mid[7]*x02_
    print(f"  Mid-row (col={i_mid}) F_22 contribution breakdown:")
    print(f"    h22 (direct stretch)       : {hh_mid[4]:+.4e}")
    print(f"    h32·x02 (perspective term) : {hh_mid[7]*x02_:+.4e}    "
          f"(h32={hh_mid[7]:+.4e}, x02={x02_:.3f})")
    print(f"    F22 numerator              : {F22_unnorm:.8f}")
    print(f"    β0 (denominator)           : {beta0_mid:.8f}")
    print(f"    ε_22^det = F22/β0 − 1      : {F22_unnorm/beta0_mid - 1:+.4e}")
    print("=" * 78)
    print()

    F_row                       = _conv.h2F(h_row, xo_pre)          # (cols, 3, 3)
    e_det_row, omega_det_row_rad = _conv.F2strain(F_row)             # both (cols, 3, 3)
    omega_det_row                = np.degrees(omega_det_row_rad)     # to degrees for display

    if plot_opts.get("plot_detector_frame_lines", False):
        fig_pre, axes_pre = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        for arr, lbl, color in [
            (e_det_row[:, 0, 0], r"$\epsilon_{11}^{det}$", "tab:blue"),
            (e_det_row[:, 1, 1], r"$\epsilon_{22}^{det}$", "tab:orange"),
            (e_det_row[:, 2, 2], r"$\epsilon_{33}^{det}$", "tab:green"),
        ]:
            axes_pre[0].plot(x_pos, arr, label=lbl, linewidth=1.5, color=color)
        axes_pre[0].set_ylim(-20e-3, 20e-3)
        _sci_fmt(axes_pre[0])
        axes_pre[0].set_ylabel("Strain")
        axes_pre[0].set_title(f"Normal strains — detector frame (from h2F·F2strain, no rotation) — row {r}")
        axes_pre[0].axhline(0, color="k", linewidth=0.6, linestyle="--")
        axes_pre[0].legend(fontsize=11)
        axes_pre[0].grid(True, linestyle="--", alpha=0.4)

        for arr, lbl, color in [
            (e_det_row[:, 0, 1], r"$\epsilon_{12}^{det}$", "tab:red"),
            (e_det_row[:, 0, 2], r"$\epsilon_{13}^{det}$", "tab:purple"),
            (e_det_row[:, 1, 2], r"$\epsilon_{23}^{det}$", "tab:brown"),
        ]:
            axes_pre[1].plot(x_pos, arr, label=lbl, linewidth=1.5, color=color)
        axes_pre[1].set_ylim(-20e-3, 20e-3)
        _sci_fmt(axes_pre[1])
        axes_pre[1].set_ylabel("Shear strain")
        axes_pre[1].set_title(f"Shear strains — detector frame (from h2F·F2strain, no rotation) — row {r}")
        axes_pre[1].axhline(0, color="k", linewidth=0.6, linestyle="--")
        axes_pre[1].legend(fontsize=11)
        axes_pre[1].grid(True, linestyle="--", alpha=0.4)

        for arr, lbl, color in [
            (omega_det_row[:, 0, 2], r"$\omega_{13}^{det}$ (°)", "tab:blue"),
            (omega_det_row[:, 1, 0], r"$\omega_{21}^{det}$ (°)", "tab:orange"),
            (omega_det_row[:, 2, 1], r"$\omega_{32}^{det}$ (°)", "tab:green"),
        ]:
            axes_pre[2].plot(x_pos, arr, label=lbl, linewidth=1.5, color=color)
        axes_pre[2].set_ylabel("Rotation (°)")
        axes_pre[2].set_xlabel(x_axis_label)
        axes_pre[2].set_title(f"Lattice rotations — detector frame (from h2F·F2strain, no rotation) — row {r}")
        axes_pre[2].axhline(0, color="k", linewidth=0.6, linestyle="--")
        axes_pre[2].legend(fontsize=11)
        axes_pre[2].grid(True, linestyle="--", alpha=0.4)

        fig_pre.suptitle(
            f"Strain line scan — DETECTOR FRAME (from h2F·F2strain, no rotation applied) — row {r}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        sp_pre = _sp("Strain_linemap_detector_frame_raw.png")
        if sp_pre:
            plt.savefig(sp_pre, dpi=200, bbox_inches="tight")
        plt.show(block=False)

    # ── Initial-guess strain line scan (FMT-FCC h_guess, pre-IC-GN refine) ───
    # Same h2F → F2strain pipeline as the detector-frame line scan above, but
    # uses the per-pattern initial-guess homography (saved by PipelineWorker
    # when init_type != NONE).  Useful for diagnosing whether IC-GN refinement
    # is moving meaningfully off the FMT-FCC starting point or just locking
    # into it.  If h_guess is unavailable (run without init guess, or older
    # results.npy file), this section is skipped silently.
    h_guess_2d = results.get("h_guess", None)
    if h_guess_2d is not None and plot_opts.get("plot_initial_guess_lines", False):
        h_guess_row = h_guess_2d[r].reshape(-1, 8).astype(np.float64)   # (cols, 8)
        F_guess_row                          = _conv.h2F(h_guess_row, xo_pre)
        e_guess_row, omega_guess_row_rad     = _conv.F2strain(F_guess_row)
        omega_guess_row                      = np.degrees(omega_guess_row_rad)

        fig_guess, axes_guess = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        for arr, lbl, color in [
            (e_guess_row[:, 0, 0], r"$\epsilon_{11}^{guess}$", "tab:blue"),
            (e_guess_row[:, 1, 1], r"$\epsilon_{22}^{guess}$", "tab:orange"),
            (e_guess_row[:, 2, 2], r"$\epsilon_{33}^{guess}$", "tab:green"),
        ]:
            axes_guess[0].plot(x_pos, arr, label=lbl, linewidth=1.5, color=color)
        axes_guess[0].set_ylim(-20e-3, 20e-3)
        _sci_fmt(axes_guess[0])
        axes_guess[0].set_ylabel("Strain")
        axes_guess[0].set_title(f"Normal strains — initial-guess h (FMT-FCC, pre-IC-GN) — row {r}")
        axes_guess[0].axhline(0, color="k", linewidth=0.6, linestyle="--")
        axes_guess[0].legend(fontsize=11)
        axes_guess[0].grid(True, linestyle="--", alpha=0.4)

        for arr, lbl, color in [
            (e_guess_row[:, 0, 1], r"$\epsilon_{12}^{guess}$", "tab:red"),
            (e_guess_row[:, 0, 2], r"$\epsilon_{13}^{guess}$", "tab:purple"),
            (e_guess_row[:, 1, 2], r"$\epsilon_{23}^{guess}$", "tab:brown"),
        ]:
            axes_guess[1].plot(x_pos, arr, label=lbl, linewidth=1.5, color=color)
        axes_guess[1].set_ylim(-20e-3, 20e-3)
        _sci_fmt(axes_guess[1])
        axes_guess[1].set_ylabel("Shear strain")
        axes_guess[1].set_title(f"Shear strains — initial-guess h (FMT-FCC, pre-IC-GN) — row {r}")
        axes_guess[1].axhline(0, color="k", linewidth=0.6, linestyle="--")
        axes_guess[1].legend(fontsize=11)
        axes_guess[1].grid(True, linestyle="--", alpha=0.4)

        for arr, lbl, color in [
            (omega_guess_row[:, 0, 2], r"$\omega_{13}^{guess}$ (°)", "tab:blue"),
            (omega_guess_row[:, 1, 0], r"$\omega_{21}^{guess}$ (°)", "tab:orange"),
            (omega_guess_row[:, 2, 1], r"$\omega_{32}^{guess}$ (°)", "tab:green"),
        ]:
            axes_guess[2].plot(x_pos, arr, label=lbl, linewidth=1.5, color=color)
        axes_guess[2].set_ylabel("Rotation (°)")
        axes_guess[2].set_xlabel(x_axis_label)
        axes_guess[2].set_title(f"Lattice rotations — initial-guess h (FMT-FCC, pre-IC-GN) — row {r}")
        axes_guess[2].axhline(0, color="k", linewidth=0.6, linestyle="--")
        axes_guess[2].legend(fontsize=11)
        axes_guess[2].grid(True, linestyle="--", alpha=0.4)

        fig_guess.suptitle(
            f"Initial-guess strain line scan — DETECTOR FRAME (from h_guess via h2F·F2strain) — row {r}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        sp_guess = _sp("Strain_linemap_initial_guess.png")
        if sp_guess:
            plt.savefig(sp_guess, dpi=200, bbox_inches="tight")
        plt.show(block=False)

    # ── Traction-free BC ──────────────────────────────────────────────────────
    if "base_quats" in results or params.get("tfbc_use_single_euler", False):
        tfbc = compute_tfbc(results, params)
        e11_abs           = tfbc["e11_abs"]
        e22_abs           = tfbc["e22_abs"]
        e33_abs           = tfbc["e33_abs"]
        tetragonal_strain = tfbc["tetragonal_strain"]
        tetragonality_ca  = tfbc["tetragonality_ca"]

        # Dump c/a map to CSV: one CSV row per scan row, one column per scan column.
        sp_ca_csv = _sp("Tetragonality_ca.csv")
        if sp_ca_csv:
            np.savetxt(
                sp_ca_csv,
                tetragonality_ca,
                delimiter=",",
                fmt="%.8f",
                header=(f"Lattice tetragonality c/a = (1+e33_abs)/(1+(e11_abs+e22_abs)/2)  "
                        f"shape=({tetragonality_ca.shape[0]} rows x {tetragonality_ca.shape[1]} cols)"),
                comments="# ",
            )
            print(f"Saved {sp_ca_csv}")

        # Long-format CSV: one row per scan position with every post-TFBC
        # strain + rotation component.  Strains are in the sample frame after
        # the σ_33 = 0 traction-free correction; rotations are in degrees.
        sp_long_csv = _sp("Strain_Rotation_TFBC.csv")
        if sp_long_csv:
            n_rows_, n_cols_ = e11_abs.shape
            rr, cc = np.meshgrid(np.arange(n_rows_), np.arange(n_cols_), indexing="ij")
            csv_cols = [
                ("row",            rr.ravel()),
                ("col",            cc.ravel()),
                ("e11",            e11_abs.ravel()),
                ("e12",            results["e12"].ravel()),
                ("e13",            results["e13"].ravel()),
                ("e22",            e22_abs.ravel()),
                ("e23",            results["e23"].ravel()),
                ("e33",            e33_abs.ravel()),
                ("w13_deg",        w13.ravel()),
                ("w21_deg",        w21.ravel()),
                ("w32_deg",        w32.ravel()),
                ("eps_T",          tetragonal_strain.ravel()),
                ("c_over_a",       tetragonality_ca.ravel()),
            ]
            header = ",".join(name for name, _ in csv_cols)
            data = np.column_stack([arr for _, arr in csv_cols])
            # Integer formats for row/col; high precision for the rest.
            fmts = ["%d", "%d"] + ["%.8e"] * (len(csv_cols) - 2)
            np.savetxt(
                sp_long_csv,
                data,
                delimiter=",",
                fmt=fmts,
                header=header,
                comments="",
            )
            print(f"Saved {sp_long_csv}")

        # Absolute strain grid (coolwarm)
        plot_component_grid(
            [
                {"data": e11_abs,         "label": r"$\epsilon_{11}$",   "vmin": -sv, "vmax": sv},
                {"data": results["e12"], "label": r"$\epsilon_{12}$",   "vmin": -sv, "vmax": sv},
                {"data": results["e13"], "label": r"$\epsilon_{13}$",   "vmin": -sv, "vmax": sv},
                {"data": w21,            "label": r"$\omega_{21}$ (°)", "vmin": -rv, "vmax": rv},
                {"data": e22_abs,         "label": r"$\epsilon_{22}$",   "vmin": -sv, "vmax": sv},
                {"data": results["e23"], "label": r"$\epsilon_{23}$",   "vmin": -sv, "vmax": sv},
                {"data": w13,            "label": r"$\omega_{13}$ (°)", "vmin": -rv, "vmax": rv},
                {"data": w32,            "label": r"$\omega_{32}$ (°)", "vmin": -rv, "vmax": rv},
                {"data": e33_abs,         "label": r"$\epsilon_{33}$",   "vmin": -sv, "vmax": sv},
            ],
            cmap="coolwarm",
            axis_off=True,
            title="Absolute strain and rotation components (traction-free BC, coolwarm)",
            save_path=_sp("Strain_Rotation_TFBC_coolwarm.png"),
        )

        # Tetragonal strain map: ε_T = ε_33 − (ε_11 + ε_22)/2 (centered on 0)
        et_mean = float(tetragonal_strain.mean())
        et_std  = float(tetragonal_strain.std())
        if plot_opts.get("plot_tetragonal_strain", False):
            et_lim = float(np.nanpercentile(np.abs(tetragonal_strain), 98))
            if et_lim == 0:
                et_lim = 1e-6
            fig_et, ax_et = plt.subplots(figsize=(8, 4))
            im_et = ax_et.imshow(tetragonal_strain, cmap="RdBu_r",
                                 vmin=-et_lim, vmax=+et_lim)
            cb_et = fig_et.colorbar(im_et, ax=ax_et, fraction=0.03, pad=0.04)
            cb_et.set_label(r"$\varepsilon_T$", fontsize=13)
            ax_et.set_title(
                r"Tetragonal strain  $\varepsilon_T = \varepsilon_{33} - "
                r"\frac{\varepsilon_{11} + \varepsilon_{22}}{2}$"
                f"   mean={et_mean:+.4e}, std={et_std:.4e}",
                fontsize=12,
            )
            ax_et.axis("off")
            plt.tight_layout()
            sp_et = _sp("Tetragonal_strain.png")
            if sp_et:
                plt.savefig(sp_et, dpi=200, bbox_inches="tight")
            plt.show(block=False)

        # ── Hydrostatic + Von Mises strain ───────────────────────────────────
        # Wrapped in try/except so any numerical/matplotlib hiccup here can't
        # take the rest of the visualization pipeline (TFBC line scan,
        # summary table, …) down with it.
        try:
            # Hydrostatic ε_h = trace(ε)/3, computed from the *absolute*
            # post-TFBC diagonals so the trace is physically meaningful (the
            # deviatoric IC-GN output has ε_33 = 0 by construction).
            eps_hydro = (np.asarray(e11_abs, dtype=np.float64)
                         + np.asarray(e22_abs, dtype=np.float64)
                         + np.asarray(e33_abs, dtype=np.float64)) / 3.0

            def _safe_pct(arr, pct, fallback=1e-6):
                vals = np.asarray(arr)[np.isfinite(np.asarray(arr))]
                if vals.size == 0:
                    return fallback
                v = float(np.percentile(np.abs(vals), pct))
                return fallback if v == 0.0 else v

            if plot_opts.get("plot_hydrostatic_strain", False):
                h_lim = _safe_pct(eps_hydro, 98)
                fig_h, ax_h = plt.subplots(figsize=(8, 4))
                im_h = ax_h.imshow(eps_hydro, cmap="RdBu_r", vmin=-h_lim, vmax=h_lim)
                cb_h = fig_h.colorbar(im_h, ax=ax_h, fraction=0.03, pad=0.04)
                cb_h.set_label(r"$\epsilon_h$", fontsize=13)
                ax_h.set_title(
                    r"Hydrostatic strain  $\epsilon_h = (\epsilon_{11}+\epsilon_{22}+\epsilon_{33}) / 3$  "
                    r"(sample frame, after TFBC)",
                    fontsize=12,
                )
                ax_h.axis("off")
                plt.tight_layout()
                sp_h = _sp("Hydrostatic_strain.png")
                if sp_h:
                    plt.savefig(sp_h, dpi=200, bbox_inches="tight")
                plt.show(block=False)

            # Von Mises  ε_VM = √( (2/3) · ε'_ij · ε'_ij ),  ε'_ij = ε_ij − ε_h δ_ij.
            # Use distinct names (e12_arr etc.) to avoid shadowing the e12/e13/e23
            # set up at the top of plot_all_results.
            e11d_arr = np.asarray(e11_abs, dtype=np.float64) - eps_hydro
            e22d_arr = np.asarray(e22_abs, dtype=np.float64) - eps_hydro
            e33d_arr = np.asarray(e33_abs, dtype=np.float64) - eps_hydro
            e12_arr  = np.asarray(results["e12"], dtype=np.float64)
            e13_arr  = np.asarray(results["e13"], dtype=np.float64)
            e23_arr  = np.asarray(results["e23"], dtype=np.float64)
            eps_vm = np.sqrt(np.maximum(
                (2.0 / 3.0) * (
                    e11d_arr ** 2 + e22d_arr ** 2 + e33d_arr ** 2
                    + 2.0 * (e12_arr ** 2 + e13_arr ** 2 + e23_arr ** 2)
                ),
                0.0,
            ))
            if plot_opts.get("plot_von_mises", False):
                vm_lim = _safe_pct(eps_vm, 98)
                fig_vm, ax_vm = plt.subplots(figsize=(8, 4))
                im_vm = ax_vm.imshow(eps_vm, cmap="magma", vmin=0.0, vmax=vm_lim)
                cb_vm = fig_vm.colorbar(im_vm, ax=ax_vm, fraction=0.03, pad=0.04)
                cb_vm.set_label(r"$\epsilon_{VM}$", fontsize=13)
                ax_vm.set_title(
                    r"Von Mises equivalent strain  "
                    r"$\epsilon_{VM} = \sqrt{\frac{2}{3}\epsilon'_{ij}\epsilon'_{ij}}$  "
                    r"(sample frame, after TFBC)",
                    fontsize=12,
                )
                ax_vm.axis("off")
                plt.tight_layout()
                sp_vm = _sp("VonMises_strain.png")
                if sp_vm:
                    plt.savefig(sp_vm, dpi=200, bbox_inches="tight")
                plt.show(block=False)
        except Exception as exc:
            print(f"[plot_all_results] Hydrostatic/Von-Mises figures skipped: {exc}")

        # TFBC line maps (normal strains + tetragonal strain ε_T + c/a along chosen row)
        if plot_opts.get("plot_tfbc_lines", False):
            fig_tf, axes_tf = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
            # Reuse the µm/column choice from the main line scan above
            x_pos_tf = x_pos.copy()

            for data, label, color in [
                (e11_abs[r], r"$\epsilon_{11}^{\mathrm{abs}}$", "tab:blue"),
                (e22_abs[r], r"$\epsilon_{22}^{\mathrm{abs}}$", "tab:orange"),
                (e33_abs[r], r"$\epsilon_{33}^{\mathrm{abs}}$ (TF BC)", "tab:green"),
            ]:
                axes_tf[0].plot(x_pos_tf, data, label=label, linewidth=1.5, color=color)
            axes_tf[0].set_ylim(-20e-3, 20e-3)
            fmt_tf = ticker.ScalarFormatter(useMathText=True)
            fmt_tf.set_scientific(True); fmt_tf.set_powerlimits((-3, -3))
            axes_tf[0].yaxis.set_major_formatter(fmt_tf)
            axes_tf[0].set_ylabel("Absolute strain")
            axes_tf[0].set_title(f"Absolute normal strains — sample frame, after TFBC — row {r}")
            axes_tf[0].axhline(0, color="k", linewidth=0.6, linestyle="--")
            axes_tf[0].legend(fontsize=11)
            axes_tf[0].grid(True, linestyle="--", alpha=0.4)

            axes_tf[1].plot(x_pos_tf, tetragonal_strain[r], color="tab:red", linewidth=1.5,
                            label=r"$\varepsilon_T = \varepsilon_{33} - (\varepsilon_{11}+\varepsilon_{22})/2$")
            axes_tf[1].axhline(0, color="k", linewidth=0.6, linestyle="--")
            axes_tf[1].axhline(et_mean, color="tab:gray", linewidth=0.8, linestyle=":",
                               label=f"mean = {et_mean:+.4e}")
            axes_tf[1].set_ylabel(r"$\varepsilon_T$")
            axes_tf[1].set_title(f"Tetragonal strain ε_T — sample frame, after TFBC — row {r}")
            axes_tf[1].legend(fontsize=10)
            axes_tf[1].grid(True, linestyle="--", alpha=0.4)

            ca_mean = float(tetragonality_ca.mean())
            axes_tf[2].plot(x_pos_tf, tetragonality_ca[r], color="tab:purple", linewidth=1.5,
                            label=r"$c/a = (1+\varepsilon_{33}) / (1+(\varepsilon_{11}+\varepsilon_{22})/2)$")
            axes_tf[2].axhline(1.0, color="k", linewidth=0.6, linestyle="--")
            axes_tf[2].axhline(ca_mean, color="tab:gray", linewidth=0.8, linestyle=":",
                               label=f"mean = {ca_mean:.6f}")
            axes_tf[2].set_ylabel(r"$c/a$")
            axes_tf[2].set_xlabel(x_axis_label)
            axes_tf[2].set_title(f"Lattice tetragonality c/a — sample frame, after TFBC — row {r}")
            axes_tf[2].legend(fontsize=10)
            axes_tf[2].grid(True, linestyle="--", alpha=0.4)

            fig_tf.suptitle(
                f"Strain line scan — sample frame, AFTER TFBC (absolute) — row {r}",
                fontsize=13, fontweight="bold",
            )
            plt.tight_layout()
            sp_tf = _sp("TFBC_linemap.png")
            if sp_tf:
                plt.savefig(sp_tf, dpi=200, bbox_inches="tight")
            plt.show(block=False)

        # ── Post-TFBC summary table (console + LaTeX) ─────────────────────────
        # Statistics on |component|.  Normal strains use absolute values
        # (post-TFBC); shears and rotations are unchanged from the deviatoric
        # output (TFBC only adds ε_33 to the diagonals).
        e12_post = results["e12"]
        e13_post = results["e13"]
        e23_post = results["e23"]
        post_tfbc_components = [
            ("e11_abs", r"$\epsilon_{11}^{\mathrm{abs}}$",   e11_abs),
            ("e22_abs", r"$\epsilon_{22}^{\mathrm{abs}}$",   e22_abs),
            ("e33_abs", r"$\epsilon_{33}^{\mathrm{abs}}$",   e33_abs),
            ("e12",     r"$\epsilon_{12}$",                  e12_post),
            ("e13",     r"$\epsilon_{13}$",                  e13_post),
            ("e23",     r"$\epsilon_{23}$",                  e23_post),
            ("eps_T",   r"$\varepsilon_T$",                  tetragonal_strain),
            ("w13",     r"$\omega_{13}$",                    w13),
            ("w21",     r"$\omega_{21}$",                    w21),
            ("w32",     r"$\omega_{32}$",                    w32),
        ]

        print()
        print("=" * 70)
        print(f"  Post-TFBC summary  ({rows} rows × {cols} cols)")
        print(f"  Sample frame, absolute strains.  Statistics on |component|.")
        print("=" * 70)
        print(f"  {'Component':<10}  {'min |·|':>16}  {'max |·|':>16}  {'mean |·|':>16}  {'std |·|':>16}")
        print(f"  {'-'*10:<10}  {'-'*16:>16}  {'-'*16:>16}  {'-'*16:>16}  {'-'*16:>16}")
        for name, _, arr in post_tfbc_components:
            arr_abs = np.abs(arr)
            print(f"  {name:<10}  "
                  f"{np.nanmin(arr_abs):>+16.8e}  "
                  f"{np.nanmax(arr_abs):>+16.8e}  "
                  f"{np.nanmean(arr_abs):>+16.8e}  "
                  f"{np.nanstd(arr_abs):>+16.8e}")
        print("=" * 70)

        # LaTeX (booktabs) table — copy-paste with \usepackage{booktabs, float}
        latex_lines = [
            r"\begin{table}[H]",
            r"  \centering",
            r"  \caption{HR-EBSD post-TFBC strain and rotation statistics on $|\cdot|$  "
                f"({rows}~rows $\\times$~{cols}~cols, sample frame, absolute strains)."
            r"}",
            r"  \label{tab:post_tfbc_summary}",
            r"  \begin{tabular}{lrrrr}",
            r"    \toprule",
            r"    Component & $\min |\cdot|$ & $\max |\cdot|$ & $\mathrm{mean}\,|\cdot|$ & $\mathrm{std}\,|\cdot|$ \\",
            r"    \midrule",
        ]
        for _, latex_lbl, arr in post_tfbc_components:
            arr_abs = np.abs(arr)
            cells = [
                _latex_sci(np.nanmin(arr_abs)),
                _latex_sci(np.nanmax(arr_abs)),
                _latex_sci(np.nanmean(arr_abs)),
                _latex_sci(np.nanstd(arr_abs)),
            ]
            latex_lines.append(
                f"    {latex_lbl} & ${cells[0]}$ & ${cells[1]}$ & ${cells[2]}$ & ${cells[3]}$ \\\\"
            )
        latex_lines += [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
        latex_table = "\n".join(latex_lines)

        print()
        print("LaTeX table (post-TFBC, copy-paste into your document):")
        print("-" * 70)
        print(latex_table)
        print("-" * 70)

        sp_tex = _sp("summary_after_TFBC.tex")
        if sp_tex:
            try:
                with open(sp_tex, "w") as f:
                    f.write(latex_table + "\n")
                print(f"Saved LaTeX table to: {sp_tex}")
            except OSError as exc:
                print(f"Could not write LaTeX file ({sp_tex}): {exc}")

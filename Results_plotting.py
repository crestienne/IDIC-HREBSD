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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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
    in-plane strains e11_abs / e22_abs and the lattice tetragonality c/a.

    Requires ``results`` to contain:
        'F_flat'     : (N, 3, 3) deformation gradients from VisWorker
        'base_quats' : (N, 4) quaternions from the .ang file
        'e11'..'e23' : 2-D relative strain maps (rows × cols)

    Requires ``params`` to contain:
        'crystal_C11', 'crystal_C12', 'crystal_C44' : elastic constants in GPa
        'crystal_structure' : 'cubic' (only cubic is supported currently)

    Returns a dict with keys:
        'e11_abs', 'e22_abs', 'e33_abs' : 2-D absolute strain maps
        'tetragonality'                  : 2-D c/a map
    """
    import utilities, rotations
    from scipy.linalg import polar as polar_decomp

    rows = results["rows"]
    cols = results["cols"]
    N    = rows * cols

    F_flat    = results["F_flat"]          # (N, 3, 3)
    base_quats = results["base_quats"]     # (N, 4)

    # ── Per-pattern quaternion: base orientation ⊗ local rotation from F ─────
    quats_total = np.empty((N, 4), dtype=np.float64)
    for idx in range(N):
        R_base         = rotations.qu2om(base_quats[idx])   # (3, 3)
        R_local, _     = polar_decomp(F_flat[idx])          # rotation part of F
        R_total        = R_base @ R_local
        quats_total[idx] = rotations.om2qu(R_total)

    # ── Rotate stiffness tensor to sample frame for each pattern ──────────────
    C11 = params["crystal_C11"]
    C12 = params["crystal_C12"]
    C44 = params["crystal_C44"]
    structure = params.get("crystal_structure", "cubic")

    C_crystal = utilities.get_stiffness_tensor(C11, C12, C44, structure=structure)
    C_rot     = utilities.rotate_stiffness_to_sample_frame(C_crystal, quats_total)  # (N, 6, 6)

    # ── Voigt index: 0=11, 1=22, 2=33, 3=23, 4=13, 5=12 ─────────────────────
    # σ₃₃ = C₃₁(e11+e33) + C₃₂(e22+e33) + C₃₃e33 + 2C₃₄e23 + 2C₃₅e13 + 2C₃₆e12 = 0
    # => e33 = -(C₃₁*e11_dev + C₃₂*e22_dev + 2*C₃₄*e23 + 2*C₃₅*e13 + 2*C₃₆*e12)
    #          / (C₃₁ + C₃₂ + C₃₃)
    C31 = C_rot[:, 2, 0]; C32 = C_rot[:, 2, 1]; C33 = C_rot[:, 2, 2]
    C34 = C_rot[:, 2, 3]; C35 = C_rot[:, 2, 4]; C36 = C_rot[:, 2, 5]

    e11_f = results["e11"].ravel()
    e22_f = results["e22"].ravel()
    e12_f = results["e12"].ravel()
    e13_f = results["e13"].ravel()
    e23_f = results["e23"].ravel()

    numerator   = C31*e11_f + C32*e22_f + 2*C34*e23_f + 2*C35*e13_f + 2*C36*e12_f
    denominator = C31 + C32 + C33
    e33_abs_f   = -numerator / denominator
    e11_abs_f   = e11_f + e33_abs_f
    e22_abs_f   = e22_f + e33_abs_f

    e33_abs = e33_abs_f.reshape(rows, cols)
    e11_abs = e11_abs_f.reshape(rows, cols)
    e22_abs = e22_abs_f.reshape(rows, cols)

    e_in_plane    = 0.5 * (e11_abs + e22_abs)
    tetragonality = (1 + e33_abs) / (1 + e_in_plane)

    return {
        "e11_abs":      e11_abs,
        "e22_abs":      e22_abs,
        "e33_abs":      e33_abs,
        "tetragonality": tetragonality,
    }


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

    h11, h12, h13 = results["h11"], results["h12"], results["h13"]
    h21, h22, h23 = results["h21"], results["h22"], results["h23"]
    h31, h32       = results["h31"], results["h32"]
    e11, e12, e13 = results["e11"], results["e12"], results["e13"]
    e22, e23, e33 = results["e22"], results["e23"], results["e33"]
    w13, w21, w32 = results["w13"], results["w21"], results["w32"]

    def _sp(name):
        return os.path.join(save, name) if save else None

    # ── Homography components ─────────────────────────────────────────────────
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

    # ── Strain + rotation (coolwarm) — shown but saved only after TFBC ────────
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
        save_path=None,   # not saved — TFBC-corrected version is saved below
    )

    # ── Strain line maps ──────────────────────────────────────────────────────
    x_pos = np.arange(cols)

    def _sci_fmt(ax):
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((-3, -3))
        ax.yaxis.set_major_formatter(fmt)

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
    axes_lm[0].set_title(f"Normal strains — row {r}")
    axes_lm[0].axhline(0, color="k", linewidth=0.6, linestyle="--")
    axes_lm[0].legend(fontsize=11)
    axes_lm[0].grid(True, linestyle="--", alpha=0.4)

    for data, label, color in shear_strains:
        axes_lm[1].plot(x_pos, data, label=label, linewidth=1.5, color=color)
    axes_lm[1].set_ylim(-20e-3, 20e-3)
    _sci_fmt(axes_lm[1])
    axes_lm[1].set_ylabel("Shear strain")
    axes_lm[1].set_title(f"Shear strains — row {r}")
    axes_lm[1].axhline(0, color="k", linewidth=0.6, linestyle="--")
    axes_lm[1].legend(fontsize=11)
    axes_lm[1].grid(True, linestyle="--", alpha=0.4)

    for data, label, color in rot_lines:
        axes_lm[2].plot(x_pos, data, label=label, linewidth=1.5, color=color)
    axes_lm[2].set_ylabel("Rotation (°)")
    axes_lm[2].set_xlabel("Column (pattern index)")
    axes_lm[2].set_title(f"Lattice rotations — row {r}")
    axes_lm[2].axhline(0, color="k", linewidth=0.6, linestyle="--")
    axes_lm[2].legend(fontsize=11)
    axes_lm[2].grid(True, linestyle="--", alpha=0.4)

    fig_lm.suptitle(f"Strain line maps — row {r}", fontsize=13)
    plt.tight_layout()
    sp = _sp("Strain_linemap.png")
    if sp:
        plt.savefig(sp, dpi=200, bbox_inches="tight")
    plt.show(block=False)

    # ── Traction-free BC ──────────────────────────────────────────────────────
    if "F_flat" in results and "base_quats" in results:
        tfbc = compute_tfbc(results, params)
        e11_abs      = tfbc["e11_abs"]
        e22_abs      = tfbc["e22_abs"]
        e33_abs      = tfbc["e33_abs"]
        tetragonality = tfbc["tetragonality"]

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

        # Tetragonality map
        ca_mean = tetragonality.mean()
        ca_std  = tetragonality.std()
        fig_ca, ax_ca = plt.subplots(figsize=(8, 4))
        im_ca = ax_ca.imshow(tetragonality, cmap="RdBu_r",
                             vmin=ca_mean - ca_std, vmax=ca_mean + 3 * ca_std)
        cb_ca = fig_ca.colorbar(im_ca, ax=ax_ca, fraction=0.03, pad=0.04)
        cb_ca.set_label(r"$c/a$", fontsize=13)
        ax_ca.set_title(
            r"Lattice tetragonality  $c/a$",
            fontsize=13,
        )
        ax_ca.axis("off")
        plt.tight_layout()
        sp_ca = _sp("Tetragonality_ca.png")
        if sp_ca:
            plt.savefig(sp_ca, dpi=200, bbox_inches="tight")
        plt.show(block=False)

        # TFBC line maps (normal strains + tetragonality along chosen row)
        fig_tf, axes_tf = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        x_pos_tf = np.arange(cols)

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
        axes_tf[0].set_title(f"Absolute normal strains — row {r}")
        axes_tf[0].axhline(0, color="k", linewidth=0.6, linestyle="--")
        axes_tf[0].legend(fontsize=11)
        axes_tf[0].grid(True, linestyle="--", alpha=0.4)

        axes_tf[1].plot(x_pos_tf, tetragonality[r], color="tab:red", linewidth=1.5, label=r"$c/a$")
        axes_tf[1].axhline(ca_mean, color="k", linewidth=0.8, linestyle="--",
                           label=f"mean = {ca_mean:.6f}")
        axes_tf[1].set_ylabel(r"$c/a$")
        axes_tf[1].set_xlabel("Column (pattern index)")
        axes_tf[1].set_title(f"Tetragonality — row {r}")
        axes_tf[1].legend(fontsize=11)
        axes_tf[1].grid(True, linestyle="--", alpha=0.4)

        fig_tf.suptitle(f"Traction-free BC results — row {r}", fontsize=13)
        plt.tight_layout()
        sp_tf = _sp("TFBC_linemap.png")
        if sp_tf:
            plt.savefig(sp_tf, dpi=200, bbox_inches="tight")
        plt.show(block=False)

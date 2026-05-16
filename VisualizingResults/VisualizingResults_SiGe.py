import matplotlib
matplotlib.use("TkAgg")  # interactive backend — pop-up windows per figure

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.linalg import expm
import conversions
import utilities


# ============================================================
# PLOTTING HELPER  (defined in Results_plotting.py)
# ============================================================

from Results_plotting import plot_component_grid
from pc_homography_correction import correct_homographies


# ============================================================
# INPUTS
# ============================================================

#EMEBSD version
#file outputed from runner script
filename = '/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/SiGe/SiGe_2rows_newup2_April_17_2026_npyfiles/SiGe_2rows_newup2_homographies_April_17_2026.npy'

#where to save the figures
foldername = f'/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/SiGe/SiGe_2rows_newup2_April_17_2026_npyfiles/'

# ---- Euler angle source ----
# Set ang_file to the path of a .ang file to read per-pattern Euler angles from it, Set to None to use the single euler_angles_deg value below for all patterns.
#ang_file = '/Users/crestiennedechaine/OriginalData/Si_Ge_Dataset/DI_largerRegion/SiGe_dp_10rows132colums_largerRegion.ang'  # e.g. '/path/to/scan.ang'
ang_file = None

e1 = np.rad2deg(2.71494)
e2 = np.rad2deg(0.03253)
e3 = np.rad2deg(2.79162)

euler_angles_deg = np.array([e1, e2, e3]) # [phi1, Phi, phi2] in degrees

print(f"Using single Euler angle {euler_angles_deg} deg for all patterns (ang_file is None)")

Rows = 2 #rows in the analysed ROI
Columns = 132 #columns in the analysed ROI

# ---- ROI offset within the full scan ----
# If the run only covered a rectangular subset of the full scan, set the
# top-left corner here so that the correct Euler angles are sliced from
# the ANG file.  Set both to 0 if the ROI starts at the scan origin.
roi_row_start = 0   # first row of the full scan included in the run
roi_col_start = 0   # first column of the full scan included in the run
patshape = np.array([512, 512])  # shape of the detector in pixels (height, width)
pixel_size = 30.0  # pixel size in microns

# ---- Tilt for the rotation matrix, in degrees
SampleTilt = 70.0
DetectorTilt = 10.0
samp_frame = True  # whether to visualize in the sample reference frame (True) or EBSD reference frame (False)
# Physical step size from the .ang file XSTEP header, in microns.
step_size_um = 2.6  # <-- update from your .ang file

# ---- PC drift correction ----
apply_pc_correction = True  # set True to remove geometric PC-drift from homographies
pc_convention = "upper_left"   # "standard"        (origin lower-right, x←, y↑)
                              # "direct_electron" (origin upper-right, x←, y↓)
                              # "upper_left"      (origin upper-left,  x→, y↓)



pattern_center_edax = np.array([0.6871, 0.8929, 1.06971])  # EDAX convention for PC from upper left
pattern_center = conversions.Edax_to_Bruker_PC(pattern_center_edax)  # Bruker convention for PC from upper left
homography_center = np.array([0.5, 0.5])  # homography center in fractional coordinates (x, y), typically (0.5, 0.5) for centred-pixel format
xo = conversions.Bruker_to_fractional_PC(pattern_center, patshape, pixel_size,homography_center)  # convert to h2F fractional PC format (x*, y*, z*), where x*, y* are fractional relative to the pattern shape
print(f"xo (fractional coordinates for h2F): {xo}")

os.makedirs(foldername, exist_ok=True)

# ============================================================
# LOAD HOMOGRAPHIES + PC CORRECTION + h2F + F2STRAIN
# ============================================================

h = np.load(filename)


if h.shape[1] != 8:
    # assume h is in shape (Rows, Columns, 8) and reshape to (Rows*Columns, 8)
    h = h.reshape(Rows * Columns, 8) 

# h is in column major order convert h to row major order for comparison
h11 = h[:, 0]
h12 = h[:, 1]
h13 = h[:, 2]
h21 = h[:, 3]
h22 = h[:, 4]
h23 = h[:, 5]
h31 = h[:, 6]
h32 = h[:, 7]

# restack h in row major order
h_calc = np.stack((h11, h12, h13, h21, h22, h23, h31, h32), axis=1)
h_before = h_calc.copy()   # snapshot before any PC correction

if apply_pc_correction:
    h_calc, TS_inv = correct_homographies(
        h                 = h_calc,
        scan_shape        = (Rows, Columns),
        step_size_um      = step_size_um,
        pc_ref            = pattern_center,
        patshape          = tuple(patshape),
        pixel_size_um     = pixel_size,
        sample_tilt_deg   = SampleTilt,
        detector_tilt_deg = DetectorTilt,
        convention        = pc_convention,
    )

    # ── TS_inv component grid ────────────────────────────────────────────────
    # TS_inv shape: (Rows, Columns, 3, 3)
    # Rows encode the geometric PC-drift correction applied at each scan point.
    # At the origin [0,0] every component equals the identity matrix.
    _ts_labels = [
        (0, 0, r"$TS^{-1}_{11}$"),
        (0, 1, r"$TS^{-1}_{12}$"),
        (0, 2, r"$TS^{-1}_{13}$"),
        (1, 0, r"$TS^{-1}_{21}$"),
        (1, 1, r"$TS^{-1}_{22}$"),
        (1, 2, r"$TS^{-1}_{23}$"),
        (2, 0, r"$TS^{-1}_{31}$"),
        (2, 1, r"$TS^{-1}_{32}$"),
        (2, 2, r"$TS^{-1}_{33}$"),
    ]
    plot_component_grid(
        components=[
            {"data": TS_inv[:, :, i, j], "label": lbl}
            for i, j, lbl in _ts_labels
        ],
        cmap="coolwarm",
        title=r"PC-drift correction matrix $TS^{-1}$ components",
        save_path=f"{foldername}/TS_inv_components.png",
    )
    print("Saved TS_inv component grid.")

_h_labels = [
    r"$h_{11}$", r"$h_{12}$", r"$h_{13}$",
    r"$h_{21}$", r"$h_{22}$", r"$h_{23}$",
    r"$h_{31}$", r"$h_{32}$",
]

# ── homography before vs after comparison (8 rows × 2 cols) ──────────────────
fig_h, axes_h = plt.subplots(8, 2, figsize=(8, 22))
fig_h.suptitle("Homography components — before vs after PC correction", fontsize=12)
axes_h[0, 0].set_title("Before correction", fontsize=11)
axes_h[0, 1].set_title("After correction",  fontsize=11)

for i, lbl in enumerate(_h_labels):
    before_2d = h_before[:, i].reshape(Rows, Columns)
    after_2d  = h_calc[:,   i].reshape(Rows, Columns)
    vmax = max(np.abs(before_2d).max(), np.abs(after_2d).max()) or 1e-9

    for ax, data in zip(axes_h[i], [before_2d, after_2d]):
        im = ax.imshow(data, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        fig_h.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_ylabel(lbl, fontsize=10)
        ax.axis("off")

fig_h.tight_layout()
fig_h.savefig(f"{foldername}/Homography_before_after_correction.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
print("Saved homography before/after comparison.")

F = conversions.h2F(h_calc, xo)

# print F shape
print(F.shape)
epsilon, omega = conversions.F2strain(F)

if samp_frame:
    # ============================================================
    # ROTATION MATRIX
    # ============================================================

    R = utilities.rotation_matrix_passive_version2(DetectorTilt, SampleTilt)

    # ============================================================
    # ROTATE TO SAMPLE FRAME
    # epsilon and omega are rotated from the detector frame to the sample
    # frame before any components are extracted or displayed.
    # R is the passive rotation from sample → detector, so the conjugate
    # R @ T @ R.T transforms a detector-frame tensor T into the sample frame.
    # ============================================================
    print(f'epsilon.shape: {epsilon.shape}')
    for i in range(epsilon.shape[0]):
        epsilon[i, :, :] = R @ epsilon[i, :, :] @ R.T
        omega[i, :, :]   = R @ omega[i, :, :]   @ R.T

    # ============================================================
    # ROTATE STRAIN BY SOLVED LATTICE ROTATION
    # R_omega rotates epsilon from the deformed frame into the reference frame.
    # Must happen before the traction-free BC so sigma_33 = 0 is enforced in
    # the correctly-oriented frame.
    for i in range(epsilon.shape[0]):
        R_omega = expm(omega[i])
        epsilon[i] = R_omega.T @ epsilon[i] @ R_omega

    # ============================================================
    # EXTRACT STRAIN / ROTATION COMPONENTS
    # ============================================================

    e11 = epsilon[:, 0, 0]
    e12 = epsilon[:, 0, 1]
    e13 = epsilon[:, 0, 2]
    e22 = epsilon[:, 1, 1]
    e23 = epsilon[:, 1, 2]
    e33 = epsilon[:, 2, 2]
    w13 = omega[:, 0, 2]
    w21 = omega[:, 1, 0]
    w32 = omega[:, 2, 1]

    # convert the rotation components to degrees
    w13 = np.degrees(w13)
    w21 = np.degrees(w21)
    w32 = np.degrees(w32)

    # ============================================================
    # RESHAPE TO (ROWS, COLUMNS)
    # ============================================================

    e11 = e11.reshape((Rows, Columns))
    e12 = e12.reshape((Rows, Columns))
    e13 = e13.reshape((Rows, Columns))
    e22 = e22.reshape((Rows, Columns))
    e23 = e23.reshape((Rows, Columns))
    e33 = e33.reshape((Rows, Columns))
    w13 = w13.reshape((Rows, Columns))
    w21 = w21.reshape((Rows, Columns))
    w32 = w32.reshape((Rows, Columns))

    # ============================================================
    # TRACTION-FREE BC + PER-PATTERN STIFFNESS TENSOR
    # ============================================================

    # ------ Silicon elastic constants (crystal frame, Voigt notation) ------
    # Si: C11=165.7, C12=63.9, C44=79.6  (GPa)  [Madelung 1982]
    C11_Si = 165.7e9  # Pa
    C12_Si =  63.9e9
    C44_Si =  79.6e9

    C_crystal = utilities.get_stiffness_tensor(C11_Si, C12_Si, C44_Si, structure="cubic")
    print("C_crystal (GPa):")
    print(np.round(C_crystal / 1e9, 2))

    # ------ Per-pattern quaternion: reference orientation ⊗ local rotation from F ------
    # The IC-GN deformation gradient F carries the local lattice rotation at each
    # scan position relative to the reference pattern.  We polar-decompose F = R_local @ U
    # and compose R_local with the base orientation so that C_rot reflects the true
    # per-pattern crystal orientation in the sample frame.
    #
    # Base orientation source:
    #   ang_file is None  → single euler_angles_deg tiled to all patterns
    #   ang_file set      → per-pattern Euler angles read from the .ang file
    import rotations
    from scipy.linalg import polar as polar_decomp

    N = Rows * Columns

    if ang_file is not None:
        ang_data = utilities.read_ang(ang_file, tuple(patshape))
        # Slice the ROI out of the full scan before reshaping.
        # ang_data.quats shape: (full_rows, full_cols, 4)
        roi_quats = ang_data.quats[
            roi_row_start : roi_row_start + Rows,
            roi_col_start : roi_col_start + Columns,
            :,
        ]
        base_quats = roi_quats.reshape(N, 4)
        print(f"Loaded per-pattern orientations from {ang_file} "
              f"(ROI rows {roi_row_start}–{roi_row_start+Rows-1}, "
              f"cols {roi_col_start}–{roi_col_start+Columns-1})")
    else:
        euler_angles_rad = np.deg2rad(euler_angles_deg)
        single_quat = rotations.eu2qu(euler_angles_rad)
        base_quats = np.tile(single_quat, (N, 1))   # (N, 4)
        print(f"Using single Euler angle {euler_angles_deg} deg for all {N} patterns")

    quats_flat = np.empty((N, 4), dtype=np.float64)
    for idx in range(N):
        R_base     = rotations.qu2om(base_quats[idx])  # (3, 3) base orientation at this point
        R_local, _ = polar_decomp(F[idx])              # rotation part of F
        R_total    = R_base @ R_local                   # compose: base then local
        quats_flat[idx] = rotations.om2qu(R_total)

    print(f"Per-pattern quats computed for {N} patterns ({Rows} rows x {Columns} cols)")

    # ------ Rotate C to sample frame for each pattern ------
    # utilities.rotate_stiffness_to_sample_frame returns shape (N, 6, 6)
    C_rot = utilities.rotate_stiffness_to_sample_frame(C_crystal, quats_flat)  # (N, 6, 6)
    print("C_rot shape:", C_rot.shape)
    print("Mean C_rot[0,0] (C11 equivalent, GPa):", np.round(C_rot[:, 0, 0].mean() / 1e9, 2))

    # ------ Traction-free BC: solve for e33_abs using per-pattern rotated C ------
    # Voigt index mapping: 0=11, 1=22, 2=33, 3=23, 4=13, 5=12
    # sigma_33 = C31*(e11_dev+e33) + C32*(e22_dev+e33) + C33*e33
    #          + 2*C34*e23 + 2*C35*e13 + 2*C36*e12 = 0
    # => e33 = -(C31*e11_dev + C32*e22_dev + 2*C34*e23 + 2*C35*e13 + 2*C36*e12)
    #          / (C31 + C32 + C33)

    # Flatten the 2D strain maps to 1D for vectorised calculation
    e11_flat = e11.ravel()   # deviatoric
    e22_flat = e22.ravel()
    e12_flat = e12.ravel()
    e13_flat = e13.ravel()
    e23_flat = e23.ravel()

    C31 = C_rot[:, 2, 0]   # (N,)
    C32 = C_rot[:, 2, 1]
    C33 = C_rot[:, 2, 2]
    C34 = C_rot[:, 2, 3]
    C35 = C_rot[:, 2, 4]
    C36 = C_rot[:, 2, 5]

    numerator   = C31*e11_flat + C32*e22_flat + 2*C34*e23_flat + 2*C35*e13_flat + 2*C36*e12_flat
    denominator = C31 + C32 + C33

    e33_abs_flat = -numerator / denominator
    e11_abs_flat = e11_flat + e33_abs_flat
    e22_abs_flat = e22_flat + e33_abs_flat

    # Reshape back to (Rows, Columns)
    e33_abs = e33_abs_flat.reshape(Rows, Columns)
    e11_abs = e11_abs_flat.reshape(Rows, Columns)
    e22_abs = e22_abs_flat.reshape(Rows, Columns)

    # Tetragonality c/a
    e_in_plane = 0.5 * (e11_abs + e22_abs)
    tetragonality_ratio = (1 + e33_abs) / (1 + e_in_plane)

    print(f"Mean e33_abs : {e33_abs.mean():.4e}")
    print(f"Mean e11_abs : {e11_abs.mean():.4e}")
    print(f"Mean e22_abs : {e22_abs.mean():.4e}")
    print(f"Mean c/a     : {tetragonality_ratio.mean():.6f}")

    # ============================================================
    # FIGURE: LATTICE ROTATIONS (coolwarm)
    # ============================================================

    _rv = 0.25
    plot_component_grid(
        components=[
            {"data": w13, "label": r"$\omega_{13}$ (deg)" + f"  (mean={w13.mean():.2e})", "vmin": -_rv, "vmax": _rv},
            {"data": w21, "label": r"$\omega_{21}$ (deg)" + f"  (mean={w21.mean():.2e})", "vmin": -_rv, "vmax": _rv},
            {"data": w32, "label": r"$\omega_{32}$ (deg)" + f"  (mean={w32.mean():.2e})", "vmin": -_rv, "vmax": _rv},
        ],
        cmap="coolwarm",
        axis_off=True,
        title="Lattice rotations",
        save_path=f"{foldername}/Rotations.png",
    )

    # ============================================================
    # FIGURE: ABSOLUTE STRAIN (coolwarm, traction-free BC, no rotations)
    # Layout mirrors the 3×3 strain tensor — upper triangle only (symmetric).
    #
    #   (0,0) ε₁₁  │  (0,1) ε₁₂  │  (0,2) ε₁₃
    #   (1,0) ───  │  (1,1) ε₂₂  │  (1,2) ε₂₃
    #   (2,0) ───  │  (2,1) ───   │  (2,2) ε₃₃
    # ============================================================

    _sv = 5e-3
    _upper_triangle = [
        (0, 0, e11_abs, r"$\epsilon_{11}^{\mathrm{abs}}$",       e11_abs.mean()),
        (0, 1, e12,     r"$\epsilon_{12}$",                       e12.mean()),
        (0, 2, e13,     r"$\epsilon_{13}$",                       e13.mean()),
        (1, 1, e22_abs, r"$\epsilon_{22}^{\mathrm{abs}}$",       e22_abs.mean()),
        (1, 2, e23,     r"$\epsilon_{23}$",                       e23.mean()),
        (2, 2, e33_abs, r"$\epsilon_{33}^{\mathrm{abs}}$ (TF)", e33_abs.mean()),
    ]

    fig_s, axes_s = plt.subplots(3, 3, figsize=(12, 7))
    fig_s.suptitle("Absolute strain tensor components (traction-free BC)", fontsize=12)

    for ax in axes_s.ravel():
        ax.axis("off")

    for row, col, data, lbl, mean_val in _upper_triangle:
        ax = axes_s[row, col]
        im = ax.imshow(data, cmap="coolwarm", vmin=-_sv, vmax=_sv)
        ax.set_title(f"{lbl}\nmean={mean_val:.2e}", fontsize=10)
        fig_s.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")

    fig_s.tight_layout()
    fig_s.savefig(f"{foldername}/Absolute_strain_grid.png", dpi=200, bbox_inches="tight")
    plt.show(block=False)
    print("Saved absolute strain tensor grid.")

    print(f"Mean e33_abs : {e33_abs.mean():.4e}")
    print(f"Mean e11_abs : {e11_abs.mean():.4e}")
    print(f"Mean e22_abs : {e22_abs.mean():.4e}")

    # ============================================================
    # FIGURE: LINE MAPS
    # ============================================================

    linemap_row = 0   # change to plot a different row

    import matplotlib.ticker as ticker

    def _strain_formatter(ax):
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((-3, -3))
        ax.yaxis.set_major_formatter(fmt)

    x_pos = np.arange(Columns)

    strain_lines = [
        (e11_abs[linemap_row, :], r"$\epsilon_{11}^{\mathrm{abs}}$", "tab:blue"),
        (e22_abs[linemap_row, :], r"$\epsilon_{22}^{\mathrm{abs}}$", "tab:orange"),
        (e33_abs[linemap_row, :], r"$\epsilon_{33}^{\mathrm{abs}}$", "tab:green"),
        (e12[linemap_row, :],     r"$\epsilon_{12}$",                "tab:red"),
        (e13[linemap_row, :],     r"$\epsilon_{13}$",                "tab:purple"),
        (e23[linemap_row, :],     r"$\epsilon_{23}$",                "tab:brown"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Panel 1: normal strains
    ax0 = axes[0]
    for data, label, color in strain_lines[:3]:
        ax0.plot(x_pos, data, label=label, linewidth=1.5, color=color)
    ax0.set_ylim(-20e-3, 20e-3)
    _strain_formatter(ax0)
    ax0.set_ylabel("Strain", fontsize=12)
    ax0.set_title(f"Normal strains along row {linemap_row}", fontsize=12)
    ax0.axhline(0, color="k", linewidth=0.6, linestyle="--")
    ax0.legend(fontsize=11, loc="upper right")
    ax0.grid(True, linestyle="--", alpha=0.4)

    # Panel 2: shear strains
    ax1 = axes[1]
    for data, label, color in strain_lines[3:]:
        ax1.plot(x_pos, data, label=label, linewidth=1.5, color=color)
    ax1.set_ylim(-20e-3, 20e-3)
    _strain_formatter(ax1)
    ax1.set_ylabel("Shear strain", fontsize=12)
    ax1.set_xlabel("Column (pattern index)", fontsize=12)
    ax1.set_title(f"Shear strains along row {linemap_row}", fontsize=12)
    ax1.axhline(0, color="k", linewidth=0.6, linestyle="--")
    ax1.legend(fontsize=11, loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle(
        f"Strain line maps — row {linemap_row}\n",
        fontsize=13,
    )
    plt.tight_layout()
    save_path = f"{foldername}/Strain_linemap_row{linemap_row}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show(block=False)
    print(f"Saved {save_path}")

else:
    print("samp_frame=False: skipping strain and rotation plots.")

# Keep all windows open until user closes them
plt.show()

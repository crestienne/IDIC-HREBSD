import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import warp
import conversions

size = (480, 640)

# ── Canonical square outline ──────────────────────────────────────────────────
# Original square corners as a closed loop in PC-relative coords (PC at image
# centre, matching warp.W / deform_image default).  We track only the four
# corners — projective transforms preserve straight lines, so connecting the
# warped corners with line segments is exact for any h (including perspective).
PC = (size[1] / 2.0, size[0] / 2.0)
sq_x_pc = np.array([-size[1] / 4, size[1] / 4, size[1] / 4, -size[1] / 4, -size[1] / 4])
sq_y_pc = np.array([-size[0] / 4, -size[0] / 4, size[0] / 4, size[0] / 4, -size[0] / 4])
orig_x_px = sq_x_pc + PC[0]
orig_y_px = sq_y_pc + PC[1]


def warped_corners(h):
    """Apply W(h) to the canonical square corners; return image-pixel coords."""
    Wh = warp.W(np.asarray(h, dtype=float))
    pts = np.vstack([sq_x_pc, sq_y_pc, np.ones_like(sq_x_pc)])
    pts_w = Wh @ pts
    pts_w = pts_w / pts_w[2:3]   # perspective normalise
    return pts_w[0] + PC[0], pts_w[1] + PC[1]


def style_outline_axis(ax):
    """White-background axis sized to the image, with image-style y-axis."""
    ax.set_xlim(0, size[1])
    ax.set_ylim(size[0], 0)         # invert y so it increases downward
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.tick_params(axis="both", which="both",
                   bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labeltop=False, labelleft=False, labelright=False)


# ── Figure 1: per-homography-component outlines ──────────────────────────────
# `homographies` are in the IC-GN / h2F convention: positive h11 = tensile in
# x (target stretched), positive h13 = target shifted +50 px right of the
# reference, etc.  Applying W(h) to corners gives the forward deformation
# (where features in R appear in T) — no inversion needed.
homographies = np.array([
    [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 50., 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 50., 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])

l = [r"$h_{11}$", r"$h_{12}$", r"$h_{13}$", r"$h_{21}$", r"$h_{22}$", r"$h_{23}$", r"$h_{31}$", r"$h_{32}$"]

ratio = size[0] / size[1]
fig, ax = plt.subplots(3, 3, figsize=(8, 8 * ratio), facecolor="white")
plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
ax = ax.ravel()
for i in range(9):
    style_outline_axis(ax[i])
    ax[i].plot(orig_x_px, orig_y_px, color="black", linewidth=1.5)
    if i != 8:
        wx, wy = warped_corners(homographies[i])
        ax[i].plot(wx, wy, color="red", linewidth=1.5)
        ax[i].text(size[1] // 2, size[0] // 2,
                   f"{l[i]} = {homographies[i, i]}",
                   va="center", ha="center", color="black")
    else:
        ax[i].text(size[1] // 2, size[0] // 2,
                   f"Image\n{(size[1], size[0])}\n\nSquare\n{(size[1] // 2, size[0] // 2)}",
                   va="center", ha="center", color="black")


# ── Figure 2: per-strain-component outlines ──────────────────────────────────
# Each panel (i, j) builds ε with only ε_ij (and ε_ji) active, forms F = I+ε,
# converts to homography via F2h with the chosen pattern centre xo, then warps
# the square corners.  Off-diagonal panels appear in symmetric pairs (ε12 ↔
# ε21, etc.) — those will look identical, which is the symmetry of ε made
# visible.  ε33 is gauge-invariant in this F2strain formulation (always 0
# because epsilon is normalised by v_stretch[2,2]) — the (2,2) panel shows
# only the original outline.
xo = (0.0, 0.0, 1000.0)
eps_magnitude = 0.1

fig2, ax2 = plt.subplots(3, 3, figsize=(8, 8 * ratio), facecolor="white")
plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
ax2 = ax2.ravel()
for panel_idx in range(9):
    i, j = divmod(panel_idx, 3)
    style_outline_axis(ax2[panel_idx])
    ax2[panel_idx].plot(orig_x_px, orig_y_px, color="black", linewidth=1.5)

    if (i, j) == (2, 2):
        label = r"$\varepsilon_{33}$ = 0" + "\n(gauge-invariant)"
    else:
        eps_input = np.zeros((3, 3))
        eps_input[i, j] = eps_magnitude
        eps_input[j, i] = eps_magnitude  # symmetrise
        F = np.eye(3) + eps_input
        h = conversions.F2h(F, xo)
        wx, wy = warped_corners(h)
        ax2[panel_idx].plot(wx, wy, color="red", linewidth=1.5)
        label = (r"$\varepsilon_{" + f"{i+1}{j+1}" + r"}$"
                 + f" = {eps_magnitude}")

    ax2[panel_idx].text(size[1] // 2, size[0] // 2, label,
                        va="center", ha="center", color="black")


# ── Figure 3: per-rotation-component outlines ────────────────────────────────
# Lattice rotation ω is antisymmetric (ω_ij = −ω_ji) — it has only 3
# independent components, but laying it out in a 3x3 grid makes the
# antisymmetry visible: panel (i, j) and panel (j, i) show opposite-sense
# rotations (mirror-image deformations), and the diagonal panels are exactly
# the original square (ω_ii ≡ 0).
#
# Pipeline per off-diagonal panel: build antisymmetric ω with ω[i,j] = +θ
# and ω[j,i] = −θ; form the EXACT rotation F = expm(ω) (so each panel is a
# pure rotation with no strain leakage from the small-angle approximation
# F ≈ I + ω); convert to homography via F2h with the same xo as fig 2; warp
# the square corners.
#
# Note: ω12 / ω21 is in-plane rotation about the detector normal (z) — visually
# a tilted square.  ω13 / ω31 and ω23 / ω32 are out-of-plane rotations about
# y / x — visually they appear as projective foreshortening on the detector
# (because rotating the lattice out of the detector plane changes how the
# Kikuchi pattern projects).
xo_rot = xo
omega_deg = 1.0           # rotation magnitude per panel
omega_rad = np.radians(omega_deg)

fig3, ax3 = plt.subplots(3, 3, figsize=(8, 8 * ratio), facecolor="white")
plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
ax3 = ax3.ravel()
for panel_idx in range(9):
    i, j = divmod(panel_idx, 3)
    style_outline_axis(ax3[panel_idx])
    ax3[panel_idx].plot(orig_x_px, orig_y_px, color="black", linewidth=1.5)

    if i == j:
        label = (r"$\omega_{" + f"{i+1}{j+1}" + r"}$ = 0"
                 + "\n(antisymmetric)")
    else:
        Omega = np.zeros((3, 3))
        Omega[i, j] = +omega_rad
        Omega[j, i] = -omega_rad        # antisymmetrise
        F = expm(Omega)                 # exact rotation matrix
        h = conversions.F2h(F, xo_rot)
        wx, wy = warped_corners(h)
        ax3[panel_idx].plot(wx, wy, color="red", linewidth=1.5)
        label = (r"$\omega_{" + f"{i+1}{j+1}" + r"}$"
                 + f" = +{omega_deg:g}°")

    ax3[panel_idx].text(size[1] // 2, size[0] // 2, label,
                        va="center", ha="center", color="black")

plt.show()

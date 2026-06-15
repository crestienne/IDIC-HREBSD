"""
figure_pc_shift_vs_strain.py — visual proof that a pattern-center (PC) shift is
NOT the same as an elastic strain.

Both a PC error and a real strain move features around on the detector, and the
two partially couple — which is exactly why a PC error biases the measured
strain.  But geometrically they are different:

  • an in-plane PC shift translates the gnomonic projection origin, so EVERY
    detector pixel moves by the SAME vector — a rigid translation (zero
    displacement gradient);
  • a strain (deformation gradient F = I + ε) produces a displacement field that
    VARIES with position — zero at the PC, growing outward as ε·r (stretch +
    shear).  No uniform PC shift can reproduce that spatial variation.

This script is purely geometric — no master pattern, no .up2, no torch.  It
reuses the pipeline's own homography math so the contrast is on the solver's
terms:

  • PC shift  → a pure-translation homography  [0,0,dx, 0,0,dy, 0,0]
                (an in-plane PC shift IS a uniform pixel translation of the
                 projection; this is exact, not a first-order approximation).
  • strain    → conversions.F2h(I + ε, X0)
  • both fields are pushed through warp.get_xi_prime() (the same per-pixel warp
    the IC-GN solver uses) so they share one code path.

The punchline panel subtracts the best-fit *uniform translation* (the spatial
mean) from each field: the PC field vanishes (it was a pure shift), while the
strain field leaves a large, structured residual — the part a PC shift can never
produce.

Run
---
    python figure_pc_shift_vs_strain.py

Writes figures/pc_shift_vs_strain/pc_shift_vs_strain.{pdf,png} and prints, for
each field, the uniform displacement t, the displacement-gradient matrix A, and
the residual RMS.  Set SHOW = True to also pop it up interactively.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import conversions
import warp


# ───────── USER INPUTS ───────────────────────────────────────────────────────

det_shape = (512, 512)            # (H, W) detector pixels
pc_edax   = (0.5, 0.5, 0.65)      # EDAX (x*, y*, z*); centred PC for a clean demo

# Representative small strain (deviatoric mix): normal + shear so both kinds of
# structure show up in the field.  ε33 is implied by the trace; only the in-plane
# 2×2 block warps the detector pattern.
eps = np.array([
    [+0.0030, +0.0020, 0.0],
    [+0.0020, -0.0015, 0.0],
    [ 0.0,     0.0,    0.0],
])

# In-plane PC shift direction (the magnitude is auto-normalised below to match
# the strain field's peak displacement, so the contrast is about STRUCTURE, not
# magnitude).
pc_shift_dir = (1.0, 0.6)         # (Δx, Δy) direction, will be rescaled

OUT_DIR  = "figures/pc_shift_vs_strain"
SAVE_PNG = True
SHOW     = False

N_QUIVER = 20                     # arrows per axis in the quiver panels


# ───────── Publication style ─────────────────────────────────────────────────

plt.rcParams.update({
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})


# ───────── Geometry ──────────────────────────────────────────────────────────

H, W = det_shape

# PC offset X0 = (x01, x02, DD) in PIXELS, the format F2h / xyt2h expect:
# vector from PC to the homography (pattern) centre, plus detector distance.
pc_bruker = conversions.Edax_to_Bruker_PC(np.asarray(pc_edax, dtype=float))
X0 = conversions.Bruker_to_fractional_PC(pc_bruker, det_shape)   # (x01, x02, DD) px

# Pixel grid in PC-centred coordinates (matches the homography convention).
xx = np.arange(W) - W / 2.0
yy = np.arange(H) - H / 2.0
XX, YY = np.meshgrid(xx, yy, indexing="xy")          # (H, W)
xi = np.vstack([XX.ravel(), YY.ravel()])             # (2, N)


def displacement(h):
    """Per-pixel displacement field u = warp(xi) - xi for homography h (8,)."""
    u = warp.get_xi_prime(xi, np.asarray(h, dtype=float)) - xi
    return u[0].reshape(H, W), u[1].reshape(H, W)


# ── Strain field: homography from F = I + ε, then the shared warp ────────────
F = np.eye(3) + eps
h_strain = conversions.F2h(F, X0)
ux_s, uy_s = displacement(h_strain)
peak_strain = float(np.sqrt(ux_s**2 + uy_s**2).max())

# ── PC shift field: a pure uniform translation, normalised to the same peak ──
# An in-plane PC shift moves every pixel by exactly (dx, dy): a rigid translation
# (homography [0,0,dx, 0,0,dy, 0,0]).  Scale (dx, dy) so |u_pc| == peak_strain,
# making the comparison purely about spatial STRUCTURE, not magnitude.
d = np.asarray(pc_shift_dir, dtype=float)
d = d / (np.linalg.norm(d) + 1e-12) * peak_strain
h_pc = np.array([0.0, 0.0, d[0], 0.0, 0.0, d[1], 0.0, 0.0])
ux_p, uy_p = displacement(h_pc)


# ───────── Quantify: uniform part t, gradient A, structured residual ──────────

def affine_fit(ux, uy):
    """Least-squares fit u(x,y) ≈ t + A·[x,y].  Returns (t (2,), A (2,2))."""
    M = np.column_stack([np.ones(XX.size), XX.ravel(), YY.ravel()])
    cx, *_ = np.linalg.lstsq(M, ux.ravel(), rcond=None)
    cy, *_ = np.linalg.lstsq(M, uy.ravel(), rcond=None)
    t = np.array([cx[0], cy[0]])
    A = np.array([[cx[1], cx[2]],
                  [cy[1], cy[2]]])
    return t, A


def residual_after_uniform(ux, uy):
    """u minus its best-fit uniform translation (the spatial mean)."""
    rx = ux - ux.mean()
    ry = uy - uy.mean()
    return rx, ry


t_p, A_p = affine_fit(ux_p, uy_p)
t_s, A_s = affine_fit(ux_s, uy_s)
rx_p, ry_p = residual_after_uniform(ux_p, uy_p)
rx_s, ry_s = residual_after_uniform(ux_s, uy_s)
resmag_p = np.sqrt(rx_p**2 + ry_p**2)
resmag_s = np.sqrt(rx_s**2 + ry_s**2)
rms = lambda a: float(np.sqrt(np.mean(a**2)))


# ───────── Figure ────────────────────────────────────────────────────────────

mag_p = np.sqrt(ux_p**2 + uy_p**2)
mag_s = np.sqrt(ux_s**2 + uy_s**2)
vmax_mag = max(mag_p.max(), mag_s.max())
vmax_res = max(resmag_p.max(), resmag_s.max())

# Quiver subsampling
step_r = max(1, H // N_QUIVER)
step_c = max(1, W // N_QUIVER)
sl = (slice(step_r // 2, None, step_r), slice(step_c // 2, None, step_c))
Xq, Yq = XX[sl], YY[sl]
# Common quiver scale so PC (uniform) and strain (growing) arrows are comparable.
q_scale = vmax_mag * N_QUIVER * 1.6


def _letter(ax, s, color="black"):
    ax.text(0.03, 0.97, s, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="top", ha="left", color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))


def _quiver_panel(ax, ux, uy, title, letter):
    # imshow extent so (0,0) is the PC at image centre; flip v for image y-down.
    ax.quiver(Xq, Yq, ux[sl], -uy[sl], color="tab:blue",
              angles="xy", scale=q_scale, width=0.004,
              headwidth=4, headlength=5, pivot="mid")
    ax.set_xlim(xx[0], xx[-1]); ax.set_ylim(yy[-1], yy[0])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    ax.plot(0, 0, "r+", ms=9, mew=1.5)          # mark the PC
    _letter(ax, letter)


def _map_panel(ax, field, title, letter, vmax, cmap="inferno"):
    im = ax.imshow(field, cmap=cmap, vmin=0, vmax=vmax,
                   extent=[xx[0], xx[-1], yy[-1], yy[0]])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
    ax.plot(0, 0, "c+", ms=9, mew=1.5)
    _letter(ax, letter, color="white")
    return im


fig = plt.figure(figsize=(11, 7.2))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.18, wspace=0.16)

# Row 1 — PC shift
ax_a = fig.add_subplot(gs[0, 0]); _quiver_panel(ax_a, ux_p, uy_p, "PC shift — displacement field", "(a)")
ax_b = fig.add_subplot(gs[0, 1]); im_mag = _map_panel(ax_b, mag_p, "PC shift — |u| (flat)", "(b)", vmax_mag)
ax_c = fig.add_subplot(gs[0, 2]); im_res = _map_panel(ax_c, resmag_p, "PC shift — residual after uniform shift", "(c)", vmax_res)

# Row 2 — strain
ax_d = fig.add_subplot(gs[1, 0]); _quiver_panel(ax_d, ux_s, uy_s, "Strain — displacement field", "(d)")
ax_e = fig.add_subplot(gs[1, 1]); _map_panel(ax_e, mag_s, "Strain — |u| (grows from PC)", "(e)", vmax_mag)
ax_f = fig.add_subplot(gs[1, 2]); _map_panel(ax_f, resmag_s, "Strain — residual after uniform shift", "(f)", vmax_res)

# Shared colorbars
cb1 = fig.colorbar(im_mag, ax=[ax_b, ax_e], fraction=0.046, pad=0.02)
cb1.set_label("|u|  (px)")
cb2 = fig.colorbar(im_res, ax=[ax_c, ax_f], fraction=0.046, pad=0.02)
cb2.set_label("residual |u − ū|  (px)")

fig.suptitle(
    "A pattern-center shift is a rigid translation; a strain is not\n"
    f"(peak |u| matched at {peak_strain:.3f} px so the contrast is structural, not magnitude)",
    fontsize=11, fontweight="bold", y=0.99,
)


# ───────── Save / report ──────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)
pdf_path = os.path.join(OUT_DIR, "pc_shift_vs_strain.pdf")
fig.savefig(pdf_path, bbox_inches="tight")
print(f"Saved {pdf_path}")
if SAVE_PNG:
    png_path = os.path.join(OUT_DIR, "pc_shift_vs_strain.png")
    fig.savefig(png_path, bbox_inches="tight")
    print(f"Saved {png_path}")

np.set_printoptions(precision=5, suppress=True)
print("\n" + "=" * 64)
print("  Displacement-field decomposition  u(x,y) ≈ t + A·[x,y]")
print("=" * 64)
for name, t, A, resmag in [
    ("PC shift", t_p, A_p, resmag_p),
    ("strain  ", t_s, A_s, resmag_s),
]:
    print(f"\n  {name}:")
    print(f"    uniform shift t      = {t}  (px)")
    print(f"    displacement grad A  = {A.tolist()}")
    print(f"    ‖A‖_F                = {np.linalg.norm(A):.6e}   "
          f"(0 ⇒ every pixel moves the same)")
    print(f"    residual RMS         = {rms(resmag):.6e} px   "
          f"(after removing best uniform shift)")
print("\n  Takeaway: PC ‖A‖≈0 and residual≈0 (a rigid shift); the strain has")
print("  ‖A‖ on the order of the imposed ε and a large structured residual —")
print("  so no uniform PC shift can reproduce a strain.\n")

if SHOW:
    plt.show()
else:
    plt.close(fig)

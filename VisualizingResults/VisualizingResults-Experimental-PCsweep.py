import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import conversions

# ============================================================
# Inputs
# ============================================================

filename = '/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/Al-MegaLargeArea-rows75to175-columns150to300_Mar122025_npyfiles/Al-MegaLargeArea-rows75to175-columns150to300_homographies_Mar122025.npy'

# Scan dimensions
Rows = 100
Columns = 150

# Nominal pattern center (xstar, ystar, zstar) in EDAX/TSL convention
PC_nominal = (0.4776, 0.5833, 0.670697)
patshape = (480, 480)
tilt = 68.0  # sample tilt in degrees

# PC sweep range: +/- this fraction around nominal xstar and ystar
pc_sweep_half_range = 0.05
print(f"PC sweep range: ±{pc_sweep_half_range*patshape[0]} pixels around nominal xstar and ystar")
n_pc_points = 11   # number of points along each axis (odd → includes nominal)

# Indices of the 5 homographies to examine
# (spread across the dataset; will be set automatically after loading)
n_patterns_to_examine = 5

foldername = '/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/PCSensitivity-Experimental/'
os.makedirs(foldername, exist_ok=True)

# ============================================================
# Helpers
# ============================================================

def rotation_matrix_from_tilt(tilt_deg: float) -> np.ndarray:
    theta_x = np.deg2rad(90.0 - tilt_deg)
    Rx = np.array([
        [1.0, 0.0,               0.0             ],
        [0.0, np.cos(theta_x), -np.sin(theta_x)  ],
        [0.0, np.sin(theta_x),  np.cos(theta_x)  ],
    ])
    Rz_180 = np.array([
        [-1.0,  0.0, 0.0],
        [ 0.0, -1.0, 0.0],
        [ 0.0,  0.0, 1.0],
    ])
    return Rx @ Rz_180


def von_mises(epsilon):
    """Von Mises equivalent strain for an (..., 3, 3) strain tensor array.
    e33 is 0 in the detector frame so the formula reduces accordingly."""
    e11 = epsilon[..., 0, 0]
    e22 = epsilon[..., 1, 1]
    e33 = epsilon[..., 2, 2]
    e12 = epsilon[..., 0, 1]
    e13 = epsilon[..., 0, 2]
    e23 = epsilon[..., 1, 2]
    return (1.0 / np.sqrt(2.0)) * np.sqrt(
        (e11 - e22) ** 2 + (e22 - e33) ** 2 + (e33 - e11) ** 2
        + 6.0 * (e12 ** 2 + e13 ** 2 + e23 ** 2)
    )


def pc_to_x0(xstar, ystar, zstar, patshape):
    return np.array([
        (xstar - 0.5) * patshape[1],
        (ystar - 0.5) * patshape[0],
        patshape[0] * zstar,
    ])


def compute_strain_for_pc(h_calc, xstar, ystar, zstar, patshape):
    """Return epsilon (N,3,3) in the detector frame for a given PC."""
    X0 = pc_to_x0(xstar, ystar, zstar, patshape)
    F = conversions.h2F(h_calc, X0)
    epsilon, omega = conversions.F2strain(F)
    return epsilon, omega


def find_optimal_pc(h_calc, PC_nominal, patshape, bound=10/480, n_grid=30,
                    components=None):
    """Find the delta_xstar and delta_ystar that minimizes mean squared strain
    across all provided homographies (in the detector frame) using a grid search.

    Args:
        h_calc (np.ndarray): Homographies, shape (N, 8).
        PC_nominal (tuple): Nominal (xstar, ystar, zstar).
        patshape (tuple): Detector shape in pixels (rows, cols).
        bound (float): Half-width of the search range in fractional PC units.
                       Default is 10 pixels / patshape width = 10/480.
        n_grid (int): Number of grid points along each axis. Default 30.
        components (list of (int,int)): Strain tensor indices to include in
            the objective. Defaults to all five measurable components
            (excludes e33 which is always 0 in the detector frame).

    Returns:
        delta_xstar (float): Optimal correction to xstar.
        delta_ystar (float): Optimal correction to ystar.
        cost_grid (np.ndarray): (n_grid, n_grid) array of objective values.
        dx_vals (np.ndarray): xstar delta values along the grid axis.
        dy_vals (np.ndarray): ystar delta values along the grid axis.
    """
    if components is None:
        # e11, e12, e13, e22, e23 — e33 is always 0 in detector frame
        components = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]

    xstar_nom, ystar_nom, zstar_nom = PC_nominal

    dx_vals = np.linspace(-bound, bound, n_grid)
    dy_vals = np.linspace(-bound, bound, n_grid)

    cost_grid = np.zeros((n_grid, n_grid))

    for xi, dx in enumerate(dx_vals):
        for yi, dy in enumerate(dy_vals):
            epsilon, _ = compute_strain_for_pc(
                h_calc,
                xstar_nom + dx,
                ystar_nom + dy,
                zstar_nom,
                patshape,
            )
            cost_grid[xi, yi] = sum(
                np.mean(epsilon[:, r, c] ** 2) for r, c in components
            )

    best = np.unravel_index(np.argmin(cost_grid), cost_grid.shape)
    delta_xstar = dx_vals[best[0]]
    delta_ystar = dy_vals[best[1]]

    return delta_xstar, delta_ystar, cost_grid, dx_vals, dy_vals

# ============================================================
# Load data
# ============================================================

h = np.load(filename)

h_calc = np.stack([h[:, i] for i in range(8)], axis=1)

N = h.shape[0]
pat_indices = np.linspace(0, N - 1, n_patterns_to_examine, dtype=int)
print(f"Examining pattern indices: {pat_indices}")

# ============================================================
# PC sweep grid
# ============================================================

xstar_nom, ystar_nom, zstar_nom = PC_nominal

xstar_vals = np.linspace(
    xstar_nom - pc_sweep_half_range,
    xstar_nom + pc_sweep_half_range,
    n_pc_points,
)
ystar_vals = np.linspace(
    ystar_nom - pc_sweep_half_range,
    ystar_nom + pc_sweep_half_range,
    n_pc_points,
)

# strain_grid[xi, yi, pat, component] where component indexes e11,e12,e13,e22,e23,e33
component_labels = ["e11", "e12", "e13", "e22", "e23", "e33"]
component_indices = [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]

strain_grid = np.zeros((n_pc_points, n_pc_points, n_patterns_to_examine, len(component_labels)))

print("Running PC sweep...")
for xi, xstar in enumerate(xstar_vals):
    for yi, ystar in enumerate(ystar_vals):
        epsilon, _ = compute_strain_for_pc(
            h_calc[pat_indices], xstar, ystar, zstar_nom, patshape
        )
        for ci, (r, c) in enumerate(component_indices):
            strain_grid[xi, yi, :, ci] = epsilon[:, r, c]

print("Done.")

XX, YY = np.meshgrid(xstar_vals, ystar_vals, indexing="ij")

# ============================================================
# Plot 1: 3D surface — one figure per pattern, showing e11 and e22
# ============================================================

for pi, pat_idx in enumerate(pat_indices):
    fig = plt.figure(figsize=(14, 5))
    for ci, comp in enumerate(["e11", "e22", "e33"]):
        ax = fig.add_subplot(1, 3, ci + 1, projection="3d")
        Z = strain_grid[:, :, pi, component_labels.index(comp)]
        ax.plot_surface(XX, YY, Z, cmap="coolwarm", edgecolor="none", alpha=0.9)
        ax.set_xlabel("xstar", fontsize=9)
        ax.set_ylabel("ystar", fontsize=9)
        ax.set_zlabel(comp, fontsize=9)
        ax.set_title(f"{comp}  |  pattern {pat_idx}", fontsize=10)
        # mark nominal PC
        ax.scatter(
            [xstar_nom], [ystar_nom],
            [strain_grid[n_pc_points//2, n_pc_points//2, pi, component_labels.index(comp)]],
            color="black", s=40, zorder=5, label="nominal PC",
        )

    plt.suptitle(f"Strain vs PC — pattern {pat_idx}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{foldername}/3D_surface_pat{pat_idx}.png", dpi=150, bbox_inches="tight")
    plt.close()

# ============================================================
# Plot 2: Heatmaps — for each component, one row per pattern
# ============================================================

for ci, comp in enumerate(component_labels):
    fig, axes = plt.subplots(1, n_patterns_to_examine, figsize=(4 * n_patterns_to_examine, 4))
    axes = np.array(axes).ravel()

    all_vals = strain_grid[:, :, :, ci]
    vmax = np.abs(all_vals).max()
    vmin = -vmax

    for pi, (ax, pat_idx) in enumerate(zip(axes, pat_indices)):
        Z = strain_grid[:, :, pi, ci]
        im = ax.imshow(
            Z.T, origin="lower", aspect="auto",
            extent=[xstar_vals[0], xstar_vals[-1], ystar_vals[0], ystar_vals[-1]],
            cmap="coolwarm", vmin=vmin, vmax=vmax,
        )
        ax.scatter(xstar_nom, ystar_nom, color="black", s=30, marker="+", linewidths=1.5, label="nominal")
        ax.set_xlabel("xstar", fontsize=9)
        ax.set_ylabel("ystar", fontsize=9)
        ax.set_title(f"pat {pat_idx}", fontsize=10)

    fig.colorbar(im, ax=axes.tolist(), shrink=0.7, label=comp)
    fig.suptitle(f"{comp} vs PC", fontsize=13)
    plt.savefig(f"{foldername}/heatmap_{comp}.png", dpi=150, bbox_inches="tight")
    plt.close()

# ============================================================
# Plot 3: Line cuts through nominal PC — xstar sweep (ystar fixed)
#         and ystar sweep (xstar fixed), for all 5 patterns
# ============================================================

mid = n_pc_points // 2  # index of nominal value

fig, axes = plt.subplots(2, len(component_labels), figsize=(4 * len(component_labels), 8))

for ci, comp in enumerate(component_labels):
    # xstar sweep at nominal ystar
    ax_x = axes[0, ci]
    for pi, pat_idx in enumerate(pat_indices):
        ax_x.plot(xstar_vals, strain_grid[:, mid, pi, ci], label=f"pat {pat_idx}")
    ax_x.axvline(xstar_nom, color="k", linestyle="--", linewidth=0.8)
    ax_x.set_xlabel("xstar")
    ax_x.set_ylabel(comp)
    ax_x.set_title(f"{comp}  (ystar fixed)")
    if ci == 0:
        ax_x.legend(fontsize=7)

    # ystar sweep at nominal xstar
    ax_y = axes[1, ci]
    for pi, pat_idx in enumerate(pat_indices):
        ax_y.plot(ystar_vals, strain_grid[mid, :, pi, ci], label=f"pat {pat_idx}")
    ax_y.axvline(ystar_nom, color="k", linestyle="--", linewidth=0.8)
    ax_y.set_xlabel("ystar")
    ax_y.set_ylabel(comp)
    ax_y.set_title(f"{comp}  (xstar fixed)")

plt.suptitle("Strain sensitivity to PC — line cuts through nominal", fontsize=13)
plt.tight_layout()
plt.savefig(f"{foldername}/line_cuts_all_components.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# Plot 4: Sensitivity summary — std of each strain component
#         across the PC grid, for each pattern
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(component_labels))
width = 0.8 / n_patterns_to_examine

for pi, pat_idx in enumerate(pat_indices):
    stds = [strain_grid[:, :, pi, ci].std() for ci in range(len(component_labels))]
    ax.bar(x_pos + pi * width, stds, width=width, label=f"pat {pat_idx}")

ax.set_xticks(x_pos + width * (n_patterns_to_examine - 1) / 2)
ax.set_xticklabels(component_labels)
ax.set_ylabel("Std of strain over PC grid")
ax.set_title("PC sensitivity: strain std across full xstar/ystar sweep")
ax.legend()
plt.tight_layout()
plt.savefig(f"{foldername}/sensitivity_summary.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"All figures saved to {foldername}")

# ============================================================
# Optimal PC correction
# ============================================================

print("\nFinding optimal PC correction...")
delta_x, delta_y, cost_grid, dx_vals, dy_vals = find_optimal_pc(h_calc, PC_nominal, patshape)

PC_corrected = (xstar_nom + delta_x, ystar_nom + delta_y, zstar_nom)
print(f"  Nominal PC:   xstar={xstar_nom:.6f}  ystar={ystar_nom:.6f}")
print(f"  Delta:        dx={delta_x:+.6f}  dy={delta_y:+.6f}")
print(f"  Corrected PC: xstar={PC_corrected[0]:.6f}  ystar={PC_corrected[1]:.6f}")

# Compare mean squared strain before and after correction
components = [(0,0),(0,1),(0,2),(1,1),(1,2)]
component_labels_5 = ["e11", "e12", "e13", "e22", "e23"]

eps_before, _ = compute_strain_for_pc(h_calc, xstar_nom, ystar_nom, zstar_nom, patshape)
eps_after,  _ = compute_strain_for_pc(h_calc, PC_corrected[0], PC_corrected[1], zstar_nom, patshape)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, eps, label in zip(axes, [eps_before, eps_after], ["Nominal PC", "Corrected PC"]):
    means = [np.mean(eps[:, r, c]) for r, c in components]
    stds  = [np.std(eps[:, r, c])  for r, c in components]
    x = np.arange(len(component_labels_5))
    ax.bar(x - 0.2, means, width=0.35, label="mean", color="steelblue")
    ax.bar(x + 0.2, stds,  width=0.35, label="std",  color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(component_labels_5)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_title(label)
    ax.set_ylabel("Strain")
    ax.legend()

fig.suptitle(
    f"Strain before/after PC correction\n"
    f"delta_xstar={delta_x:+.5f}  delta_ystar={delta_y:+.5f}",
    fontsize=12,
)
plt.tight_layout()
plt.savefig(f"{foldername}/optimal_pc_correction.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved optimal_pc_correction.png")

# Cost landscape
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(
    cost_grid.T, origin="lower", aspect="auto",
    extent=[dx_vals[0], dx_vals[-1], dy_vals[0], dy_vals[-1]],
    cmap="viridis",
)
ax.scatter(delta_x, delta_y, color="red", s=60, marker="x", linewidths=2, label="minimum")
ax.set_xlabel("delta xstar")
ax.set_ylabel("delta ystar")
ax.set_title("PC optimisation cost landscape")
fig.colorbar(im, ax=ax, label="mean squared strain")
ax.legend()
plt.tight_layout()
plt.savefig(f"{foldername}/optimal_pc_cost_landscape.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved optimal_pc_cost_landscape.png")

# ============================================================
# Strain heatmaps before and after correction
# ============================================================

vmin = -1e-1
vmax =  1e-1

for (r, c), comp in zip(components, component_labels_5):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, eps, label in zip(axes, [eps_before, eps_after], ["Nominal PC", "Corrected PC"]):
        data = eps[:, r, c].reshape(Rows, Columns)
        im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{label}", fontsize=12)
        ax.axis("off")

    fig.suptitle(
        f"{comp}  —  before vs after PC correction\n"
        f"delta_xstar={delta_x:+.5f}  delta_ystar={delta_y:+.5f}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(f"{foldername}/heatmap_correction_{comp}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap_correction_{comp}.png")

# ============================================================
# Strain gradient quiver plot
# Computes, for a 10x10 grid of patterns, the direction in
# (delta_xstar, delta_ystar) space that most reduces e11.
# Arrows are overlaid on the e11 heatmap.
# ============================================================

n_quiver = 15           # grid points per axis (15x15 = 225 patterns)
h_step   = 1.0 / patshape[1]   # 1 pixel step in fractional PC units

# Build 15x15 grid of scan indices
row_idx = np.linspace(0, Rows    - 1, n_quiver, dtype=int)
col_idx = np.linspace(0, Columns - 1, n_quiver, dtype=int)

# Pre-compute flat pattern indices for all grid points
flat_indices = np.ravel_multi_index(
    np.array(np.meshgrid(row_idx, col_idx, indexing="ij")).reshape(2, -1),
    (Rows, Columns),
)

print("\nComputing per-pattern strain gradients...")
grad_x_map = np.zeros(n_quiver * n_quiver)
grad_y_map = np.zeros(n_quiver * n_quiver)

for k, idx in enumerate(flat_indices):
    h_single = h_calc[idx : idx + 1]   # shape (1, 8)

    eps_px, _ = compute_strain_for_pc(h_single, xstar_nom + h_step, ystar_nom,          zstar_nom, patshape)
    eps_mx, _ = compute_strain_for_pc(h_single, xstar_nom - h_step, ystar_nom,          zstar_nom, patshape)
    eps_py, _ = compute_strain_for_pc(h_single, xstar_nom,          ystar_nom + h_step, zstar_nom, patshape)
    eps_my, _ = compute_strain_for_pc(h_single, xstar_nom,          ystar_nom - h_step, zstar_nom, patshape)

    # Numerical gradient of von_mises^2 — negate for steepest descent
    grad_x_map[k] = -(von_mises(eps_px)[0] ** 2 - von_mises(eps_mx)[0] ** 2) / (2 * h_step)
    grad_y_map[k] = -(von_mises(eps_py)[0] ** 2 - von_mises(eps_my)[0] ** 2) / (2 * h_step)

print("Done.")

# Reshape for quiver — note: imshow uses (row, col) = (y, x)
# col_idx → x axis, row_idx → y axis
col_grid, row_grid = np.meshgrid(col_idx, row_idx)   # both (n_quiver, n_quiver)
gx = grad_x_map.reshape(n_quiver, n_quiver)
gy = grad_y_map.reshape(n_quiver, n_quiver)

# Normalize arrows to uniform length so all directions are visible
mag = np.sqrt(gx ** 2 + gy ** 2)
mag[mag == 0] = 1.0
gx_norm = gx / mag
gy_norm = gy / mag

fig, ax = plt.subplots(figsize=(8, 6))
vm_map = von_mises(eps_before).reshape(Rows, Columns)
im = ax.imshow(vm_map, cmap="viridis", vmin=0, vmax=vmax, origin="upper")
fig.colorbar(im, ax=ax, label="Von Mises strain")

# Arrows: u = delta_xstar direction (horizontal), v = delta_ystar direction (vertical)
# gy maps to v (row direction on image), gx maps to u (col direction on image)
ax.quiver(
    col_grid, row_grid,
    gx_norm, -gy_norm,   # flip v because imshow y-axis increases downward
    color="black", scale=n_quiver * 1.5, width=0.005,
    headwidth=5, headlength=5,
)
ax.set_title(
    "Von Mises strain with steepest-descent PC direction per pattern\n"
    "(arrows point in the (delta_xstar, delta_ystar) direction that most reduces von Mises strain)",
    fontsize=10,
)
ax.axis("off")
plt.tight_layout()
plt.savefig(f"{foldername}/strain_gradient_quiver.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved strain_gradient_quiver.png")

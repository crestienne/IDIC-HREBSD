import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import conversions
import os
import ErnouldsMethod


def rotation_matrix_from_tilt(tilt_deg: float) -> np.ndarray:
    theta_x = np.deg2rad(90.0 - tilt_deg)
    Rx = np.array([
        [1.0, 0.0,               0.0              ],
        [0.0, np.cos(theta_x), -np.sin(theta_x)   ],
        [0.0, np.sin(theta_x),  np.cos(theta_x)   ],
    ])
    Rz_180 = np.array([
        [-1.0,  0.0, 0.0],
        [ 0.0, -1.0, 0.0],
        [ 0.0,  0.0, 1.0],
    ])
    return Rx @ Rz_180


# ============================================================
# Inputs
# ============================================================

#EMEBSD version
#file that contains all of the deformation gradient data exported from Al_results visulalization
filename = '/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/Al-MegaLargeArea-rows75to175-columns150to300_Mar122025_npyfiles/Al-MegaLargeArea-rows75to175-columns150to300_homographies_Mar122025.npy'
#How many Rows and Columns were in the original EBSD scan?
Rows = 100    # y columns
Columns = 150 # x columns

# Pattern center (xstar, ystar, zstar) in EDAX/TSL convention
PC = (0.4776, 0.5833, 0.670697)
patshape = (480, 480)
tilt = 68.0  # sample tilt in degrees

#results folder path
foldername = f'/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/Al-MegaLargeArea-rows75to175-columns150to300_homographies_Mar122025/'

os.makedirs(foldername, exist_ok=True)

# grain_ids = np.load('/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/Al-x150to175y75to125_segment_Mar102025_npyfiles/Al-x150to175y75to125_segment_ids_Mar102025.npy')
# kam = np.load('/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/Al-x150to175y75to125_segment_Mar102025_npyfiles/Al-x150to175y75to125_segment_kam_Mar102025.npy')

# img = grain_ids.reshape(Rows, Columns)
# masked = np.ma.masked_where(img > 10, img)
# plt.figure(figsize=(10,8))
# plt.imshow(masked, cmap='tab20')
# plt.colorbar(label='Grain ID')
# plt.title('Grain IDs ≤ 10')
# plt.xlabel('Column Index')
# plt.ylabel('Row Index')
# plt.show()

# kam = kam.reshape(Rows, Columns)
# plt.figure(figsize=(10,8))
# im = plt.imshow(kam, cmap='viridis', vmin=0, vmax=2)
# plt.colorbar(im, label='KAM (degrees)')
# plt.title('KAM Map')
# plt.xlabel('Column Index')
# plt.ylabel('Row Index')
# plt.show()

# ============================================================
# Conversion: Homography to Strain
# ============================================================

h = np.load(filename)

# h is in column major order — convert to row major order
h11 = h[:, 0]
h12 = h[:, 1]
h13 = h[:, 2]
h21 = h[:, 3]
h22 = h[:, 4]
h23 = h[:, 5]
h31 = h[:, 6]
h32 = h[:, 7]

h_calc = np.stack((h11, h12, h13, h21, h22, h23, h31, h32), axis=1)

F = conversions.h2F(h_calc, np.array([(PC[0]-0.5)*patshape[1], (PC[1]-0.5)*patshape[0], patshape[0]*PC[2]]))

print(F.shape)
epsilon, omega = conversions.F2strain(F)


R = rotation_matrix_from_tilt(tilt)

samp_frame = True
if samp_frame:
    for i in range(epsilon.shape[0]):
        epsilon[i, :, :] = R.T @ epsilon[i, :, :] @ R
        omega[i, :, :]   = R.T @ omega[i, :, :]   @ R

e11 = epsilon[:, 0, 0]
e12 = epsilon[:, 0, 1]
e13 = epsilon[:, 0, 2]
e22 = epsilon[:, 1, 1]
e23 = epsilon[:, 1, 2]
e33 = epsilon[:, 2, 2]
w13 = omega[:, 0, 2]
w21 = omega[:, 1, 0]
w32 = omega[:, 2, 1]

# Convert rotation components to degrees
w13 = np.degrees(w13)
w21 = np.degrees(w21)
w32 = np.degrees(w32)

# Reshape all components to Rows x Columns
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
# Plotting
# ============================================================

vmin = -5e-2
vmax = 5e-2
vmin_rot = -1.0
vmax_rot = 1.0

# --- Spectral colormap ---
fig, ax = plt.subplots(3, 3, figsize=(15, 10))

ax[0, 0].imshow(e11, cmap="Spectral", vmin=vmin, vmax=vmax)
cb1 = fig.colorbar(ax[0,0].imshow(e11, cmap="Spectral", vmin=vmin, vmax=vmax), ax=ax[0,0])

ax[0, 1].imshow(e12, cmap="Spectral", vmin=vmin, vmax=vmax)
cb2 = fig.colorbar(ax[0,1].imshow(e12, cmap="Spectral", vmin=vmin, vmax=vmax), ax=ax[0,1])
ax[0, 1].set_title("ε12")

ax[0, 2].imshow(e13, cmap="Spectral", vmin=vmin, vmax=vmax)
cb3 = fig.colorbar(ax[0,2].imshow(e13, cmap="Spectral", vmin=vmin, vmax=vmax), ax=ax[0,2])
ax[0, 2].set_title("ε13")

ax[1, 0].imshow(w21, cmap="Spectral", vmin=vmin, vmax=vmax)
cb4 = fig.colorbar(ax[1,0].imshow(w21, cmap="Spectral", vmin=vmin_rot, vmax=vmax_rot), ax=ax[1,0])
ax[1, 0].set_title("ω21 (degrees)")

ax[1, 1].imshow(e22, cmap="Spectral", vmin=vmin, vmax=vmax)
cb5 = fig.colorbar(ax[1,1].imshow(e22, cmap="Spectral", vmin=vmin, vmax=vmax), ax=ax[1,1])
ax[1, 1].set_title("ε22")

ax[1, 2].imshow(e23, cmap="Spectral", vmin=vmin, vmax=vmax)
cb6 = fig.colorbar(ax[1,2].imshow(e23, cmap="Spectral", vmin=vmin, vmax=vmax), ax=ax[1,2])
ax[1, 2].set_title("ε23")

ax[2, 0].imshow(w13, cmap="Spectral", vmin=vmin, vmax=vmax)
cb7 = fig.colorbar(ax[2,0].imshow(w13, cmap="Spectral", vmin=vmin_rot, vmax=vmax_rot), ax=ax[2,0])
ax[2, 0].set_title("ω13 (degrees)")

ax[2, 1].imshow(w32, cmap="Spectral", vmin=vmin, vmax=vmax)
cb8 = fig.colorbar(ax[2,1].imshow(w32, cmap="Spectral", vmin=vmin_rot, vmax=vmax_rot), ax=ax[2,1])
ax[2, 1].set_title("ω32 (degrees)")
ax[2, 2].axis("off")

fig.suptitle("Strain and Rotation Components", fontsize=16)
plt.tight_layout()
plt.savefig(f"{foldername}/Strain_and_Rotation_Calculated.png")
plt.close()

# --- Viridis colormap ---
vmin_rot = -2.0
vmax_rot = 2.0

fig, ax = plt.subplots(3, 3, figsize=(15, 10))

ax[0, 0].imshow(e11, cmap="viridis", vmin=vmin, vmax=vmax)
cb1 = fig.colorbar(ax[0,0].imshow(e11, cmap="viridis", vmin=vmin, vmax=vmax), ax=ax[0,0])
ax[0, 0].set_title(r"$ \epsilon_{11}$", fontsize=16)
ax[0, 0].axis('off')

ax[0, 1].imshow(e12, cmap="viridis", vmin=vmin, vmax=vmax)
cb2 = fig.colorbar(ax[0,1].imshow(e12, cmap="viridis", vmin=vmin, vmax=vmax), ax=ax[0,1])
ax[0, 1].set_title(r"$ \epsilon_{12}$", fontsize=16)
ax[0, 1].axis('off')

ax[0, 2].imshow(e13, cmap="viridis", vmin=vmin, vmax=vmax)
cb3 = fig.colorbar(ax[0,2].imshow(e13, cmap="viridis", vmin=vmin, vmax=vmax), ax=ax[0,2])
ax[0, 2].set_title(r"$ \epsilon_{13}$", fontsize=16)
ax[0, 2].axis('off')

ax[1, 0].imshow(w21, cmap="viridis", vmin=vmin_rot, vmax=vmax_rot)
cb4 = fig.colorbar(ax[1,0].imshow(w21, cmap="viridis", vmin=vmin_rot, vmax=vmax_rot), ax=ax[1,0])
ax[1, 0].set_title(r"$ \omega_{21}$ (degrees)", fontsize=16)
ax[1, 0].axis('off')

ax[1, 1].imshow(e22, cmap="viridis", vmin=vmin, vmax=vmax)
cb5 = fig.colorbar(ax[1,1].imshow(e22, cmap="viridis", vmin=vmin, vmax=vmax), ax=ax[1,1])
ax[1, 1].set_title(r"$ \epsilon_{22}$", fontsize=16)
ax[1, 1].axis('off')

ax[1, 2].imshow(e23, cmap="viridis", vmin=vmin, vmax=vmax)
cb6 = fig.colorbar(ax[1,2].imshow(e23, cmap="viridis", vmin=vmin, vmax=vmax), ax=ax[1,2])
ax[1, 2].set_title(r"$ \epsilon_{23}$", fontsize=16)
ax[1, 2].axis('off')

ax[2, 0].imshow(w13, cmap="viridis", vmin=vmin_rot, vmax=vmax_rot)
cb7 = fig.colorbar(ax[2,0].imshow(w13, cmap="viridis", vmin=vmin_rot, vmax=vmax_rot), ax=ax[2,0])
ax[2, 0].set_title(r"$ \omega_{13}$ (degrees)", fontsize=16)
ax[2, 0].axis('off')

ax[2, 1].imshow(w32, cmap="viridis", vmin=vmin_rot, vmax=vmax_rot)
cb8 = fig.colorbar(ax[2,1].imshow(w32, cmap="viridis", vmin=vmin_rot, vmax=vmax_rot), ax=ax[2,1])
ax[2, 1].set_title(r"$ \omega_{32}$ (degrees)", fontsize=16)
ax[2, 1].axis('off')
ax[2, 2].axis("off")

plt.tight_layout()
plt.savefig(f"{foldername}/Strain_and_Rotation_Calculated - viridis.png")
plt.close()

# --- Save each component individually ---
components = {
    "e11": e11,
    "e12": e12,
    "e13": e13,
    "e22": e22,
    "e23": e23,
    "e33": e33,
    "w13": w13,
    "w21": w21,
    "w32": w32,
}

for name, data in components.items():
    plt.figure(figsize=(6, 5))
    plt.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
    cb = plt.colorbar()
    plt.title(f"{name} Component")
    plt.savefig(f"{foldername}/{name}_Calculated.png")
    plt.close()

from tabnanny import verbose
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

import Data
import utilities
import get_homography_cpu as core
import conversions

# want to calc the difference between the true strain and the calculated strain
# ----------- Code for reading in the inputted values ------------
Testcasesfilename = '/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/Inputs/Altering_e13.csv'
strain = True #set to True for strain version, False for w version

verbose = False  # set to True to show plots, False to skip showing plots


name = "TestRun"
component = 'e13'  # specify the component being analyzed
date = "Dec22025"
up2 = (
   '/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/Inputs/E13_Ernould_Nov102025.up2' 
)
# up2 = "/Users/jameslamb/Documents/research/data/GaN-DED/20240508_27238_512x512_flipX.up2"
ang = "/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/Inputs/ErnouldMethod_ang.ang"
x0 = (0, 0)

pat_obj = Data.UP2(up2)
print(pat_obj.patshape)
ang_data = utilities.read_ang(ang, pat_obj.patshape, segment_grain_threshold=None)
x0 = np.ravel_multi_index(x0, ang_data.shape)
print(x0)
shape = ang_data.shape
calc = False  # set to True to perform calculation, False to load saved data


if calc:
    h, iterations, residuals, dp_norms = core.optimize(
        pat_obj, x0, crop_fraction=0.9, max_iter=50, n_jobs=16, verbose=True
    )

    np.save(
        "E:/SiGe_edax/10 mu squares map no averaging3a_spherical_homographies.npy",
        h,
    )
    np.save(
        "E:/SiGe_edax/10 mu squares map no averaging3a_spherical_iterations.npy",
        iterations,
    )
    np.save(
        "E:/SiGe_edax/10 mu squares map no averaging3a_spherical_residuals.npy",
        residuals,
    )
    np.save(
        "E:/SiGe_edax/10 mu squares map no averaging3a_spherical_dp_norms.npy",
        dp_norms,
    )
else:
    h = np.load(
        "/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/TestRun_homographies_Nov272025.npy"
    )
    iterations = np.load(
        "/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/TestRun_iterations_Nov272025.npy"
    )
    residuals = np.load(
        "/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/TestRun_residuals_Nov272025.npy"
    )
    dp_norms = np.load(
        "/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/TestRun_dp_norms_Nov272025.npy"
    )
    h_guess = np.load(
        "/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/TestRun_h_guess_Nov272025.npy"
    )

h = h.reshape((shape[0], shape[1], 8))
iterations = iterations.reshape(shape)
residuals = residuals.reshape(shape)
dp_norms = dp_norms.reshape(shape)

folder = '/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results'



fig, ax = plt.subplots(4, 3, figsize=(15, 10))
ax[0, 0].imshow(iterations)
ax[0, 1].imshow(residuals)
ax[0, 2].imshow(dp_norms)
ax[1, 0].imshow(h[:, :, 0], cmap="gray")
ax[1, 1].imshow(h[:, :, 1], cmap="gray")
ax[1, 2].imshow(h[:, :, 2], cmap="gray")
ax[2, 0].imshow(h[:, :, 3], cmap="gray")
ax[2, 1].imshow(h[:, :, 4], cmap="gray")
ax[2, 2].imshow(h[:, :, 5], cmap="gray")
ax[3, 0].imshow(h[:, :, 6], cmap="gray")
ax[3, 1].imshow(h[:, :, 7], cmap="gray")
names = [
    "Iterations",
    "Residuals",
    "DP Norms",
    "H11",
    "H12",
    "H13",
    "H21",
    "H22",
    "H23",
    "H31",
    "H32",
    "H33",
]
for a in ax.ravel():
    a.axis("off")
    a.set_title(names.pop(0))

plt.tight_layout()
plt.savefig(f"{folder}/homography_results_overview_{date}.png")
if verbose:
    plt.show()

#plot iterations, residuals, dp_norms as subplots
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(iterations)
ax[0].set_title("Iterations")
#add colorbars to each subplot
fig.colorbar(ax[0].imshow(iterations), ax=ax[0])
ax[1].imshow(residuals)
ax[1].set_title("Residuals")
fig.colorbar(ax[1].imshow(residuals), ax=ax[1])
ax[2].imshow(dp_norms)
ax[2].set_title("DP Norms")
fig.colorbar(ax[2].imshow(dp_norms), ax=ax[2])
plt.savefig(f"{folder}/optimization_metrics_{date}.png")
plt.tight_layout()

# convert homographies to deformation gradients and save as tiff files
F = conversions.h2F(h, np.array([0, 0, 800]))
#print the shape of F
print("F shape:", F.shape)

#plot F components as subplots
fig, ax = plt.subplots(3, 3, figsize=(15, 10))
ax[0, 0].imshow(F[:, :, 0, 0], cmap="gray")
ax[0, 1].imshow(F[:, :, 0, 1], cmap="gray")
ax[0, 2].imshow(F[:, :, 0, 2], cmap="gray")
ax[1, 0].imshow(F[:, :, 1, 0], cmap="gray")
ax[1, 1].imshow(F[:, :, 1, 1], cmap="gray")
ax[1, 2].imshow(F[:, :, 1, 2], cmap="gray")
ax[2, 0].imshow(F[:, :, 2, 0], cmap="gray")
ax[2, 1].imshow(F[:, :, 2, 1], cmap="gray")
ax[2, 2].imshow(F[:, :, 2, 2], cmap="gray")
names = [
    "F11",
    "F12",
    "F13",
    "F21",
    "F22",
    "F23",
    "F31",
    "F32",
    "F33",
]
#save the figure
for a in ax.ravel():
    a.axis("off")
    a.set_title(names.pop(0))
plt.tight_layout()
plt.savefig(f"{folder}/deformation_gradients.png")
if verbose:
    plt.show()


#convert F to strain components and save as tiff files
epsilon, omega = conversions.F2strain(F)

e11 = epsilon[:, :, 0, 0]
e12 = epsilon[:, :, 0, 1]
e13 = epsilon[:, :, 0, 2]
e22 = epsilon[:, :, 1, 1]
e23 = epsilon[:, :, 1, 2]
w21 = omega[:, :, 1, 0]
w13 = omega[:, :, 0, 2]
w32 = omega[:, :, 2, 1]

# ...existing code...
#plot strain components as subplots
fig, ax = plt.subplots(3, 3, figsize=(15, 10))

vmin = -5e-2
vmax = 5e-2

ax[0, 0].imshow(e11, cmap="viridis", vmin=vmin, vmax=vmax)
cb1 = fig.colorbar(ax[0,0].imshow(e11, cmap="viridis", vmin=vmin, vmax=vmax), ax=ax[0,0])
cb1.set_ticks([-5e-2, -2.5e-2, 0, 2.5e-2, 5e-2])
ax[0, 0].set_title("ε11")
ax[0, 1].imshow(e12, cmap="viridis", vmin=vmin, vmax=vmax)
cb2 = fig.colorbar(ax[0,1].imshow(e12, cmap="viridis", vmin=vmin, vmax=vmax), ax=ax[0,1])
cb2.set_ticks([-5e-2, -2.5e-2, 0, 2.5e-2, 5e-2])
ax[0, 1].set_title("ε12")
ax[0, 2].imshow(e13, cmap="viridis", vmin=vmin, vmax=vmax)
cb3 = fig.colorbar(ax[0,2].imshow(e13, cmap="viridis", vmin=vmin, vmax=vmax), ax=ax[0,2])
cb3.set_ticks([-5e-2, -2.5e-2, 0, 2.5e-2, 5e-2])
ax[0, 2].set_title("ε13")
ax[1, 0].imshow(w21, cmap="viridis", vmin=vmin, vmax=vmax)
cb4 = fig.colorbar(ax[1,0].imshow(w21, cmap="viridis", vmin=vmin, vmax=vmax), ax=ax[1,0])
cb4.set_ticks([-5e-2, -2.5e-2, 0, 2.5e-2, 5e-2])
ax[1, 0].set_title("ω21")
ax[1, 1].imshow(e22, cmap="viridis", vmin=vmin, vmax=vmax)
cb5 = fig.colorbar(ax[1,1].imshow(e22, cmap="viridis", vmin=vmin, vmax=vmax), ax=ax[1,1])
cb5.set_ticks([-5e-2, -2.5e-2, 0, 2.5e-2, 5e-2])
ax[1, 1].set_title("ε22")
ax[1, 2].imshow(e23, cmap="viridis", vmin=vmin, vmax=vmax)
cb6 = fig.colorbar(ax[1,2].imshow(e23, cmap="viridis", vmin=vmin, vmax=vmax), ax=ax[1,2])
cb6.set_ticks([-5e-2, -2.5e-2, 0, 2.5e-2, 5e-2])
ax[1, 2].set_title("ε23")
ax[2, 0].imshow(w13, cmap="viridis", vmin=vmin, vmax=vmax)
cb7 = fig.colorbar(ax[2,0].imshow(w13, cmap="viridis", vmin=vmin, vmax=vmax), ax=ax[2,0])
cb7.set_ticks([-5e-2, -2.5e-2, 0, 2.5e-2, 5e-2])
ax[2, 0].set_title("ω13")
ax[2, 1].imshow(w32, cmap="viridis", vmin=vmin, vmax=vmax)
cb8 = fig.colorbar(ax[2,1].imshow(w32, cmap="viridis", vmin=vmin, vmax=vmax), ax=ax[2,1])
cb8.set_ticks([-5e-2, -2.5e-2, 0, 2.5e-2, 5e-2])
ax[2, 1].set_title("ω32")
ax[2, 2].axis("off")
#set a supertitle
fig.suptitle(f"Strain and Rotation Components for {component}", fontsize=16)

plt.tight_layout()
plt.savefig(f"{folder}/Strain_and_Rotation_Calculated.png")

#save the h, F, epsilon, and omega arrays in a single .csv file with appropriate headers


#first need to reshape h, F, epsilon, and omega to 2D arrays
h = h.reshape((shape[0]*shape[1], 8))
F = F.reshape((shape[0]*shape[1], 9))
epsilon = epsilon.reshape((shape[0]*shape[1], 9))
e = np.stack((epsilon[:, 0], epsilon[:, 1], epsilon[:, 2], epsilon[:, 4], epsilon[:, 5], epsilon[:, 8]), axis=-1)
print("e shape:", e.shape)
omega = omega.reshape((shape[0]*shape[1], 9))
w = np.stack((omega[:, 7], omega[:, 6], omega[:, 2]), axis=-1)

export = np.hstack((w, e, F, h))

output_filename = f"{folder}/Deformation_Gradients_and_Strain_{component}_{date}.csv"
np.savetxt(output_filename, export, delimiter=",", header="w32,w13,w21,e11,e12,e13,e22,e23,e33, F11,F12,F13,F21,F22,F23,F31,F32,F33,H11,H12,H13,H21,H22,H23,H31,H32", comments="")

#---- plotting the initial guess homographies -----
h_guess = h_guess.reshape((shape[0], shape[1], 8))
#convert to deformation gradients
F_guess = conversions.h2F(h_guess, np.array([600, 600, 800]))
epsilon_guess, omega_guess = conversions.F2strain(F_guess)
#convert omega values to degrees
omega_guess = np.degrees(omega_guess)
#plot strain components as subplots
fig, ax = plt.subplots(3, 3, figsize=(15, 10))
vmin = -5e-2
vmax = 5e-2

vmin_rot = -2
vmax_rot = 2
ax[0, 0].imshow(epsilon_guess[:, :, 0, 0], cmap="RdBu", vmin=vmin, vmax=vmax)
cb1 = fig.colorbar(ax[0,0].imshow(epsilon_guess[:, :, 0, 0], cmap="RdBu", vmin=vmin, vmax=vmax), ax=ax[0,0])
cb1.set_ticks([-5e-2, -2.5e-2, 0, 2.5e-2, 5e-2])
ax[0, 0].set_title("ε11 (Initial Guess)")
ax[0, 1].imshow(epsilon_guess[:, :, 0, 1], cmap="RdBu", vmin=vmin, vmax=vmax)
cb2 = fig.colorbar(ax[0,1].imshow(epsilon_guess[:, :, 0, 1], cmap="RdBu", vmin=vmin, vmax=vmax), ax=ax[0,1])
cb2.set_ticks([-5e-2, -2.5e-2, 0, 2.5e-2, 5e-2])
ax[0, 1].set_title("ε12 (Initial Guess)")
ax[0, 2].imshow(epsilon_guess[:, :, 0, 2], cmap="RdBu", vmin=vmin, vmax=vmax)
cb3 = fig.colorbar(ax[0,2].imshow(epsilon_guess[:, :, 0, 2], cmap="RdBu", vmin=vmin, vmax=vmax), ax=ax[0,2])
cb3.set_ticks([-5e-2, -2.5e-2, 0, 2.5e-2, 5e-2])
ax[0, 2].set_title("ε13 (Initial Guess)")
ax[1, 0].imshow(omega_guess[:, :, 1, 0], cmap="RdBu", vmin=vmin_rot, vmax=vmax_rot)
cb4 = fig.colorbar(ax[1,0].imshow(omega_guess[:, :, 1, 0], cmap="RdBu", vmin=vmin_rot, vmax=vmax_rot), ax=ax[1,0])
cb4.set_ticks([-2, -1.5, 0, 1.5, 2])
ax[1, 0].set_title("ω21 (Initial Guess)")
ax[1, 1].imshow(epsilon_guess[:, :, 1, 1], cmap="RdBu", vmin=vmin, vmax=vmax)
cb5 = fig.colorbar(ax[1,1].imshow(epsilon_guess[:, :, 1, 1], cmap="RdBu", vmin=vmin, vmax=vmax), ax=ax[1,1])
cb5.set_ticks([-5e-2, -2.5e-2, 0, 2.5e-2, 5e-2])
ax[1, 1].set_title("ε22 (Initial Guess)")
ax[1, 2].imshow(epsilon_guess[:, :, 1, 2], cmap="RdBu", vmin=vmin, vmax=vmax)
cb6 = fig.colorbar(ax[1,2].imshow(epsilon_guess[:, :, 1, 2], cmap="RdBu", vmin=vmin, vmax=vmax), ax=ax[1,2])
cb6.set_ticks([-5e-2, -2.5e-2, 0, 2.5e-2, 5e-2])
ax[1, 2].set_title("ε23 (Initial Guess)")
ax[2, 0].imshow(omega_guess[:, :, 0, 2], cmap="RdBu", vmin=vmin_rot, vmax=vmax_rot)
cb7 = fig.colorbar(ax[2,0].imshow(omega_guess[:, :, 0, 2], cmap="RdBu", vmin=vmin_rot, vmax=vmax_rot), ax=ax[2,0])
cb7.set_ticks([-2, -1.5, 0, 1.5, 2])
ax[2, 0].set_title("ω13 (Initial Guess)")
ax[2, 1].imshow(omega_guess[:, :, 2, 1], cmap="RdBu", vmin=vmin_rot, vmax=vmax_rot)
cb8 = fig.colorbar(ax[2,1].imshow(omega_guess[:, :, 2, 1], cmap="RdBu", vmin=vmin_rot, vmax=vmax_rot), ax=ax[2,1])
cb8.set_ticks([-2, -1.5, 0, 1.5, 2])
ax[2, 1].set_title("ω32 (Initial Guess)")
ax[2, 2].axis("off")
#set a supertitle
fig.suptitle(f"Initial Guess Strain and Rotation Components for {component}", fontsize=16)
plt.tight_layout()
plt.savefig(f"{folder}/Initial_Guess_Strain_and_Rotation_{date}.png")
if verbose:
    plt.show()
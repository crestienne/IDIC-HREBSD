import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

import Data
import utilities
import get_homography_cpu as core
import os
import datetime


component = "Al-lookingatgrads-ignoreresult"
date = "Mar142026" 
up2 = (
   '/Volumes/Extreme SSD/DONOTEDIT_originaldata/Zehua-Recrystallized Al/HR.up2' 
)
# up2 = "/Users/jameslamb/Documents/research/data/GaN-DED/20240508_27238_512x512_flipX.up2"
ang = '/Volumes/Extreme SSD/DONOTEDIT_originaldata/Zehua-Recrystallized Al/HR.ang'
x0 = (100, 160) #order is y,x

#order for roi_slice: [slice(y_start, y_stop), slice(x_start, x_stop)], set to none if want to look at whole pattern 
roi_slice= [slice(100, 110), slice(160, 170)] 

base_folder_name = f'{component}_{date}_npyfiles'
foldername = f'/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/{base_folder_name}/'

os.makedirs(foldername, exist_ok=True)  # Set to False since we want unique folders


pat_obj = Data.UP2(up2)
print(pat_obj.patshape)

# ------ Simulated reference options (set use_simulated_reference=True to enable) ------
use_simulated_reference = False
master_pattern_path = ''      # e.g. '/path/to/Al-master-20kV.h5'
tilt_deg = 70.0               # sample tilt in degrees
debug_gradients = True        # set True to save debug/gradient_comparison.png

# ------ Run optimization ------

ang_data = utilities.read_ang(ang, pat_obj.patshape)
x0 = np.ravel_multi_index(x0, ang_data.shape)
print("Initial index and coordinates:")
print(x0)

# Read the Euler angles and PC for the reference pattern (used only if use_simulated_reference=True)
euler_angles_ref = ang_data.eulers[np.unravel_index(x0, ang_data.shape)]  # shape (3,) in radians
print("Reference pattern Euler angles (radians):", euler_angles_ref)
pc_ref = ang_data.pc  # (xstar, ystar, zstar)
print("Reference pattern PC:", pc_ref)

optimize_params = dict(
    init_type='partial',
    crop_fraction=0.9,
    max_iter=25,
    n_jobs=19,
    verbose=True,
    roi_slice=roi_slice,
    scan_shape=ang_data.shape,
    mask=pat_obj.get_mask(),
    use_simulated_reference=use_simulated_reference,
    master_pattern_path=master_pattern_path,
    euler_angles_ref=euler_angles_ref,
    pc_ref=pc_ref,
    tilt_deg=tilt_deg,
    debug_gradients=debug_gradients,
)

# ------ Plot real vs simulated reference pattern ------
if use_simulated_reference:
    os.makedirs("debug", exist_ok=True)
    real_pat = pat_obj.read_pattern(x0, process=True)
    sim_pat  = core.simulate_reference_pattern(
        master_pattern_path=master_pattern_path,
        euler_angles=euler_angles_ref,
        PC=pc_ref,
        patshape=pat_obj.patshape,
        tilt_deg=tilt_deg,
        pat_obj=pat_obj,
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(real_pat, cmap="gray")
    axes[0].set_title("Real reference pattern")
    axes[0].axis("off")
    axes[1].imshow(sim_pat, cmap="gray")
    axes[1].set_title("Simulated reference pattern")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("debug/real_vs_simulated_reference.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved debug/real_vs_simulated_reference.png")

h, h_guess, iterations, residuals, dp_norms = core.optimize(
    pat_obj, x0, **optimize_params
)


np.save(
    f"{foldername}{component}_homographies_{date}.npy",
    h,
)
np.save(
    f"{foldername}{component}_iterations_{date}.npy",
    iterations,
)
np.save(
    f"{foldername}{component}_h_guess_{date}.npy",
    h_guess,
)
np.save(
    f"{foldername}{component}_residuals_{date}.npy",
    residuals,
)
np.save(
    f"{foldername}{component}_dp_norms_{date}.npy",
    dp_norms,
)

# np.save(
#     f"{foldername}{component}_ids_{date}.npy",
#     ids,
# )
# np.save(
#     f"{foldername}{component}_kam_{date}.npy",
#     kam,
# )





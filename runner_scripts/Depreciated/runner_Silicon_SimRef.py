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
from optimize_reference import optimize_pc_and_euler


component = "Si-TestRegion-SimulatedReference"
date = "Apr072026" 
up2 = (
   '/Users/crestiennedechaine/OriginalData/Si-Indent/001_Si_spherical_indent_20kV.up2' 
)
# up2 = "/Users/jameslamb/Documents/research/data/GaN-DED/20240508_27238_512x512_flipX.up2"
ang = '/Users/crestiennedechaine/OriginalData/Si-Indent/dp2-refined.ang'
x0 = (0, 0) # reference pattern, order is y,x

#order for roi_slice: [slice(y_start, y_stop), slice(x_start, x_stop)], set to none if want to look at whole pattern 
roi_slice= [slice(0, 5), slice(0, 10)]
#roi_slice = None

base_folder_name = f'{component}_{date}_npyfiles'
foldername = f'/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/{base_folder_name}/'

os.makedirs(foldername, exist_ok=True)  # Set to False since we want unique folders


# ------ Pattern Processing Parameters ------
pat_obj = Data.UP2(up2)
pat_obj.set_processing(
    low_pass_sigma=0.0,
    high_pass_sigma=15.0,
    truncate_std_scale=3.0,
    mask_type="center_cross",       # options: "circular", "center_cross", None
    center_cross_half_width=6,      # only used when mask_type="center_cross"
    clahe_kernel=(5, 5),
    clahe_clip=0.01,
    clahe_nbins=256,
)
print(pat_obj.patshape)

pat_obj.plot_parameter_sweep(
    pattern_idx=0,
    high_pass_sigmas=[3, 5, 7, 10, 15, 50, 80],
    clahe_kernels=[(3, 3), (4, 4), (5, 5), (6, 6), (8, 8), (12, 12)],
)

# ------ Simulated reference options (set use_simulated_reference=True to enable) ------
use_simulated_reference = True
master_pattern_path = '/Users/crestiennedechaine/OriginalData/Si-Indent/Si-master-20kV.h5'      # e.g. '/path/to/Si-master-20kV.h5'
tilt_deg = 70.0                 # sample tilt in degrees
debug_gradients = True        # set True to save debug/gradient_comparison.png

# ------ Run optimization ------

ang_data = utilities.read_ang(ang, pat_obj.patshape, segment_grain_threshold=None)
x0 = np.ravel_multi_index(x0, ang_data.shape)
print("Initial index and coordinates:")
print(x0)

# Read the Euler angles and PC for the reference pattern (used only if use_simulated_reference=True)
euler_angles_ref = ang_data.eulers[np.unravel_index(x0, ang_data.shape)]  # shape (3,) in radians
print("Reference pattern Euler angles (radians):", euler_angles_ref)
pc_ref = ang_data.pc  # (xstar, ystar, zstar)
print ("Reference pattern PC:", pc_ref)

# ------ Refine PC and Euler angles against the experimental reference ------
if use_simulated_reference:
    euler_angles_ref, pc_ref = optimize_pc_and_euler(
        pat_obj=pat_obj,
        x0=x0,
        master_pattern_path=master_pattern_path,
        euler_angles_init=euler_angles_ref,
        pc_init=pc_ref,
        tilt_deg=tilt_deg,
        max_iter=300,
        save_dir="debug",
    )
    print("Refined Euler angles (rad):", euler_angles_ref)
    print("Refined PC:", pc_ref)

optimize_params = dict(
    init_type='partial',
    crop_fraction=0.9,
    max_iter=100,
    n_jobs=8,
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



# ------ Save results ------

optimize_params_str = "\n".join(f"  {k}: {v}" for k, v in optimize_params.items())
params_txt = f"""\
Run Parameters
==============
Component:              {component}
Date:                   {date}

Files
-----
UP2 file:               {up2}
ANG file:               {ang}

Pattern Info
------------
Pattern shape:          {pat_obj.patshape}
N patterns:             {pat_obj.nPatterns}
Reference pattern x0:   {x0}

Processing Parameters
---------------------
low_pass_sigma:         {pat_obj.low_pass_sigma}
high_pass_sigma:        {pat_obj.high_pass_sigma}
truncate_std_scale:     {pat_obj.truncate_std_scale}
mask_type:              {pat_obj.mask_type}
center_cross_half_width:{pat_obj.center_cross_half_width}
clahe_kernel:           {pat_obj.clahe_kernel}
clahe_clip:             {pat_obj.clahe_clip}
clahe_nbins:            {pat_obj.clahe_nbins}

Optimize Parameters
-------------------
{optimize_params_str}
"""

with open(f"{foldername}{component}_params_{date}.txt", "w") as f:
    f.write(params_txt)

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



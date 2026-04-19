import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

import Data
import utilities
import get_homography_cpu as core
import datetime
import time
from optimize_reference import optimize_pc_and_euler
from ipf_map import plot_ipf_map

# ---- Inputs ----
component = "SiIndent"
date = "April_15_2026" 
up2 = (
   '/Users/crestiennedechaine/OriginalData/Si-Indent/001_Si_spherical_indent_20kV.up2' 
)
# up2 = "/Users/jameslamb/Documents/research/data/GaN-DED/20240508_27238_512x512_flipX.up2"
ang = '/Users/crestiennedechaine/OriginalData/Si-Indent/dp2-refined.ang'
x0 = (0, 0) # reference pattern, order is y,x

#do you want to do a prarameter sweep to see the effect of different processing parameters on the pattern quality? If so, set to True and it will save a figure showing the results in the output folder. If False, it will skip this step.
do_parameter_sweep = False

# plot iterations and residuals for the first row of patterns as a line map and save to results folder?
plot_first_row_linemap = True

#order for roi_slice: [slice(y_start, y_stop), slice(x_start, x_stop)], set to none if want to look at whole pattern 
#roi_slice= [slice(0, 2), slice(0, 50)]
roi_slice = None

# ------ Simulated reference options (set use_simulated_reference=True to enable) ------
use_simulated_reference = False
master_pattern_path = '/Users/crestiennedechaine/OriginalData/Si_Ge_Dataset/DI_largerRegion/MCoutput.h5'      # e.g. '/path/to/SiGe-master-20kV.h5'
tilt_deg = 70.0 + 2.0               # sample tilt in degrees
debug_gradients = True        # set True to save debug/gradient_comparison.png
# Set euler_angles_override to a (phi1, Phi, phi2) tuple in DEGREES to use a
# fixed orientation instead of reading from the .ang file at x0.
# Set to None to use the .ang file value (default).
#euler_angles_override = euler_angles_deg = np.array([44.96, 90.14, 357.09])  # [phi1, Phi, phi2] in degrees # e.g. (0.0, 45.0, 0.0)
euler_angles_override = None


# ---- Output folder setup ----
base_folder_name = f'{component}_{date}_npyfiles'
foldername = f'/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/Si-Indent/{base_folder_name}/'

os.makedirs(foldername, exist_ok=True)  # Set to False since we want unique folders


# ------ Pattern Processing Parameters ------
pat_obj = Data.UP2(up2)
pat_obj.set_processing(
    low_pass_sigma=1.5,
    high_pass_sigma=10.0,
    truncate_std_scale=3.0,
    mask_type="center_cross",       # options: "circular", "center_cross", "none"
    center_cross_half_width=6,
    clahe_kernel=(4, 4),
    clahe_clip=0.02,
    clahe_nbins=256,
    use_clahe=False,                 # applied before low-pass, after high-pass
    flip_x=False,
    rescale_to_uint16=True,
    unsharp_sigma=0.0,
    unsharp_strength=1.5,
)
print(pat_obj)

# ------ Save first pattern to debug folder ------
os.makedirs("debug", exist_ok=True)
first_pat = pat_obj.read_pattern(0, process=True)
plt.imsave("debug/first_pattern_processed.png", first_pat, cmap="gray")
plt.imsave("debug/first_pattern_raw.png", pat_obj.read_pattern(0, process=False), cmap="gray")
print("Saved first pattern to debug/")

if do_parameter_sweep:
    pat_obj.plot_parameter_sweep(
        pattern_idx=0,
        high_pass_sigmas=[3, 5, 7, 10, 15, 50, 80],
        clahe_kernels=[(3, 3), (4, 4), (5, 5), (6, 6), (8, 8), (12, 12)],
        save_dir=foldername,
    )
    pat_obj.plot_parameter_sweep(
        pattern_idx=0,
        high_pass_sigmas=[3, 5, 7, 10, 15, 50, 80],
        clahe_kernels=[(3, 3), (4, 4), (5, 5), (6, 6), (8, 8), (12, 12)],
        save_dir=foldername,
    )

# ------ Run optimization ------

ang_data = utilities.read_ang(ang, pat_obj.patshape, segment_grain_threshold=None)
x0 = np.ravel_multi_index(x0, ang_data.shape)
print("Initial index and coordinates:")
print(x0)

# Read the Euler angles and PC for the reference pattern (used only if use_simulated_reference=True)
if euler_angles_override is not None:
    euler_angles_ref = np.deg2rad(np.array(euler_angles_override, dtype=np.float64))
    print("Using overridden Euler angles (degrees):", euler_angles_override)
else:
    euler_angles_ref = ang_data.eulers[np.unravel_index(x0, ang_data.shape)]  # shape (3,) in radians
    print("Reference pattern Euler angles (degrees):", np.rad2deg(euler_angles_ref))
pc_ref = ang_data.pc  # (xstar, ystar, zstar)
print("Reference pattern PC:", pc_ref)

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

print(f"Scan shape from ang_data: {ang_data.shape}")
optimize_params = dict(
    init_type='partial',  # 'full' or 'partial' (partial uses the reference pattern as the initial guess, full starts from identity homography)
    crop_fraction=0.9,
    max_iter=150,
    conv_tol=1e-3,
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

t_start = time.perf_counter()
_result = core.optimize(pat_obj, x0, **optimize_params)
if len(_result) == 5:
    h, h_guess, iterations, residuals, dp_norms = _result
else:
    h, iterations, residuals, dp_norms = _result
    h_guess = None
t_end = time.perf_counter()
total_time_s = t_end - t_start
n_patterns = iterations.size
time_per_pattern_ms = (total_time_s / n_patterns) * 1000
print(f"Optimization complete: {total_time_s:.1f} s total, {time_per_pattern_ms:.2f} ms/pattern")

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

Reference Pattern
-----------------
Real or Simulated:         {"Simulated" if use_simulated_reference else "Real"}
Master pattern path:       {master_pattern_path if use_simulated_reference else "N/A"}
Reference pattern Euler angles (radians): {euler_angles_ref if use_simulated_reference else "N/A"}
Reference pattern PC:      {pc_ref if use_simulated_reference else "N/A"}
Sample tilt (degrees):     {tilt_deg if use_simulated_reference else "N/A"}

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
use_clahe:              {pat_obj.use_clahe}
rescale_to_uint16:      {pat_obj.rescale_to_uint16}
unsharp_sigma:          {pat_obj.unsharp_sigma}
unsharp_strength:       {pat_obj.unsharp_strength}
flip_x:                 {pat_obj.flip_x}

Optimize Parameters
-------------------
{optimize_params_str}

Timing
------
Total optimization time:    {total_time_s:.2f} s  ({total_time_s/60:.2f} min)
Number of patterns:         {n_patterns}
Mean time per pattern:      {time_per_pattern_ms:.2f} ms
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
if h_guess is not None:
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

# ------ Plot first-row line maps ------
if plot_first_row_linemap:
    # Results are already 2D (roi_rows, roi_cols) when roi_slice is used,
    # or 1D (nPatterns,) for a full scan — reshape the full-scan case.
    if iterations.ndim == 1:
        side = int(np.sqrt(iterations.size))
        iters_2d = iterations.reshape(side, -1)
        resid_2d = residuals.reshape(side, -1)
    else:
        iters_2d = iterations
        resid_2d = residuals
    first_row_iters = iters_2d[0, :]
    first_row_resid = resid_2d[0, :]
    x_positions = np.arange(first_row_iters.shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(x_positions, first_row_iters, marker='o', markersize=4, linewidth=1.5)
    axes[0].set_ylabel("Iterations to converge")
    axes[0].set_title("First row — iterations")
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].plot(x_positions, first_row_resid, marker='o', markersize=4, linewidth=1.5, color='tab:orange')
    axes[1].set_ylabel("Final residual")
    axes[1].set_xlabel("Pattern index along first row")
    axes[1].set_title("First row — residuals")
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    linemap_path = f"{foldername}{component}_first_row_linemap_{date}.png"
    plt.savefig(linemap_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved first-row line map to {linemap_path}")

# ------ IPF map ------
print("Generating IPF map from ANG file...")
for direction in ("ND", "RD", "TD"):
    plot_ipf_map(
        ang_path        = ang,
        direction_label = direction,
        patshape        = pat_obj.patshape,
        save_path       = f"{foldername}{component}_IPF_{direction}_{date}.png",
        show            = False,
    )
    plt.close("all")
print("IPF maps saved.")

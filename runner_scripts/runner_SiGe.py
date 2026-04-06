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


component = "TestRun_SiGe"
date = "April_5_2026" 
up2 = (
   '/Users/crestiennedechaine/OriginalData/Si_Ge_Dataset/SiGe_smallerRegion_20260322_512x512.up2'
)
# up2 = "/Users/jameslamb/Documents/research/data/GaN-DED/20240508_27238_512x512_flipX.up2"
ang = '/Users/crestiennedechaine/OriginalData/Si_Ge_Dataset//DictionaryIndexing/SiGe_smallerRegion_20260322_512x512.ang'
x0 = (0, 0) # reference pattern, order is y,x

#order for roi_slice: [slice(y_start, y_stop), slice(x_start, x_stop)], set to none if want to look at whole pattern 
#roi_slice= [slice(0, 5), slice(0, 5)]
roi_slice = None

base_folder_name = f'{component}_{date}_npyfiles'
foldername = f'/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/SiGe/{base_folder_name}/'

os.makedirs(foldername, exist_ok=True)  # Set to False since we want unique folders


# ------ Pattern Processing Parameters ------
pat_obj = Data.UP2(up2)
pat_obj.set_processing(
    low_pass_sigma=0.0,
    high_pass_sigma=15.0,
    truncate_std_scale=3.0,
    mask_type="None",       # options: "circular", "center_cross", None
    center_cross_half_width=6,      # only used when mask_type="center_cross"
    clahe_kernel=(5, 5),
    clahe_clip=0.01,
    clahe_nbins=256,
    flip_x=True,            # flip pattern about x axis (reverses rows)
)
print(pat_obj.patshape)

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

optimize_params = dict(
    init_type='partial',
    crop_fraction=0.9,
    max_iter=100,
    n_jobs=8,
    verbose=True,
    roi_slice=roi_slice,
    mask=pat_obj.get_mask(),
)

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



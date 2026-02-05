import numpy as np
import matplotlib.pyplot as plt

import Data
import utilities
import get_homography_cpu as core
import os
import datetime


component = "Al-x150to250y75to150"
date = "Jan222025" 
up2 = (
   '/Volumes/Extreme SSD/DONOTEDIT_originaldata/Zehua-Recrystallized Al/HR.up2' 
)
# up2 = "/Users/jameslamb/Documents/research/data/GaN-DED/20240508_27238_512x512_flipX.up2"
ang = '/Volumes/Extreme SSD/DONOTEDIT_originaldata/Zehua-Recrystallized Al/HR.ang'
x0 = (110, 200) #order is y,x
#order for roi_slice: [slice(y_start, y_stop), slice(x_start, x_stop)], set to none if want to look at whole pattern 
roi_slice= [slice(75, 150), slice(150, 250)]

base_folder_name = f'{component}_{date}_npyfiles'
foldername = f'/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/{base_folder_name}/'

os.makedirs(foldername, exist_ok=True)  # Set to False since we want unique folders


pat_obj = Data.UP2(up2)
print(pat_obj.patshape)
ang_data = utilities.read_ang(ang, pat_obj.patshape, segment_grain_threshold=None)
x0 = np.ravel_multi_index(x0, ang_data.shape)
print("Initial index and coordinates:")
print(x0)

h, h_guess, iterations, residuals, dp_norms = core.optimize(
    pat_obj, x0, init_type='partial', crop_fraction=0.9, max_iter=25, n_jobs=19, verbose=True, roi_slice=roi_slice
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



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
ang_data = utilities.read_ang(ang, pat_obj.patshape)
x0 = np.ravel_multi_index(x0, ang_data.shape)
print("Initial index and coordinates:")
print(x0)

h, h_guess, iterations, residuals, dp_norms = core.optimize(
    pat_obj, x0, init_type='partial', crop_fraction=0.9, max_iter=25, n_jobs=19, verbose=True, roi_slice=roi_slice,
    mask=pat_obj.get_mask(),
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





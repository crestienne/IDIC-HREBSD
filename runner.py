import numpy as np
import matplotlib.pyplot as plt

import Data
import utilities
import get_homography_cpu as core


name = "TestRun"
date = "Nov262025"
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

h, h_guess, iterations, residuals, dp_norms = core.optimize(
    pat_obj, x0, init_type='partial', crop_fraction=0.9, max_iter=25, n_jobs=19, verbose=True
)


np.save(
    f"/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/{name}_homographies_{date}.npy",
    h,
)
np.save(
    f"/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/{name}_iterations_{date}.npy",
    iterations,
)
np.save(
    f"/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/{name}_residuals_{date}.npy",
    residuals,
)
np.save(
    f"/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/{name}_dp_norms_{date}.npy",
    dp_norms,
)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(iterations)
ax[1].imshow(residuals)
ax[2].imshow(dp_norms)
plt.tight_layout()
plt.show()

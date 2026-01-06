'''
Crestienne's Runner Script, testing the functionality of Ernould's code with simulated patterns 
From Chapter 4 of Book



'''
import sys
import os
import numpy as np
import matplotlib.pyplot as plt


# Add the parent directory to the Python path
sys.path.append(os.path.abspath("/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/"))

import utilities
import Data
import utilities
import rotations
import get_homography_gpu as gh_gpu



'''
Need to also install scipy 
install tqdm 
install skikit-image
dill 
torch
kornia
joblib
mpire 
h5py (for master pattern)
Need to set the number of cores on line 264 of the cpu code

'''

if __name__ == "__main__":
    ############################
    # Load the pattern object
    ang = '/Users/crestiennedechaine/Scripts/pyHREBSD/001_Si_spherical_indent_20kV.ang'


    up2 = '/Users/crestiennedechaine/Scripts/pyHREBSD/results/SimulatedPatterns-Aug42025.up2'
    name = "001_Si_spherical_indent_20kV"
    # Set the geometry parameters
    pixel_size = 20.0  # The pixel size in um, taking binning into account (so 4xpixel_size for 4x4 binning)
    sample_tilt = 70.0  # The sample tilt in degrees
    detector_tilt = 0.0  # The detector tilt in degrees
    step_size = 1.8  # The step size in um
    subset_size = 512 # The size of the subset to use for the homography calculation
    fixed_projection = True #don't want to alter the pc 
    # Set the initial guess parameters
    init_type = "partial"  # The type of initial guess to use, "none", "full", or "partial"
    initial_guess_subset_size = 512
    # Set the roi parameters
    start = (0, 0)  # The pixel location to start the ROI
    span = None  # None is the full scan
    x0 = (0, 0)  # The location of the reference within the ROI
    # Set the image processing parameters
    high_pass_sigma = 0
    low_pass_sigma = 0
    truncate_std_scale = 0
    # Set the small strain flag
    small_strain = False
    # Set the stiffness tensor
    C = utilities.get_stiffness_tensor(165.77, 63.94, 79.62, structure="cubic") #stiffness values for Si in GPa from mason 1958
    traction_free = False #
    # Calculate or read
    calc = True
    # Whether to view the reference image
    view_reference = False
    # Number of cores, max iterations, and convergence tolerance if calculating
    n_cores = 1
    max_iter = 100
    conv_tol = 1e-3
    # Verbose
    verbose = False
    gpu = False
    ############################
    Euler = np.array([75, 125, 15]) #euler angles in degrees, from Ernould chpt 4
    Rows = 6 #defined in AlDatasetGeneration July222025.ipynb
    Columns = 58 #defined in AlDatasetGeneration July222025.ipynb
    N = Rows * Columns #number of patterns to generate
    detector_shape = (1200, 1200) #shape of the detector, defined in AlDatasetGeneration July222025.ipynb


    '''
    SUPER IMPORTANT, the PC value as per chapter 3 is the vector that points from the pattern center to the center of the image
    '''
    PC = np.array([0, 0, 800]) #pattern center absolute coordinates in Pixels, according to chapter 3


    ################################


    get_homography = gh_gpu

    # Load the pattern object
    # pat_obj, ang_data = utilities.get_scan_data(up2, ang)

    pat_obj = Data.UP2(up2)
    ang_data = utilities.read_ang(ang, pat_obj.patshape, segment_grain_threshold=None) #don't have an ang file for this yet so altering all the values 
    
    # ========= Values read in from the ang file that are then used ===========



    # Defining the values for the simulated patterns 
    #create a tensor called quatst that is 500 by 4
    quats = np.zeros((N, 4), dtype=float)
    euler = np.deg2rad(Euler) #converting euler angles to radians
    #convert E to a tensor 
    Q = rotations.eu2qu(euler) #converting to quaternions via ebsdtorch function
    #convert Q to a numpy array
    #set each row of quats to be the same quaternion
    for i in range(N):
        quats[i] = Q
    #reshape quats to be 20 by 25 by 4
    quats = quats.reshape((Rows, Columns, 4))



    # Rotate the stiffness tensor into the sample frame
    if C is not None:
        C = utilities.rotate_stiffness_to_sample_frame(C, quats)
        # C = np.ones(ang_data.shape + (6, 6), dtype=float) * C


    #Defining the pattern center to be that for the simulated images 



    if calc:
        # Create the optimizer

        print ('the optimizer is running ---')
        optimizer = get_homography.ICGNOptimizer(
            pat_obj=pat_obj,
            x0=x0,
            PC=PC, #defined in units of pixels from upper left corner of the detector
            sample_tilt=sample_tilt,
            detector_tilt=detector_tilt,
            pixel_size=pixel_size,
            step_size=step_size,
            scan_shape=(Rows, Columns),
            small_strain=small_strain,
            # Not setting a value for C C=C,
            fixed_projection=fixed_projection,
            traction_free=traction_free,
        )
        
        # Set the image processing parameters
        optimizer.set_image_processing_kwargs(
            low_pass_sigma=low_pass_sigma,
            high_pass_sigma=high_pass_sigma,
            truncate_std_scale=truncate_std_scale
            
        )
        #optimizer.set_simRefpatFlag(False)
        # Set the region of interest
        optimizer.set_roi(start=start, span=span)
        # Set the homography subset (the size of the number of pixels )
        optimizer.set_homography_subset(subset_size, "image")
        # Set the initial guess parameters
        optimizer.set_initial_guess_params(
            subset_size=initial_guess_subset_size, init_type=init_type
        )
        # #steps for the simulated dataset 
        # masterpatternpath = '/Users/crestiennedechaine/Scripts/pyHREBSD/Si-master-20kV.h5'
        # optimizer.set_masterpattern(masterpatternpath)
        # optimizer.set_euler_angles(quats[0, 0, :]) #setting euler angles for pattern simulation in radians
        # #setting the se3 vector 
        # optimizer.set_se3vector(PC) #setting the se3 vector for pattern simulation, should not require any changes


        # Run the optimizer
        optimizer.extra_verbose = True
        optimizer.run()
        #believe this code is for the gpu
        #optimizer.run(
        #    batch_size=16, max_iter=max_iter, conv_tol=conv_tol
        #)
        results = optimizer.results
        results.save(f"results/SimData-Results_Nov3.pkl")
        results.calculate()
        results.save(f"results/SimData-Results_Nov3.pkl")


        
    h23 = results.homographies[0, :, 5]
    h22 = results.homographies[0, :, 4]
    h13 = results.homographies[0, :, 2]
    e = results.strains
    e11 = e[0, :, 0, 0]
    e12 = e[0, :, 0, 1]
    e13 = e[0, :, 0, 2]
    e22 = e[0, :, 1, 1]
    e23 = e[0, :, 1, 2]
    e33 = e[0, :, 2, 2]

    e_t = e33 - (e11 + e22) / 2
    w21 = -np.rad2deg(results.rotations[0, :, 0, 1])
    w13 = np.rad2deg(results.rotations[0, :, 0, 2])
    w32 = -np.rad2deg(results.rotations[0, :, 1, 2])
    res = results.residuals[0]
    itr = results.num_iter[0]
    x = np.arange(len(e11))

    e11_total = e[:, :, 0, 0].flatten()
    e12_total = e[:, :, 0, 1].flatten()
    e13_total = e[:, :, 0, 2].flatten()
    e22_total = e[:, :, 1, 1].flatten()
    e23_total = e[:, :, 1, 2].flatten()
    e33_total = e[:, :, 2, 2].flatten()

    #combine the total strain results into a single array
    strain_results_total = np.zeros((len(e11_total), 6))
    strain_results_total[:, 0] = e11_total
    strain_results_total[:, 1] = e12_total
    strain_results_total[:, 2] = e13_total
    strain_results_total[:, 3] = e22_total
    strain_results_total[:, 4] = e23_total
    strain_results_total[:, 5] = e33_total

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax[0, 0].plot(x, e11, lw=3, c="r", label=r"$\epsilon_{11}$")
    ax[0, 0].plot(x, e22, lw=3, c="g", label=r"$\epsilon_{22}$")
    ax[0, 0].plot(x, e33, lw=3, c="b", label=r"$\epsilon_{33}$")
    ax[0, 1].plot(x, e12, lw=3, c="m", label=r"$\epsilon_{12}$")
    ax[0, 1].plot(x, e13, lw=3, c="y", label=r"$\epsilon_{13}$")
    ax[0, 1].plot(x, e23, lw=3, c="c", label=r"$\epsilon_{23}$")
    ax[0, 2].plot(x, e_t, lw=3, c="k", label=r"$\epsilon_{tetragonal}$")
    ax[1, 0].plot(x, res, lw=3, c="k", label="Residuals")
    ax[1, 1].plot(x, itr, lw=3, c="k", label="Num Iterations")
    # ax[1, 2].plot(x, w13, lw=3, c="tab:orange", label=r"$\omega_{13}$")
    # ax[1, 2].plot(x, w21, lw=3, c="tab:purple", label=r"$\omega_{21}$")
    # ax[1, 2].plot(x, w32, lw=3, c="tab:brown", label=r"$\omega_{32}$")
    ax[1, 2].plot(x, h23, lw=3, c="tab:orange", label=r"$h_{23}$")
    ax[1, 2].plot(x, h22, lw=3, c="tab:purple", label=r"$h_{22}$")
    ax[1, 2].plot(x, h13, lw=3, c="tab:brown", label=r"$h_{13}$")

    bound = 0.02
    for a in [ax[0, 0], ax[0, 1], ax[0, 2]]:
        a.set_ylim(-bound, bound)

    args = [dict(ncols=3), dict(ncols=3), dict(ncols=1), dict(ncols=1), dict(ncols=1), dict(ncols=3)]
    for i, a in enumerate(ax.flatten()):
        utilities.standardize_axis(a)
        utilities.make_legend(a, columnspacing=0.8, handlelength=1, **args[i])

    plt.subplots_adjust(wspace=0.3, hspace=0.15, left=0.08, right=0.99, top=0.95, bottom=0.05)
    plt.savefig("/Users/crestiennedechaine/Scripts/pyHREBSD/results/Aug42025/Simpattern-OptResults-Nov3.png", dpi=300)

    m = results.num_iter > 0
    # Generate maps
    if span is None:
        #span = ang_data.shape
        span = (Rows, Columns)
    xy = (x0[0] - start[0], x0[1] - start[1])
    save_dir = "/Users/crestiennedechaine/Scripts/pyHREBSD/results/Aug42025/"
    utilities.view_tensor_images(results.F[m].reshape(span + (3, 3)),     "deformation", xy, save_dir, name, "all",   "local", "viridis")
    utilities.view_tensor_images(results.strains[m].reshape(span + (3, 3)),    "strain",      xy, save_dir, name, "upper", "local", "viridis")
    utilities.view_tensor_images(results.rotations[m].reshape(span + (3, 3)),   "rotation",      xy, save_dir, name, "upper", "local", "viridis")
    utilities.view_tensor_images(results.homographies[m].reshape(span + (8,)),  "homography",  xy, save_dir, name, "all",   "local", "viridis")
    plt.close("all")



    Fe_calc = results.F[m]
    #save the Fe_calc results to a csv file
    np.savetxt('/Users/crestiennedechaine/Scripts/pyHREBSD/results/Aug42025/Fe_calc_Nov3.csv', Fe_calc.reshape((Fe_calc.shape[0], 9)), delimiter=',')

    #save the homography results
    np.savetxt('/Users/crestiennedechaine/Scripts/pyHREBSD/results/Aug42025/homography_results_Nov3.csv', results.homographies[m], delimiter=',')

    #combine rotation results into a single array
    rotation_results_total = np.zeros((results.rotations[m].shape[0], 3))
    rotation_results_total[:, 0] = w13.flatten()
    rotation_results_total[:, 1] = w21.flatten()
    rotation_results_total[:, 2] = w32.flatten()
    combined_results = np.hstack((strain_results_total, rotation_results_total))
    #save the combined results to a csv file
    np.savetxt('/Users/crestiennedechaine/Scripts/pyHREBSD/results/Aug42025/Strain_Rotation_Results_Nov3.csv', combined_results, delimiter=',', header='e11,e12,e13,e22,e23,e33,w13,w21,w32', comments='')
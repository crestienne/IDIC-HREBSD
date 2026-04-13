# pyHREBSD

HR-EBSD pipeline to determine absolute elastic strain via the use of dynamically simulated EBSD patterns. Is implemented using an inverse compositional Gauss-Newton optimization routine to determine the linear homography required to warp a target EBSP to match a simulated reference EBSP in python. This code follows the HR-EBSD calculations outlined in the [ATEX](http://www.atex-software.eu) EBSD software developed by Jean-Jacques Fundenberger and Benoit Beausir. The linear homography approach was developed by Clement Ernould during his PhD research and more information can be found here: 

Insert link to Ernould's papers

The following code builds upon the HR-EBSD implementation by Dr. James Lamb, whose implementation can be found here: [https://github.com/lambjames18/pyHREBSD] 

The code is designed specifically to support the use of a dynamically simulated reference patterns generated via EMsoft. The following implementation can be run in two ways either using a GUI or alternately through a runner script. 


## Quick Start


### 1 — Install Miniconda
If you don't already have conda, it can be downloaded and installed via  **Miniconda** from:
https://docs.anaconda.com/miniconda/

Follow the installer defaults. When it finishes, open a terminal:
- **Mac**: open the **Terminal** app (search for it in Spotlight with ⌘ Space)
- **Windows**: open **Anaconda Prompt** from the Start menu

### 2 — Download the code
In the terminal, navigate to wherever you want to save the project, then clone the repository:
```
cd ~/Documents
git clone https://github.com/crestienne/IDIC-HREBSD.git
cd IDIC-HREBSD
```

### 3 — Create the conda environment
Once the repository has been downloaded the conda environment must be created. 

Run the commands for your operating system from the **Conda Env** sections below. The following commands only need to be run once. 

### 4 — Launch the GUI
The GUI can be launched from the terminal via the following commands. 

Every time you want to use the software, open a terminal, activate the environment, and run:
```
conda activate hrebsd
cd ~/InsertPathHere/IDIC-HREBSD
python Run_GUI.py
```
A window titled **DIC-HREBSD Pipeline** should appear.
---


HR-EBSD calculations implementing the inverse compositional Gauss-Newton optimization routine for determining the linear homography required to warp a target EBSP to match a reference EBSP in python. This code follows the HR-EBSD calculations outlined in the [ATEX](http://www.atex-software.eu) EBSD software developed by Jean-Jacques Fundenberger and Benoit Beausir. The code supports both vectorized GPU routines (through the `pytorch` package) and parallelized CPU routines (through the `mpire` package).

### Conda Env (windows, cuda version = 12.4)
```
conda create -n hrebsd python=3.12 -y
conda activate hrebsd
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install numpy matplotlib tqdm scipy scikit-image joblib kornia -c conda-forge -y

```
### Conda Env (Mac or no CUDA)
```
conda create -n hrebsd python=3.12 numpy matplotlib tqdm scipy scikit-image pytorch kornia joblib -c pytorch -c conda-forge
conda activate hrebsd
pip install PyQt6
```

### Important Geometry Instructions

All functions currently run using the EDAX sample frame and a Bruker detector frame. This is the same as utilized by kikuchipy and thus if the reader would like more information regarding these two sample frames they are highly encouraged to look there


### File descripitions

The following section needs to be updated....

- `get_homography_cpu.py`: Contains the code for running the inverse compositional Gauss–Newton (IC-GN) algorithm for determining the homographies that warp target patterns to a reference pattern. Inside this file, the method `run_single` of the `ICGNOptimizer` class contains the actual algorithm useed to determine the homographies.
- `get_homography_gpu.py`: Same thing but for the GPU. Note that the GPU version currently does not support creating an initial guess of the homography. Inside this file, the method `run` of the `ICGNOptimizer` class contains the actual algorithm useed to determine the homographies.
- `bspline_gpu.py`: Contains all of the core GPU functions that are used during the IC-GN algorithm on the GPU.
- `warp.py`: Contains helper functions for image warping, coordinate warping, homography shape functions, and a custom Spline class. Functions for the CPU and GPU are contained here.
- `Data.py`: Contains a simgple UP2 class for reading up2 files and processing the EBSPs contained within.
- `utilities.py`: Numerous helper functions for viewing results, reading/manipulating patterns, pattern center conversions, ang/up2 reader, patttern sharpness calculation, stiffness tensor creating, etc.
- `segment.py`: Methods for segmenting grains in an EBSD dataset. This is experimental. Future use will use this function to separate an HR-EBSD calculation into individual grains.
- `rotations.py`: Conversions between different orientation representations. Copied from the pyebsdindex python package developed by Dave Rowenhorst.
- `conversions.py`: Contains conversion functions needed for the IC-GN algorithm. Such as homography to deformation gradient, deformation gradient to strain, translation and rotation to homography, etc.
- `[...]_runner.py`: A script for running the HREBSD calculation. These scripts are tailored for specific experiments/datasets, but showcase the inputs needed in order to run a scan.

All other scripts are either deprecated, for development, or for testing features. Note that this repository is actively under development and there will likely be breaking changes.

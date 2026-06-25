# IDIC - HREBSD

HR-EBSD pipeline to determine absolute elastic strain via the use of dynamically simulated EBSD patterns. It is implemented using an inverse compositional Gauss-Newton optimization routine to determine the linear homography required to warp a target EBSP to match a simulated reference EBSP in Python. This code follows the HR-EBSD calculations outlined in the [ATEX](http://www.atex-software.eu) EBSD software developed by Jean-Jacques Fundenberger and Benoit Beausir. The linear homography approach was developed by Clement Ernould during his PhD research, and more information can be found [here](http://www.atex-software.eu/papers/EBF22_ch2.pdf).

This code builds upon the HR-EBSD implementation by Dr. James Lamb, whose implementation, pyHREBSD, can be found [here](https://github.com/lambjames18/pyHREBSD).

IDIC-HREBSD is designed specifically to support the use of dynamically simulated reference patterns generated via EMsoft. This work also utilizes a modified version of EBSDtorch, developed by Dr. Zachary Varley, linked [here](https://github.com/ZacharyVarley/ebsdtorch).

This work was supported by the ARO MURI Program (ARO W911NF-25-2-0164).

## Quick Start

### 1 — Install Miniconda

If you don't already have conda, it can be downloaded and installed via **Miniconda** from:
https://docs.anaconda.com/miniconda/

Follow the installer defaults. When it finishes, open a terminal:

- **Mac**: open the **Terminal** app (search for it in Spotlight with ⌘ Space)
- **Windows**: open **Anaconda Prompt** from the Start menu

### 2 — Download the code

In the terminal, navigate to wherever you want to save the project, then clone the repository:

```
cd ~/insert/path/here
git clone https://github.com/crestienne/IDIC-HREBSD.git
cd IDIC-HREBSD
```

### 3 — Create the conda environment

Once the repository has been downloaded, the conda environment must be created. Run the commands for your operating system from the sections below. These commands only need to be run once.

### Conda Env (Windows, CUDA version 12.4)

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

### 4 — Launch the GUI

The GUI can be launched from the terminal via the following commands. Every time you want to use the software, open a terminal and activate the environment:

```
conda activate hrebsd
```

Then navigate to where IDIC-HREBSD is stored:

```
cd ~/insert/your/path/IDIC-HREBSD
```

Then run:

```
python Run_GUI.py
```

A window titled **DIC-HREBSD Pipeline** should appear.

#### Running scripts (non-GUI)

The codebase is organized as packages (`core/`, `fileio/`, `gui/`, `analysis/`,
`scripts/`). Standalone scripts must be run as modules **from the repo root** so the
packages resolve, e.g.:

```
python -m scripts.runner
python -m analysis.figure_spectral_match
```

(Running `python scripts/runner.py` directly will not put the repo root on the path.)

---

These HR-EBSD calculations implement the inverse compositional Gauss-Newton optimization routine for determining the linear homography required to warp a target EBSP to match a reference EBSP in Python. This code follows the HR-EBSD calculations outlined in the [ATEX](http://www.atex-software.eu) EBSD software developed by Jean-Jacques Fundenberger and Benoit Beausir. The code supports both vectorized GPU routines (through the `pytorch` package) and parallelized CPU routines (through the `mpire` package).

### Conda Env (Windows, CUDA version 12.4)

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

All functions currently run using the EDAX sample frame and a Bruker detector frame. This is the same setup utilized by kikuchipy, so if the reader would like more information regarding these two sample frames, they are highly encouraged to look [there](https://kikuchipy.org/en/stable/tutorials/reference_frames.html).

### Layout

`Run_GUI.py` (repo root) is the GUI entry point. Everything else is grouped into
role-based packages:

**`core/`** — the HR-EBSD engine and shared math:
- get_homography_cpu.py - The primary pipeline for the IC-GN algorithm. Contains all code pertaining to the IC-GN algorithm
- get_homography_cpu_reversed.py - an experimental script so the reversibility of the IC-GN algorithm can be tested
- conversions.py - handles all conversions for the pattern center to internal pattern center conventions. The pattern center is defined internally utilizing the Bruker pattern center convention.
- HREBSD.py - All code related to pattern simulation
- warp.py, rotations.py, segment.py - image warping, rotation/quaternion math, grain segmentation
- optimize_reference.py - pattern-center / Euler reference optimization
- ErnouldsMethod.py - linear-homography HR-EBSD (EMEBSD-based), not used by the current GUI
- multiple_ref.py, pc_homography_correction.py, pc_plane_fit.py - multi-reference handling and PC correction helpers
- utilities.py - shared helpers (processing, plotting, elastic-constant math, Results class)

**`fileio/`** — data I/O:
- Data.py - reading in and processing experimental EBSD patterns, including .up2 reading and pre-processing
- ebsd_io.py - low-level .up2 / .ang readers (split out of utilities.py)
- write_up2.py - writing/binning .up2 files

**`gui/`** — the PyQt6 wizard (plus `Materials/` stiffness-tensor presets):
- gui_pages.py, gui_workers.py, gui_visualization.py, gui_theme.py
- gui_help.py - help menu descriptions and formatting
- gui_materials.py - materials GUI for adding stiffness tensors
- gui_settings.py - in development; font size and theme settings

**`analysis/`** — results visualization and figures:
- Results_plotting.py, runner_results_vis.py
- ipf_map.py - IPF map plotting
- viz_samp2detectorATEX.py, figure_pc_shift_vs_strain.py, figure_spectral_match.py

**`scripts/`** — standalone runnable scripts (run as `python -m scripts.<name>`):
- runner.py, optimization_test.py, homography_validation.py
- fmt_initial_shifts_scan.py, put_sharpness_in_ang.py

**`PatternSimulation/`** — EMsoft master-pattern projection backend (`SimPatGen.py`).
- warp.py 
- write_up2.py

### Main Todos

- Add support for non cubic materials (specifically regarding the traction free boundary condition and the materials stiffness tensor)
- Add support for non EDAX files (specfically Oxford)



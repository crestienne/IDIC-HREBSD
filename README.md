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

'''
SimPatGen — EBSD pattern generation backed by HREBSD.py.

Per-pixel energy-weighted projection (EMsoft CalcEBSDPatternSingleFull_ equivalent):
the master patterns are kept 3-D with the energy axis preserved, and the
Monte-Carlo accum_e is projected onto the detector grid at the current PC
inside GenPattern with the EMsoft GenerateDetector_ geometric obliquity
factor applied — so the rendered pattern carries both the spatial falloff
in the BSE energy distribution and the cos³(off-axis) intensity envelope.

Public API kept identical to the previous version:

    sim = patternSimulation()
    sim.detector_height / detector_width / det_shape    # set if non-default
    sim.detector_tilt_deg / sample_tilt_deg / azimuthal_deg
    sim.mastersetup(master_pattern_h5_path)
    sim.EandPCSet(euler_rad_bunge, pc_bruker, verbose=True)
    pats = sim.GenPattern()        # (batch, H*W) tensor — reshape downstream
'''

import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import sys
import h5py
import numpy as np
import torch
from torch import Tensor

# Make the parent project importable so we can pull in HREBSD.py
_PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from HREBSD import (
    bu2qu_emsoft,
    project_HREBSD_pattern_energy_weighted,
    accum_e_to_detector,
)


def _read_master_pattern_h5(file_path: str) -> tuple:
    """Load energy-resolved NH/SH master patterns and the MC-grid accum_e
    from an EMsoft-style .h5 master file.

    Returns:
        (mLPNH, mLPSH, accum_e_mc) — all torch.float32 tensors with the
        energy axis as the FIRST dimension:
            mLPNH, mLPSH : (nE, H, W)
            accum_e_mc    : (nE, ns, ns)

    EMsoft stores accum_e as (ns, ns, nE) and the master patterns as
    (1, nE, H, W); both are normalized to (nE, *) here.
    """
    with h5py.File(file_path, "r") as f:
        e_accum = f["EMData/MCOpenCL/accum_e"][:].astype(np.float32)
        nh = f["EMData/EBSDmaster/mLPNH"][:].astype(np.float32).squeeze()
        sh = f["EMData/EBSDmaster/mLPSH"][:].astype(np.float32).squeeze()

    if nh.ndim != 3 or sh.ndim != 3:
        raise ValueError(
            f"Master patterns must be 3-D (nE, H, W) after squeeze; "
            f"got mLPNH={nh.shape}, mLPSH={sh.shape}.  Is this an "
            f"energy-resolved EMsoft master pattern?"
        )
    nE_master = nh.shape[0]

    if e_accum.ndim != 3:
        raise ValueError(
            f"accum_e must be 3-D; got shape {e_accum.shape}."
        )

    # EMsoft writes accum_e_save(numEbins, numsx, numsy) in Fortran row-major.
    # HDF5+numpy produce shape (numsy, numsx, nE) — i.e., nE on the last axis.
    # Some master-pattern dumps keep nE first.  Detect by matching one axis
    # to the master-pattern nE.
    if e_accum.shape[-1] == nE_master and e_accum.shape[0] == e_accum.shape[1]:
        accum_e_mc = np.transpose(e_accum, (2, 0, 1))  # (ns, ns, nE) -> (nE, ns, ns)
    elif e_accum.shape[0] == nE_master and e_accum.shape[1] == e_accum.shape[2]:
        accum_e_mc = e_accum                           # already (nE, ns, ns)
    else:
        raise ValueError(
            f"Cannot match accum_e shape {e_accum.shape} to master-pattern "
            f"nE={nE_master}.  Expected one accum_e axis of length {nE_master}."
        )

    return (
        torch.from_numpy(nh).contiguous(),
        torch.from_numpy(sh).contiguous(),
        torch.from_numpy(accum_e_mc).contiguous(),
    )


class patternSimulation:
    def __init__(self):
        self.batch  = 1
        self.device = torch.device("cpu")
        self.dtype  = torch.float32

        self.detector_height   = 516
        self.detector_width    = 516
        self.det_shape         = (self.detector_height, self.detector_width)
        self.detector_tilt_deg = 70.0
        self.azimuthal_deg     = 0.0
        self.sample_tilt_deg   = 0.0

        # Deformation gradient (identity = no deformation).
        # project_HREBSD_pattern_energy_weighted inverts internally.
        self.F = torch.eye(3, dtype=self.dtype, device=self.device)[None, :, :]

        # populated by EandPCSet
        self.quats              = torch.zeros(1, 4, device=self.device, dtype=self.dtype)
        self.pattern_centerInit = None

        # populated by mastersetup — all energy-resolved (nE-first axis)
        self.mLPNH      = None        # (nE, H, W)
        self.mLPSH      = None        # (nE, H, W)
        self.accum_e_mc = None        # (nE, ns, ns)

    def mastersetup(self, masterpatternpath: str):
        mLPNH, mLPSH, accum_e_mc = _read_master_pattern_h5(masterpatternpath)
        self.mLPNH      = mLPNH.to(self.device).to(self.dtype)
        self.mLPSH      = mLPSH.to(self.device).to(self.dtype)
        self.accum_e_mc = accum_e_mc.to(self.device).to(self.dtype)
        print(f"Master Pattern Loaded  (nE={self.mLPNH.shape[0]}, "
              f"Lambert={tuple(self.mLPNH.shape[1:])}, "
              f"MC grid={tuple(self.accum_e_mc.shape[1:])})")

    def EandPCSet(self, Euler, PC, D=None, verbose: bool = True):
        '''
        Args:
            Euler: Bunge ZXZ Euler angles in radians, shape (3,).
            PC:    Pattern center in Bruker convention (pcx, pcy, pcz).
            D:     Optional deformation gradient (3, 3) — defaults to identity.
        '''
        E = torch.tensor(Euler, dtype=self.dtype, device=self.device)[None, :]
        qu = bu2qu_emsoft(E)
        qu = qu / torch.norm(qu, dim=1, keepdim=True)
        self.quats = torch.nn.Parameter(qu.repeat(self.batch, 1))

        pc_t = torch.tensor(PC, dtype=self.dtype, device=self.device)[None, :]
        self.pattern_centerInit = torch.nn.Parameter(pc_t.repeat(self.batch, 1))

        if D is not None:
            F = torch.as_tensor(D, dtype=self.dtype, device=self.device)
            if F.ndim == 2:
                F = F[None, ...]
            self.F = F

        if verbose:
            print('Initial Quaternions: ' + str(self.quats))
            print('Initial PC (Bruker): ' + str(self.pattern_centerInit))

    def GenPattern(self) -> Tensor:
        """Project the master pattern through the current geometry / orientation,
        using the EMsoft-equivalent per-pixel energy weighting + obliquity.

        Returns:
            Tensor of shape (batch, H*W) — caller reshapes to (H, W) as before.
        """
        if self.mLPNH is None or self.mLPSH is None or self.accum_e_mc is None:
            raise RuntimeError("mastersetup(...) must be called before GenPattern().")
        if self.pattern_centerInit is None:
            raise RuntimeError("EandPCSet(...) must be called before GenPattern().")

        # Ensure unit quaternions even if Parameter has drifted during optim
        quats = self.quats / torch.norm(self.quats, dim=1, keepdim=True)

        H, W = self.det_shape

        # Project MC-grid accum_e onto the detector for this PC / geometry,
        # including EMsoft GenerateDetector_ obliquity (cos³-like) weighting.
        # PC changes during Nelder-Mead refinement, so this is recomputed on
        # every GenPattern call.
        accum_e_det = accum_e_to_detector(
            self.accum_e_mc,
            self.pattern_centerInit,
            int(H), int(W),
            float(self.detector_tilt_deg),
            float(self.azimuthal_deg),
            float(self.sample_tilt_deg),
        )  # (nE, H, W)

        pats = project_HREBSD_pattern_energy_weighted(
            pcs                   = self.pattern_centerInit,
            n_rows                = int(H),
            n_cols                = int(W),
            tilt                  = float(self.detector_tilt_deg),
            azimuthal             = float(self.azimuthal_deg),
            sample_tilt           = float(self.sample_tilt_deg),
            quaternions           = quats,
            deformation_gradients = self.F,
            master_pattern_MSLNH  = self.mLPNH,
            master_pattern_MSLSH  = self.mLPSH,
            accum_e               = accum_e_det,
            signal_mask           = None,
        )
        return pats

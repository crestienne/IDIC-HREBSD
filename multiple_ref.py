"""
multiple_ref.py — ReferencePatternSet container and auto-selection logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ReferenceEntry:
    grain_id:    int
    ref_row:     int
    ref_col:     int
    ref_pat_idx: int                         # flat index into UP2
    pc:          Tuple[float, float, float]
    euler:       Tuple[float, float, float]  # radians (phi1, Phi, phi2)


class ReferencePatternSet:
    """One reference pattern entry per grain, indexed by grain_id."""

    def __init__(self, entries: List[ReferenceEntry] = None):
        self._entries: List[ReferenceEntry] = sorted(
            entries or [], key=lambda e: e.grain_id
        )

    # ── container protocol ────────────────────────────────────────────────────

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, idx):
        return self._entries[idx]

    # ── helpers ───────────────────────────────────────────────────────────────

    def grain_ids(self) -> List[int]:
        return [e.grain_id for e in self._entries]

    def by_grain(self, grain_id: int) -> Optional[ReferenceEntry]:
        for e in self._entries:
            if e.grain_id == grain_id:
                return e
        return None

    def update_ref(
        self,
        grain_id:    int,
        ref_row:     int,
        ref_col:     int,
        ref_pat_idx: int,
        euler:       Tuple[float, float, float] = None,
    ):
        """Override the reference position for a specific grain."""
        for e in self._entries:
            if e.grain_id == grain_id:
                e.ref_row     = ref_row
                e.ref_col     = ref_col
                e.ref_pat_idx = ref_pat_idx
                if euler is not None:
                    e.euler = euler
                return
        raise KeyError(f"Grain {grain_id} not found in ReferencePatternSet")


def select_references(
    grain_ids: np.ndarray,
    ang_data,
    scan_cols: int,
    strategy: str = "mean",
    kam: np.ndarray = None,
    iq:  np.ndarray = None,
    interior_erode: int = 2,
    selected_grain_ids: list = None,
) -> ReferencePatternSet:
    """For each grain label > 0, pick a representative reference pixel.

    Strategies
    ----------
    ``strategy = "mean"`` (default)
        Pick the pixel whose quaternion is closest to the grain's mean
        quaternion (max ``|q_mean · q_i|`` over candidates).
    ``strategy = "kam_min"``
        Pick the pixel with the lowest in-grain KAM (most orientation-uniform
        local neighbourhood).  Requires ``kam`` — the ``average_misorientation``
        array returned by ``segment.segment_grains``.
    ``strategy = "iq_max"``
        Pick the pixel with the highest pattern quality (image quality / IQ)
        within the grain.  Requires ``iq`` — the IQ column from the .ang
        file (``ang_data.iq``).

    Interior filter
    ---------------
    Before picking, each grain's boolean mask is eroded by ``interior_erode``
    pixels via ``scipy.ndimage.binary_erosion`` to keep the reference away
    from grain boundaries and the scan edge.  If the eroded mask is empty
    (very small grain), the full mask is used as a fallback.

    Args:
        grain_ids:      (rows, cols) int array of grain labels; 0 = unassigned
        ang_data:       namedtuple from utilities.read_ang — needs .quats / .eulers / .pc
        scan_cols:      number of columns in the full scan (for flat index)
        strategy:       "mean" or "kam_min"
        kam:            (rows, cols) float array of per-pixel in-grain KAM
                        (required if strategy="kam_min")
        interior_erode: erosion radius in pixels for the interior-only filter
                        (0 = no erosion).

    Returns:
        ReferencePatternSet with one entry per non-zero grain, sorted by grain_id
    """
    if strategy not in ("mean", "kam_min", "iq_max"):
        raise ValueError(
            f"strategy must be 'mean', 'kam_min', or 'iq_max', got {strategy!r}"
        )
    if strategy == "kam_min" and kam is None:
        raise ValueError("strategy='kam_min' requires a kam array")
    if strategy == "iq_max" and iq is None:
        raise ValueError("strategy='iq_max' requires an iq array")

    from scipy.ndimage import binary_erosion

    quats  = ang_data.quats    # (rows, cols, 4)
    eulers = ang_data.eulers   # (rows, cols, 3)
    pc     = ang_data.pc       # single PC for the whole scan

    # If the caller passed an explicit list of grain IDs (e.g. the user's
    # interactive Step 4 selection), only build entries for those grains.
    if selected_grain_ids is not None:
        _allowed = set(int(g) for g in selected_grain_ids if int(g) != 0)
    else:
        _allowed = None

    entries = []
    for gid in np.unique(grain_ids):
        if gid == 0:
            continue
        if _allowed is not None and int(gid) not in _allowed:
            continue
        grain_mask = (grain_ids == gid)

        # Interior filter — erode then fall back to the full grain if the
        # erosion ate everything.
        if interior_erode > 0:
            interior = binary_erosion(grain_mask, iterations=int(interior_erode))
            if not interior.any():
                interior = grain_mask
        else:
            interior = grain_mask

        row_idx, col_idx = np.where(interior)
        grain_quats = quats[row_idx, col_idx]   # (N, 4)

        if strategy == "kam_min":
            kam_vals = kam[row_idx, col_idx]
            # If KAM is NaN everywhere (single-pixel grains, no neighbours)
            # fall back to mean-strategy on the same candidates.
            if np.all(np.isnan(kam_vals)):
                strategy_used = "mean"
            else:
                strategy_used = "kam_min"
        elif strategy == "iq_max":
            iq_vals = iq[row_idx, col_idx]
            # All-NaN / non-finite IQ → fall back to mean.
            if not np.any(np.isfinite(iq_vals)):
                strategy_used = "mean"
            else:
                strategy_used = "iq_max"
        else:
            strategy_used = "mean"

        if strategy_used == "kam_min":
            kam_safe = np.where(np.isnan(kam_vals), np.inf, kam_vals)
            best = int(np.argmin(kam_safe))
        elif strategy_used == "iq_max":
            iq_safe = np.where(np.isfinite(iq_vals), iq_vals, -np.inf)
            best = int(np.argmax(iq_safe))
        else:
            q_mean = grain_quats.mean(axis=0)
            norm   = np.linalg.norm(q_mean)
            if norm < 1e-12:
                best = 0
            else:
                q_mean /= norm
                dots = np.abs(grain_quats @ q_mean)
                best = int(np.argmax(dots))

        ref_row = int(row_idx[best])
        ref_col = int(col_idx[best])
        entries.append(ReferenceEntry(
            grain_id=int(gid),
            ref_row=ref_row,
            ref_col=ref_col,
            ref_pat_idx=ref_row * scan_cols + ref_col,
            pc=tuple(pc),
            euler=tuple(eulers[ref_row, ref_col]),
        ))

    return ReferencePatternSet(entries)

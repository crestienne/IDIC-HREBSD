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
) -> ReferencePatternSet:
    """For each grain label > 0, pick the scan point whose quaternion is
    closest to the grain's mean quaternion (minimum geodesic misorientation).

    Args:
        grain_ids:  (rows, cols) int array of grain labels; 0 = unassigned
        ang_data:   namedtuple from utilities.read_ang — needs .quats and .eulers
        scan_cols:  number of columns in the full scan (for flat index)

    Returns:
        ReferencePatternSet with one entry per non-zero grain, sorted by grain_id
    """
    quats  = ang_data.quats    # (rows, cols, 4)
    eulers = ang_data.eulers   # (rows, cols, 3)
    pc     = ang_data.pc       # single PC for the whole scan

    entries = []
    for gid in np.unique(grain_ids):
        if gid == 0:
            continue
        row_idx, col_idx = np.where(grain_ids == gid)
        grain_quats = quats[row_idx, col_idx]   # (N, 4)

        # Quaternion mean: average then renormalise.
        # Good approximation for grains with <~10° spread (typical HREBSD case).
        q_mean = grain_quats.mean(axis=0)
        norm   = np.linalg.norm(q_mean)
        if norm < 1e-12:
            best = 0
        else:
            q_mean /= norm
            # cos(geodesic_angle / 2) = |q_mean · q_i| — maximise this
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

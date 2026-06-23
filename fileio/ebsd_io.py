# Description: I/O helpers for reading EBSD pattern (.up2) and orientation (.ang) files.
# Split out of utilities.py.
# Author: James Lamb

import os
import re
import struct
from collections import namedtuple

import numpy as np

import segment
import rotations


NUMERIC = r"[-+]?\d*\.\d+|\d+"

_EDAX_DEFAULT_COLUMNS = ["Phi1", "Phi", "Phi2", "x", "y", "IQ", "CI", "Phase"]


def read_up2(up2: str) -> namedtuple:
    """Read in patterns and a pattern center from an ang file and a pattern file.
    Only supports a up2 file using the EDAX/TSL convention.

    Args:
        up2 (str): Path to the pattern file.

    Returns:
        namedtuple: Pattern file object with fields patshape, filesize, nPatterns, and datafile.
                    patshape is a tuple of the pattern dimensions.
                    filesize is the size of the pattern file.
                    nPatterns is the number of patterns in the file.
                    datafile is the file object to read the patterns."""
    # Get patterns
    upFile = open(up2, "rb")
    chunk_size = 4
    tmp = upFile.read(chunk_size)
    FirstEntryUpFile = struct.unpack("i", tmp)[0]
    tmp = upFile.read(chunk_size)
    sz1 = struct.unpack("i", tmp)[0]
    tmp = upFile.read(chunk_size)
    sz2 = struct.unpack("i", tmp)[0]
    tmp = upFile.read(chunk_size)
    bitsPerPixel = struct.unpack("i", tmp)[0]
    sizeBytes = os.path.getsize(up2) - 16
    sizeString = str(round(sizeBytes / 1e6, 1)) + " MB"
    bytesPerPixel = 2
    nPatternsRecorded = int((sizeBytes / bytesPerPixel) / (sz1 * sz2))
    out = namedtuple("up2_file", ["patshape", "filesize", "nPatterns", "datafile"])
    out = out((sz1, sz2), sizeString, nPatternsRecorded, upFile)
    return out


def read_ang(
    path: str,
    patshape: tuple | list | np.ndarray = None,
    segment_grain_threshold: float = None,
    column_names: list = None,
) -> namedtuple:
    """Reads in the pattern center from an ang file.
    Only supports EDAX/TSL.

    To print the data columns in the ang file, use the following:
    >>> ang_data = read_ang("path/to/ang/file.ang")
    >>> print(ang_data._fields)

    Args:
        ang (str): Path to the ang file.
        patshape (tuple): The shape of the patterns. If None, the pattern center will be
                          (xstar, ystar, zstar) and not (xpc, ypc, L).
        segment_grain_threshold (bool): Grain boundary threshold for segmenting grains.
                                        If None (default), grains will not be segmented.

    Returns:
        namedtuple: The data read in from the ang file with the following fields:
                    - quats: The quaternions.
                    - eulers: The Euler angles.
                    - shape: The shape of the data.
                    - pc: The pattern center.
                    - pidx: The index of the pattern in the pattern file.
                    - ids: The grain IDs. Will be all ones if segment_grains is False.
                    - all data columns in the ang file
                      (i.e. x, y, iq, ci, sem, phase_index, etc.)
    """
    header_lines = 0
    names = None
    # Safe defaults in case any header field is missing
    xstar = ystar = zstar = 0.0
    rows = cols = cols_even = 1
    step_size = 1.0
    grid_type = "SqrGrid"

    with open(path, "r") as ang:
        for line in ang:
            if not line.startswith("#"):
                break
            header_lines += 1
            if "x-star" in line:
                xstar = float(re.findall(NUMERIC, line)[0])

            elif "y-star" in line:
                ystar = float(re.findall(NUMERIC, line)[0])

            elif "z-star" in line:
                zstar = float(re.findall(NUMERIC, line)[0])

            elif "NROWS" in line:
                rows = int(re.findall(NUMERIC, line)[0])

            elif "NCOLS_ODD" in line:
                cols = int(re.findall(NUMERIC, line)[0])

            elif "NCOLS_EVEN" in line:
                cols_even = int(re.findall(NUMERIC, line)[0])

            elif "GRID" in line and "HEADER" not in line:
                grid_type = line.split(":")[-1].strip()

            elif "XSTEP" in line:
                step_size = float(re.findall(NUMERIC, line)[0])
                print(f"X-step size: {step_size}")
            elif "COLUMN_HEADERS" in line:
                names = line.replace("\n", "").split(":")[1].strip().split(", ")
            elif "HEADER: End" in line:
                break

    if names is None:
        names = column_names if column_names is not None else _EDAX_DEFAULT_COLUMNS

    # Package the header data — always store original fractional (xstar, ystar, zstar)
    PC = (xstar, ystar, zstar)
    print(f"Pattern center (xstar, ystar, zstar): {PC}")
    print(f"Grid type: {grid_type},  NROWS: {rows},  NCOLS_ODD: {cols},  NCOLS_EVEN: {cols_even}")

    # Read in the data
    ang_data = np.genfromtxt(path, skip_header=header_lines)
    n_cols_data = ang_data.shape[1]

    is_hex = (grid_type.strip().lower() == "hexgrid") and (cols_even != cols)

    if is_hex:
        # HexGrid: odd-indexed rows (0, 2, 4, …) have `cols` points,
        # even-indexed rows (1, 3, 5, …) have `cols_even` points.
        # We reconstruct a rectangular (rows × cols) array, padding shorter
        # rows with NaN so downstream code can keep the same 2-D indexing.
        n_odd = (rows + 1) // 2   # rows 0, 2, 4, ...
        n_even = rows // 2        # rows 1, 3, 5, ...
        expected = n_odd * cols + n_even * cols_even
        if ang_data.shape[0] != expected:
            print(
                f"Warning: HexGrid expected {expected} data lines "
                f"but got {ang_data.shape[0]}. Proceeding anyway."
            )

        rect = np.full((rows, cols, n_cols_data), np.nan)
        src_idx = 0
        for r in range(rows):
            ncols_this_row = cols if (r % 2 == 0) else cols_even
            rect[r, :ncols_this_row, :] = ang_data[src_idx : src_idx + ncols_this_row, :]
            src_idx += ncols_this_row
        ang_data = rect
        shape = (rows, cols)
    else:
        shape = (rows, cols)
        ang_data = ang_data.reshape(shape + (n_cols_data,))
    euler = ang_data[..., 0:3]
    ang_data = ang_data[..., 3:]

    # Build column names, dropping euler angles
    data_col_names = [
        name.replace(" ", "_").lower()
        for name in names
        if name.lower() not in ["phi1", "phi", "phi2"]
    ]
    # If parsed names don't match actual column count, fall back to generic names
    if len(data_col_names) != ang_data.shape[-1]:
        print(
            f"Warning: {len(data_col_names)} column names but {ang_data.shape[-1]} data columns. "
            "Falling back to generic names."
        )
        data_col_names = [f"col_{i}" for i in range(ang_data.shape[-1])]

    names = data_col_names + ["eulers", "quats", "shape", "pc", "step_size", "pidx"]
    qu = rotations.eu2qu(euler)
    pidx = np.arange(np.prod(shape)).reshape(shape)
    if segment_grain_threshold is not None:
        ids, kam = segment.segment_grains(qu, segment_grain_threshold)
        #args = (euler, qu, shape, PC, step_size, pidx, ids, kam)

    args = (euler, qu, shape, PC, step_size, pidx)
    ang_data = np.moveaxis(ang_data, 2, 0)

    print("FIELD COUNT:", len(names))
    print("VALUES COUNT:", len(ang_data) + len(args))


    # Package everything into a namedtuple
    out = namedtuple("ang_file", names)(*ang_data, *args)
    if segment_grain_threshold is not None:
        print(
            f"Segmented grains with threshold {segment_grain_threshold}. Number of grains: {len(np.unique(ids))}"
        )
        return out, ids, kam
    else:
        return out


def get_scan_data(up2: str, ang: str) -> tuple:
    """Reads in patterns and orientations from an ang file and a pattern file.
    Only supports EDAX/TSL.

    Args:
        up2 (str): Path to the pattern file.
        ang (str): Path to the ang file.

    Returns:
        np.ndarray: The patterns.
        namedtuple: The orientations. namedtuple with fields corresponding to the columns in the ang file + eulers, quats, shape, pc.
    """
    # Get the patterns
    pat_obj = read_up2(up2)

    # Get the ang data
    ang_data = read_ang(ang, pat_obj.patshape)

    return pat_obj, ang_data


def get_patterns(pat_obj: namedtuple, idx: np.ndarray | list | tuple = None) -> tuple:
    """Read in patterns from a pattern file object.

    Args:
        pat_obj (namedtuple): Pattern file object.
        idx (np.ndarray | list | tuple): Indices of patterns to read in. If None, reads in all patterns.

    Returns:
        np.ndarray: Patterns."""
    # Handle inputs
    if idx is None:
        idx = range(pat_obj.nPatterns)
        reshape = False
    else:
        idx = np.asarray(idx)
        reshape = False
        if idx.ndim >= 2:
            reshape = True
            out_shape = idx.shape + pat_obj.patshape
            idx = idx.flatten()

    # Read in the patterns
    start_byte = np.int64(16)
    pattern_bytes = np.int64(pat_obj.patshape[0] * pat_obj.patshape[1] * 2)
    pats = np.zeros((len(idx), *pat_obj.patshape), dtype=np.uint16)
    # for i in tqdm(range(len(idx)), desc="Reading patterns", unit="pats"):
    for i in range(len(idx)):
        pat = np.int64(idx[i])
        seek_pos = np.int64(start_byte + pat * pattern_bytes)
        pat_obj.datafile.seek(seek_pos)
        pats[i] = np.frombuffer(
            pat_obj.datafile.read(pat_obj.patshape[0] * pat_obj.patshape[1] * 2),
            dtype=np.uint16,
        ).reshape(pat_obj.patshape)

    # Reshape the patterns
    pats = np.squeeze(pats)
    if reshape:
        pats = pats.reshape(out_shape)

    return pats


def get_pattern(pat_obj: namedtuple, idx: int = None) -> tuple:
    """Read in patterns from a pattern file object.

    Args:
        pat_obj (namedtuple): Pattern file object.
        idx (int): Indice of pattern to read in.

    Returns:
        np.ndarray: Patterns."""
    # Read in the patterns
    start_byte = np.int64(16)
    pattern_bytes = np.int64(pat_obj.patshape[0] * pat_obj.patshape[1] * 2)
    # for i in tqdm(range(len(idx)), desc="Reading patterns", unit="pats"):
    seek_pos = np.int64(start_byte + np.int64(idx) * pattern_bytes)
    pat_obj.datafile.seek(seek_pos)
    pat = np.frombuffer(
        pat_obj.datafile.read(pat_obj.patshape[0] * pat_obj.patshape[1] * 2),
        dtype=np.uint16,
    ).reshape(pat_obj.patshape)
    return pat

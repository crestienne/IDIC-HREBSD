"""
segment.py  —  grain segmentation by flood-fill on misorientation
"""

from collections import deque

import numpy as np
from tqdm.auto import tqdm

import rotations


# ── Cubic symmetry operators (24 proper rotations of Oh / m-3m) ──────────────
R2 = 0.7071067811865475244008443621048490392848359376884740365883398689
R3 = 0.8660254037844386467637231707529361834714026269051903140279034897

LAUE_O = np.array(
    [
        [1,   0,   0,   0  ],
        [0,   1,   0,   0  ],
        [0,   0,   1,   0  ],
        [0,   0,   0,   1  ],
        [R2,  0,   0,   R2 ],
        [R2,  0,   0,  -R2 ],
        [0,   R2,  R2,  0  ],
        [0,  -R2,  R2,  0  ],
        [0.5, 0.5,-0.5, 0.5],
        [0.5, 0.5, 0.5,-0.5],
        [0.5, 0.5,-0.5,-0.5],
        [0.5,-0.5,-0.5,-0.5],
        [0.5,-0.5, 0.5, 0.5],
        [0.5,-0.5, 0.5,-0.5],
        [0.5,-0.5,-0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
        [R2,  R2,  0,   0  ],
        [R2, -R2,  0,   0  ],
        [R2,  0,   R2,  0  ],
        [R2,  0,  -R2,  0  ],
        [0,   R2,  0,   R2 ],
        [0,  -R2,  0,   R2 ],
        [0,   0,   R2,  R2 ],
        [0,   0,  -R2,  R2 ],
    ],
    dtype=np.float64,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2-D grain segmentation
# ─────────────────────────────────────────────────────────────────────────────

def segment_grains(quaternions: np.ndarray, threshold: float):
    """
    Segment a 2-D EBSD scan into grains by flood-fill on misorientation.

    Parameters
    ----------
    quaternions : (rows, cols, 4) float array
        Per-pixel quaternions in scalar-first convention [w, x, y, z].
    threshold : float
        Misorientation threshold in degrees.  Neighbours whose
        misorientation from the seed is ≤ threshold are merged into the
        same grain.

    Returns
    -------
    grain_ids : (rows, cols) int array
        Grain label for each pixel (1-indexed; 0 = unvisited, should not
        remain after the function returns).
    average_misorientation : (rows, cols) float array
        Mean misorientation (deg) of each pixel with respect to its
        within-grain neighbours.  NaN where no within-grain neighbour
        exists.
    """
    rows, cols = quaternions.shape[:2]
    grain_ids              = np.zeros((rows, cols), dtype=int)
    average_misorientation = np.full((rows, cols), np.nan, dtype=float)
    current_grain_id       = 1

    coords       = np.array(np.meshgrid(np.arange(rows), np.arange(cols))).T.reshape(-1, 2)
    progress_bar = tqdm(total=rows * cols, desc="Segmenting grains")

    for i, j in coords:
        if grain_ids[i, j] != 0:
            progress_bar.update(1)
            continue

        # ── Start a new grain by flood-fill ───────────────────────────────
        grain_ids[i, j] = current_grain_id
        queue = deque([(i, j)])

        while queue:
            x, y = queue.popleft()

            # Build neighbour list
            nbr = []
            if x - 1 >= 0:    nbr.append((x - 1, y))
            if x + 1 < rows:  nbr.append((x + 1, y))
            if y - 1 >= 0:    nbr.append((x, y - 1))
            if y + 1 < cols:  nbr.append((x, y + 1))

            if not nbr:
                continue

            nbr      = np.array(nbr)                          # (K, 2)
            nbr_quat = quaternions[nbr[:, 0], nbr[:, 1]]     # (K, 4)

            angles = misorientation(
                quaternions[x, y],
                nbr_quat,
                degrees=True,
                symmetry=True,
                both=False,          # 24 products — sufficient for segmentation
            )
            angles = np.atleast_1d(np.abs(angles)[..., 3])   # (K,) degrees

            in_grain    = grain_ids[nbr[:, 0], nbr[:, 1]] == current_grain_id
            unvisited   = grain_ids[nbr[:, 0], nbr[:, 1]] == 0
            below_thr   = angles <= threshold

            # Record average misorientation for this pixel
            within = below_thr & (in_grain | unvisited)
            if within.any():
                average_misorientation[x, y] = float(np.nanmean(angles[within]))

            # Flood-fill into unvisited neighbours below the threshold
            for _, (nx, ny) in enumerate(nbr[below_thr & unvisited]):
                grain_ids[nx, ny] = current_grain_id
                queue.append((nx, ny))

        current_grain_id += 1
        progress_bar.update(1)

    progress_bar.close()
    return grain_ids, average_misorientation


# ─────────────────────────────────────────────────────────────────────────────
# 3-D grain segmentation
# ─────────────────────────────────────────────────────────────────────────────

def segment_grains_3d(
    quaternions: np.ndarray,
    threshold: float,
    mask: np.ndarray = None,
):
    """
    Segment a 3-D voxel dataset into grains by flood-fill on misorientation.

    Parameters
    ----------
    quaternions : (rows, cols, depth, 4) float array
    threshold   : misorientation threshold in degrees
    mask        : (rows, cols, depth) bool array, True = valid voxel.
                  If None, all voxels are treated as valid.

    Returns
    -------
    grain_ids              : (rows, cols, depth) int array
    average_misorientation : (rows, cols, depth) float array
    """
    rows, cols, depth = quaternions.shape[:3]

    if mask is None:
        mask = np.ones((rows, cols, depth), dtype=bool)

    grain_ids              = np.zeros((rows, cols, depth), dtype=int)
    average_misorientation = np.full((rows, cols, depth), np.nan, dtype=float)
    current_grain_id       = 1

    coords = np.array(
        np.meshgrid(np.arange(rows), np.arange(cols), np.arange(depth))
    ).T.reshape(-1, 3)
    progress_bar = tqdm(total=rows * cols * depth, desc="Segmenting grains 3-D")

    for i, j, k in coords:
        if grain_ids[i, j, k] != 0 or not mask[i, j, k]:
            progress_bar.update(1)
            continue

        grain_ids[i, j, k] = current_grain_id
        queue = deque([(i, j, k)])

        while queue:
            x, y, z = queue.popleft()

            nbr = []
            if x - 1 >= 0     and mask[x-1, y,   z  ]: nbr.append((x-1, y,   z  ))
            if x + 1 < rows   and mask[x+1, y,   z  ]: nbr.append((x+1, y,   z  ))
            if y - 1 >= 0     and mask[x,   y-1, z  ]: nbr.append((x,   y-1, z  ))
            if y + 1 < cols   and mask[x,   y+1, z  ]: nbr.append((x,   y+1, z  ))
            if z - 1 >= 0     and mask[x,   y,   z-1]: nbr.append((x,   y,   z-1))
            if z + 1 < depth  and mask[x,   y,   z+1]: nbr.append((x,   y,   z+1))

            if not nbr:
                continue

            nbr      = np.array(nbr)
            nbr_quat = quaternions[nbr[:, 0], nbr[:, 1], nbr[:, 2]]

            angles = misorientation(
                quaternions[x, y, z],
                nbr_quat,
                degrees=True,
                symmetry=True,
                both=False,
            )
            angles = np.atleast_1d(np.abs(angles)[..., 3])

            in_grain  = grain_ids[nbr[:, 0], nbr[:, 1], nbr[:, 2]] == current_grain_id
            unvisited = grain_ids[nbr[:, 0], nbr[:, 1], nbr[:, 2]] == 0
            below_thr = angles <= threshold

            within = below_thr & (in_grain | unvisited)
            if within.any():
                average_misorientation[x, y, z] = float(np.nanmean(angles[within]))

            for _, (nx, ny, nz) in enumerate(nbr[below_thr & unvisited]):
                grain_ids[nx, ny, nz] = current_grain_id
                queue.append((nx, ny, nz))

        current_grain_id += 1
        progress_bar.update(1)

    progress_bar.close()
    return grain_ids, average_misorientation


# ─────────────────────────────────────────────────────────────────────────────
# Misorientation
# ─────────────────────────────────────────────────────────────────────────────

def misorientation(q1, q2, degrees=True, symmetry=True, both=False):
    """
    Compute the minimum misorientation angle (and axis) between q1 and q2.

    Parameters
    ----------
    q1 : (..., 4) array — one or more quaternions
    q2 : (..., 4) array — one or more quaternions
    degrees : if True, return angle in degrees
    symmetry : if True, minimise over cubic (m-3m) symmetry operators
    both : if True, apply symmetry to *both* operands (576 products for
           cubic); if False, apply only to q2 (24 products) — sufficient
           for grain segmentation and 24× faster.

    Returns
    -------
    axis_angle : (..., 4) array  [ax, ay, az, angle]
    """
    q2_inv = inverse_qu(q2)           # returns a copy — does NOT mutate q2

    if symmetry:
        axangle = misorientation_sym(q1, q2_inv, both=both)
    else:
        mis     = quaternion_raw_multiply(q1, q2_inv)
        if mis.ndim == 1:
            mis = mis.reshape(1, 1, *mis.shape)
        axangle = rotations.qu2ax(mis)
        axangle[axangle[..., 2] < 0] *= -1

    # axangle shape: (N1, N2, 576, 4) both; (N1, N2, 24, 4) no-both; (N1, N2, 4) no-sym
    axangle = axangle.reshape(axangle.shape[0], axangle.shape[1], -1, axangle.shape[-1])
    argmin  = np.abs(axangle[..., 3]).argmin(axis=-1)   # (N1, N2)

    min_axangles = axangle[
        np.arange(axangle.shape[0])[:, None],
        np.arange(axangle.shape[1])[None, :],
        argmin,
    ]   # (N1, N2, 4)

    if degrees:
        min_axangles = min_axangles.copy()
        min_axangles[..., 3] = np.rad2deg(min_axangles[..., 3])

    return np.squeeze(min_axangles)


def misorientation_sym(q1, q2, both=False):
    """
    Return all symmetry-equivalent misorientations.

    q1, q2: already q1 and inv(q2) respectively.
    Returns axangle array of shape (N1, N2, 24, 4) or (N1, N2, 576, 4).
    """
    if both:
        q1s = quaternion_raw_multiply(q1, LAUE_O)   # (N1, 24, 4)
    else:
        q1s = np.atleast_2d(q1)                      # (N1, 4) → use as-is
        if q1s.ndim == 2:
            q1s = q1s[:, np.newaxis, :]              # (N1, 1, 4)

    q2s  = quaternion_raw_multiply(q2, LAUE_O)       # (N2, 24, 4)
    mis  = quaternion_raw_multiply(q1s, q2s)

    axangle                      = rotations.qu2ax(mis)
    axangle[axangle[..., 2] < 0] = -axangle[axangle[..., 2] < 0]
    return axangle


# ─────────────────────────────────────────────────────────────────────────────
# Quaternion utilities
# ─────────────────────────────────────────────────────────────────────────────

def quaternion_raw_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two arrays of quaternions, with broadcasting."""
    if a.shape[-1] != 4 or b.shape[-1] != 4:
        raise ValueError("Last dimension of both inputs must be 4.")

    if a.ndim == b.ndim + 1:
        b = b[np.newaxis]
    elif a.ndim + 1 == b.ndim:
        a = a[np.newaxis]
    elif a.ndim != b.ndim:
        raise ValueError("Arrays must have the same ndim, or differ by 1.")

    if a.ndim == 1:
        a = a[np.newaxis]
        b = b[np.newaxis]

    # Set up broadcasting: insert singleton axes so each pair broadcasts
    a = a.reshape(a.shape[:1] + (1,) + a.shape[1:])
    b = b.reshape((1,) + b.shape)

    if a.ndim > 3:
        a = a.reshape(a.shape[:-1] + (1,) + a.shape[-1:])
        b = b.reshape(b.shape[:-2] + (1,) + b.shape[-2:])

    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    ow = aw*bw - ax*bx - ay*by - az*bz
    ox = aw*bx + ax*bw + ay*bz - az*by
    oy = aw*by - ax*bz + ay*bw + az*bx
    oz = aw*bz + ax*by - ay*bx + az*bw

    return standardize_qu(np.stack((ow, ox, oy, oz), axis=-1))


def standardize_qu(q: np.ndarray) -> np.ndarray:
    """Normalise quaternions and ensure scalar part ≥ 0."""
    q_out = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q_out = np.where(q_out[..., :1] < 0, -q_out, q_out)
    return q_out + 0.0   # flush -0 → +0


def inverse_qu(qu: np.ndarray) -> np.ndarray:
    """
    Return the inverse (conjugate) of unit quaternions.

    Returns a *copy* — the input array is never modified.
    (The original code modified the array in-place, which corrupted the
    quaternions grid every time misorientation() was called.)
    """
    inv = qu.copy()
    inv[..., 1:] *= -1
    return inv


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import utilities

    ang_path  = '/Users/crestiennedechaine/OriginalData/Si_Ge_Dataset/DI_largerRegion/SiGe_dp_10rows132colums_largerRegion.ang'
    patshape  = (512, 512)
    threshold = 2.0   # degrees

    ang_data = utilities.read_ang(ang_path, patshape, segment_grain_threshold=None)
    quaternions = ang_data.quats   # (rows, cols, 4)

    grain_ids, kam = segment_grains(quaternions, threshold)

    from scipy.ndimage import find_objects
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    axes[0].imshow(grain_ids, cmap="tab20b", interpolation="nearest")
    axes[0].set_title(f"Grain IDs  (threshold = {threshold}°)", fontsize=12)
    im = axes[1].imshow(kam, cmap="hot_r", interpolation="nearest")
    axes[1].set_title("KAM  (mean within-grain misorientation, °)", fontsize=12)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Draw bounding boxes around each grain on both panels
    slices = find_objects(grain_ids)   # list indexed by (grain_id - 1)
    for grain_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        row_sl, col_sl = sl
        x = col_sl.start - 0.5
        y = row_sl.start - 0.5
        w = col_sl.stop - col_sl.start
        h = row_sl.stop - row_sl.start
        for ax in axes:
            rect = mpatches.Rectangle(
                (x, y), w, h,
                linewidth=0.8, edgecolor="white", facecolor="none",
            )
            ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig("grain_segmentation.png", dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Found {grain_ids.max()} grains.")

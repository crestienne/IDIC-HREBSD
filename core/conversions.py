import itertools
import numpy as np
import rotations
from scipy.linalg import polar


def xyt2h_partial(measurements: np.ndarray):
    """Convert a translation and rotation to a homography.

    Args:
        x (float | np.ndarray): The x-translation. Can be a scalar or an array.
        y (float | np.ndarray): The y-translation. Can be a scalar or an array.
        theta (float | np.ndarray): The rotation angle in radians. Can be a scalar or an array.

    Returns:
        np.ndarray: The homography parameters. Shape is (8,) if scalars provided. Shape is (..., 8) if arrays provided."""
    x, y, theta = measurements[..., 0], measurements[..., 1], measurements[..., 2]
    if all(isinstance(i, (int, float)) for i in (x, y, theta)):
        return np.array([np.cos(theta) - 1, -np.sin(theta), x*np.cos(theta) - y*np.sin(theta),
                         np.sin(theta), np.cos(theta) - 1, x*np.sin(theta) + y*np.cos(theta),
                         0, 0])
    else:
        _0 = np.zeros(x.shape[0])
        _c = np.cos(theta)
        _s = np.sin(theta)
        return np.array([_c - 1, -_s, x*_c-y*_s, _s, _c - 1, x*_s+y*_c, _0, _0]).T


def xyt2h(shifts: np.ndarray, PC: tuple | list | np.ndarray) -> np.ndarray:
    """Convert a translation and rotation to a homography.

    Args:
        shifts (np.ndarray): The shifts. Shape is (N, 3).  First entry is x-shift,
                             second entry is y-shift, third entry is rotation angle.
        PC (tuple | list | np.ndarray): The pattern center.
        tilt (float | int): The tilt angle of the sample in degrees.

    Returns:
        np.ndarray: The homography parameters. Shape is (8,) if scalars provided. Shape is (..., 8) if arrays provided."""
    # Check the shape of the inputs
    if shifts.ndim == 1:
        shifts = shifts[None]
    ## First convert the shifts to a global rotation matrix in the detector frame
    # Decompose inputs
    x01, x02, DD = PC
    x, y, theta = shifts[..., 0], shifts[..., 1], shifts[..., 2]
    # Get xy hats (we rotate w.r.t. the PC so we don't need to change the x and y)
    # x_hat = x + x01 * (np.cos(theta) - 1) + x02 * np.sin(theta)
    # y_hat = y - x01 * np.sin(theta) + x02 * (np.cos(theta) - 1)
    x_hat = x
    y_hat = y
    # Get omegas

    w1 = np.around(np.arctan(-y_hat / DD), 3)  # w32
    w2 = np.around(np.arctan(x_hat / DD), 3)  # W13
    w3 = np.around(theta, 3)  # w21
    # Get the global rotation matrix in the sample frame
    c1, c2, c3 = np.cos(w1), np.cos(w2), np.cos(w3)
    s1, s2, s3 = np.sin(w1), np.sin(w2), np.sin(w3)
    # Rs = np.array([[c2*c3, s1*s2*c3 - c1*s3, c1*s2*c3 + s1*s3],
    #                [c2*s3, s1*s2*s3 + c1*c3, c1*s2*s3 - s1*c3],
    #                [-s2, s1*c2, c1*c2]])
    Rs = np.array([[c2*c3, -c2*s3, s2],
                   [c1*s3 + c3*s1*s2, c1*c3 - s1*s2*s3, -c2*s1],
                   [s1*s3 - c1*c3*s2, c3*s1 + c1*s2*s3, c1*c2]])
    Rs = Rs.transpose(2, 0, 1)
    # Get the global rotation matrix in the detector frame
    # Psr = rotations.eu2om(np.array([0.0, tilt * np.pi / 180, 0.0], dtype=float))
    # Rr = np.matmul(Psr.T, np.matmul(Rs, Psr))
    # Rr = np.matmul(Psr, np.matmul(Rs, Psr.T))
    Rr = Rs
    Rr = Rr / Rr[..., 2, 2][:, None, None]

    ## Now get the homography
    # Decompose inputs
    # m11, m12, m13 = Rr[..., 0, 0], Rr[..., 0, 1], Rr[..., 0, 2]
    # m21, m22, m23 = Rr[..., 1, 0], Rr[..., 1, 1], Rr[..., 1, 2]
    m31, m32      = Rr[..., 2, 0], Rr[..., 2, 1]
    # Compose shape function matrix values
    g0 = DD + m31 * x01 + m32 * x02
    # g11 = DD * m11 - m31 * x01 - g0
    # g22 = DD * m22 - m32 * x02 - g0
    # g13 = DD * ((m11 - 1) * x01 + m12 * x02 + m13 * DD) + x01 * (DD - g0)
    # g23 = DD * (m21 * x01 + (m22 - 1) * x02 + m23 * DD) + x02 * (DD - g0)
    # Compose homography
    # h11 = g11 / g0
    # h12 = (DD * m12 - m32 * x01) / g0
    # h13 = g13 / g0
    # h21 = (DD * m21 - m31 * x02) / g0
    # h22 = g22 / g0
    # h23 = g23 / g0
    h31 = m31 / g0
    h32 = m32 / g0
    homographies = xyt2h_partial(shifts)
    homographies[..., 6] = h31
    homographies[..., 7] = h32
    return np.squeeze(homographies)


def h2F(H, X0):
    """Calculate the deviatoric deformation gradient from a homography using the projection geometry (pattern center).
    Note that the deformation gradient is insensitive to hydrostatic dilation.
    Within the PC, the detector distance, DD (PC[2]), must be positive. The calculation requires the distance to be negative,
    as the homography is calculated from the detector to the sample. This function will negate the provided DD distance.

    Args:
        H (np.ndarray): The homography matrix.
        PC (np.ndarray): The pattern center.

    Returns:
        np.ndarray: The deviatoric deformation gradient."""
    # Reshape the homography if necessary
    if H.ndim == 1:
        H.reshape(1, 8)

    # Extract the data from the inputs
    if np.asarray(X0).ndim == 1:
        input_pc = np.array(X0)
        X0 = np.ones(H.shape[:-1] + (3,))
        X0[..., :3] = input_pc[:3]
    x01, x02, DD = X0[..., 0], X0[..., 1], X0[..., 2]
    h11, h12, h13, h21, h22, h23, h31, h32 = H[..., 0], H[..., 1], H[..., 2], H[..., 3], H[..., 4], H[..., 5], H[..., 6], H[..., 7]

    # Negate the detector distance becase our coordinates have +z pointing from the sample towards the detector
    # The calculation is the opposite, so we need to negate the distance
    #DD = -DD done earlier in the pipeline now

    # Calculate the deformation gradient
    beta0 = 1 - h31 * x01 - h32 * x02
    Fe11 = 1 + h11 + h31 * x01
    Fe12 = h12 + h32 * x01
    Fe13 = (h13 - (h11 * x01) - (h12 * x02) + x01*(beta0 - 1))/DD
    Fe21 = h21 + h31 * x02
    Fe22 = 1 + h22 + (h32 * x02)
    Fe23 = (h23 - (h21 * x01) - (h22 * x02) + x02*(beta0 - 1))/DD
    Fe31 = DD * h31
    Fe32 = DD * h32
    Fe33 = beta0
    Fe = np.array([[Fe11, Fe12, Fe13], [Fe21, Fe22, Fe23], [Fe31, Fe32, Fe33]]) / beta0

    # Reshape the output if necessary
    if Fe.ndim == 4:
        Fe = np.moveaxis(Fe, (0, 1, 2, 3), (2, 3, 0, 1))
    elif Fe.ndim == 3:
        #Fe = np.squeeze(np.moveaxis(Fe, (0, 1, 2), (1, 2, 0)))
        Fe = np.moveaxis(Fe, -1, 0)  # move last axis (N) to first
    # Apply a tolerance to avoid numerical issues with very small values esp negative 0  
    eps = 1e-12
    Fe = np.where(np.abs(Fe) < eps, 0.0, Fe)

    return Fe


def F2h(Fe: np.ndarray, X0: tuple | list | np.ndarray) -> np.ndarray:
    """Calculate the homography from a deformation gradient using the projection geometry (pattern center).
    'currently inaccurate when more than one Fe value is passed at a time - do not use for batch calculations'
    Args:
        Fe (np.ndarray): The deformation gradient.
        X0 (tuple | list | np.ndarray): The distance from the pattern center to the homography center (x01, x02, DD), per Ernould's method.

    Returns:
        np.ndarray: The homography matrix."""
    # Reshape the deformation gradient if necessary
    if Fe.ndim == 3:
        Fe = Fe[None, ...]
    elif Fe.ndim == 2:
        Fe = Fe[None, None, ...]

    # Extract the data from the inputs
    x01, x02, DD = X0
    F11, F12, F13, F21, F22, F23, F31, F32 = Fe[..., 0, 0], Fe[..., 0, 1], Fe[..., 0, 2], Fe[..., 1, 0], Fe[..., 1, 1], Fe[..., 1, 2], Fe[..., 2, 0], Fe[..., 2, 1]

    # Negate the detector distance becase our coordinates have +z pointing from the sample towards the detector
    # The calculation is the opposite, so we need to negate the distance
    # DD = -DD

    # Calculate the homography
    g0 = DD + F31 * x01 + F32 * x02
    g11 = DD * F11 - F31 * x01 - g0
    g22 = DD * F22 - F32 * x02 - g0
    g13 = DD * ((F11 - 1) * x01 + F12 * x02 + F13 * DD) + x01 * (DD - g0)
    g23 = DD * (F21 * x01 + (F22 - 1) * x02 + F23 * DD) + x02 * (DD - g0)
    h11 = g11 / g0
    h12 = (DD * F12 - F32 * x01) / g0
    h13 = g13 / g0
    h21 = (DD * F21 - F31 * x02) / g0
    h22 = g22 / g0
    h23 = g23 / g0
    h31 = F31 / g0
    h32 = F32 / g0
    H = np.array([h11, h12, h13, h21, h22, h23, h31, h32])

    # Reshape the output if necessary
    if H.ndim == 3:
        H = np.squeeze(np.moveaxis(H, (0, 1, 2), (1, 2, 0)))
    if H.ndim == 2:
        H = np.squeeze(H.T)

    return H

def F2strain(
    Fe: np.ndarray,
    C: np.ndarray = None,
    small_strain: bool = False,
):
    """
    Compute elastic strain and lattice rotation from deformation gradient.

    NOTE:
    This code is implemented exactly as perscribed in Ernould's optical distortions paper
    -shown to be correct and matches ATEX RESULTS - do not edit this code!! 
    """

    Fe = np.asarray(Fe, dtype=float)
    if Fe.shape[-2:] != (3, 3):
        raise ValueError("Fe must have shape (..., 3, 3)")

    # Replace any NaN/Inf 3×3 blocks with the identity so SVD doesn't
    # blow up.  We restore NaN in the outputs at the end so downstream
    # NaN-aware plotting still skips those pixels.  Without this guard,
    # a single NaN homography (e.g. grain-mask skipped pixel) raises
    # LinAlgError("SVD did not converge") and aborts the whole map.
    bad_mask = ~np.all(np.isfinite(Fe).reshape(Fe.shape[:-2] + (-1,)), axis=-1)
    if np.any(bad_mask):
        Fe = Fe.copy()
        Fe[bad_mask] = np.eye(3)

    I = np.eye(3)

    # -------------------------
    # Small strain formulation
    # -------------------------
    if small_strain:
        d = Fe - I
        epsilon = 0.5 * (d + np.swapaxes(d, -1, -2))
        omega   = 0.5 * (d - np.swapaxes(d, -1, -2))
        if np.any(bad_mask):
            epsilon[bad_mask] = np.nan
            omega[bad_mask]   = np.nan
        return epsilon, omega

    # -------------------------
    # Finite strain formulation
    # -------------------------

    # Polar decomposition via SVD
    U, S, Vt = np.linalg.svd(Fe)
    R = U @ Vt

    # -------------------------
    # Extract rotation angles (unchanged math)
    # -------------------------
    w2 = np.arctan2(
        -R[..., 2, 0],
        np.sqrt(R[..., 0, 0]**2 + R[..., 1, 0]**2)
    )

    w1 = np.arctan2(R[..., 2, 1], R[..., 2, 2])

    s1 = np.sin(w1)
    c1 = np.cos(w1)

    w3 = np.arctan2(
        s1 * R[..., 0, 2] - c1 * R[..., 0, 1],
        c1 * R[..., 1, 1] - s1 * R[..., 1, 2]
    )

    # -------------------------
    # HR-EBSD elastic strain
    # -------------------------
    # Build diagonal stretch tensor from S
    Sigma = np.zeros_like(Fe)
    idx = np.arange(3)
    Sigma[..., idx, idx] = S

    v_stretch = U @ Sigma @ np.swapaxes(U, -1, -2)

    epsilon = (v_stretch - v_stretch[..., 2, 2][..., None, None] * I) / v_stretch[..., 2, 2][..., None, None]

    # -------------------------
    # Lattice rotation tensor
    # -------------------------
    omega = np.zeros_like(Fe)
    omega[..., 0, 1] = -w3
    omega[..., 0, 2] =  w2 
    omega[..., 1, 0] =  w3
    omega[..., 1, 2] = -w1
    omega[..., 2, 0] = -w2
    omega[..., 2, 1] =  w1


    #just making sure that we don't end up with negative 0 (added Feb25, 2026)
    eps = 1e-12
    epsilon = np.where(np.abs(epsilon) < eps, 0.0, epsilon)
    omega = np.where(np.abs(omega) < eps, 0.0, omega)

    # Restore NaN for the pixels we substituted identity into so
    # downstream plotting / statistics skip them.
    if np.any(bad_mask):
        epsilon[bad_mask] = np.nan
        omega[bad_mask]   = np.nan

    return epsilon, omega


# ─────────────────────────────────────────────────────────────────────────────
# Rotation matrix ↔ Bunge Euler angle utilities
#
# Bunge convention (ZXZ, passive / crystal-frame):
#     g(phi1, Phi, phi2) = Rz(phi2) · Rx(Phi) · Rz(phi1)
#
# omega here is a skew-symmetric small-rotation tensor (the antisymmetric
# part of the displacement gradient that F2strain returns).  R = exp(omega)
# via Rodrigues, then Bunge angles are extracted.
# ─────────────────────────────────────────────────────────────────────────────


def bunge_to_rotation_matrix(phi1, Phi, phi2, degrees=False):
    """Inverse: build the Bunge rotation matrix from (phi1, Phi, phi2).

    Vectorised: phi1/Phi/phi2 can be scalars or matching-shape arrays.
    Returns an array of shape (..., 3, 3).
    """
    phi1 = np.asarray(phi1, dtype=float)
    Phi  = np.asarray(Phi,  dtype=float)
    phi2 = np.asarray(phi2, dtype=float)
    if degrees:
        phi1 = np.radians(phi1)
        Phi  = np.radians(Phi)
        phi2 = np.radians(phi2)
    c1, s1 = np.cos(phi1), np.sin(phi1)
    cP, sP = np.cos(Phi),  np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)

    R = np.empty(phi1.shape + (3, 3), dtype=float)
    R[..., 0, 0] =  c1*c2 - s1*s2*cP
    R[..., 0, 1] =  s1*c2 + c1*s2*cP
    R[..., 0, 2] =  s2*sP
    R[..., 1, 0] = -c1*s2 - s1*c2*cP
    R[..., 1, 1] = -s1*s2 + c1*c2*cP
    R[..., 1, 2] =  c2*sP
    R[..., 2, 0] =  s1*sP
    R[..., 2, 1] = -c1*sP
    R[..., 2, 2] =  cP
    return R


def rotation_matrix_to_bunge(R, degrees=False, tol=1e-7, check=True):
    """Convert a single 3x3 rotation matrix to Bunge Euler angles."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3, got shape {R.shape}")

    if check:
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-5):
            raise ValueError("R is not orthogonal (R.T @ R != I).")
        if not np.isclose(np.linalg.det(R), 1.0, atol=1e-5):
            raise ValueError("R is not a proper rotation (det(R) != +1).")

    cosPhi = np.clip(R[2, 2], -1.0, 1.0)
    Phi = np.arccos(cosPhi)
    sinPhi = np.sin(Phi)

    if abs(sinPhi) < tol:
        phi2 = 0.0
        if cosPhi > 0:
            phi1 = np.arctan2(R[0, 1], R[0, 0])
        else:
            phi1 = np.arctan2(-R[0, 1], R[0, 0])
    else:
        phi1 = np.arctan2(R[2, 0], -R[2, 1])
        phi2 = np.arctan2(R[0, 2],  R[1, 2])

    twopi = 2.0 * np.pi
    phi1 = phi1 % twopi
    phi2 = phi2 % twopi

    if degrees:
        return np.degrees(phi1), np.degrees(Phi), np.degrees(phi2)
    return phi1, Phi, phi2


def rotation_matrix_array_to_bunge(R, degrees=False, lock_tol=1e-7):
    """Vectorised Bunge extraction from a stack of rotation matrices.

    Parameters
    ----------
    R : array_like, shape (..., 3, 3)
        Proper rotation matrices in Bunge form.
    degrees : bool
        If True, return angles in degrees.
    lock_tol : float
        Threshold for sin(Phi) below which the gimbal-lock branch is taken.

    Returns
    -------
    angles : ndarray, shape (..., 3)
        Stack of (phi1, Phi, phi2).
    """
    R = np.asarray(R, dtype=float)
    if R.ndim < 2 or R.shape[-2:] != (3, 3):
        raise ValueError(f"R must have shape (..., 3, 3), got {R.shape}")

    cosPhi = np.clip(R[..., 2, 2], -1.0, 1.0)
    Phi    = np.arccos(cosPhi)
    sinPhi = np.sin(Phi)

    phi1 = np.arctan2(R[..., 2, 0], -R[..., 2, 1])
    phi2 = np.arctan2(R[..., 0, 2],  R[..., 1, 2])

    locked      = np.abs(sinPhi) < lock_tol
    phi1_at_0   = np.arctan2( R[..., 0, 1], R[..., 0, 0])
    phi1_at_pi  = np.arctan2(-R[..., 0, 1], R[..., 0, 0])
    phi1_locked = np.where(cosPhi > 0, phi1_at_0, phi1_at_pi)
    phi1 = np.where(locked, phi1_locked, phi1)
    phi2 = np.where(locked, 0.0,         phi2)

    twopi = 2.0 * np.pi
    phi1 = phi1 % twopi
    phi2 = phi2 % twopi

    angles = np.stack([phi1, Phi, phi2], axis=-1)
    if degrees:
        angles = np.degrees(angles)
    return angles


def skew_to_rotation_matrix_array(W, small_tol=1e-12):
    """Vectorised Rodrigues: exponentiate a stack of skew tensors to rotations.

    Parameters
    ----------
    W : array_like, shape (..., 3, 3)
        Skew-symmetric matrices.  Symmetrised internally (W -> (W - W^T)/2)
        so numerical drift does not corrupt the result.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim < 2 or W.shape[-2:] != (3, 3):
        raise ValueError(f"W must have shape (..., 3, 3), got {W.shape}")
    W = 0.5 * (W - np.swapaxes(W, -1, -2))

    w1 = W[..., 2, 1]
    w2 = W[..., 0, 2]
    w3 = W[..., 1, 0]
    theta2 = w1*w1 + w2*w2 + w3*w3
    theta  = np.sqrt(theta2)

    small  = theta < small_tol
    safe_t = np.where(small, 1.0, theta)
    a = np.where(small, 1.0 - theta2/6.0,  np.sin(theta) / safe_t)
    b = np.where(small, 0.5 - theta2/24.0, (1.0 - np.cos(theta)) / (safe_t * safe_t))

    WW = W @ W
    R  = np.eye(3) + a[..., None, None] * W + b[..., None, None] * WW
    return R


def omega_array_to_bunge(omegas, degrees=False, small_tol=1e-12, lock_tol=1e-7):
    """Convert an array of skew-symmetric lattice-rotation tensors to Bunge angles.

    Each omega exponentiates (Rodrigues) to a proper rotation R; Bunge Euler
    angles (phi1, Phi, phi2) are extracted from R.

    Parameters
    ----------
    omegas : array_like, shape (..., 3, 3)
        Skew-symmetric matrices.
    degrees : bool
        If True, return angles in degrees.

    Returns
    -------
    angles : ndarray, shape (..., 3)
        phi1, Phi, phi2 — phi1 & phi2 in [0, 2π); Phi in [0, π].
    """
    R = skew_to_rotation_matrix_array(omegas, small_tol=small_tol)
    return rotation_matrix_array_to_bunge(R, degrees=degrees, lock_tol=lock_tol)


def Edax_to_Bruker_PC(edax_pc):
    """
    Convert EDAX fractional PC coordinates to Bruker convention.

    EDAX convention: (x, y, z) with origin at upper left, x right, y down, z into sample
    Bruker convention: (x', y', z') with origin at upper left, x' right, y' up, z' out of sample

    Conversion:
        x' = x
        y' = 1 - y
        z' = 1 - z

    Parameters
    ----------
    edax_pc : array-like of shape (3,)
        Fractional PC coordinates in EDAX convention.

    Returns
    -------
    bruker_pc : ndarray of shape (3,)
        Fractional PC coordinates in Bruker convention.
    """
    edax_pc = np.asarray(edax_pc)
    if edax_pc.shape != (3,):
        raise ValueError("edax_pc must be a 3-element vector.")
    
    x_bruker = edax_pc[0]
    y_bruker = 1.0 - edax_pc[1]
    z_bruker = edax_pc[2]

    return np.array([x_bruker, y_bruker, z_bruker])
# -------- Calculated parameters --------

def Bruker_to_Edax_PC(bruker_pc):
    """
    Convert Bruker fractional PC coordinates to EDAX convention.

    Bruker convention: (x', y', z') with origin at upper left, x' right, y' up, z' out of sample
    EDAX convention: (x, y, z) with origin at upper left, x right, y down, z into sample

    Conversion:
        x = x'
        y = 1 - y'
        z = z'

    Parameters
    ----------
    bruker_pc : array-like of shape (3,)
        Fractional PC coordinates in Bruker convention.

    Returns
    -------
    edax_pc : ndarray of shape (3,)
        Fractional PC coordinates in EDAX convention.
    """
    bruker_pc = np.asarray(bruker_pc)
    if bruker_pc.shape != (3,):
        raise ValueError("bruker_pc must be a 3-element vector.")
    
    x_edax = bruker_pc[0]
    y_edax = 1.0 - bruker_pc[1]
    z_edax = bruker_pc[2]

    return np.array([x_edax, y_edax, z_edax])

def Bruker_to_fractional_PC(bruker_pc, patshape, pixel_size=None, homography_center = np.array([0.5, 0.5])):
    """
    Convert Bruker fractional PC to the h2F fractional PC format. Is the vector that goes from the pattern center to the homography center, in fractional coordinates relative to the pattern shape. This is the format that h2F expects for the PC input.

    Bruker convention: (x', y', z') with origin at upper left, x' right, y' up, z' out of sample
    h2F fractional PC convention: (x*, y*, z*) with origin at upper left, x* right, y* down, z* into sample, and x*, y* are fractional relative to the pattern shape

    Conversion:
        x* = x'
        y* = 1 - y'
        z* = z'

    Parameters'''
    bruker_pc : array-like of shape (3,)
        Fractional PC coordinates in Bruker convention. 
    patshape : tuple
        (height_px, width_px) of the pattern.
    homography_center : array-like of shape (2,), optional
        Fractional coordinates of the homography center (x, y) relative to the pattern shape. Default is (0.5, 0.5) for centred-pixel format.
    """
    # Vector from the pattern center to the homography center, in pixels.
    xo = np.array([(homography_center[0] - bruker_pc[0]) * patshape[0], (homography_center[1] - bruker_pc[1]) * patshape[1], (bruker_pc[2] * patshape[1])])

    return xo


    







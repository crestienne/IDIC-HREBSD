import torch
from torch import Tensor
from typing import Optional

"""

Adopted from PyTorch3D (project from Meta Research - formerly Facebook Research) and from EMsoft

https://github.com/facebookresearch/pytorch3d

https://github.com/marcdegraef/3Drotations

List of acronyms used in the code:

cu: cubochoric
ho: homochoric
ax: axis-angle
qu: quaternion
om: orientation matrix
eu: Euler angles
ro: Rodrigues vector

For neo-Eulerian representations these are the functions of rotation angle w:

ax: w
cu:
ho: [(3 / 4) * (w - sin(w))]^(1/3)
ro: tan(w / 2)

"""

# -------------------------------------------------------------------
# -------------------------- sphere functions -----------------------
# -------------------------------------------------------------------


@torch.jit.script
def theta_phi_to_xyz(theta: Tensor, phi: Tensor) -> Tensor:
    """
    Convert spherical coordinates to cartesian coordinates.
    :param theta: torch tensor of shape (n, ) containing the polar declination angles
    :param phi: torch tensor of shape (n, ) containing the azimuthal angles
    :return: torch tensor of shape (n, 3) containing the cartesian coordinates
    """
    return torch.stack(
        (
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(phi),
        ),
        dim=1,
    )


@torch.jit.script
def xyz_to_theta_phi(xyz: Tensor) -> Tensor:
    """
    Convert cartesian coordinates to latitude and longitude.
    :param xyz: torch tensor of shape (n, 3) containing the cartesian coordinates
    :return: torch tensor of shape (n, 2) containing the polar declination and azimuthal angles
    """
    return torch.stack(
        (
            torch.atan2(torch.norm(xyz[:, :2], dim=1), xyz[:, 2]),
            torch.atan2(xyz[:, 1], xyz[:, 0]),
        ),
        dim=1,
    )


# -------------------------------------------------------------------
# ------------------------ quaternion functions ---------------------
# -------------------------------------------------------------------


@torch.jit.script
def standardize_quaternion(quaternions: Tensor) -> Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non-negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


@torch.jit.script
def quaternion_raw_multiply(a: Tensor, b: Tensor) -> Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


@torch.jit.script
def quaternion_multiply(a: Tensor, b: Tensor) -> Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


@torch.jit.script
def quaternion_real_of_prod(a: Tensor, b: Tensor) -> Tensor:
    """
    Multiply two quaternions and return the positive real part of the product.


    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The positive real part of the product of a and b, a tensor of shape (..., 1).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    return ow.abs()


@torch.jit.script
def quaternion_invert(quaternion: Tensor) -> Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling


@torch.jit.script
def quaternion_apply(quaternion: Tensor, point: Tensor) -> Tensor:
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


@torch.jit.script
def normalize_quaternion(quaternion: Tensor) -> Tensor:
    """
    Normalize a quaternion to a unit quaternion.

    Args:
        quaternion: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Normalized quaternions as tensor of shape (..., 4).
    """
    return quaternion / torch.norm(quaternion, dim=-1, keepdim=True)


@torch.jit.script
def norm_standard_quaternion(quaternion: Tensor) -> Tensor:
    """
    Normalize a quaternion to a unit quaternion and standardize it.

    Args:
        quaternion: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Normalized and standardized quaternions as tensor of shape (..., 4).
    """
    return standardize_quaternion(normalize_quaternion(quaternion))


@torch.jit.script
def quaternion_rotate_sets_sphere(points_start: Tensor, points_finish) -> Tensor:
    """
    Determine the quaternions that rotate the points_start to the points_finish.
    All points are assumed to be on the unit sphere. The cross product is used
    as the axis of rotation, but there are an infinite number of quaternions that
    fulfill the requirement as the points can be rotated around their axis by
    an arbitrary angle, and they will still have the same latitude and longitude.

    Args:
        points_start: Starting points as tensor of shape (..., 3).
        points_finish: Ending points as tensor of shape (..., 3).

    Returns:
        The quaternions, as tensor of shape (..., 4).

    """
    # determine mask for numerical stability
    valid = torch.abs(torch.sum(points_start * points_finish, dim=-1)) < 0.999999
    # get the cross product of the two sets of points
    cross = torch.cross(points_start[valid], points_finish[valid], dim=-1)
    # get the dot product of the two sets of points
    dot = torch.sum(points_start[valid] * points_finish[valid], dim=-1)
    # get the angle
    angle = torch.atan2(torch.norm(cross, dim=-1), dot)
    # add tau to the angle if the cross product is negative
    angle[angle < 0] += 2 * torch.pi
    # set the output
    out = torch.empty(
        (points_start.shape[0], 4), dtype=points_start.dtype, device=points_start.device
    )
    out[valid, 0] = torch.cos(angle / 2)
    out[valid, 1:] = torch.sin(angle / 2)[:, None] * (
        cross / torch.norm(cross, dim=-1, keepdim=True)
    )
    out[~valid, 0] = 1
    out[~valid, 1:] = 0
    return out


@torch.jit.script
def misorientation_angle(quaternion: Tensor) -> Tensor:
    """
    Compute the misorientation angle for a quaternion.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The misorientation angle, a tensor of shape (...).
    """
    return 2 * torch.acos(quaternion[..., 0])


# -------------------------------------------------------------------
# ------------------------- octonion functions ----------------------
# -------------------------------------------------------------------


@torch.jit.script
def octionion_standardize(octonions: Tensor) -> Tensor:
    """
    Convert a unit octonion to a standard form: one in which the real
    part is non-negative.

    Args:
        octonions: Octonions with real part first,
            as tensor of shape (..., 8).

    Returns:
        Standardized octonions as tensor of shape (..., 8).
    """
    return torch.where(octonions[..., 0:1] < 0, -octonions, octonions)


@torch.jit.script
def octonion_raw_multiply(a: Tensor, b: Tensor) -> Tensor:
    """
    Multiply two octonions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Octonions as tensor of shape (..., 8), real part first.
        b: Octonions as tensor of shape (..., 8), real part first.

    Returns:
        The product of a and b, a tensor of octonions shape (..., 8).
    """
    # Unbind octonions
    aw, ax, ay, az, al, am, an, ao = torch.unbind(a, -1)
    bw, bx, by, bz, bl, bm, bn, bo = torch.unbind(b, -1)

    # Compute multiplication
    ow = aw * bw - ax * bx - ay * by - az * bz - al * bl - am * bm - an * bn - ao * bo
    ox = aw * bx + ax * bw + ay * bz - az * by + al * bo - am * bn + an * bm - ao * bl
    oy = aw * by - ax * bz + ay * bw + az * bx + al * bn + am * bo + an * bl + ao * bm
    oz = aw * bz + ax * by - ay * bx + az * bw + al * bm - am * bl - an * bo + ao * bn
    ol = aw * bl - ax * bo - ay * bn + az * bm - al * bw + am * bx - an * by + ao * az
    om = aw * bm + ax * bn - ay * bl - az * bo - al * bx - am * bw + an * bx + ao * ay
    on = aw * bn - ax * bm + ay * bo + az * bl + al * by - am * az - an * bw - ao * ax
    oo = aw * bo + ax * bl + ay * bm - az * bn - al * az + am * ay + an * ax + ao * bw

    return torch.stack((ow, ox, oy, oz, ol, om, on, oo), -1)


@torch.jit.script
def octonion_multiply(a: Tensor, b: Tensor) -> Tensor:
    """
    Multiply two octonions representing rotations, returning the octonion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Octonions as tensor of shape (..., 8), real part first.
        b: Octonions as tensor of shape (..., 8), real part first.

    Returns:
        The product of a and b, a tensor of octonions of shape (..., 8).
    """
    ab = octonion_raw_multiply(a, b)
    return octionion_standardize(ab)


@torch.jit.script
def octonion_invert(octonion: Tensor) -> Tensor:
    """
    Given an octonion representing rotation, get the octonion representing
    its inverse.

    Args:
        octonion: Octonions as tensor of shape (..., 8), with real part
            first, which must be versors (unit octonions).

    Returns:
        The inverse, a tensor of octonions of shape (..., 8).
    """

    scaling = torch.tensor([1, -1, -1, -1, -1, -1, -1, -1], device=octonion.device)
    return octonion * scaling


@torch.jit.script
def octonion_misorientation_angle(octonion: Tensor) -> Tensor:
    """
    Compute the misorientation angle for an octonion.

    Args:
        octonion: Octonions as tensor of shape (..., 8), with real part
            first, which must be versors (unit octonions).

    Returns:
        The misorientation angle, a tensor of shape (...).
    """
    return 2 * torch.acos(torch.clamp(octonion[..., 0], min=-1, max=1))


# -------------------------------------------------------------------
# ------------------------ conversion functions ---------------------
# -------------------------------------------------------------------


def _sqrt_positive_part(x: Tensor) -> Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


@torch.jit.script
def om2qu(matrix: Tensor) -> Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    best_indices = torch.argmax(q_abs, dim=-1)

    return quat_candidates[torch.arange(quat_candidates.shape[0]), best_indices]


@torch.jit.script
def cu2ho(cu: Tensor) -> Tensor:
    """
    Convert cubochoric coordinates to homochoric coordinates.

    Args:
        cu: Cubochoric coordinates as tensor of shape (B, 3)

    Returns:
        Homochoric coordinates as tensor of shape (B, 3)
    """
    ho = cu.clone()
    cu_abs = torch.abs(ho)
    x_abs, y_abs, z_abs = torch.unbind(cu_abs, dim=1)

    # Determine pyramid
    pyramid = torch.zeros(ho.shape[0], dtype=torch.uint8)
    pyramid[(x_abs <= z_abs) & (y_abs <= z_abs)] = 1
    pyramid[(x_abs <= -z_abs) & (y_abs <= -z_abs)] = 2
    pyramid[(z_abs <= x_abs) & (y_abs <= x_abs)] = 3
    pyramid[(z_abs <= -x_abs) & (y_abs <= -x_abs)] = 4
    pyramid[(x_abs <= y_abs) & (z_abs <= y_abs)] = 5
    pyramid[(x_abs <= -y_abs) & (z_abs <= -y_abs)] = 6

    # move everything to correct pyramid
    mask_34 = (pyramid == 3) | (pyramid == 4)
    mask_56 = (pyramid == 5) | (pyramid == 6)
    ho[mask_34] = torch.roll(ho[mask_34], shifts=-1, dims=1)
    ho[mask_56] = torch.roll(ho[mask_56], shifts=1, dims=1)

    # Scale
    ho = ho * torch.pi ** (1.0 / 6.0) / 6.0 ** (1.0 / 6.0)

    # Process based on conditions
    x, y, z = torch.unbind(ho, dim=1)
    prefactor = (
        (3 * torch.pi / 4) ** (1 / 3)
        * 2 ** (1 / 4)
        / (torch.pi ** (5 / 6) / 6 ** (1 / 6) / 2)
    )
    sqrt2 = 2**0.5

    # abs(y) <= abs(x) condition
    mask_y_leq_x = torch.abs(y) <= torch.abs(x)
    q_y_leq_x = (torch.pi / 12.0) * y[mask_y_leq_x] / x[mask_y_leq_x]
    cosq_y_leq_x = torch.cos(q_y_leq_x)
    sinq_y_leq_x = torch.sin(q_y_leq_x)
    q_val_y_leq_x = prefactor * x[mask_y_leq_x] / torch.sqrt(sqrt2 - cosq_y_leq_x)
    t1_y_leq_x = (sqrt2 * cosq_y_leq_x - 1) * q_val_y_leq_x
    t2_y_leq_x = sqrt2 * sinq_y_leq_x * q_val_y_leq_x
    c_y_leq_x = t1_y_leq_x**2 + t2_y_leq_x**2
    s_y_leq_x = torch.pi * c_y_leq_x / (24 * z[mask_y_leq_x] ** 2)
    c_y_leq_x = torch.pi**0.5 * c_y_leq_x / 24**0.5 / z[mask_y_leq_x]
    q_y_leq_x = torch.sqrt(1 - s_y_leq_x)

    ho[mask_y_leq_x, 0] = t1_y_leq_x * q_y_leq_x
    ho[mask_y_leq_x, 1] = t2_y_leq_x * q_y_leq_x
    ho[mask_y_leq_x, 2] = (6 / torch.pi) ** 0.5 * z[mask_y_leq_x] - c_y_leq_x

    # abs(y) > abs(x) condition
    mask_y_gt_x = ~mask_y_leq_x
    q_y_gt_x = (torch.pi / 12.0) * x[mask_y_gt_x] / y[mask_y_gt_x]
    cosq_y_gt_x = torch.cos(q_y_gt_x)
    sinq_y_gt_x = torch.sin(q_y_gt_x)
    q_val_y_gt_x = prefactor * y[mask_y_gt_x] / torch.sqrt(sqrt2 - cosq_y_gt_x)
    t1_y_gt_x = sqrt2 * sinq_y_gt_x * q_val_y_gt_x
    t2_y_gt_x = (sqrt2 * cosq_y_gt_x - 1) * q_val_y_gt_x
    c_y_gt_x = t1_y_gt_x**2 + t2_y_gt_x**2
    s_y_gt_x = torch.pi * c_y_gt_x / (24 * z[mask_y_gt_x] ** 2)
    c_y_gt_x = torch.pi**0.5 * c_y_gt_x / 24**0.5 / z[mask_y_gt_x]
    q_y_gt_x = torch.sqrt(1 - s_y_gt_x)

    ho[mask_y_gt_x, 0] = t1_y_gt_x * q_y_gt_x
    ho[mask_y_gt_x, 1] = t2_y_gt_x * q_y_gt_x
    ho[mask_y_gt_x, 2] = (6 / torch.pi) ** 0.5 * z[mask_y_gt_x] - c_y_gt_x

    # Roll the array based on the pyramid values
    ho[mask_34] = torch.roll(ho[mask_34], shifts=1, dims=1)
    ho[mask_56] = torch.roll(ho[mask_56], shifts=-1, dims=1)

    # wherever cu had all zeros, ho should be set to be (0, 0, 0)
    mask_zero = torch.abs(cu).sum(dim=1) == 0
    ho[mask_zero] = 0

    # wherever cu had (0, 0, z) ho should be set to be (0, 0, np.sqrt(6 / np.pi) * z)
    mask_z = torch.abs(cu[:, :2]).sum(dim=1) == 0
    ho[mask_z, :2] = 0
    ho[mask_z, 2] = (6.0 / torch.pi) ** 0.5 * cu[mask_z, 2]

    return ho


@torch.jit.script
def ho2cu(ho: Tensor) -> Tensor:
    """
    Homochoric vector to cubochoric vector.
    """
    cu = ho.clone()

    # Wherever ho had all zeros, cu should be set to be (0, 0, 0)
    mask_zero = torch.abs(cu).sum(dim=1) == 0

    cu_abs = torch.abs(cu)
    x_abs, y_abs, z_abs = torch.unbind(cu_abs, dim=-1)

    # Determine pyramid
    pyramid = torch.zeros(cu.shape[0], dtype=torch.uint8)
    pyramid[(x_abs <= z_abs) & (y_abs <= z_abs)] = 1
    pyramid[(x_abs <= -z_abs) & (y_abs <= -z_abs)] = 2
    pyramid[(z_abs <= x_abs) & (y_abs <= x_abs)] = 3
    pyramid[(z_abs <= -x_abs) & (y_abs <= -x_abs)] = 4
    pyramid[(x_abs <= y_abs) & (z_abs <= y_abs)] = 5
    pyramid[(x_abs <= -y_abs) & (z_abs <= -y_abs)] = 6

    # Move everything to correct pyramid
    mask_34 = (pyramid == 3) | (pyramid == 4)
    mask_56 = (pyramid == 5) | (pyramid == 6)
    cu[mask_34] = torch.roll(cu[mask_34], shifts=-1, dims=1)
    cu[mask_56] = torch.roll(cu[mask_56], shifts=1, dims=1)

    # Process based on conditions

    # Roll back to the original pyramid order
    cu[mask_34] = torch.roll(cu[mask_34], shifts=1, dims=1)
    cu[mask_56] = torch.roll(cu[mask_56], shifts=-1, dims=1)

    cu[mask_zero] = 0

    return cu


@torch.jit.script
def ho2ax(ho: Tensor) -> Tensor:
    """
    Converts a set of homochoric vectors to axis-angle representation.

    Args:
        ho (Tensor): A tensor of shape (N, 3) containing N homochoric vectors.

    Returns:
        torch.Tensor: A tensor of shape (N, 4) containing the corresponding axis-angle representations.
            The first three columns contain the axis of rotation, and the fourth column contains the angle
            of rotation in radians.
    """
    # Constants stolen directly from EMsoft
    fit_parameters = torch.tensor(
        [
            0.9999999999999968,
            -0.49999999999986866,
            -0.025000000000632055,
            -0.003928571496460683,
            -0.0008164666077062752,
            -0.00019411896443261646,
            -0.00004985822229871769,
            -0.000014164962366386031,
            -1.9000248160936107e-6,
            -5.72184549898506e-6,
            7.772149920658778e-6,
            -0.00001053483452909705,
            9.528014229335313e-6,
            -5.660288876265125e-6,
            1.2844901692764126e-6,
            1.1255185726258763e-6,
            -1.3834391419956455e-6,
            7.513691751164847e-7,
            -2.401996891720091e-7,
            4.386887017466388e-8,
            -3.5917775353564864e-9,
        ],
        dtype=ho.dtype,
        device=ho.device,
    )

    ho_magnitude = torch.sum(ho**2, dim=1)

    mask_zero = torch.abs(ho_magnitude) < 1e-8
    ax = torch.zeros((ho.shape[0], 4), dtype=ho.dtype, device=ho.device)
    ax[mask_zero, 2] = 1

    ho_magnitude = torch.sum(ho**2, dim=1)
    ho_magnitude[mask_zero] = 1.0  # Avoid division by zero

    hom = ho_magnitude
    s = fit_parameters[0] + fit_parameters[1] * hom
    for i in range(2, 21):
        hom = hom * ho_magnitude
        s = s + fit_parameters[i] * hom

    hon = ho / torch.sqrt(ho_magnitude).unsqueeze(1)
    s = 2 * torch.acos(s)

    # the axis is inherited no matter what
    ax[~mask_zero, :3] = hon[~mask_zero]
    # if we are at pi condition
    mask_pi = torch.abs(s - torch.pi) < 1e-8
    ax[~mask_zero & mask_pi, 3] = torch.pi
    # rest are normal
    ax[~mask_zero & ~mask_pi, 3] = s[~mask_zero & ~mask_pi]
    return ax


@torch.jit.script
def ax2ho(ax: Tensor) -> Tensor:
    """
    Converts axis-angle representation to homochoric representation.

    Args:
        ax (Tensor): Tensor of shape (N, 4) where N is the number of axis-angle representations.
            Each row represents an axis-angle representation in the format (x, y, z, angle).

    Returns:
        torch.Tensor: Tensor of shape (N, 3) where N is the number of homochoric vectors.
    """
    f = (0.75 * (ax[..., 3:4] - torch.sin(ax[..., 3:4]))) ** (1.0 / 3.0)
    ho = ax[..., :3] * f
    return ho


@torch.jit.script
def ax2ro(ax: Tensor) -> Tensor:
    """
    Converts axis-angle representation to rotation matrix representation.

    Args:
        ax (Tensor): Tensor of shape (N, 4) representing N axis-angle vectors.

    Returns:
        torch.Tensor: Tensor of shape (N, 4) representing N Rodrigues vectors.
    """
    ro = torch.zeros(ax.shape[0], 4, dtype=ax.dtype, device=ax.device)
    angle = ax[:, 3]
    mask_zero_angle = torch.abs(angle) < 1e-8
    ro[mask_zero_angle, 2] = 1.0

    mask_nonzero_angle = ~mask_zero_angle
    ro[mask_nonzero_angle, :3] = ax[mask_nonzero_angle, :3]

    mask_pi = torch.abs(angle - torch.pi) < 1e-7
    ro[mask_pi, 3] = float("inf")

    mask_else = mask_nonzero_angle & (~mask_pi)
    ro[mask_else, 3] = torch.tan(angle[mask_else] * 0.5)
    return ro


@torch.jit.script
def ro2ax(ro: Tensor) -> Tensor:
    """
    Converts a rotation vector to an axis-angle representation.

    Args:
        ro (Tensor): A tensor of shape (N, 4) Rodrigues vectors.

    Returns:
        torch.Tensor: A tensor of shape (N, 4) axis-angle representations.
    """
    ax = torch.zeros(ro.shape[0], 4, dtype=torch.float64)
    mask_zero_ro = torch.abs(ro[:, 3]) < 1e-8
    ax[mask_zero_ro] = torch.tensor([0, 0, 1, 0], dtype=torch.float64)

    mask_inf_ro = torch.isinf(ro[:, 3])
    ax[mask_inf_ro, :3] = ro[mask_inf_ro, :3]
    ax[mask_inf_ro, 3] = torch.pi

    mask_else = ~(mask_zero_ro | mask_inf_ro)
    norm = torch.sqrt(torch.sum(ro[mask_else, :3] ** 2, dim=1))
    ax[mask_else, :3] = ro[mask_else, :3] / norm.unsqueeze(1)
    ax[mask_else, 3] = 2 * torch.atan(ro[mask_else, 3])
    return ax


@torch.jit.script
def ax2qu(ax: Tensor) -> Tensor:
    """
    Converts axis-angle representation to quaternion representation.

    Args:
        ax (Tensor): Tensor of shape (N, 4) where N is the number of axis-angle representations.
            Each row represents an axis-angle representation in the format (x, y, z, angle).

    Returns:
        torch.Tensor: Tensor of shape (N, 4) where N is the number of quaternions.
            Each row represents a quaternion in the format (w, x, y, z).
    """
    qu = torch.zeros(ax.shape[0], 4, dtype=ax.dtype, device=ax.device)
    angle = ax[:, 3]
    mask_zero_angle = (angle > -1e-8) & (angle < 1e-8)
    qu[mask_zero_angle, 0] = 1.0

    mask_nonzero_angle = ~mask_zero_angle
    c = torch.cos(angle[mask_nonzero_angle] * 0.5)
    s = torch.sin(angle[mask_nonzero_angle] * 0.5)
    qu[mask_nonzero_angle, 0] = c
    qu[mask_nonzero_angle, 1:] = ax[mask_nonzero_angle, :3] * s.unsqueeze(1)

    # normalize the quaternions
    qu_norm = qu / torch.norm(qu, dim=1, keepdim=True)
    return qu_norm


@torch.jit.script
def qu2ax(qu: Tensor) -> Tensor:
    """
    Converts quaternion representation to axis-angle representation.

    Args:
        qu (Tensor): Tensor of shape (N, 4) where N is the number of quaternions.
            Each row represents a quaternion in the format (w, x, y, z).

    Returns:
        torch.Tensor: Tensor of shape (N, 4) where N is the number of axis-angle representations.
            Each row represents an axis-angle representation in the format (x, y, z, angle).
    """
    ax = torch.zeros(qu.shape[0], 4, dtype=qu.dtype, device=qu.device)

    w, x, y, z = torch.unbind(qu, dim=1)

    # Scale similar to the numpy version
    s = torch.sign(w) / torch.sqrt(x**2 + y**2 + z**2)

    # Omega computation, which is the angle
    omega = 2.0 * torch.acos(torch.clamp(w, min=-1.0, max=1.0))

    # Assign values based on the scalar part condition
    mask = w.abs() < 1.0e-8
    ax[mask, :3] = torch.stack([x, y, z], dim=1)[mask]
    ax[~mask, :3] = torch.stack([x, y, z], dim=1)[~mask] * s[~mask].unsqueeze(1)
    ax[~mask, 3] = omega[~mask]

    # Handle the special case where scalar part is close to 1
    mask_special_case = (w - 1.0).abs() < 1.0e-12
    ax[mask_special_case] = torch.tensor(
        [0.0, 0.0, 1.0, 0.0], dtype=qu.dtype, device=qu.device
    )
    return ax


@torch.jit.script
def qu2ro(qu: Tensor) -> Tensor:
    """
    Converts quaternion representation to Rodrigues-Frank vector representation.

    Args:
        qu (Tensor): Tensor of shape (N, 4) where N is the number of quaternions.
            Each row represents a quaternion in the format (w, x, y, z).

    Returns:
        torch.Tensor: Tensor of shape (N, 4) where N is the number of Rodrigues-Frank vectors.
            Each row represents a Rodrigues-Frank vector representation.
    """
    s = torch.norm(qu[:, 1:], dim=1, keepdim=True)
    ro = torch.zeros_like(qu)

    # Handle general case
    w_clipped = torch.clamp(qu[:, 0:1], min=-1.0, max=1.0)
    tan_part = torch.tan(torch.acos(w_clipped))
    ro[:, :3] = qu[:, 1:4] / s
    ro[:, 3] = tan_part[:, 0]
    return ro


@torch.jit.script
def qu2om(quaternions: Tensor) -> Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


@torch.jit.script
def _copysign(a: Tensor, b: Tensor) -> Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


@torch.jit.script
def _axis_angle_rotation(axis: str, angle: Tensor) -> Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


@torch.jit.script
def eu2om(euler_angles: Tensor, convention: str) -> Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


@torch.jit.script
def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


@torch.jit.script
def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


@torch.jit.script
def om2eu(matrix: Tensor, convention: str) -> Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


@torch.jit.script
def zh2om(zh: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = zh[..., :3], zh[..., 3:]
    b1 = a1 / torch.norm(a1, p=2, dim=-1, keepdim=True)
    b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
    b2 = b2 / torch.norm(b2, p=2, dim=-1, keepdim=True)
    b3 = torch.cross(b1, b2, dim=-1)
    b3 = b3 / torch.norm(b3, p=2, dim=-1, keepdim=True)
    return torch.stack((b1, b2, b3), dim=-2)


@torch.jit.script
def om2zh(matrix: Tensor) -> Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


@torch.jit.script
def zh2qu(zh: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to quaternion
    representation using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of quaternions of size (*, 4)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    return om2qu(zh2om(zh))


@torch.jit.script
def qu2zh(quaternions: Tensor) -> Tensor:
    """
    Converts quaternion representation to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        quaternions: batch of quaternions of size (*, 4)

    Returns:
        6D rotation representation, of size (*, 6)
    """

    return om2zh(qu2om(quaternions))


@torch.jit.script
def bu2qu_emsoft(bu: Tensor) -> Tensor:
    """
    Bunge (ZXZ) Euler angles → quaternion, EMsoft P=+1 canonical form.

    Matches the project's rotations.eu2qu (and ebsdtorch's bu2qu) so the
    resulting quaternion is consistent with the rest of the DIC-HREBSD
    pipeline (IPF maps, .ang reader, Results_plotting, etc.).

    NOTE: this is NOT the same as eu2qu(bu, "ZXZ") in this file — that
    chains through om2qu(eu2om(...)) and produces the conjugate (inverse
    rotation).  Use this function for any Bunge angles that came from
    or will be compared to the .ang file's orientations.

    Args:
        bu (Tensor): shape (..., 3) Bunge ZXZ Euler angles in radians.

    Returns:
        Tensor of quaternions (w, x, y, z), shape (..., 4), with w ≥ 0.
    """
    sigma = 0.5 * (bu[..., 0] + bu[..., 2])
    delta = 0.5 * (bu[..., 0] - bu[..., 2])
    c = torch.cos(0.5 * bu[..., 1])
    s = torch.sin(0.5 * bu[..., 1])

    qu = torch.empty(bu.shape[:-1] + (4,), dtype=bu.dtype, device=bu.device)
    qu[..., 0] =  c * torch.cos(sigma)
    qu[..., 1] = -s * torch.cos(delta)
    qu[..., 2] = -s * torch.sin(delta)
    qu[..., 3] = -c * torch.sin(sigma)

    # Force the scalar part non-negative for canonical form.
    sign = torch.where(qu[..., 0:1] < 0, -1.0, 1.0)
    return qu * sign


@torch.jit.script
def qu2cu(quaternions: Tensor) -> Tensor:
    """
    Convert rotations given as quaternions to cubochoric vectors.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Cubochoric vectors as tensor of shape (..., 3).
    """
    return ho2cu(ax2ho(qu2ax(quaternions)))


@torch.jit.script
def om2ax(matrix: Tensor) -> Tensor:
    """
    Converts rotation matrices to axis-angle representation.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Axis-angle representation as tensor of shape (..., 4).
    """
    return qu2ax(om2qu(matrix))


@torch.jit.script
def om2ho(matrix: Tensor) -> Tensor:
    """
    Converts rotation matrices to homochoric vector representation.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Homochoric vector representation as tensor of shape (..., 3).
    """
    return ax2ho(om2ax(matrix))


@torch.jit.script
def qu2ho(quaternions: Tensor) -> Tensor:
    """
    Converts quaternions to homochoric vector representation.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Homochoric vector representation as tensor of shape (..., 3).
    """
    return ax2ho(qu2ax(quaternions))


@torch.jit.script
def ax2om(ax: Tensor) -> Tensor:
    """
    Converts axis-angle representation to rotation matrix representation.

    Args:
        ax: Axis-angle representation as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return qu2om(ax2qu(ax))


@torch.jit.script
def ax2cu(ax: Tensor) -> Tensor:
    """
    Converts axis-angle representation to cubochoric vector representation.

    Args:
        ax: Axis-angle representation as tensor of shape (..., 4).

    Returns:
        Cubochoric vectors as tensor of shape (..., 3).
    """
    return qu2cu(ax2qu(ax))


@torch.jit.script
def ax2eu(ax: Tensor, convention: str) -> Tensor:
    """
    Converts axis-angle representation to Euler angles in radians.

    Args:
        ax: Axis-angle representation as tensor of shape (..., 4).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    return om2eu(ax2om(ax), convention)


@torch.jit.script
def ro2qu(ro: Tensor) -> Tensor:
    """
    Converts Rodrigues-Frank vector representation to quaternions.

    Args:
        ro: Rodrigues-Frank vector representation as tensor of shape (..., 4).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    return ax2qu(ro2ax(ro))


@torch.jit.script
def ro2om(ro: Tensor) -> Tensor:
    """
    Converts Rodrigues-Frank vector representation to rotation matrices.

    Args:
        ro: Rodrigues-Frank vector representation as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return ax2om(ro2ax(ro))


@torch.jit.script
def ro2cu(ro: Tensor) -> Tensor:
    """
    Converts Rodrigues-Frank vector representation to cubochoric vectors.

    Args:
        ro: Rodrigues-Frank vector representation as tensor of shape (..., 4).

    Returns:
        Cubochoric vectors as tensor of shape (..., 3).
    """
    return ax2cu(ro2ax(ro))


@torch.jit.script
def ro2ho(ro: Tensor) -> Tensor:
    """
    Converts Rodrigues-Frank vector representation to homochoric vectors.

    Args:
        ro: Rodrigues-Frank vector representation as tensor of shape (..., 4).

    Returns:
        Homochoric vectors as tensor of shape (..., 3).
    """
    return ax2ho(ro2ax(ro))


@torch.jit.script
def ro2eu(ro: Tensor, convention: str) -> Tensor:
    """
    Converts Rodrigues-Frank vector representation to Euler angles in radians.

    Args:
        ro: Rodrigues-Frank vector representation as tensor of shape (..., 4).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    return om2eu(ro2om(ro), convention)


@torch.jit.script
def cu2ax(cubochoric_vectors: Tensor) -> Tensor:
    """
    Converts cubochoric vector representation to axis-angle representation.

    Args:
        cubochoric_vectors: Cubochoric vectors as tensor of shape (..., 3).

    Returns:
        Axis-angle representation as tensor of shape (..., 4).
    """
    return ho2ax(cu2ho(cubochoric_vectors))


@torch.jit.script
def cu2qu(cubochoric_vectors: Tensor) -> Tensor:
    """
    Converts cubochoric vector representation to quaternions.

    Args:
        cubochoric_vectors: Cubochoric vectors as tensor of shape (..., 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    return ax2qu(ho2ax(cu2ho(cubochoric_vectors)))


@torch.jit.script
def cu2om(cubochoric_vectors: Tensor) -> Tensor:
    """
    Converts cubochoric vector representation to rotation matrices.

    Args:
        cubochoric_vectors: Cubochoric vectors as tensor of shape (..., 3).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return ax2om(ho2ax(cu2ho(cubochoric_vectors)))


@torch.jit.script
def cu2ro(cubochoric_vectors: Tensor) -> Tensor:
    """
    Converts cubochoric vector representation to Rodrigues-Frank vector representation.

    Args:
        cubochoric_vectors: Cubochoric vectors as tensor of shape (..., 3).

    Returns:
        Rodrigues-Frank vector representation as tensor of shape (..., 4).
    """
    return ax2ro(ho2ax(cu2ho(cubochoric_vectors)))


@torch.jit.script
def cu2eu(cubochoric_vectors: Tensor, convention: str) -> Tensor:
    """
    Converts cubochoric vector representation to Euler angles in radians.

    Args:
        cubochoric_vectors: Cubochoric vectors as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    return om2eu(cu2om(cubochoric_vectors), convention)


@torch.jit.script
def ho2qu(homochoric_vectors: Tensor) -> Tensor:
    """
    Converts homochoric vector representation to quaternions.

    Args:
        homochoric_vectors: Homochoric vectors as tensor of shape (..., 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    return ax2qu(ho2ax(homochoric_vectors))


@torch.jit.script
def ho2om(homochoric_vectors: Tensor) -> Tensor:
    """
    Converts homochoric vector representation to rotation matrices.

    Args:
        homochoric_vectors: Homochoric vectors as tensor of shape (..., 3).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return ax2om(ho2ax(homochoric_vectors))


@torch.jit.script
def ho2ro(homochoric_vectors: Tensor) -> Tensor:
    """
    Converts homochoric vector representation to Rodrigues-Frank vector representation.

    Args:
        homochoric_vectors: Homochoric vectors as tensor of shape (..., 3).

    Returns:
        Rodrigues-Frank vector representation as tensor of shape (..., 4).
    """
    return ax2ro(ho2ax(homochoric_vectors))


@torch.jit.script
def ho2eu(homochoric_vectors: Tensor, convention: str) -> Tensor:
    """
    Converts homochoric vector representation to Euler angles in radians.

    Args:
        homochoric_vectors: Homochoric vectors as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    return om2eu(ho2om(homochoric_vectors), convention)


@torch.jit.script
def eu2ax(euler_angles: Tensor, convention: str) -> Tensor:
    """
    Convert rotations given as Euler angles in radians to axis-angle representation.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Axis-angle representation as tensor of shape (..., 4).
    """
    return om2ax(eu2om(euler_angles, convention))


@torch.jit.script
def eu2qu(euler_angles: Tensor, convention: str) -> Tensor:
    """
    Convert rotations given as Euler angles in radians to quaternions.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    return om2qu(eu2om(euler_angles, convention))


@torch.jit.script
def eu2ro(euler_angles: Tensor, convention: str) -> Tensor:
    """
    Convert rotations given as Euler angles in radians to Rodrigues-Frank vector representation.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rodrigues-Frank vector representation as tensor of shape (..., 4).
    """
    return ax2ro(eu2ax(euler_angles, convention))


@torch.jit.script
def eu2cu(euler_angles: Tensor, convention: str) -> Tensor:
    """
    Convert rotations given as Euler angles in radians to cubochoric vectors.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Cubochoric vectors as tensor of shape (..., 3).
    """
    return qu2cu(eu2qu(euler_angles, convention))


@torch.jit.script
def qu2eu(quaternions: Tensor, convention: str) -> Tensor:
    """
    Convert rotations given as quaternions to Euler angles in radians.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    return om2eu(qu2om(quaternions), convention)


@torch.jit.script
def eu2ho(euler_angles: Tensor, convention: str) -> Tensor:
    """
    Convert rotations given as Euler angles in radians to homochoric vectors.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.
    Returns:
        Homochoric vectors as tensor of shape (..., 3).
    """
    return qu2ho(eu2qu(euler_angles, convention))


@torch.jit.script
def om2ro(matrix: Tensor) -> Tensor:
    """
    Converts rotation matrices to Rodrigues-Frank vector representation.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rodrigues-Frank vector representation as tensor of shape (..., 4).
    """
    return qu2ro(om2qu(matrix))


@torch.jit.script
def om2cu(matrix: Tensor) -> Tensor:
    """
    Converts rotation matrices to cubochoric vector representation.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Cubochoric vector representation as tensor of shape (..., 3).
    """
    return qu2cu(om2qu(matrix))


@torch.jit.script
def square_lambert(pts: Tensor) -> Tensor:
    """
    Map unit sphere to (-1, 1) X (-1, 1) square via square lambert projection.
    :param pts: torch tensor of shape (..., 3) containing the points
    :return: torch tensor of shape (..., 2) containing the projected points
    """

    # constants
    TWO_DIV_SQRT8 = 0.7071067811865475  # 2 / sqrt(8)
    TWOSQRT2_DIV_PI = 0.9003163161571062  # 2 * sqrt(2) / pi

    shape_in = pts.shape[:-1]

    # x-axis and y-axis on the plane are labeled a and b
    x, y, z = torch.unbind(pts.reshape(-1, 3), dim=-1)

    # Define output tensor
    out = torch.empty((len(x), 2), dtype=pts.dtype, device=pts.device)

    # Define conditions and calculations
    cond = torch.abs(y) <= torch.abs(x)
    factor = torch.sqrt(2.0 * (1.0 - torch.abs(z)))

    # instead of precalcuating each branch, just use the condition to select the correct branch
    out[cond, 0] = torch.sign(x[cond]) * factor[cond] * TWO_DIV_SQRT8
    out[cond, 1] = (
        torch.sign(x[cond])
        * factor[cond]
        * torch.atan2(
            y[cond] * torch.sign(x[cond]),
            x[cond] * torch.sign(x[cond]),
        )
        * TWOSQRT2_DIV_PI
    )
    out[~cond, 0] = (
        torch.sign(y[~cond])
        * factor[~cond]
        * torch.atan2(
            x[~cond] * torch.sign(y[~cond]),
            y[~cond] * torch.sign(y[~cond]),
        )
        * TWOSQRT2_DIV_PI
    )
    out[~cond, 1] = torch.sign(y[~cond]) * factor[~cond] * TWO_DIV_SQRT8

    # where close to (0, 0, 1), map to (0, 0)
    at_pole = torch.abs(z) > 0.99999999
    out[at_pole] = 0.0

    return out.reshape(shape_in + (2,))


@torch.jit.script
def inv_square_lambert(pts: Tensor) -> Tensor:
    """
    Map (-1, 1) X (-1, 1) square to Northern hemisphere via inverse square lambert projection.

    :param pts: torch tensor of shape (..., 2) containing the points
    :return: torch tensor of shape (..., 3) containing the projected points

    """

    # move points form [-1, 1] X [-1, 1] to [0, 1] X [0, 1]
    pi = torch.pi

    a = pts[..., 0] * 1.25331413732  # sqrt(pi / 2)
    b = pts[..., 1] * 1.25331413732  # sqrt(pi / 2)

    # mask for branch
    go = torch.abs(b) <= torch.abs(a)

    output = torch.empty((pts.shape[0], 3), dtype=pts.dtype, device=pts.device)

    output[go, 0] = (
        (2 * a[go] / pi)
        * torch.sqrt(pi - a[go] ** 2)
        * torch.cos((pi * b[go]) / (4 * a[go]))
    )
    output[go, 1] = (
        (2 * a[go] / pi)
        * torch.sqrt(pi - a[go] ** 2)
        * torch.sin((pi * b[go]) / (4 * a[go]))
    )
    output[go, 2] = 1 - (2 * a[go] ** 2 / pi)

    output[~go, 0] = (
        (2 * b[~go] / pi)
        * torch.sqrt(pi - b[~go] ** 2)
        * torch.sin((pi * a[~go]) / (4 * b[~go]))
    )
    output[~go, 1] = (
        (2 * b[~go] / pi)
        * torch.sqrt(pi - b[~go] ** 2)
        * torch.cos((pi * a[~go]) / (4 * b[~go]))
    )
    output[~go, 2] = 1 - (2 * b[~go] ** 2 / pi)

    return output


@torch.jit.script
def detector_coords_to_ksphere_via_pc(
    pcs: Tensor,
    n_rows: int,
    n_cols: int,
    tilt: float,
    azimuthal: float,
    sample_tilt: float,
    signal_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Return sets of direction cosines for varying projection centers.

    This should be viewed as a transformation of coordinates specified by n_rows and
    n_cols in the detector plane to points on the sphere.

    Args:
        pcs: Projection centers. Shape (n_pcs, 3)
        n_rows: Number of detector rows.
        n_cols: Number of detector columns.
        tilt: Detector tilt from horizontal in degrees.
        azimuthal: Sample tilt about the sample RD axis in degrees.
        sample_tilt: Sample tilt from horizontal in degrees.
        signal_mask: 1D signal mask with ``True`` values for pixels to get direction

    Returns:
        The direction cosines for each detector pixel for each PC. Shape (n_pcs, n_det_pixels, 3)

    """
    # Generate row and column coordinates
    nrows_array = torch.arange(n_rows - 1, -1, -1, device=pcs.device).float()
    ncols_array = torch.arange(n_cols, device=pcs.device).float()

    # Calculate cosines and sines
    alpha_rad = torch.tensor(
        [(torch.pi / 2.0) + (tilt - sample_tilt) * (torch.pi / 180.0)],
        device=pcs.device,
    )
    azimuthal_rad = torch.tensor([azimuthal * (torch.pi / 180.0)], device=pcs.device)
    cos_alpha = torch.cos(alpha_rad)
    sin_alpha = torch.sin(alpha_rad)
    cos_omega = torch.cos(azimuthal_rad)
    sin_omega = torch.sin(azimuthal_rad)

    # Extract pcx, pcy, pcz from the pc tensor
    pcx_bruker, pcy_bruker, pcz_bruker = torch.unbind(pcs, dim=-1)

    # Convert Bruker (origin top-left, pcx,pcy in [0,1]) → EMsoft detector
    # coordinates.  Matches kikuchipy / EMsoft GenerateDetector:
    #     xpc =  Nx · (pcx − 0.5)
    #     ypc = -Ny · (pcy − 0.5)  ≡  Ny · (0.5 − pcy)
    # The xpc sign was previously (0.5 − pcx), which mirrored the detector
    # x-axis and required a post-hoc np.fliplr downstream.
    pcx_ems = n_cols * (pcx_bruker - 0.5)
    pcy_ems = n_rows * (0.5 - pcy_bruker)
    pcz_ems = n_rows * pcz_bruker

    # det_x is shape (n_pcs, n_cols)
    det_x = pcx_ems[:, None] + (1 - n_cols) * 0.5 + ncols_array[None, :]
    det_y = pcy_ems[:, None] - (1 - n_rows) * 0.5 - nrows_array[None, :]

    # Calculate Ls (n_pcs, n_cols)
    Ls = -sin_omega * det_x + pcz_ems[:, None] * cos_omega
    # Calculate Lc (n_pcs, n_rows)
    Lc = cos_omega * det_x + pcz_ems[:, None] * sin_omega

    # Generate 2D grid indices
    row_indices, col_indices = torch.meshgrid(
        torch.arange(n_rows, device=pcs.device),
        torch.arange(n_cols, device=pcs.device),
        indexing="ij",
    )

    # Flatten the 2D grid indices to 1D
    rows_flat = row_indices.flatten()
    cols_flat = col_indices.flatten()

    # Apply signal mask if it exists
    if signal_mask is not None:
        rows = rows_flat[signal_mask]
        cols = cols_flat[signal_mask]
    else:
        rows = rows_flat
        cols = cols_flat

    # Vectorize the computation
    r_g_x = det_y[:, rows] * cos_alpha + sin_alpha * Ls[:, cols]
    r_g_y = Lc[:, cols]
    r_g_z = -sin_alpha * det_y[:, rows] + cos_alpha * Ls[:, cols]

    # Stack and reshape
    r_g_array = torch.stack([r_g_x, r_g_y, r_g_z], dim=-1)

    # Normalize
    r_g_array = r_g_array / torch.linalg.norm(r_g_array, dim=-1, keepdim=True)

    return r_g_array


@torch.jit.script
def project_pattern_single_geometry(
    master_pattern_MSLNH: Tensor,
    master_pattern_MSLSH: Tensor,
    quaternions: Tensor,
    direction_cosines: Tensor,
) -> Tensor:
    """
    Args:
        master_pattern_MSLNH: modified Square Lambert projection for the Northern Hemisphere. Shape (H, W)
        master_pattern_MSLSH: modified Square Lambert projection for the Southern Hemisphere. Shape (H, W)
        quaternions: Quaternions for each crystalline orientation. Shape (n_orientations, 4)
        direction_cosines: Direction cosines for each pixel in the detector. Shape (n_det_pixels, 3)

    Returns:
        The projected master pattern. Shape (n_orientations, n_det_pixels)

    """
    # sanitize inputs
    assert master_pattern_MSLNH.ndim == 2
    assert master_pattern_MSLSH.ndim == 2
    assert quaternions.ndim == 2
    assert direction_cosines.ndim == 2
    assert direction_cosines.shape[-1] == 3

    n_orientations = quaternions.shape[0]
    n_det_pixels = direction_cosines.shape[0]

    output = torch.empty(
        (n_orientations, n_det_pixels),
        dtype=master_pattern_MSLNH.dtype,
        device=master_pattern_MSLNH.device,
    )

    # rotate the outgoing vectors on the K-sphere according to the crystal orientations
    rotated_vectors = quaternion_apply(
        quaternions[:, None, :], direction_cosines[None, :, :]
    )

    # mask for positive z component
    mask = rotated_vectors[..., 2] > 0

    # where the z component is negative, use the Southern Hemisphere projection
    coords_within_square = square_lambert(rotated_vectors)

    # where the z component is positive, use the Northern Hemisphere projection
    output[mask] = torch.nn.functional.grid_sample(
        master_pattern_MSLNH[None, None, ...],
        coords_within_square[mask][None, None, :],
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    ).squeeze()

    # where the z component is negative, use the Southern Hemisphere projection
    output[~mask] = torch.nn.functional.grid_sample(
        master_pattern_MSLSH[None, None, ...],
        coords_within_square[~mask][None, None, :],
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    ).squeeze()

    return output


@torch.jit.script
def project_pattern_multiple_geometry(
    master_pattern_MSLNH: Tensor,
    master_pattern_MSLSH: Tensor,
    quaternions: Tensor,
    direction_cosines: Tensor,
) -> Tensor:
    """

    This function projects the master pattern onto the detector for each crystalline orientation.
    It is called "paired" because each orientation is paired with another pattern center triplet of
    direction cosines. This function would make sense to use in the context of indexing a map of
    EBSD patterns. Each crystalline orientation would be paired with a pattern center triplet that
    corresponds to that location on the sample.

    Args:
        master_pattern_MSLNH: modified Square Lambert projection for the Northern Hemisphere. Shape (H, W)
        master_pattern_MSLSH: modified Square Lambert projection for the Southern Hemisphere. Shape (H, W)
        quaternions: Quaternions for each crystalline orientation. Shape (n_orientations, 4)
        direction_cosines: Direction cosines for each pixel in the detector. Shape (n_pcs, n_det_pixels, 3)

    Returns:
        The projected master pattern. Shape (n_pcs, n_orientations, n_det_pixels)

    """
    # sanitize inputs
    assert master_pattern_MSLNH.ndim == 2
    assert master_pattern_MSLSH.ndim == 2
    assert quaternions.ndim == 2
    assert direction_cosines.ndim == 3
    assert direction_cosines.shape[-1] == 3

    n_orientations = quaternions.shape[0]
    n_pcs, n_det_pixels, _ = direction_cosines.shape

    output = torch.empty(
        (n_pcs, n_orientations, n_det_pixels),
        dtype=master_pattern_MSLNH.dtype,
        device=master_pattern_MSLNH.device,
    )

    # rotate the outgoing vectors on the K-sphere according to the crystal orientations
    rotated_vectors = quaternion_apply(
        quaternions[None, :, None, :], direction_cosines[:, None, :, :]
    )

    # mask for positive z component
    mask = rotated_vectors[..., 2] > 0

    # where the z component is negative, use the Southern Hemisphere projection
    coords_within_square = square_lambert(rotated_vectors)

    # where the z component is positive, use the Northern Hemisphere projection
    output[mask] = torch.nn.functional.grid_sample(
        master_pattern_MSLNH[None, None, ...],
        coords_within_square[mask][None, :, None],
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    ).squeeze()

    # where the z component is negative, use the Southern Hemisphere projection
    output[~mask] = torch.nn.functional.grid_sample(
        master_pattern_MSLSH[None, None, ...],
        coords_within_square[~mask][None, :, None],
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    ).squeeze()

    return output


@torch.jit.script
def project_HREBSD_pattern(
    pcs: Tensor,
    n_rows: int,
    n_cols: int,
    tilt: float,
    azimuthal: float,
    sample_tilt: float,
    quaternions: Tensor,
    deformation_gradients: Tensor,
    master_pattern_MSLNH: Tensor,
    master_pattern_MSLSH: Tensor,
    signal_mask: Optional[Tensor] = None,
) -> Tensor:
    """

    This function projects the master pattern onto the detector for each crystalline orientation.
    It is called "paired" because each orientation is paired with another pattern center triplet of
    direction cosines. This function would make sense to use in the context of indexing a map of
    EBSD patterns. Each crystalline orientation would be paired with a pattern center triplet that
    corresponds to that location on the sample.

    Args:
        pcs: Projection centers. Shape (n, 3)
        n_rows: Number of detector rows.
        n_cols: Number of detector columns.
        tilt: Detector tilt from horizontal in degrees.
        azimuthal: Sample tilt about the sample RD axis in degrees.
        sample_tilt: Sample tilt from horizontal in degrees.
        quaternions: Quaternions for each crystalline orientation. Shape (n, 4)
        deformation_gradients: Deformation gradients for each crystalline orientation. Shape (n, 3, 3)
        master_pattern_MSLNH: modified Square Lambert projection for the Northern Hemisphere. Shape (H, W)
        master_pattern_MSLSH: modified Square Lambert projection for the Southern Hemisphere. Shape (H, W)


    Returns:
        The projected master pattern. Shape (n, n_det_pixels)

    """
    # sanitize inputs
    if not pcs.ndim == 2 or not pcs.shape[1] == 3:
        raise ValueError(f"pcs must be shape (1, 3) or (n, 3) but got {pcs.shape}")
    if pcs.ndim == 1:
        pcs = pcs[None, :]
    if not quaternions.ndim == 2 or not quaternions.shape[1] == 4:
        raise ValueError(
            f"quaternions must be shape (1, 4) or (n, 4) but got {quaternions.shape}"
        )
    if quaternions.ndim == 1:
        quaternions = quaternions[None, :]
    if not deformation_gradients.ndim == 3 or not deformation_gradients.shape[1:] == (
        3,
        3,
    ):
        raise ValueError(
            f"deformation_gradients must be shape (1, 3, 3) or (n, 3, 3) but got {deformation_gradients.shape}"
        )
    if deformation_gradients.ndim == 2:
        deformation_gradients = deformation_gradients[None, ...]

    # check that the shapes are broadcastable
    if (
        (not pcs.shape[0] == quaternions.shape[0])
        and pcs.shape[0] != 1
        and quaternions.shape[0] != 1
    ):
        raise ValueError(
            f"Not broadcastable: pcs shaped {pcs.shape} and quaternions {quaternions.shape}"
        )
    if (
        (not pcs.shape[0] == deformation_gradients.shape[0])
        and pcs.shape[0] != 1
        and deformation_gradients.shape[0] != 1
    ):
        raise ValueError(
            f"Not broadcastable: pcs shaped {pcs.shape} and deformation gradient {deformation_gradients.shape}"
        )
    if (
        (not quaternions.shape[0] == deformation_gradients.shape[0])
        and quaternions.shape[0] != 1
        and deformation_gradients.shape[0] != 1
    ):
        raise ValueError(
            f"Not broadcastable: quaternions shaped {quaternions.shape} and deformation gradient {deformation_gradients.shape}"
        )

    if not master_pattern_MSLNH.ndim == 2:
        raise ValueError(
            f"master_pattern_MSLNH must be shape (H, W) but got {master_pattern_MSLNH.shape}"
        )
    if not master_pattern_MSLSH.ndim == 2:
        raise ValueError(
            f"master_pattern_MSLSH must be shape (H, W) but got {master_pattern_MSLSH.shape}"
        )

    # get direction cosines
    direction_cosines = detector_coords_to_ksphere_via_pc(
        pcs,
        n_rows,
        n_cols,
        tilt,
        azimuthal,
        sample_tilt,
        signal_mask=signal_mask,
    )

    n_orientations = quaternions.shape[0]
    n_det_pixels = direction_cosines.shape[1]

    output = torch.empty(
        (n_orientations, n_det_pixels),
        dtype=master_pattern_MSLNH.dtype,
        device=master_pattern_MSLNH.device,
    )

    # rotate the outgoing vectors on the K-sphere according to the crystal orientations
    rotated_vectors = quaternion_apply(quaternions[:, None, :], direction_cosines)

    # apply the inverse of the deformation gradients to the rotated vectors
    rotated_vectors = torch.matmul(
        torch.inverse(deformation_gradients), rotated_vectors[:, :, :, None]
    ).squeeze(-1)

    # renormalize the rotated vectors
    rotated_vectors = rotated_vectors / torch.linalg.norm(
        rotated_vectors, dim=-1, keepdim=True
    )

    # mask for positive z component
    mask = rotated_vectors[..., 2] > 0

    # get the coordinates within the image square
    coords_within_square = square_lambert(rotated_vectors)

    # where the z component is positive, use the Northern Hemisphere projection
    output[mask] = torch.nn.functional.grid_sample(
        master_pattern_MSLNH[None, None, ...],
        coords_within_square[mask][None, None, :],
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    ).squeeze()

    # where the z component is negative, use the Southern Hemisphere projection
    output[~mask] = torch.nn.functional.grid_sample(
        master_pattern_MSLSH[None, None, ...],
        coords_within_square[~mask][None, None, :],
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    ).squeeze()

    return output


# ---------------------------------------------------------------------------
# Energy-weighted EBSD pattern projection (matches EMsoft CalcEBSDPatternSingleFull_)
#
#   Per-pixel:  pattern(ii, jj) = Σ_kk  accum_e(kk, ii, jj) * mLPNH/SH(nix, niy, kk)
#
# Use these when you need the spatial variation of the Monte-Carlo energy
# weights across the detector.  For a quick energy-collapsed approximation,
# use `_1d` variant.
# ---------------------------------------------------------------------------


def accum_e_to_detector(
    accum_e_mc: Tensor,           # (nE, ns, ns) Monte-Carlo Lambert grid
    pcs: Tensor,                  # (n_pcs, 3) Bruker PCs — typically n_pcs=1
    n_rows: int,
    n_cols: int,
    tilt: float,
    azimuthal: float,
    sample_tilt: float,
) -> Tensor:
    """
    Project Monte-Carlo accum_e from its MC Lambert grid onto the detector
    grid.  Mirrors EMsoft's GenerateDetector_ — for each detector pixel,
    take the direction cosine on the K-sphere, project to the square Lambert
    coordinates, and sample accum_e at that coordinate for every energy bin.

    This is run separately from the master-pattern lookup because accum_e
    lives in the sample reference frame (it does not depend on grain
    orientation), while the master pattern is looked up in the crystal
    frame (which requires the rotation by `quaternions`).

    Args:
        accum_e_mc: (nE, ns, ns) Monte-Carlo accum_e in the upper hemisphere
            of the sample reference frame, in square-Lambert coordinates.
        pcs, n_rows, n_cols, tilt, azimuthal, sample_tilt: same conventions
            as `detector_coords_to_ksphere_via_pc`.

    Returns:
        accum_e_detector: (nE, n_rows, n_cols) with the same energy binning
            as accum_e_mc.  Only n_pcs=1 is currently supported — multi-PC
            mode would return (nE, n_pcs, n_rows, n_cols) which downstream
            functions don't yet consume.
    """
    if accum_e_mc.ndim != 3:
        raise ValueError(f"accum_e_mc must be 3D (nE, ns, ns); got shape {accum_e_mc.shape}")
    if pcs.ndim == 1:
        pcs = pcs[None, :]
    if pcs.shape[0] != 1:
        raise NotImplementedError(
            f"accum_e_to_detector currently supports a single PC; got n_pcs={pcs.shape[0]}."
        )

    import math

    nE = accum_e_mc.shape[0]
    device = accum_e_mc.device
    dtype = accum_e_mc.dtype

    # Direction cosines for each detector pixel, in the sample reference frame.
    # accum_e is in the same frame — no quaternion rotation needed.
    direction_cosines = detector_coords_to_ksphere_via_pc(
        pcs.to(device), n_rows, n_cols, tilt, azimuthal, sample_tilt,
        signal_mask=None,
    )  # (1, n_det_pixels, 3)

    # ── EMsoft GenerateDetector_ geometric weighting ───────────────────────
    #   alpha  = atan(delta/L/sqrt(pi))
    #   calpha = cos(alpha)
    #   dp     = pcvec · dc
    #   g      = ((calpha^2 + dp^2 - 1)^1.5) / (calpha^3) * 0.25
    # delta/L (angular size of one pixel) follows from Bruker pcz convention:
    #   pcz_bruker = L / (n_rows * delta)  ⇒  delta/L = 1 / (pcz_bruker * n_rows)
    pcz_bruker = float(pcs[0, 2].item())
    delta_over_L = 1.0 / (pcz_bruker * float(n_rows))
    alpha_pix = math.atan(delta_over_L / math.sqrt(math.pi))
    calpha = math.cos(alpha_pix)

    # pcvec in sample frame: at the PC pixel, det_x = det_y = 0, so the
    # direction cosine from detector_coords_to_ksphere_via_pc reduces to
    #   (sin(α_geo)·cos(ω), sin(ω), cos(α_geo)·cos(ω))
    # where α_geo = π/2 + (tilt − sample_tilt)·π/180 and ω = azimuthal·π/180.
    alpha_geo_rad = (math.pi / 2.0) + (tilt - sample_tilt) * (math.pi / 180.0)
    omega_rad     = azimuthal * (math.pi / 180.0)
    pcvec = torch.tensor(
        [
            math.sin(alpha_geo_rad) * math.cos(omega_rad),
            math.sin(omega_rad),
            math.cos(alpha_geo_rad) * math.cos(omega_rad),
        ],
        device=device, dtype=dtype,
    )
    pcvec = pcvec / torch.linalg.norm(pcvec)

    # dp = pcvec · dc, broadcast over detector pixels.
    dp = (direction_cosines.squeeze(0) * pcvec).sum(dim=-1)  # (n_det_pixels,)

    # g factor.  Clamp the cube-root argument to ≥ 0 so floating-point grit
    # at pixels nearly orthogonal to pcvec doesn't return NaN.
    calpha_t = torch.tensor(calpha, device=device, dtype=dtype)
    inside = (calpha_t ** 2 + dp ** 2 - 1.0).clamp(min=0.0)
    g = inside.pow(1.5) / (calpha_t ** 3) * 0.25       # (n_det_pixels,)

    # ── Lambert lookup of accum_e ──────────────────────────────────────────
    # square_lambert uses |z|, so upper- and lower-hemisphere rays collapse
    # to the same 2-D coord — fine here because accum_e is only populated
    # for the upper hemisphere (the back-scatter direction in EMsoft's MC).
    coords = square_lambert(direction_cosines).to(dtype=dtype)

    # EMsoft samples accum_e on a Lambert grid that is rotated 90° relative
    # to the master-pattern Lambert grid.  See EMsoftOO mod_EBSD.f90 ~2466-2470:
    #     x = ixy(1); ixy(1) = ixy(2); ixy(2) = -x
    # i.e. (x, y) → (y, -x).  Without this, accum_e is sampled at the wrong
    # angular position and the per-pixel energy-weighting structure smears
    # out — visually, the centro-symmetric envelope is intact but the MC
    # asymmetry projects onto the wrong detector axis.
    coords = torch.stack([coords[..., 1], -coords[..., 0]], dim=-1)

    grid = coords.unsqueeze(1)                          # (1, 1, n_det_pixels, 2)

    # accum_e_mc here is (nE, ipy, ipx) after the read transpose in
    # SimPatGen._read_master_pattern_h5.  PyTorch's grid_sample reads the
    # input as (N, C, H, W) with grid[..., 0] (Lambert x) indexing W and
    # grid[..., 1] (Lambert y) indexing H, so the existing axis order
    # already maps ipy → H and ipx → W correctly.  No transpose needed.
    sampled = torch.nn.functional.grid_sample(
        accum_e_mc.unsqueeze(0),     # (1, nE, ipy_H, ipx_W)
        grid,                         # (1, 1, n_det_pixels, 2)
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    ).squeeze(0).squeeze(1)           # (nE, n_det_pixels)

    # accum_e is a non-negative count — clamp interpolation negatives
    # near sharp features so the multiplied output stays physical.
    sampled = torch.clamp(sampled, min=0.0)

    # Apply the geometric obliquity factor per detector pixel.
    sampled = sampled * g.unsqueeze(0)

    return sampled.reshape(nE, n_rows, n_cols).contiguous()


def project_HREBSD_pattern_energy_weighted(
    pcs: Tensor,
    n_rows: int,
    n_cols: int,
    tilt: float,
    azimuthal: float,
    sample_tilt: float,
    quaternions: Tensor,
    deformation_gradients: Tensor,
    master_pattern_MSLNH: Tensor,   # (nE, H, W)
    master_pattern_MSLSH: Tensor,   # (nE, H, W)
    accum_e: Tensor,                # (nE, n_rows, n_cols)
    e_min_idx: int = 0,
    e_max_idx: Optional[int] = None,
    signal_mask: Optional[Tensor] = None,
    interp_mode: str = "bilinear",
) -> Tensor:
    """
    Energy-resolved EBSD pattern projection, matching EMsoft's
    CalcEBSDPatternSingleFull_.

    Args:
        pcs: Projection centers (Bruker convention). Shape (n, 3) or (1, 3).
        n_rows, n_cols: Detector pixel dimensions.
        tilt: Detector tilt (theta_c) in degrees.
        azimuthal: Sample azimuthal (omega) about RD in degrees.
        sample_tilt: Sample tilt (sigma) in degrees, typically 70.0.
        quaternions: Crystal orientations, shape (n, 4), real-first.
        deformation_gradients: F tensors, shape (n, 3, 3). Pass identity for
            an undeformed pattern.
        master_pattern_MSLNH: Northern-hemisphere Lambert master pattern with
            an energy axis. Shape (nE, H, W).
        master_pattern_MSLSH: Southern-hemisphere counterpart. Shape (nE, H, W).
        accum_e: Per-detector-pixel energy weights from EMsoft's Monte Carlo
            (the `accum_e_detector` array). Shape (nE, n_rows, n_cols).
        e_min_idx: Lowest energy bin index to include (inclusive). Default 0.
        e_max_idx: Highest energy bin index to include (exclusive, Python slice
            semantics). Default None means use all bins above e_min_idx.
        signal_mask: Optional 1D bool mask of length n_rows*n_cols selecting
            which detector pixels to project. Same convention as the existing
            HREBSD.py functions.
        interp_mode: 'bilinear' (matches EMsoft) or 'bicubic'. Default
            'bilinear'.

    Returns:
        Energy-integrated patterns. Shape (n_orientations, n_det_pixels) where
        n_det_pixels = n_rows * n_cols (or `signal_mask.sum()` if a mask was
        provided). Reshape to (n_rows, n_cols) to view as an image.
    """
    # ---- input validation ----
    if pcs.ndim == 1:
        pcs = pcs[None, :]
    if quaternions.ndim == 1:
        quaternions = quaternions[None, :]
    if deformation_gradients.ndim == 2:
        deformation_gradients = deformation_gradients[None, ...]

    if pcs.shape[-1] != 3:
        raise ValueError(f"pcs must end in shape 3, got {pcs.shape}")
    if quaternions.shape[-1] != 4:
        raise ValueError(f"quaternions must end in shape 4, got {quaternions.shape}")
    if deformation_gradients.shape[-2:] != (3, 3):
        raise ValueError(
            f"deformation_gradients must end in (3,3), got {deformation_gradients.shape}"
        )
    if master_pattern_MSLNH.ndim != 3 or master_pattern_MSLSH.ndim != 3:
        raise ValueError(
            "master_pattern_MSLNH and _MSLSH must be 3D (nE, H, W); "
            f"got {master_pattern_MSLNH.shape}, {master_pattern_MSLSH.shape}"
        )
    if master_pattern_MSLNH.shape != master_pattern_MSLSH.shape:
        raise ValueError("NH and SH master patterns must have the same shape.")
    if accum_e.ndim != 3:
        raise ValueError(f"accum_e must be (nE, n_rows, n_cols), got {accum_e.shape}")
    if accum_e.shape[0] != master_pattern_MSLNH.shape[0]:
        raise ValueError(
            f"accum_e energy axis ({accum_e.shape[0]}) does not match master "
            f"pattern energy axis ({master_pattern_MSLNH.shape[0]})."
        )
    if accum_e.shape[1] != n_rows or accum_e.shape[2] != n_cols:
        raise ValueError(
            f"accum_e detector shape {accum_e.shape[1:]} does not match "
            f"(n_rows, n_cols) = ({n_rows}, {n_cols})."
        )

    # ---- energy bin slicing (mimics Emin / Emax in the Fortran) ----
    if e_max_idx is None:
        e_max_idx = master_pattern_MSLNH.shape[0]
    e_min_idx = max(0, e_min_idx)
    e_max_idx = min(master_pattern_MSLNH.shape[0], e_max_idx)
    if e_max_idx <= e_min_idx:
        raise ValueError(f"empty energy range: [{e_min_idx}, {e_max_idx})")

    mp_nh = master_pattern_MSLNH[e_min_idx:e_max_idx]   # (nE_sel, H, W)
    mp_sh = master_pattern_MSLSH[e_min_idx:e_max_idx]
    accum = accum_e[e_min_idx:e_max_idx]                # (nE_sel, n_rows, n_cols)

    nE_sel = mp_nh.shape[0]
    device = mp_nh.device
    dtype = mp_nh.dtype

    # ---- detector direction cosines ----
    direction_cosines = detector_coords_to_ksphere_via_pc(
        pcs.to(device),
        n_rows,
        n_cols,
        tilt,
        azimuthal,
        sample_tilt,
        signal_mask=signal_mask,
    )  # (n_pcs, n_det_pixels, 3)

    n_orientations = quaternions.shape[0]
    n_det_pixels = direction_cosines.shape[1]

    # ---- rotate by orientation, then apply F^{-1} ----
    rotated = quaternion_apply(
        quaternions.to(device)[:, None, :], direction_cosines
    )  # broadcast: (n_orientations, n_det_pixels, 3)

    F_inv = torch.inverse(deformation_gradients.to(device))
    rotated = torch.matmul(
        F_inv[:, None, :, :], rotated[:, :, :, None]
    ).squeeze(-1)
    rotated = rotated / torch.linalg.norm(rotated, dim=-1, keepdim=True)

    # ---- hemisphere selection and Lambert square coords ----
    mask_nh = rotated[..., 2] >= 0                       # (n_orientations, n_det_pixels)
    coords = square_lambert(rotated)                     # (n_orientations, n_det_pixels, 2)

    # ---- sample master pattern at every energy bin in one shot ----
    # grid_sample input shape: (1, nE_sel, H, W); grid shape:
    # (1, n_orientations, n_det_pixels, 2). Output: (1, nE_sel, n_o, n_p).
    grid = coords.unsqueeze(0).to(dtype=dtype)

    sampled_nh = torch.nn.functional.grid_sample(
        mp_nh.unsqueeze(0),
        grid,
        mode=interp_mode,
        align_corners=True,
        padding_mode="border",
    ).squeeze(0)  # (nE_sel, n_orientations, n_det_pixels)

    sampled_sh = torch.nn.functional.grid_sample(
        mp_sh.unsqueeze(0),
        grid,
        mode=interp_mode,
        align_corners=True,
        padding_mode="border",
    ).squeeze(0)  # (nE_sel, n_orientations, n_det_pixels)

    # Pick hemisphere per pixel. mask_nh broadcasts over the energy axis.
    sampled = torch.where(mask_nh.unsqueeze(0), sampled_nh, sampled_sh)
    #   shape: (nE_sel, n_orientations, n_det_pixels)

    # ---- apply per-pixel energy weights and sum over energy ----
    # accum is (nE_sel, n_rows, n_cols). Flatten the detector to match
    # the pixel ordering used by detector_coords_to_ksphere_via_pc, then
    # apply the same signal_mask the geometry code applied.
    accum_flat = accum.reshape(nE_sel, n_rows * n_cols).to(device=device, dtype=dtype)
    if signal_mask is not None:
        accum_flat = accum_flat[:, signal_mask.to(device)]

    # (nE_sel, 1, n_det_pixels) * (nE_sel, n_o, n_det_pixels) -> sum over E
    pattern = (sampled * accum_flat.unsqueeze(1)).sum(dim=0)
    #   shape: (n_orientations, n_det_pixels)

    return pattern


def project_HREBSD_pattern_energy_weighted_1d(
    pcs: Tensor,
    n_rows: int,
    n_cols: int,
    tilt: float,
    azimuthal: float,
    sample_tilt: float,
    quaternions: Tensor,
    deformation_gradients: Tensor,
    master_pattern_MSLNH: Tensor,   # (nE, H, W)
    master_pattern_MSLSH: Tensor,   # (nE, H, W)
    energy_weights: Tensor,         # (nE,)
    signal_mask: Optional[Tensor] = None,
    interp_mode: str = "bilinear",
) -> Tensor:
    """
    Approximate energy weighting using a single 1D weight vector. Equivalent
    to first collapsing the master pattern via
        mp_2d = sum_k energy_weights[k] * mp_3d[k]
    and then running the standard projector. Faster and uses less memory,
    but ignores how `accum_e` varies across the detector (it falls off toward
    corners). Use this for a quick check; use the per-pixel function above
    for a real EMsoft match.
    """
    if energy_weights.ndim != 1:
        raise ValueError(f"energy_weights must be 1D, got {energy_weights.shape}")
    if energy_weights.shape[0] != master_pattern_MSLNH.shape[0]:
        raise ValueError(
            f"energy_weights length {energy_weights.shape[0]} does not match "
            f"master pattern nE = {master_pattern_MSLNH.shape[0]}"
        )

    w = energy_weights.to(device=master_pattern_MSLNH.device,
                          dtype=master_pattern_MSLNH.dtype)
    mp_nh_2d = (master_pattern_MSLNH * w[:, None, None]).sum(dim=0)
    mp_sh_2d = (master_pattern_MSLSH * w[:, None, None]).sum(dim=0)

    # Build a flat accum_e equivalent so we can reuse the per-pixel function
    # without duplicating geometry code. Each pixel gets the same energy
    # weights -> we promote to a (1, ...) energy axis with weight 1.
    accum_pseudo = torch.ones(
        (1, n_rows, n_cols),
        device=mp_nh_2d.device,
        dtype=mp_nh_2d.dtype,
    )

    return project_HREBSD_pattern_energy_weighted(
        pcs=pcs,
        n_rows=n_rows,
        n_cols=n_cols,
        tilt=tilt,
        azimuthal=azimuthal,
        sample_tilt=sample_tilt,
        quaternions=quaternions,
        deformation_gradients=deformation_gradients,
        master_pattern_MSLNH=mp_nh_2d.unsqueeze(0),
        master_pattern_MSLSH=mp_sh_2d.unsqueeze(0),
        accum_e=accum_pseudo,
        signal_mask=signal_mask,
        interp_mode=interp_mode,
    )


def load_emsoft_master_and_accum(h5_path: str, device: str = "cpu",
                                 return_energies: bool = False):
    """
    Read the energy-resolved master patterns and the Monte-Carlo accum_e
    array out of an EMsoft master-pattern HDF5 file. Field names follow
    standard EMsoft layout; adjust if your file uses different paths.

    Args:
        h5_path: Path to the EMsoft master-pattern HDF5.
        device: Torch device for the returned tensors.
        return_energies: If True, also return the per-bin energy axis (keV).
            If False (default), preserves the previous 3-tuple return for
            backward compatibility.

    Returns:
        When return_energies is False:
            (mLPNH, mLPSH, accum_e)
        When return_energies is True:
            (mLPNH, mLPSH, accum_e, energies_keV)
            with energies_keV of shape (nE,) — the bin centers in keV.

    Note: `accum_e` here is the *Monte Carlo grid* version, not the
    detector-projected `accum_e_detector`.  Pass it through
    `accum_e_to_detector(...)` to get the per-pixel weights, which
    HREBSD.project_HREBSD_pattern_energy_weighted then consumes.
    """
    import h5py
    import numpy as np

    with h5py.File(h5_path, "r") as f:
        mLPNH = np.asarray(f["EMData/EBSDmaster/mLPNH"][...])
        mLPSH = np.asarray(f["EMData/EBSDmaster/mLPSH"][...])
        # EMsoft stores mLPNH as (1, nE, H, W) — drop the leading axis if so.
        if mLPNH.ndim == 4 and mLPNH.shape[0] == 1:
            mLPNH = mLPNH[0]
            mLPSH = mLPSH[0]
        # accum_e is in the MC group of the same file (or in the linked MC file)
        accum_e = np.asarray(f["EMData/MCOpenCL/accum_e"][...])

        # Energy axis metadata.  EMsoft writes Ehistmin (lower edge), EkeV
        # (beam energy), and Ebinsize into the NML group.  Fall back to
        # integer bin indices if the keys aren't present.
        energies = None
        if return_energies:
            try:
                E_min  = float(f["NMLparameters/MCCLNameList/Ehistmin"][()])
                E_max  = float(f["NMLparameters/MCCLNameList/EkeV"][()])
                E_bin  = float(f["NMLparameters/MCCLNameList/Ebinsize"][()])
                energies = np.arange(E_min, E_max + 0.5 * E_bin, E_bin, dtype=np.float32)
            except KeyError:
                energies = np.arange(mLPNH.shape[0], dtype=np.float32)

    nh_t = torch.from_numpy(mLPNH).to(device=device, dtype=torch.float32)
    sh_t = torch.from_numpy(mLPSH).to(device=device, dtype=torch.float32)
    ae_t = torch.from_numpy(accum_e).to(device=device, dtype=torch.float32)
    if return_energies:
        en_t = torch.from_numpy(energies).to(device=device)
        return nh_t, sh_t, ae_t, en_t
    return nh_t, sh_t, ae_t


def emsoft_keV_to_bin(energies_keV: Tensor, energy_keV: float) -> int:
    """Convert an energy in keV to the index of the nearest master-pattern bin.

    Args:
        energies_keV: 1-D tensor of bin-center energies (shape (nE,)) — get
            this from `load_emsoft_master_and_accum(..., return_energies=True)`.
        energy_keV: Target energy in keV.

    Returns:
        Integer bin index that minimises |energies_keV[i] − energy_keV|.
    """
    return int(torch.argmin(torch.abs(energies_keV - energy_keV)).item())


# ---------------------------------------------------------------------------
# Demo / smoke-test (kept guarded so `import HREBSD` doesn't run it)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import kikuchipy as kp
    import matplotlib.pyplot as plt
    import numpy as np

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # get the master pattern
    mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")

    # to torch tensor
    mLPNH = torch.from_numpy(mp.data[0, :, :]).to(torch.float32)
    mLPSH = torch.from_numpy(mp.data[1, :, :]).to(torch.float32)

    # normalize each master pattern to 0 to 1
    mLPNH = ((mLPNH - torch.min(mLPNH)) / (torch.max(mLPNH) - torch.min(mLPNH))).to(device)
    mLPSH = ((mLPSH - torch.min(mLPSH)) / (torch.max(mLPSH) - torch.min(mLPSH))).to(device)

    # get the signal mask
    signal_mask = kp.filters.Window("circular", (1080, 1080)).astype(bool).reshape(-1)
    signal_mask = torch.from_numpy(signal_mask).to(torch.bool)

    # get the deformation gradients
    deformation_gradients = torch.eye(3, dtype=torch.float32).to(device)[None, :, :]

    pattern_center = torch.tensor([0.4221, 0.2179, 0.4954], device=device)[None, :]
    detector_height = 1080
    detector_width = 1080
    det_shape = (detector_height, detector_width)
    detector_tilt_deg = 0.0
    azimuthal_deg = 0.0
    sample_tilt_deg = 70.0

    # demo: identity F vs y-stretch
    for tag, F in (("identity", torch.eye(3, dtype=torch.float32, device=device)[None, :, :]),
                   ("ystretch", torch.eye(3, dtype=torch.float32, device=device)[None, :, :].clone())):
        if tag == "ystretch":
            F[:, 1, 1] = 1.05
        from HREBSD import eu2qu  # self-import for the demo
        quats = eu2qu(torch.zeros(1, 3, device=device), "ZXZ")
        patterns = project_HREBSD_pattern(
            pattern_center,
            detector_height,
            detector_width,
            detector_tilt_deg,
            azimuthal_deg,
            sample_tilt_deg,
            quats,
            F,
            mLPNH,
            mLPSH,
            signal_mask=signal_mask,
        )
        canvas = np.zeros(det_shape)
        canvas[signal_mask.reshape(det_shape)] = patterns[0].cpu().numpy()
        plt.imshow(canvas, cmap="gray"); plt.axis("off")
        plt.savefig(f"HREBSD_demo_{tag}.png", bbox_inches="tight"); plt.clf()

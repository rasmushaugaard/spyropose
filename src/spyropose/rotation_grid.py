from functools import cache

import healpy as hp
import numpy as np
from scipy.spatial.transform import Rotation

from . import utils


def vec2pix(R_z, r):
    """R_z to nested healpix index at recursion r"""
    return hp.vec2pix(nside=2**r, x=R_z[..., 0], y=R_z[..., 1], z=R_z[..., 2], nest=True)


@cache
def get_R_base():
    """
    We want a mapping from rotation -> pixel index
    use one of the axes as vec for easy pix lookup (as implicit_pdf, but with z)
    and define tilt relative to arbitrary, but defined and constant base pixel
    frames to avoid singularities and fast changes in frame
    """
    base_z = np.stack(
        hp.pix2vec(1, np.arange(12), nest=True), axis=1
    )  # (12 base pixels, 3 z axis)
    # define by a non-zero cross product (any direction not in base_z)
    base_x = utils.normalize_vectors(np.cross(np.ones(3), base_z))  # (12, 3)
    R_base = np.stack((base_x, np.cross(base_z, base_x), base_z), axis=-1)
    assert R_base.shape == (12, 3, 3)
    return R_base  # (12, 3, 3)


def get_local_frame(R_z, pix_base=None):
    if pix_base is None:
        pix_base = vec2pix(R_z, r=0)
    R_base = get_R_base()[pix_base]
    # find "closest" R that rotates R_base_z into R_z
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    v = np.cross(R_base[..., 2], R_z)  # (..., 3)
    vx = np.cross(np.eye(3), v[..., None, :])  # (..., 3, 3)
    c = (R_base[..., 2] * R_z).sum(axis=-1)[..., None, None]  # (..., 1, 1)
    R_offset = np.eye(3) + vx + (vx @ vx) / (1 + c)
    return R_offset @ R_base


def get_closest_pix(R, r):  # (..., 3, 3), (,)
    R_z = R[..., 2]
    pix = vec2pix(R_z, r=r)
    R_frame = get_local_frame(R_z)
    R_local = np.swapaxes(R_frame, -1, -2) @ R
    tilt = Rotation.from_matrix(R_local).as_rotvec()[..., 2]
    n_tilts = 6 * 2**r
    tilt = np.floor(tilt % (2 * np.pi) * n_tilts / (2 * np.pi)).astype(int)

    # find nested indexing
    pix = np.unravel_index(pix, [12] + [4] * r)
    tilt = np.unravel_index(tilt, [6] + [2] * r)
    flat = []
    for i in range(r + 1):
        flat.append(pix[i])
        flat.append(tilt[i])
    idx = np.ravel_multi_index(flat, [12, 6] + [4, 2] * r)
    return idx


def generate_rotation_grid(recursion_level=None, size=None, return_xyz=False):
    """
    Modified version of:
    https://github.com/google-research/google-research/blob/master/implicit_pdf/models.py

    Generates an equivolumetric grid on SO(3) following Yershova et al. (2010).

    Uses a Healpix grid on the 2-sphere as a starting point and then tiles it
    along the 'tilt' direction 6*2**recursion_level times over 2pi.

    Args:
      recursion_level: An integer which determines the level of resolution of the
        grid.  The final number of points will be 72*8**recursion_level.  A
        recursion_level of 2 (4k points) was used for training and 5 (2.4M points)
        for evaluation.
      size: A number of rotations to be included in the grid.  The nearest grid
        size in log space is returned.

    Returns:
      (N, 3, 3) array of rotation matrices, where N=72*8**recursion_level.
    """
    # TODO: clearly distinguish between hp pix and SO3 pix
    assert not (recursion_level is None and size is None)
    if size:
        recursion_level = max(int(np.round(np.log(size / 72.0) / np.log(8.0))), 0)
    number_per_side = 2**recursion_level
    # 12 * (2**recursion_level) ** 2 = 12 * 4 ** recursion_level
    number_pix = hp.nside2npix(number_per_side)
    # nest=True
    R_z = np.stack(
        hp.pix2vec(number_per_side, np.arange(number_pix), nest=True), axis=1
    )  # (number_pix, 3)
    R_pix = get_local_frame(R_z)

    n_tilts = 6 * 2**recursion_level
    # add (2 pi) / (2 n_tilts) to get the region-splitting property
    tilts = np.linspace(0, 2 * np.pi, n_tilts, endpoint=False) + np.pi / n_tilts
    R_tilt = Rotation.from_rotvec(
        np.stack((np.zeros(n_tilts), np.zeros(n_tilts), tilts), axis=-1)
    ).as_matrix()

    # rotate in object frame (righthand matmul)
    # and keep nest-property (reshaping to nested shape)
    return (
        (
            R_pix.reshape([12, 1] + [4, 1] * recursion_level + [3, 3])
            @ R_tilt.reshape([1, 6] + [1, 2] * recursion_level + [3, 3])
        )
        .reshape(number_pix * n_tilts, 3, 3)
        .astype(np.float32)
    )

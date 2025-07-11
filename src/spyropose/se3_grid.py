"""
Cartesian product of the Healpix rotation grid and the translation grid.

We define the SE3 recursion level 0 at:
    Healpix recursion level 0 (72 bins)
    and translation grid recursion leve 1 (8 bins)
leading to 72 * 8 = 576 bins in the SE3 grid at r = 0.

For rotations, nested indexing is used, since it allows for easy expansion and indexing,
when storing the full mapping from indices to rotations.
For positions, we use a per-dimension index, since it's easy to convert back and forth
between positions and indices.

An SE3 bin is then indexed by the four indices:
    (rot_idx, pos_idx_x, pos_idx_y, pos_idx_z).
"""

import math
from dataclasses import dataclass
from functools import cached_property
from typing import List

import einops
import numpy as np
import torch
from torch import Tensor

from . import translation_grid


def get_idx_recursion_0(b, device, extended=False):
    """
    Extended: SE3 recursion 0, but with a 3x3x3 positional grid instead of 2x2x2.
    This enables training with a grid offset up to one grid spacing while keeping
    the original volume covered, ensuring that any pose within the volume
    is valid during inference at any recursion level.
    """
    n_rot = 72

    pos_sidelen = 3 if extended else 2
    n_pos = pos_sidelen**3
    rot_idx = einops.repeat(
        torch.arange(n_rot, device=device),
        "n_rot -> b (n_rot n_pos)",
        b=b,
        n_pos=n_pos,
    )
    pos_idx = torch.arange(pos_sidelen, device=device)
    pos_idx = torch.stack(torch.meshgrid([pos_idx] * 3, indexing="ij"), dim=-1).view(n_pos, 3)
    pos_idx = einops.repeat(pos_idx, "n_pos d -> b (n_rot n_pos) d", b=b, n_rot=n_rot)
    return rot_idx, pos_idx


def expand(rot_idx, pos_idx, flat=False):
    """Every SE3 bin is expanded to 8 * 8 = 64 bins"""
    device = rot_idx.device
    b, n = rot_idx.shape
    assert pos_idx.shape == (b, n, 3)
    rot_idx = einops.repeat(
        rot_idx[..., None] * 8 + torch.arange(8, device=device),
        "b n r -> b n (r p)",
        p=8,
    )
    pos_idx = einops.repeat(
        translation_grid.expand_grid(pos_idx),  # (b, n, 8, 3)
        "b n p d -> b n (r p) d",
        r=8,
    )
    if flat:
        rot_idx = rot_idx.view(b, n * 64)
        pos_idx = pos_idx.view(b, n * 64, 3)
    return rot_idx, pos_idx


def log_bin_count(r):
    # 72 rotation bins x 8 position bins at recursion 0, branch factor 64.
    # n = (72 * 8) * 64 ** r
    # log(n) = log(72 * 8) + log(64 ** r) = log(72 * 8) + r * log(64)
    return math.log(72 * 8) + r * math.log(64)


def locate_poses_in_pyramid(
    q_rot_idx_rlast: Tensor,
    rot_idxs: List[Tensor],
    log_probs: List[Tensor],
    position_scale: float,
    q_pos: Tensor,
    t_est: Tensor,
    pos_grid_frame: Tensor,
    pos_idxs: List[Tensor],
):
    """
    Locates the recursion level and idx of the bin representing the query pose.
    rlast refers to the last recursion.

    Locates multiple (n) query poses in multiple (b) se3 pyramids.

    TODO: refactor to use SpyroPyramid
    """
    device = q_rot_idx_rlast.device
    recursion_depth = len(rot_idxs)
    rlast = recursion_depth - 1
    b, n = q_rot_idx_rlast.shape

    log_rot_volume = torch.full(size=(b,), fill_value=np.log(np.pi**2), device=device)
    assert t_est.shape == (b, 3, 1)
    assert q_pos.shape == (b, n, 3, 1)
    assert pos_grid_frame.shape == (b, 3, 3)
    assert len(pos_idxs) == recursion_depth

    t_est = t_est.unsqueeze(1)
    pos_grid_frame = pos_grid_frame.unsqueeze(1)
    q_pos_idx_rlast = translation_grid.pos2grid(
        pos=q_pos, t_est=t_est, grid_frame=pos_grid_frame, r=rlast + 1
    )  # (b, n, 3)

    # determinant of frame is the volume of the parallelogram
    # spanning the grid volume
    log_pos_volume = torch.logdet(pos_grid_frame * position_scale).view(b)
    log_volume = log_pos_volume + log_rot_volume

    # traverse the grid top down and return results at different layers
    idx_match = torch.full(
        size=(b, n, recursion_depth), fill_value=-1, dtype=torch.long, device=device
    )
    ll = torch.empty(size=(b, n, recursion_depth), dtype=torch.float, device=device)

    for r in range(recursion_depth):
        # rotation index expansion is 8
        q_rot_idx = q_rot_idx_rlast.div(8 ** (rlast - r), rounding_mode="trunc").unsqueeze(2)
        assert q_rot_idx.shape == (b, n, 1)
        rot_idx = rot_idxs[r].unsqueeze(1)
        l = rot_idx.shape[-1]
        assert rot_idx.shape == (b, 1, l)
        match = rot_idx == q_rot_idx

        # position index expansion is 2 per dim
        q_pos_idx = q_pos_idx_rlast.div(2 ** (rlast - r), rounding_mode="trunc").unsqueeze(2)
        assert q_pos_idx.shape == (b, n, 1, 3)
        pos_idx = pos_idxs[r].unsqueeze(1)
        assert pos_idx.shape == (b, 1, l, 3)
        # should match all four indices (1 rot and 3 pos indices)
        match = match & (pos_idx == q_pos_idx).all(dim=-1)

        b_idx, n_idx, l_idx = torch.where(match)
        # l_idx are the pose indices in the pyramid layers
        idx_match[b_idx, n_idx, r] = l_idx
        ll[..., r] = ll[..., r - 1]

        bin_volume = log_volume - log_bin_count(r=r)
        ll[b_idx, n_idx, r] = log_probs[r][b_idx, l_idx] - bin_volume[b_idx]

    # all queries should match at layer 0
    assert (idx_match[..., 0] != -1).all()
    return idx_match, ll


@dataclass
class PosePyramid:
    """
    n_r denotes n_expanded_{r-1} * 2^6 (top_k * 2^6)
    """

    world_t_frame_est: Tensor  # (b, 3, 1)
    position_scale: float  # e.g. 1e-3 if units are mm
    translation_grid_basis: Tensor  # (b, 3, 3)
    rotation_grids: list[Tensor]  # R x (m_r, 3, 3)
    obj_t_frame: Tensor  # (3, 1)

    rot_idxs: list[Tensor]  # R x (b, n_r)
    pos_idxs: list[Tensor]  # R x (b, n_r)
    log_probs: list[Tensor]  # R x (b, n_r)
    expand_idxs: list[Tensor]  # R x (b, n_expanded_r)

    @property
    def recursion_depth(self):
        return len(self.rot_idxs)

    @property
    def b(self):
        return self.rot_idxs[0].shape[0]

    @property
    def device(self):
        return self.rot_idxs[0].device

    @cached_property
    def leaf_idxs(self):
        """R x (b, n_leaf_r)
        Those bins which have not been expanded are leaf nodes"""
        leaf_idxs: list[Tensor] = []
        for x, expand_idx in zip(self.rot_idxs, self.expand_idxs, strict=True):
            b, n = x.shape
            _, k = expand_idx.shape
            mask = torch.ones((b, n), device=self.device, dtype=torch.bool)
            mask.scatter_(dim=1, index=expand_idx, value=False)
            idx = mask.nonzero(as_tuple=True)[1].view(b, n - k)
            leaf_idxs.append(idx)
        return leaf_idxs

    @cached_property
    def n_leaves_per_recursion(self):
        return torch.tensor([leaf_idx.shape[1] for leaf_idx in self.leaf_idxs], device=self.device)

    @cached_property
    def n_leaves(self):
        return self.n_leaves_per_recursion.sum()

    def _get_leaves(self, xl: list[Tensor]):
        """(b, n_leaves)"""
        return torch.cat(
            [
                x[torch.arange(self.b, device=self.device).view(self.b, 1), leaf_idx]
                for x, leaf_idx in zip(xl, self.leaf_idxs, strict=True)
            ],
            dim=1,
        )

    @cached_property
    def leaf_rot_idxs(self):
        """(b, n_leaves)"""
        return self._get_leaves(self.rot_idxs)

    @cached_property
    def leaf_pos_idxs(self):
        """(b, n_leaves)"""
        return self._get_leaves(self.pos_idxs)

    @cached_property
    def leaf_log_probs(self):
        """(b, n_leaves)"""
        return self._get_leaves(self.log_probs)

    @cached_property
    def leaf_probs(self):
        """(b, n_leaves)"""
        return self.leaf_log_probs.exp()

    @cached_property
    def leaf_recursion_level(self):
        """(n_leaves,)"""
        return torch.repeat_interleave(
            torch.arange(self.recursion_depth, device=self.device),
            self.n_leaves_per_recursion,
        )

    @cached_property
    def log_bounded_se3_volume(self):
        """(b, 1)"""
        log_so3_volume: float = np.log(np.pi**2).item()
        # determinant of frame is the volume of the parallelogram spanning the grid volume
        log_r3_volume = torch.logdet(self.translation_grid_basis * self.position_scale)
        log_se3_volume = log_r3_volume + log_so3_volume
        return log_se3_volume.unsqueeze(1)

    @cached_property
    def leaf_log_volume(self):
        """(b, n_leaves)"""
        return self.log_bounded_se3_volume - log_bin_count(self.leaf_recursion_level)

    @cached_property
    def leaf_log_density(self):
        """(b, n_leaves)"""
        return self.leaf_log_probs - self.leaf_log_volume

    @cached_property
    def _world_R_frame(self):
        """R x (b, n_r, 3, 3)"""
        return [self.rotation_grids[r][rot_idx] for r, rot_idx in enumerate(self.rot_idxs)]

    @cached_property
    def _world_t_frame(self):
        """R x (b, n_r, 3, 1)"""
        return [
            translation_grid.grid2pos(
                grid=pos_idx,
                t_est=self.world_t_frame_est,
                grid_frame=self.translation_grid_basis,
                r=r + 1,
            )
            for r, pos_idx in enumerate(self.pos_idxs)
        ]

    @cached_property
    def leaf_world_t_frame(self):
        """(b, n_leaves, 3, 1)"""
        return self._get_leaves(self._world_t_frame)

    @cached_property
    def leaf_world_R_frame(self):
        """(b, n_leaves, 3, 3)"""
        return self._get_leaves(self._world_R_frame)

    @cached_property
    def leaf_world_t_obj(self):
        """(b, n_leaves, 3, 1)"""
        frame_t_obj = -self.obj_t_frame  # obj_R_frame = I
        return self.leaf_world_t_frame + self.leaf_world_R_frame @ frame_t_obj

    @property
    def leaf_world_R_obj(self):
        """(b, n_leaves, 3, 3)"""
        return self.leaf_world_R_frame  # obj_R_frame = I

    @cached_property
    def expected_world_R_frame(self) -> Tensor:
        """(b, 3, 3)"""
        x = self.leaf_world_R_frame * einops.rearrange(self.leaf_probs, "b n -> b n 1 1")
        x = x.sum(dim=1)  # (b, 3, 3)
        # orthogonal procrustes
        u, _, vh = torch.linalg.svd(x)
        s = torch.eye(3, device=self.device).unsqueeze(0).expand(self.b, 3, 3).clone()
        s[:, 2, 2] = torch.sign(torch.linalg.det(u @ vh))
        return u @ s @ vh

    @property
    def expected_world_R_obj(self):
        """(b, 3, 3)"""
        return self.expected_world_R_frame

    @cached_property
    def expected_world_t_frame(self):
        """(b, 3, 1)"""
        return (self.leaf_world_t_frame * einops.rearrange(self.leaf_probs, "b n -> b n 1 1")).sum(
            dim=1
        )

    @cached_property
    def expected_world_t_obj(self):
        """(b, 3, 1)"""
        frame_t_obj = -self.obj_t_frame  # obj_R_frame = I
        return self.expected_world_t_frame + self.expected_world_R_frame @ frame_t_obj

    @cached_property
    def _expected_frame_angle_leaf_frames(self):
        """(b, n_leaves)"""
        expected_frame_R_world = self.expected_world_R_frame.mT  # (b, 3, 3)
        expected_frame_R_leaf_frames = (
            expected_frame_R_world[:, None] @ self.leaf_world_R_frame
        )  # (b, 1, 3, 3) @ (b, n_leaf, 3, 3) = (b, n_leaf, 3, 3)
        # https://en.wikipedia.org/wiki/Rotation_matrix#Determining_the_angle
        traces = expected_frame_R_leaf_frames.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (b, n_leaf)
        angles = torch.arccos(((traces - 1) / 2).clamp_(-1, 1))  # (b, n_leaves)
        return angles

    @cached_property
    def angle_variance(self):
        """(b,)"""
        angles_sqr = self._expected_frame_angle_leaf_frames**2
        expected_squared_angles = (angles_sqr * self.leaf_probs).sum(dim=1)
        return expected_squared_angles

    @cached_property
    def _expected_R_angles(self):
        """2 x (b, n_leaves)"""
        angles, indices = torch.sort(self._expected_frame_angle_leaf_frames)  # (b, n_leaves)
        sorted_probs = self.leaf_probs.gather(dim=1, index=indices)  # (b, n_leaves)
        probs_cumsum = sorted_probs.cumsum(dim=1)  # (b, n_leaves)
        return angles, probs_cumsum

    def expected_R_confidence_interval(self, confidence_levels: Tensor):
        """Returns confidence levels in radians wrt. expected rotation. Shape (b, n_levels).
        Args:
            confidence_level: (b, n_levels)
        """
        angles, probs_cumsum = self._expected_R_angles
        indices = torch.searchsorted(probs_cumsum, confidence_levels, right=True)  # (b, n_levels)
        return angles.gather(1, indices)  # (b, n_levels)

    @cached_property
    def _expected_t_dists(self):
        """2 x (b, n_leaves)"""
        dists = torch.norm(
            self.expected_world_t_frame[:, None] - self.leaf_world_t_frame, dim=2
        ).squeeze(-1)  # (b, n_leaves)
        dists, indices = torch.sort(dists)
        sorted_probs = self.leaf_probs.gather(dim=1, index=indices)  # (b, n_leaves)
        probs_cumsum = sorted_probs.cumsum(dim=1)  # (b, n_leaves)
        return dists, probs_cumsum

    def expected_t_confidence_interval(self, confidence_levels: Tensor):
        """Returns confidence levels in radians wrt. expected rotation. Shape (b, n_levels).
        Args:
            confidence_level: (b, n_levels)
        """
        dists, probs_cumsum = self._expected_t_dists
        indices = torch.searchsorted(probs_cumsum, confidence_levels, right=True)  # (b, n_levels)
        return dists.gather(1, indices)  # (b, n_levels)

"""
We need to define a translation grid.
A regular grid with a random rotation and translation will allow a straight forward multi-view extension.
To allow an effective coarse-to-fine pyramid however, at a given recursion, it's desired that moving in any direction
in the 6-dimensional grid leads to a visual change of similar scale (such that ambiguity and samples are not dominated by one dimension).
To relate rotation resolution with in-plane translation resolution, we define:
    in_plane_grid_spacing = angle_grid_spacing * obj_radius
From a single image, there's typically more depth-ambiguity than in-plane translational ambiguity,
which motivates both extending the grid further in z than xy and having lower resolution along z.
For the crops, we aim to keep the size_appearance, s, constant from an estimated depth / detection:
    s = d * f / z
where d is the object diameter, by choosing:
    f = s_desired * z_est / d
We can see that the change in appeared size from a change in z depends on z:
    ds / dz = - d * f / z ** 2
And we can thus choose to scale the grid along z by c:
    ds / dz * c = du / dx,  u = x * f / z,  du / dx = f / z
    - d * f / z ** 2 * c = f / z
    |c| = z / d
    z_spacing = in_plane_grid_spacing * z_est / d
If we want to use the trained model in a multiview setup (with c = 1) the grid used during training should be c >= 1,
such that probability is not "lost" between bins during inference (at c = 1).
We can see that c >= 1 as long as z_est >= d, which is easily satisfied and not violated in any training sets I'm aware of.

To align the grid with the view frustum, we can sheer the grid, and sheer does not not change the volume of the cells.

Bounds and prior:

The translation grid does not need to be a cube and can have more elements along certain dimensions if desired.
Since the resolution is chosen to be relative to estimated ambiguity, for simplicity, we choose to have the same number of elements per translation dimension.
At Healpix recursion level = 0 (72 elements), the angle resolution is approx 60 deg, 60/180 * pi ~= 1, which means in_plane_grid_spacing ~= obj_radius.
Having a 2x2x2 translation grid with in_plane_grid_spacing ~= obj_radius seems reasonable, leading to 2^3 * 72 = 576 elements at SE3 rec. level 0.
This grid covers an in-plane square of sidelength d.

Sample the crop offsets from a normal distribution truncated to guarantee representation of gt in the grid
and with a low probability at the boundary, since most often, the crop should be reasonable.
Sample the depth offset similarly.

The grid is now fully defined. Hooray.
Start coding!

angle_grid_spacing is close to 1 radian at rotation recursion 0,
so for simplicity, let:
in_plane_grid_spacing = obj_radius

It would be nice to have the same nested indexing for the translation grid
We need to be able to
* Expandthis indexing (easy): x_{i+1} = x_i * 8 + j
* Go from this nested indexing to a translation (using an intermediate grid index (x, y, z))
* Go from a translation to nearest translation pixel (also using the grid index)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import Tensor


def get_translation_grid_frame(
    frame_radius: float, t_frame_est: np.ndarray, random_rotation=True, regular=False
):
    """
    Initialises a grid around an approximate translation of the spyro frame.
    When the number of recursions go to infinity, the rotation-independent coverage
    becomes the sphere with diameter one in the grid space.
    """
    plane_grid_spacing = frame_radius * 2
    assert t_frame_est.shape == (3, 1)
    x_est, y_est, z_est = t_frame_est[:, 0]

    if random_rotation:
        grid_frame = Rotation.random().as_matrix()
    else:
        grid_frame = np.eye(3)

    grid_frame *= plane_grid_spacing
    if not regular:
        # depth scale
        grid_frame[2] *= z_est / (frame_radius * 2)
        # shear
        grid_frame[0] += grid_frame[2] * x_est / z_est
        grid_frame[1] += grid_frame[2] * y_est / z_est

    return grid_frame.astype(np.float32)


def get_grid_origin(t_est: Tensor, grid_frame: Tensor) -> Tensor:
    """The grid origin is defined as the zero-corner of the base pix"""
    shape = t_est.shape[:-2]
    assert t_est.shape == (*shape, 3, 1)
    assert grid_frame.shape == (*shape, 3, 3)
    return t_est - 0.5 * grid_frame @ torch.ones(3, 1, device=t_est.device)


def get_grid_frame_r(grid_frame: Tensor, r: int) -> Tensor:
    """the frame gets divided by two at each recursion"""
    return grid_frame / (2**r)


def expand_grid(grid: Tensor) -> Tensor:  # (..., 3) -> (..., 8, 3)
    cells = torch.arange(2, device=grid.device)
    cells = torch.stack(torch.meshgrid(cells, cells, cells, indexing="ij"), dim=-1)  # (2, 2, 2, 3)
    return (grid * 2)[..., None, :] + cells.view(8, 3)


def grid2pos(grid: Tensor, t_est: Tensor, grid_frame: Tensor, r: int):
    """returns the centers of bins"""
    grid_frame_r = get_grid_frame_r(grid_frame, r)
    pos = get_grid_origin(t_est, grid_frame) + grid_frame_r @ (grid.mT + 0.5)
    return pos.mT.unsqueeze(-1)  # (b, n, 3, 1)


def pos2grid(pos: Tensor, t_est: Tensor, grid_frame: Tensor, r: int):
    assert pos.shape[-2:] == (3, 1), pos.shape
    assert t_est.shape[-2:] == (3, 1), t_est.shape
    assert grid_frame.shape[-2:] == (3, 3)
    grid_frame_r = get_grid_frame_r(grid_frame, r)
    grid_origin = get_grid_origin(t_est, grid_frame)
    grid = (grid_frame_r.inverse() @ (pos - grid_origin)).long()
    return grid.squeeze(-1)  # (b, n, 3)


### code related to pix indexing, which can be removed


def to_binary(x: Tensor, bits: int, dim=-1):
    """most significant bit to the right"""
    bitmask = 2 ** torch.arange(bits, device=x.device)
    shape = [1 for _ in range(x.ndim + 1)]
    shape[dim] = bits
    return x.unsqueeze(dim).bitwise_and(bitmask.view(*shape)).ne_(0)


def to_decimal(x: Tensor, dim=-1):
    bitmask = 2 ** torch.arange(x.shape[dim], device=x.device)
    shape = [1 for _ in range(x.ndim)]
    shape[dim] = x.shape[dim]
    return (x * bitmask.view(*shape)).sum(dim=dim)


def pix2grid(pix: Tensor, r: int, d=3):
    pix_binary = to_binary(pix, bits=r * d)  # (..., r * d: (eg. [x0, y0, x1, y1, ...]))
    pix_grid = pix_binary.view(*pix.shape, r, d)
    return to_decimal(pix_grid, dim=-2)  # (..., d)


def grid2pix(grid: Tensor, r: int):
    *shape, d = grid.shape
    grid_binary = to_binary(grid, bits=r, dim=-2)  # (..., r, d)
    pix_binary = grid_binary.view(*shape, r * d)
    return to_decimal(pix_binary)


def pix2pos(pix: Tensor, t_est: Tensor, grid_frame: Tensor, r: int):
    """returns the center position of the pixel (explaining the + 0.5)"""
    grid_frame_r = get_grid_frame_r(grid_frame, r)
    grid = pix2grid(pix, r=r).mT
    pos = get_grid_origin(t_est, grid_frame) + grid_frame_r @ (grid + 0.5)
    return pos  # (b, 3, n)


def pos2pix(pos: Tensor, t_est: Tensor, grid_frame: Tensor, r: int):
    grid_frame_r = get_grid_frame_r(grid_frame, r)
    grid_origin = get_grid_origin(t_est, grid_frame)
    grid = (grid_frame_r.inverse() @ (pos - grid_origin)).long()
    return grid2pix(grid.mT, r=r)  # (..., n)


def expand_pix(pix: Tensor):  # (...) -> (..., 8)
    return (pix * 8)[..., None] + torch.arange(8, device=pix.device)


def _main():
    import matplotlib.pyplot as plt

    t_est = torch.tensor([0.2, 0.2, 2.0]).view(3, 1)
    idxs = 0, 2
    grid_frame = torch.from_numpy(
        get_translation_grid_frame(
            frame_radius=0.5,
            t_frame_est=t_est.numpy(),
            random_rotation=False,
            regular=False,
        )
    )
    plt.scatter(*t_est[idxs,], zorder=10, c="k", marker="x")

    grid = torch.zeros(1, 3, dtype=torch.long)
    for r in range(6):
        if r > 0:
            grid = expand_grid(grid).view(8**r, 3)
        pos = grid2pos(grid, t_est, grid_frame, r)  # (n, 3, 1)
        print(grid.shape, t_est.shape, grid_frame.shape)
        pos_ = grid2pos(
            grid + (torch.rand_like(grid, dtype=torch.float) - 0.5) * 0.999,
            t_est,
            grid_frame,
            r,
        )
        grid_ = pos2grid(pos_, t_est=t_est, grid_frame=grid_frame, r=r)
        assert torch.all(grid == grid_)

        # assert torch.allclose(grid, grid_), grid
        plt.scatter(*pos.squeeze(-1).mT[idxs,])
    plt.grid()
    plt.gca().set_aspect(1)
    # print(pos.shape)
    plt.show()


def __main():
    import utils

    obj_radius = 0.5

    t_est = np.array([1, 0, 3])
    grid_frame = get_translation_grid_frame(
        obj_radius, t_est, random_rotation=False, regular=False
    )
    grid = np.stack(np.meshgrid(*([[-0.5, 0.5]] * 3), indexing="ij"), axis=-1)
    grid = grid.reshape(8, 3) @ grid_frame.T + t_est

    axes = "xyz"
    ax0, ax1 = 0, 2

    x, y = t_est[ax0], t_est[ax1]
    plt.plot([0, x], [0, y], c="k")
    plt.gca().add_patch(plt.Circle((x, y), obj_radius, color="k"))

    bound = utils.sample_truncated_normal(1000, std=0.5, trunc=1) @ grid_frame.T + t_est
    plt.scatter(bound[:, ax0], bound[:, ax1], c="r", s=1, zorder=10, alpha=0.3)

    s = 10
    num_rec = 5
    cmap = plt.get_cmap("plasma")
    for i in range(num_rec):
        plt.scatter(*grid[:, (ax0, ax1)].T, s=s, color=cmap(i / (num_rec)))
        grid_frame *= 0.5
        s *= 0.5
        grid = (
            grid.reshape(-1, 1, 3)
            + (
                np.stack(np.meshgrid(*([[-0.5, 0.5]] * 3), indexing="ij"), axis=-1).reshape(-1, 3)
                @ grid_frame.T
            ).reshape(1, -1, 3)
        ).reshape(-1, 3)

        print(len(grid))

    plt.xlabel(axes[ax0])
    plt.ylabel(axes[ax1])
    plt.gca().set_aspect(1)
    plt.show()


if __name__ == "__main__":
    _main()

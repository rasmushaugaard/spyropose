from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation


@dataclass
class SO3VisResult:
    ax: Axes
    canonical_rotation: np.ndarray
    show_marker: Callable
    scatter_tree: KDTree
    colors: np.ndarray
    alpha: np.ndarray


def visualize_so3_probabilities(
    rotations: np.ndarray,
    probabilities: np.ndarray,
    rotations_gt: np.ndarray | None = None,
    ax: Axes | None = None,
    rot_offset=np.eye(3),
    canonical_rotation=Rotation.from_euler("xyz", [0.4] * 3).as_matrix(),
    scatter_alpha=1.0,
    cmap=plt.get_cmap("hsv"),
    s=2,
    long_offset=0.0,
    lat_offset=0.0,
    fill_gt=False,
    marker_size=2000,
    marker_linewidth=2,
    gamma=1.0,
    c=None,
    scatter_zorder=10,
) -> SO3VisResult:
    assert rotations.ndim == 3 and rotations.shape[-2:] == (3, 3), (
        f"expected shape (n, 3, 3) for rotations. Got {rotations.shape}"
    )

    if rotations_gt is None:
        rotations_gt = np.empty((0, 3, 3))
    assert rotations_gt.ndim == 3, rotations_gt.shape
    assert rotations_gt.shape[1:] == (3, 3), rotations_gt.shape
    rotations_gt = np.asarray(rotations_gt)

    if fill_gt:
        marker_linewidth = marker_linewidth * 2

    if ax is None:
        fig = plt.figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111, projection="mollweide")
    assert ax.name == "mollweide", "provided Axes must have mollweide projection"

    def get_vis_rot(R):
        R = rot_offset @ R @ canonical_rotation
        xyz = R[..., 2]
        longitude = (np.arctan2(xyz[..., 0], -xyz[..., 1]) + np.pi + long_offset) % (
            2 * np.pi
        ) - np.pi
        latitude = (np.arcsin(xyz[..., 2]) + np.pi / 2 + lat_offset) % np.pi - np.pi / 2
        tilt_angles = Rotation.from_matrix(R).as_euler("zyx")[..., 0]
        # TODO: Choose a continuously varying reference frame for tilt visualization.
        return longitude, latitude, tilt_angles

    def show_marker(rotation=np.eye(3), marker="o", s=marker_size, fill=False):
        scatter = ax.scatter(
            0,
            0,
            s=s,
            zorder=11,
            edgecolors=cmap(0),
            facecolors="w" if fill else "none",
            marker=marker,
            linewidth=0 if fill else marker_linewidth,
        )

        def set_rotation(rotation):
            long, lat, tilt = get_vis_rot(rotation)
            color = cmap(0.5 + tilt / 2 / np.pi)
            scatter.set_offsets((long, lat))
            scatter.set_edgecolor(color)  # type: ignore

        set_rotation(rotation)
        return set_rotation

    if rotations_gt is not None:
        for rotation in rotations_gt:
            show_marker(rotation)
        if fill_gt:
            for rotation in rotations_gt:
                show_marker(rotation, fill=True)

    longitudes, latitudes, tilt_angles = get_vis_rot(rotations)

    ax.grid(visible=True, zorder=-10)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    alpha = (probabilities / probabilities.max()) ** gamma
    colors = cmap(0.5 + tilt_angles / 2.0 / np.pi)
    ax.scatter(
        longitudes,
        latitudes,
        s=s,
        c=colors if c is None else c,
        alpha=alpha * scatter_alpha,
        edgecolors="none",
        zorder=scatter_zorder,
    )

    return SO3VisResult(
        ax=ax,
        canonical_rotation=canonical_rotation,
        show_marker=show_marker,
        scatter_tree=KDTree(np.stack((longitudes, latitudes), axis=1)),
        colors=colors,
        alpha=alpha,
    )

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation


def visualize_so3_probabilities(
    rotations,
    probabilities,
    rotations_gt=None,
    ax=None,
    fig=None,
    display_threshold_probability=0,
    show_color_wheel=False,
    rot_offset=np.eye(3),
    canonical_rotation=Rotation.from_euler("xyz", [0.4] * 3).as_matrix(),
    scatter_alpha=1.0,
    scatterpoint_scaling=4e3,
    scatter_edgecolor="none",
    visualize_prob_by_alpha=False,
    cmap=plt.cm.hsv,
    s=5,
    long_offset=0.0,
    lat_offset=0.0,
    fill_gt=False,
    marker_size=2000,
    marker_linewidth=2,
    gamma=0.5,
    c=None,
    scatter_zorder=10,
):
    if rotations_gt is None:
        rotations_gt = np.empty((0, 3, 3))

    assert rotations_gt.ndim == 3, rotations_gt.shape
    assert rotations_gt.shape[1:] == (3, 3), rotations_gt.shape

    rotations = np.asarray(rotations)
    rotations_gt = np.asarray(rotations_gt)
    if fill_gt:
        marker_linewidth = marker_linewidth * 2

    if ax is None:
        assert fig is None
        fig = plt.figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111, projection="mollweide")

    def get_vis_rot(R):
        R = rot_offset @ R @ canonical_rotation
        xyz = R[..., 2]
        longitude = (np.arctan2(xyz[..., 0], -xyz[..., 1]) + np.pi + long_offset) % (
            2 * np.pi
        ) - np.pi
        latitude = (np.arcsin(xyz[..., 2]) + np.pi / 2 + lat_offset) % np.pi - np.pi / 2
        tilt_angles = Rotation.from_matrix(R).as_euler("zyx")[..., 0]
        # TODO: Choose a continously varying reference frame for tilt visualization.
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
            scatter.set_edgecolor(color)

        set_rotation(rotation)
        return set_rotation

    if rotations_gt is not None:
        for rotation in rotations_gt:
            show_marker(rotation)
        if fill_gt:
            for rotation in rotations_gt:
                show_marker(rotation, fill=True)

    rotation_mask = probabilities > display_threshold_probability
    longitudes, latitudes, tilt_angles = get_vis_rot(rotations[rotation_mask])

    ax.grid(True, zorder=-10)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if visualize_prob_by_alpha:
        # WIP
        alpha = probabilities[rotation_mask]
        alpha = (alpha / alpha.max()) ** gamma
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
    else:
        ax.scatter(
            longitudes,
            latitudes,
            # rlha: s is scatter point area (so facearea *is* proportional to probability)
            s=scatterpoint_scaling * probabilities[rotation_mask],
            c=cmap(0.5 + tilt_angles / 2.0 / np.pi),
            # rlha: overlapping areas can accumulate and better represent dist. with alpha < 1
            alpha=scatter_alpha,
            # rlha: the edge expands the area proportional to the circumference, not the area,
            #       and thus, the visualization were overrepresenting low-probability points.
            edgecolors=scatter_edgecolor,
        )

    return dict(
        fig=fig,
        ax=ax,
        canonical_rotation=canonical_rotation,
        show_marker=show_marker,
        scatter_tree=KDTree(np.stack((longitudes, latitudes), axis=1)),
        rotation_mask=rotation_mask,
        colors=colors,
        alpha=alpha,
    )

"""
SYMSOL I objects
* Cone
* Cylinder
* Tetrahedron
* Cube
* Icosphere

Higher number of symmetries for continuous symmetries such as cone, cylinder etc,
helps reduce variance of evaluation metric.
"""

import icosphere
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def get_z_rotations(n):
    return Rotation.from_rotvec(
        np.stack(
            np.broadcast_arrays(0, 0, np.linspace(0, 2 * np.pi, n, endpoint=False)),
            axis=1,
        )
    ).as_matrix()


def normalize(a, axis=-1):
    return a / np.linalg.norm(a, axis=axis, keepdims=True)


def normalize_mesh(mesh: trimesh.Trimesh):
    "1 m in bounding sphere diameter"
    diameter = mesh.bounding_sphere.primitive.radius * 2
    mesh = mesh.copy()
    mesh.apply_scale(1 / diameter)
    return mesh


def get_face_symmetries(mesh, sym_per_face):
    """
    assumes the object is centered around the symmetries
    calculates a frame based on each face and splits a full rotation
    into "sym_per_face" around that face's normal
    """
    vts = mesh.vertices[mesh.faces]  # (n_faces, 3 verts, 3 xyz)
    a, b, c = vts.transpose(1, 0, 2)  # 3 verts x (n_faces, 3xyz)

    # find the columns of the frames that are "the same" in the obj frame
    x = normalize(c - a)
    z = normalize(np.cross(x, b - a))
    y = np.cross(z, x)
    obj_R_frames = np.stack((x, y, z), axis=-1)

    # for each frame, add additional frames around the z axis of that frame
    frame_R_z_frames = get_z_rotations(sym_per_face)
    obj_R_frames = (obj_R_frames[:, None] @ frame_R_z_frames).reshape(-1, 3, 3)

    obj_R_obj_frames = obj_R_frames[0] @ obj_R_frames.transpose(0, 2, 1)
    syms = np.zeros((len(obj_R_obj_frames), 4, 4))
    syms[:, :3, :3] = obj_R_obj_frames
    syms[:, 3, 3] = 1
    return syms


def get_cone(radius=1, height=2.5, sections=360, n_syms=360):
    top = np.array([[0, 0, height / 2]])
    bottom = -top
    theta = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    S, C = np.sin(theta), np.cos(theta)
    vertices = np.concatenate(
        [
            np.stack(np.broadcast_arrays(S * radius, C * radius, -height / 2), axis=1),
            top,
            bottom,
        ]
    )
    side_pairs = (np.arange(sections)[:, None] + np.arange(2)) % sections
    side_faces = np.concatenate(
        (
            np.full((sections, 1), sections),
            side_pairs[:, ::-1],
        ),
        axis=1,
    )
    bottom_faces = np.concatenate(
        (
            np.full((sections, 1), sections + 1),
            side_pairs,
        ),
        axis=1,
    )
    faces = np.concatenate((side_faces, bottom_faces))
    mesh = trimesh.Trimesh(vertices, faces)

    # symmetry: rotation around z axis
    syms = np.zeros((n_syms, 4, 4))
    syms[:, :3, :3] = get_z_rotations(n_syms)
    syms[:, 3, 3] = 1

    mesh.apply_translation(-mesh.bounding_sphere.primitive.center)

    return normalize_mesh(mesh), syms


def get_cylinder(radius=1.0, height=2.0, sections=360, n_syms=360):
    mesh = trimesh.primitives.Cylinder(radius=radius, height=height, sections=sections)

    # symmetry: rotation around z axis
    syms = np.zeros((2 * n_syms, 4, 4))
    R = get_z_rotations(n_syms)
    syms[:, :3, :3] = np.concatenate(
        (
            Rotation.from_rotvec([np.pi, 0, 0]).as_matrix() @ R,
            R,
        )
    )
    syms[:, 3, 3] = 1

    return normalize_mesh(mesh), syms


def get_tetrahedron():
    vertices = np.array(
        [
            [-1, -1 / 3**0.5, -1 / 6**0.5],
            [1, -1 / 3**0.5, -1 / 6**0.5],
            [0, 2 / 3**0.5, -1 / 6**0.5],
            [0, 0, 3 / 6**0.5],
        ]
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [0, 1, 3],
            [1, 2, 3],
        ]
    )
    mesh = trimesh.Trimesh(vertices, faces)
    return normalize_mesh(mesh), get_face_symmetries(mesh, 3)


def get_cube():
    mesh = trimesh.primitives.Box()
    idx = np.unique(mesh.face_normals, axis=0, return_index=True)[1]
    mesh_ = trimesh.Trimesh(mesh.vertices, mesh.faces[idx,])
    return normalize_mesh(mesh), get_face_symmetries(mesh_, 4)


def get_icosahedral():
    mesh = trimesh.Trimesh(*icosphere.icosphere(1))
    return normalize_mesh(mesh), get_face_symmetries(mesh, 3)


obj_gen_dict = dict(
    cone=get_cone,
    cyl=get_cylinder,
    tet=get_tetrahedron,
    cube=get_cube,
    ico=get_icosahedral,
)


def _main():
    """
    For verifying / debugging the object generation.
        * visual verification that
    """
    import argparse

    import matplotlib.pyplot as plt

    from .renderer import SimpleRenderer

    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=obj_gen_dict.keys())
    name = parser.parse_args().name

    mesh, syms = obj_gen_dict[name]()

    crop_res = 224
    f = crop_res * 2
    K = np.array(((f, 0, crop_res / 2 - 0.5), (0, f, crop_res / 2 - 0.5), (0, 0, 1)))

    z = mesh.bounding_sphere.primitive.radius * 5
    base_pose = np.eye(4)
    base_pose[:3, :3] = Rotation.from_euler("xyz", [np.pi / 2, 0, 0]).as_matrix()
    base_pose[2, 3] = z

    renderer = SimpleRenderer(mesh, near=z * 0.5, far=z * 2, w=crop_res)

    n = len(syms)
    cols = int(np.ceil(n**0.5))
    rows = int(np.ceil(n / cols))
    s = 1
    axs = plt.subplots(rows, cols, figsize=(s * cols, s * rows))[1]
    axs = np.asarray(axs)

    syms_dist = np.linalg.norm(
        syms.reshape(1, -1, 16) - syms.reshape(-1, 1, 16), axis=2
    )
    syms_dist[np.arange(n), np.arange(n)] = 1

    for sym, ax in zip(syms, axs.reshape(-1)):
        pose = base_pose @ sym
        ax.imshow(renderer.render(K, pose[:3, :3], t=pose[:3, 3:])[..., :3])

    for ax in axs.reshape(-1):
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    assert (syms_dist > 1e-3).all()
    assert np.allclose(mesh.bounding_sphere.primitive.center, 0)


if __name__ == "__main__":
    _main()

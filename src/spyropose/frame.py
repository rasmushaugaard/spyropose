from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class SpyroFrame:
    obj_t_frame: tuple[float, float, float]  # position of spyro frame in obj frame
    radius: float  # frame radius - determines crop size and positional bounds
    padding_ratio: float = 1.5  # determines nominal size of object in crop

    @staticmethod
    def from_mesh_bounding_sphere(mesh_original: trimesh.Trimesh):
        sphere = mesh_original.bounding_sphere.primitive
        return SpyroFrame(obj_t_frame=tuple(sphere.center), radius=sphere.radius)

    def get_mesh_in_frame(self, mesh_original: trimesh.Trimesh):
        mesh = mesh_original.copy()
        mesh.apply_translation(-np.array(self.obj_t_frame))
        return mesh

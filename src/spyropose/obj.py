import json
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from spyropose import utils
from spyropose.frame import SpyroFrame


def init_keypoints_from_mesh(
    mesh: trimesh.Trimesh, frame: SpyroFrame, n_keypoints: int, tol=1.001
):
    frame_vertices = mesh.vertices - np.asarray(frame.obj_t_frame)
    mask = np.linalg.norm(frame_vertices, axis=1) < frame.radius * tol
    frame_vertices = frame_vertices[mask]
    return utils.farthest_point_sampling(frame_vertices, n_keypoints)  # (n, 3)


class SpyroObjectConfig:
    def __init__(
        self,
        dataset: str,
        obj: str,
        frame: SpyroFrame | None = None,
        crop_res: int = 224,
        recursion_depth: int = 7,
        keypoints: int | tuple[tuple[float, float, float], ...] = 16,
    ):
        self.dataset = dataset
        self.obj = obj
        self.crop_res = crop_res
        self.recursion_depth = recursion_depth

        if frame is not None:
            self.frame = frame
        else:
            self.frame = SpyroFrame.from_mesh_bounding_sphere(self.mesh)

        if isinstance(keypoints, int):
            self.keypoints = init_keypoints_from_mesh(
                mesh=self.mesh, frame=self.frame, n_keypoints=keypoints
            )
        else:
            self.keypoints = np.asarray(keypoints)

    @property
    def root_dir(self):
        root = Path("data/bop") / self.dataset
        assert root.is_dir(), root
        return root

    @property
    def meshes_dir(self):
        return self.root_dir / "models"

    @cached_property
    def mesh_info(self):
        with (self.meshes_dir / "models_info.json").open() as f:
            meshes_info: dict[str, dict[str, Any]] = json.load(f)

        try:
            # by index (bop format)
            obj_id = int(self.obj)
            mesh_path = self.meshes_dir / f"obj_{obj_id:06d}.ply"
        except ValueError:
            # by name
            mesh_path = self.meshes_dir / f"{self.obj}"
            mesh_name2id = dict()
            for idx_str, mesh_info in meshes_info.items():
                if "mesh_name" in mesh_info:
                    mesh_name2id[mesh_info.get("mesh_name")] = int(idx_str)
            if mesh_path.name not in mesh_name2id:
                raise RuntimeError(
                    f"Object name '{self.obj}' not found. "
                    f"Available names: {list(mesh_name2id.keys())}"
                )
            obj_id = mesh_name2id[mesh_path.name]

        assert mesh_path.exists(), f"{mesh_path}"
        mesh_info = meshes_info[str(obj_id)]
        mesh_info["path"] = mesh_path
        mesh_info["id"] = obj_id
        return mesh_info

    @property
    def mesh_path(self) -> Path:
        return self.mesh_info["path"]

    @property
    def obj_id(self) -> int:
        return self.mesh_info["id"]

    @cached_property
    def mesh(self) -> trimesh.Trimesh:
        return trimesh.load_mesh(self.mesh_path)

    @cached_property
    def mesh_frame(self) -> trimesh.Trimesh:
        return self.frame.get_mesh_in_frame(self.mesh)

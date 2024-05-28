import json
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import torch.utils.data
import trimesh
from tqdm import tqdm

from .config import DatasetConfig


class BopInstanceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root: Path,
        obj_id: int,
        pbr: bool,
        test: bool,
        cfg: DatasetConfig,
        auxs: Sequence["BopInstanceAux"] = tuple(),
        scene_ids=None,
        min_visib_fract=0.1,
        min_px_count_visib=1024,
        offset_pbr_half_px=False,
    ):
        self.pbr, self.test, self.cfg = pbr, test, cfg
        if pbr:
            assert not test
            self.data_folder = dataset_root / "train_pbr"
            self.img_folder = "rgb"
            self.depth_folder = "depth"
            self.img_ext = "jpg"
            self.depth_ext = "png"
        else:
            self.data_folder = dataset_root / (
                cfg.test_folder if test else cfg.train_folder
            )
            self.img_folder = cfg.img_folder
            self.depth_folder = cfg.depth_folder
            self.img_ext = cfg.img_ext
            self.depth_ext = cfg.depth_ext

        self.mesh = trimesh.load_mesh(
            dataset_root / cfg.model_folder / f"obj_{obj_id:06d}.ply"
        )
        self.obj_radius = self.mesh.bounding_sphere.primitive.radius
        self.obj_center = self.mesh.bounding_sphere.primitive.center.reshape(3, 1)

        self.auxs = auxs
        self.instances = []
        if scene_ids is None:
            scene_ids = sorted([int(p.name) for p in self.data_folder.glob("*")])
        for scene_id in tqdm(scene_ids, "loading crop info"):
            scene_folder = self.data_folder / f"{scene_id:06d}"
            scene_gt = json.load((scene_folder / "scene_gt.json").open())
            scene_gt_info = json.load((scene_folder / "scene_gt_info.json").open())
            scene_camera = json.load((scene_folder / "scene_camera.json").open())

            for img_id, poses in scene_gt.items():
                img_info = scene_gt_info[img_id]
                K = np.array(scene_camera[img_id]["cam_K"]).reshape((3, 3)).copy()
                if pbr and offset_pbr_half_px:
                    warnings.warn(
                        "Altering camera matrix,"
                        " since PBR camera matrix doesnt seem to be correct"
                    )
                    K[:2, 2] -= 0.5

                for pose_idx, pose in enumerate(poses):
                    if pose["obj_id"] != obj_id:
                        continue
                    pose_info = img_info[pose_idx]
                    if pose_info["visib_fract"] < min_visib_fract:
                        continue
                    if pose_info["px_count_visib"] < min_px_count_visib:
                        continue

                    bbox_visib = pose_info["bbox_visib"]
                    bbox_obj = pose_info["bbox_obj"]

                    cam_R_obj = np.array(pose["cam_R_m2c"]).reshape(3, 3)
                    cam_t_obj = np.array(pose["cam_t_m2c"]).reshape(3, 1)
                    cam_t_ctr = cam_R_obj @ self.obj_center + cam_t_obj

                    self.instances.append(
                        dict(
                            scene_id=scene_id,
                            img_id=int(img_id),
                            K=K,
                            obj_id=obj_id,
                            pose_idx=pose_idx,
                            bbox_visib=bbox_visib,
                            bbox_obj=bbox_obj,
                            cam_R_obj=cam_R_obj,
                            cam_t_obj=cam_t_obj,
                            cam_t_ctr=cam_t_ctr,
                            obj_radius=self.obj_radius,
                        )
                    )

        for aux in self.auxs:
            aux.init(self)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        instance = self.instances[i].copy()
        for aux in self.auxs:
            instance = aux(instance, self)
        return instance


class BopInstanceAux:
    def init(self, dataset: BopInstanceDataset):
        pass

    def __call__(self, data: dict, dataset: BopInstanceDataset) -> dict:
        pass

import json

import numpy as np
import torch.utils.data
from tqdm import tqdm

from .auxs import get_auxs
from .cfg import SpyroDataConfig


class SpyroDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: SpyroDataConfig):
        self.auxs = get_auxs(cfg)
        self.instances: list[dict] = []
        obj_t_frame = np.asarray(cfg.obj.frame.obj_t_frame).reshape(3, 1)

        for scene_id in tqdm(cfg.scene_ids, "loading crop info"):
            scene_folder = cfg.split_dir / f"{scene_id:06d}"
            scene_gt = json.load((scene_folder / "scene_gt.json").open())
            scene_gt_info = json.load((scene_folder / "scene_gt_info.json").open())
            scene_camera = json.load((scene_folder / "scene_camera.json").open())

            for img_id, poses in scene_gt.items():
                img_info = scene_gt_info[img_id]
                K = np.array(scene_camera[img_id]["cam_K"]).reshape((3, 3)).copy()
                for pose_idx, pose in enumerate(poses):
                    if pose["obj_id"] != cfg.obj.obj_id:
                        continue
                    pose_info = img_info[pose_idx]
                    if pose_info["visib_fract"] < cfg.min_visib_fract:
                        continue
                    if pose_info["px_count_visib"] < cfg.min_px_count_visib:
                        continue

                    bbox_visib = pose_info["bbox_visib"]
                    bbox_obj = pose_info["bbox_obj"]

                    cam_R_obj = np.array(pose["cam_R_m2c"]).reshape(3, 3)
                    cam_t_obj = np.array(pose["cam_t_m2c"]).reshape(3, 1)
                    cam_t_frame = cam_t_obj + cam_R_obj @ obj_t_frame

                    self.instances.append(
                        dict(
                            scene_id=scene_id,
                            img_id=int(img_id),
                            K=K,
                            obj_id=cfg.obj.obj_id,
                            pose_idx=pose_idx,
                            bbox_visib=bbox_visib,
                            bbox_obj=bbox_obj,
                            cam_R_obj=cam_R_obj,
                            cam_t_obj=cam_t_obj,
                            cam_t_frame=cam_t_frame,
                            frame_radius=cfg.obj.frame.radius,
                        )
                    )

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        instance = self.instances[i].copy()
        for aux in self.auxs:
            instance = aux(instance)
        return instance

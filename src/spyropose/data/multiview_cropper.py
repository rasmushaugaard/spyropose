from typing import Sequence

import cv2
import numpy as np

from . import auxs

# TODO


class MultiViewCropper:
    def __init__(self, dataset, obj_radius, crop_res=224):
        self.dataset = dataset
        self.rgb_loader = auxs.RgbLoader()
        self.crop_res = crop_res
        self.obj_radius = obj_radius

    def get(
        self,
        scene_id: int,
        world_p_obj_est: np.ndarray,
        img_ids: Sequence[int],
        cams_t_world: np.ndarray,
        Ks: Sequence[np.ndarray],
    ):
        cams_p_obj_est = (
            cams_t_world[:, :3, :3] @ world_p_obj_est + cams_t_world[:, :3, 3:]
        )
        imgs, K_crops = [], []
        for img_id, cam_t_obj_est, K in zip(img_ids, cams_p_obj_est, Ks):
            img = self.rgb_loader(
                dict(
                    scene_id=scene_id,
                    img_id=img_id,
                ),
                self.dataset,
            )["rgb"]
            h, w = img.shape[:2]
            cm = auxs.calculate_crop_matrix(
                t_est=cam_t_obj_est,
                crop_res=self.crop_res,
                obj_radius=self.obj_radius,
                padding=1.5,
                K=K,
                random_rotation=False,
                h=h,
                w=w,
            )
            img = cv2.warpAffine(
                src=img,
                M=cm["M_crop"][:2],
                dsize=(self.crop_res, self.crop_res),
                flags=cv2.INTER_LINEAR,
            )
            img = img.astype(np.float32) / 255.0
            imgs.append(img)
            K_crops.append(cm["K_crop"])
        return np.stack(imgs), np.stack(K_crops)

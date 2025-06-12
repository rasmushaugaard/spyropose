import json

import albumentations as A
import cv2
import numpy as np
import torch.utils.data
from matplotlib import pyplot as plt

from ..data.cfg import SpyroDataConfig


class SpyroDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: SpyroDataConfig):
        self.cfg = cfg
        self.data = []

        for scene_id in cfg.scene_ids:
            scene_path = cfg.split_dir / f"{scene_id:06d}"
            with (scene_path / "scene_camera.json").open() as f:
                scene_camera = json.load(f)
            with (scene_path / "scene_gt.json").open() as f:
                scene_gt = json.load(f)
            with (scene_path / "scene_gt_info.json").open() as f:
                scene_gt_info = json.load(f)
            for img_idx, cam in scene_camera.items():
                self.data.append((
                    scene_path,
                    int(img_idx),
                    np.asarray(cam["cam_K"]).reshape(3, 3),
                    scene_gt[img_idx],
                    scene_gt_info[img_idx],
                ))

        if cfg.img_aug.enabled:
            ia = cfg.img_aug
            self.augs = A.Compose([
                A.CoarseDropout(),
                ia.color_jitter_aug(),
                A.GaussianBlur(blur_limit=(1, 3)),
                A.ISONoise(),
                A.GaussNoise(),
                A.GaussianBlur(blur_limit=(1, 3)),
            ])
        else:
            self.augs = A.Compose([])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        scene_path, img_idx, K, img_gt, img_gt_info = self.data[i]
        img_name = f"{img_idx:06d}.{self.cfg.img_ext}"
        img = cv2.imread(str((scene_path / "rgb" / img_name)), cv2.IMREAD_COLOR_RGB)
        img = self.augs(image=img)["image"]
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        # normalization is part of the model

        obj_t_frame = np.asarray(self.cfg.obj.frame.obj_t_frame).reshape(3, 1)
        f = abs(np.linalg.det(K[:2, :2])) ** 0.5

        bboxes = []
        targets = []
        for inst_gt, inst_info in zip(img_gt, img_gt_info):
            if inst_gt["obj_id"] != self.cfg.obj.obj_id:
                continue
            if inst_info["px_count_visib"] < self.cfg.min_px_count_visib:
                continue
            if inst_info["visib_fract"] < self.cfg.min_visib_fract:
                continue
            l, t, w, h = inst_info["bbox_obj"]

            cam_R_obj = np.asarray(inst_gt["cam_R_m2c"]).reshape(3, 3)
            cam_t_obj = np.asarray(inst_gt["cam_t_m2c"]).reshape(3, 1)

            cam_t_frame = cam_t_obj + cam_R_obj @ obj_t_frame
            z = cam_t_frame[2, 0]

            p = K @ cam_t_frame
            p = p[:2, 0] / p[2, 0]

            half_bbox_size = self.cfg.obj.frame.radius * f / z
            l, t = p - half_bbox_size
            r, b = p + half_bbox_size
            bboxes.append((l, t, r, b))
            # targets.append(inst_gt["obj_id"])
            targets.append(1)

        return (
            img,
            np.asarray(bboxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(targets, dtype=np.int64),
        )


def _main(cfg: SpyroDataConfig, annotate=True):
    dataset = SpyroDetectionDataset(cfg)
    for i in range(len(dataset)):
        img, bboxes, targets = dataset[i]
        img = img.transpose(1, 2, 0).copy()
        cmap = plt.get_cmap("tab10")

        for bbox, target in zip(bboxes, targets):
            left, top, right, bottom = np.round(bbox).astype(int)
            if annotate:
                c = tuple(cmap(target % cmap.N))
                cv2.rectangle(img, (left, top), (right, bottom), c)
                cv2.putText(
                    img=img,
                    text=f"{target}",
                    org=(left + 2, bottom - 2),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1.0,
                    color=c,
                )

        cv2.imshow("", img)
        if cv2.waitKey() == ord("q"):
            break


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.auto_cli(_main)

from dataclasses import dataclass
from typing import Set

import albumentations as A
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ... import rotation_grid, translation_grid, utils
from . import tfms
from .instance import BopInstanceAux, BopInstanceDataset


class RgbLoader(BopInstanceAux):
    def __init__(self, copy=False):
        self.copy = copy

    def __call__(self, inst: dict, dataset: BopInstanceDataset) -> dict:
        scene_id, img_id = inst["scene_id"], inst["img_id"]
        fname = f"{img_id:06d}.{dataset.img_ext}"
        fp = dataset.data_folder / f"{scene_id:06d}" / dataset.img_folder / fname
        rgb = cv2.imread(str(fp), cv2.IMREAD_COLOR)[..., ::-1]
        assert rgb is not None
        inst["rgb"] = rgb.copy() if self.copy else rgb
        return inst


_default_interpolation_methods = (
    cv2.INTER_NEAREST,
    cv2.INTER_LINEAR,
    cv2.INTER_AREA,
    cv2.INTER_CUBIC,
)


class RandomRotatedMaskCrop(BopInstanceAux):
    def __init__(
        self,
        crop_res: int,
        grid_random_rotate=True,
        regular_grid=False,
        translation_std=0.1,
        padding=1.5,
        rgb_interpolation=_default_interpolation_methods,
        random_rotation=True,
    ):
        self.crop_res = crop_res
        self.rgb_interpolation = rgb_interpolation
        self.grid_random_rotate = grid_random_rotate
        self.grid_regular = regular_grid
        self.translation_std = translation_std
        self.padding = padding
        self.definition_aux = RandomRotatedMaskCropDefinition(self)
        self.apply_aux = RandomRotatedMaskCropApply(self)
        self.random_rotation = random_rotation

    def __call__(self, inst: dict, _) -> dict:
        inst = self.definition_aux(inst, _)
        inst = self.apply_aux(inst, _)
        return inst


def calculate_crop_matrix(
    t_est, crop_res, obj_radius, padding, K, random_rotation, h, w
):
    # get the intrinsics of the crop around the estimated position
    #   scale: f * obj_diameter * pad_multiplier / z = res
    f = crop_res / (obj_radius * 2 * padding) * t_est[2, 0]
    #   center: fx/z + cx = res / 2 - 0.5
    c = crop_res / 2 - 0.5 - f * t_est[:2, 0] / t_est[2, 0]
    K_des = np.array(
        (
            (f, 0, c[0]),
            (0, f, c[1]),
            (0, 0, 1),
        )
    ).astype(np.float32)
    # move center to origin
    K_des[:2, 2] -= crop_res / 2 - 0.5

    M_crop = K_des @ np.linalg.inv(K)

    # random rotation
    theta = np.random.uniform(0.0, 2 * np.pi) if random_rotation else 0
    S, C = np.sin(theta), np.cos(theta)
    Rz = np.array(
        (
            (C, -S, 0),
            (S, C, 0),
            (0, 0, 1),
        )
    )
    M_crop = Rz @ M_crop
    # and move origin to center
    M_crop[:2, 2] += (crop_res / 2 - 0.5) * M_crop[2, 2]

    # calculate axis aligned bounding box (AABB) in the original image of the rotated
    # crop to allow only applying image augmentations on the AABB
    mi, ma = -0.5, crop_res - 0.5
    corners = (
        np.linalg.inv(M_crop)
        @ np.array(((mi, mi, 1), (mi, ma, 1), (ma, ma, 1), (ma, mi, 1))).T
    )
    corners = corners[:2] / corners[2:]
    left, top = np.floor(corners.min(axis=1)).astype(int)
    right, bottom = np.ceil(corners.max(axis=1)).astype(int)
    AABB_crop = max(0, left), max(0, top), min(right, w - 1), min(bottom, h - 1)

    return dict(
        M_crop=M_crop.astype(np.float32),
        K_crop=(M_crop @ K).astype(np.float32),
        AABB_crop=AABB_crop,
    )


class RandomRotatedMaskCropDefinition(BopInstanceAux):
    def __init__(self, parent: RandomRotatedMaskCrop):
        self.p = parent

    def __call__(self, inst: dict, _) -> dict:
        # get grid frame based on true position
        t_ctr = inst["cam_t_ctr"]
        obj_radius = inst["obj_radius"]

        t_grid_frame = translation_grid.get_translation_grid_frame(
            obj_radius=obj_radius,
            t_est=t_ctr,
            random_rotation=self.p.grid_random_rotate,
            regular=self.p.grid_regular,
        )  # (3, 3)
        # sample offset within the rotation-independent grid region
        # this offset serves to simulate a 2D detector with a depth estimate
        t_offset = utils.sample_truncated_normal(
            n=1, std=self.p.translation_std, trunc=0.5
        )[0, :, None]
        t_offset = t_grid_frame @ t_offset
        t_ctr_est = t_ctr + t_offset
        inst["t_ctr_est"] = t_ctr_est
        inst["t_grid_frame"] = t_grid_frame

        h, w = inst["rgb"].shape[:2]
        for key, val in calculate_crop_matrix(
            t_est=t_ctr_est,
            crop_res=self.p.crop_res,
            obj_radius=obj_radius,
            padding=self.p.padding,
            K=inst["K"],
            random_rotation=self.p.random_rotation,
            h=h,
            w=w,
        ).items():
            inst[key] = val
        return inst


class RandomRotatedMaskCropApply(BopInstanceAux):
    def __init__(self, parent: RandomRotatedMaskCrop):
        self.p = parent

    def __call__(self, inst: dict, _) -> dict:
        inst["rgb_crop"] = cv2.warpPerspective(
            inst["rgb"],
            inst["M_crop"],
            (self.p.crop_res, self.p.crop_res),
            flags=np.random.choice(self.p.rgb_interpolation),
        )
        return inst


class TransformsAux(BopInstanceAux):
    def __init__(self, tfms, key: str, crop_key=None):
        self.key = key
        self.tfms = tfms
        self.crop_key = crop_key

    def __call__(self, inst: dict, _) -> dict:
        if self.crop_key is not None:
            left, top, right, bottom = inst[self.crop_key]
            img_slice = slice(top, bottom), slice(left, right)
        else:
            img_slice = slice(None)
        img = inst[self.key]
        img[img_slice] = self.tfms(image=img[img_slice])["image"]
        return inst


class NormalizeAux(BopInstanceAux):
    def __init__(self, recursion_depth: int, random_offset_rotation=False):
        self.random_offset_rotation = random_offset_rotation
        self.rlast = recursion_depth - 1

    def __call__(self, inst: dict, _) -> dict:
        inst = dict(
            img=(inst["rgb_crop"].astype(np.float32) / 255.0).transpose((2, 0, 1)),
            K=inst["K_crop"].astype(np.float32),
            R=inst["cam_R_obj"].astype(np.float32),
            t=inst["cam_t_ctr"].astype(np.float32),
            t_est=inst["t_ctr_est"].astype(np.float32),
            t_grid_frame=inst["t_grid_frame"].astype(np.float32),
        )
        if self.random_offset_rotation:
            R_offset = Rotation.random().as_matrix()
        else:
            R_offset = np.eye(3)
        R_offset = R_offset.astype(np.float32)
        inst["R_offset"] = R_offset
        inst[f"rot_idx_target_{self.rlast}"] = rotation_grid.get_closest_pix(
            R_offset @ inst["R"][None], self.rlast
        )
        return inst


class KeyFilterAux(BopInstanceAux):
    def __init__(self, keys=Set[str]):
        self.keys = keys

    def __call__(self, inst: dict, _) -> dict:
        return {k: v for k, v in inst.items() if k in self.keys}


class Identity(BopInstanceAux):
    def __call__(self, inst: dict, _) -> dict:
        return inst


@dataclass
class ImgAugConfig:
    enabled = True
    cj_p = 1.0
    cj_hue = 0.1
    cj_brightness = 0.5
    cj_contrast = 0.5
    cj_saturation = 0.5


def get_auxs(
    crop_res: int,
    recursion_depth: int,
    img_aug_cfg: ImgAugConfig,
):
    random_crop_aux = RandomRotatedMaskCrop(
        crop_res=crop_res,
        random_rotation=img_aug_cfg.enabled,
    )

    auxs = [
        RgbLoader(),
        random_crop_aux.definition_aux,
    ]

    if img_aug_cfg.enabled:
        auxs.append(
            TransformsAux(
                key="rgb",
                crop_key="AABB_crop",
                tfms=A.Compose(
                    [
                        A.GaussianBlur(blur_limit=(1, 3)),
                        A.ISONoise(),
                        A.GaussNoise(),
                        tfms.DebayerArtefacts(),
                        tfms.Unsharpen(),
                        A.CLAHE(),
                        A.GaussianBlur(blur_limit=(1, 3)),
                    ]
                ),
            )
        )

    auxs.append(random_crop_aux.apply_aux)

    if img_aug_cfg.enabled:
        auxs.append(
            TransformsAux(
                key="rgb_crop",
                tfms=A.Compose(
                    [
                        A.CoarseDropout(
                            max_height=16, max_width=16, min_width=8, min_height=8
                        ),
                        A.ColorJitter(
                            p=img_aug_cfg.cj_p,
                            hue=img_aug_cfg.cj_hue,
                            brightness=img_aug_cfg.cj_brightness,
                            contrast=img_aug_cfg.cj_contrast,
                            saturation=img_aug_cfg.cj_saturation,
                        ),
                    ]
                ),
            )
        )

    auxs.append(
        NormalizeAux(
            recursion_depth=recursion_depth,
            random_offset_rotation=img_aug_cfg.enabled,
        )
    )
    return auxs

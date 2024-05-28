from pathlib import Path

import albumentations as A

from . import auxs, tfms
from .config import config
from .instance import BopInstanceDataset


def get_auxs(
    crop_res,
    recursion_depth: int,
    image_augmentations=False,
    cj_p=1.0,
    cj_hue=0.1,
    cj_brightness=0.5,
    cj_contrast=0.5,
    cj_saturation=0.5,
    random_offset_rotation=False,
    regular_grid=False,
):
    random_crop_aux = auxs.RandomRotatedMaskCrop(
        crop_res=crop_res,
        regular_grid=regular_grid,
        random_rotation=image_augmentations,
    )
    return [
        auxs.RgbLoader(),
        random_crop_aux.definition_aux,
        auxs.TransformsAux(
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
        if image_augmentations
        else auxs.Identity(),
        random_crop_aux.apply_aux,
        auxs.TransformsAux(
            key="rgb_crop",
            tfms=A.Compose(
                [
                    A.CoarseDropout(
                        max_height=16, max_width=16, min_width=8, min_height=8
                    ),
                    A.ColorJitter(
                        p=cj_p,
                        hue=cj_hue,
                        brightness=cj_brightness,
                        contrast=cj_contrast,
                        saturation=cj_saturation,
                    ),
                ]
            ),
        )
        if image_augmentations
        else auxs.Identity(),
        auxs.NormalizeAux(
            recursion_depth=recursion_depth,
            random_offset_rotation=random_offset_rotation,
        ),
    ]


def get_bop_dataset(
    dataset_name: str,
    name: str,
    split: str,
    recursion_depth: int,
    crop_res=224,
    scene_ids_train=list(range(0, 48)),
    scene_ids_valid=list(range(48, 50)),
    image_augmentations=False,
    cj_p=1.0,
    cj_hue=0.1,
    cj_brightness=0.5,
    cj_contrast=0.5,
    cj_saturation=0.5,
    regular_grid=False,
    random_offset_rotation=True,
):
    obj_id = int(name)

    data_kwargs = dict(
        dataset_root=Path("data") / "bop" / dataset_name,
        cfg=config[dataset_name],
        pbr=True,
        test=False,
        obj_id=obj_id,
    )
    aux_kwargs = dict(
        crop_res=crop_res,
        regular_grid=regular_grid,
        recursion_depth=recursion_depth,
    )
    if split == "train":
        return BopInstanceDataset(
            scene_ids=scene_ids_train,
            auxs=get_auxs(
                image_augmentations=image_augmentations,
                cj_p=cj_p,
                cj_brightness=cj_brightness,
                cj_contrast=cj_contrast,
                cj_saturation=cj_saturation,
                cj_hue=cj_hue,
                random_offset_rotation=random_offset_rotation,
                **aux_kwargs,
            ),
            **data_kwargs,
        )
    if split == "valid":
        return BopInstanceDataset(
            scene_ids=scene_ids_valid, auxs=get_auxs(**aux_kwargs), **data_kwargs
        )
    if split == "test":
        data_kwargs["pbr"] = False
        data_kwargs["test"] = True
        return BopInstanceDataset(auxs=get_auxs(**aux_kwargs), **data_kwargs)

    raise RuntimeError()

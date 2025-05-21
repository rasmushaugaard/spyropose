from dataclasses import dataclass
from pathlib import Path

from .auxs import ImgAugConfig, get_auxs
from .instance import BopInstanceDataset


@dataclass
class DatasetConfig:
    root_path: Path
    model: int | str
    scene_id_range: tuple[int, int]
    image_folder = "train_pbr"
    models_folder = "models"
    img_aug_config = ImgAugConfig()

    @property
    def model_path(self):
        models_path = 


def get_bop_dataset(cfg: DatasetConfig):
    obj_id = int(name)

    data_kwargs = dict(
        dataset_root=Path("data") / "bop" / dataset_name,
        cfg=cfg[dataset_name],
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

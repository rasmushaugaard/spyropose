from dataclasses import dataclass, field
from functools import cached_property

import albumentations as A

from ..obj import SpyroObjectConfig


@dataclass
class ImgAugConfig:
    enabled: bool = True
    cj_p: float = 1.0
    cj_hue: float = 0.1
    cj_brightness: float = 0.5
    cj_contrast: float = 0.5
    cj_saturation: float = 0.5

    def color_jitter_aug(self):
        return A.ColorJitter(
            brightness=self.cj_brightness,
            contrast=self.cj_contrast,
            saturation=self.cj_saturation,
            hue=self.cj_hue,
        )


@dataclass
class SpyroDataConfig:
    obj: SpyroObjectConfig
    split_dir_name: str = "train_pbr"
    scene_id_range: tuple[int, int] | None = None
    img_aug: ImgAugConfig = field(default_factory=lambda: ImgAugConfig())
    min_visib_fract: float = 0.1
    min_px_count_visib: int = 1024
    img_ext: str = "jpg"

    @property
    def split_dir(self):
        return self.obj.root_dir / self.split_dir_name

    @cached_property
    def scene_ids(self):
        if self.scene_id_range is None:
            return tuple(sorted([int(p.name) for p in self.split_dir.glob("*")]))
        else:
            return tuple(range(*self.scene_id_range))

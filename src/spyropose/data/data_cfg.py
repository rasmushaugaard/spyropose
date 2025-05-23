import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path


@dataclass
class ImgAugConfig:
    enabled: bool = True
    cj_p: float = 1.0
    cj_hue: float = 0.1
    cj_brightness: float = 0.5
    cj_contrast: float = 0.5
    cj_saturation: float = 0.5


@dataclass
class DatasetConfig:
    root_dir: Path
    obj: int | str
    scene_id_range: tuple[int, int] = None
    sub_dir_name: str = "train_pbr"
    img_ext: str = "jpg"
    models_dir_name: str = "models"
    img_aug_cfg: ImgAugConfig = field(default_factory=lambda: ImgAugConfig())
    crop_res: int = 224
    recursion_depth: int = 6  # more a model detail which the dataset depends on?
    min_visib_fract: float = 0.2
    min_px_count_visib: int = 512

    @property
    def models_dir(self):
        return self.root_dir / self.models_dir_name

    @property
    def sub_dir(self):
        return self.root_dir / self.sub_dir_name

    @cached_property
    def scene_ids(self):
        if self.scene_id_range is None:
            return tuple(sorted([int(p.name) for p in self.sub_dir.glob("*")]))
        else:
            return tuple(range(*self.scene_id_range))

    @cached_property
    def model_info(self):
        with (self.models_dir / "models_info.json").open() as f:
            models_info: dict[str, dict[str]] = json.load(f)

        try:
            # by index (bop format)
            obj_id = int(self.obj)
            model_path = self.models_dir / f"obj_{obj_id:06d}.ply"
        except ValueError:
            # by name
            model_path = self.models_dir / f"{self.obj}"
            mesh_name2id = dict()
            for idx_str, model_info in models_info.items():
                if "mesh_name" in model_info:
                    mesh_name2id[model_info.get("mesh_name")] = int(idx_str)
            if model_path.name not in mesh_name2id:
                raise RuntimeError(
                    f"Model name '{self.obj}' not found. "
                    f"Available names: {list(mesh_name2id.keys())}"
                )
            obj_id = mesh_name2id[model_path.name]

        assert model_path.exists(), f"{model_path}"
        model_info = models_info[str(obj_id)]
        model_info["path"] = model_path
        model_info["id"] = obj_id
        return model_info

    @property
    def model_path(self) -> Path:
        return self.model_info["path"]

    @property
    def obj_id(self) -> int:
        return self.model_info["id"]

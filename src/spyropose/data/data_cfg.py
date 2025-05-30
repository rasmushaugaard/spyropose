import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from ..frame import SpyroFrame


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
    dataset: str
    obj: str
    frame: SpyroFrame = None
    scene_id_range: tuple[int, int] = None
    sub_dir_name: str = "train_pbr"
    img_ext: str = "jpg"
    meshes_dir_name: str = "models"
    img_aug_cfg: ImgAugConfig = field(default_factory=lambda: ImgAugConfig())
    crop_res: int = 224
    recursion_depth: int = 7  # more a model detail
    min_visib_fract: float = 0.2
    min_px_count_visib: int = 512

    @property
    def root_dir(self):
        return Path("data/bop") / self.dataset

    @property
    def meshes_dir(self):
        return self.root_dir / self.meshes_dir_name

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
    def mesh_info(self):
        with (self.meshes_dir / "models_info.json").open() as f:
            meshes_info: dict[str, dict[str]] = json.load(f)

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

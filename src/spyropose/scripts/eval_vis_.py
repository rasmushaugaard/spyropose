from pathlib import Path

import jsonargparse

from ..data.cfg import ImgAugConfig, SpyroDataConfig
from ..model import SpyroPoseModel


def main():
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_class_arguments(
        SpyroDataConfig, "data", skip={"obj"}, default={"img_aug.enabled": False}
    )

    cfg = parser.parse_args()

    model = SpyroPoseModel.load_from_checkpoint(cfg.model_path, cfg.device)
    cfg.data.img_aug = ImgAugConfig(**cfg.data.img_aug)
    cfg.data = SpyroDataConfig(obj=model.cfg.obj, **cfg.data)


if __name__ == "__main__":
    main()

import jsonargparse
import numpy as np
import pytorch_lightning as pl
import torch.utils.data
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers.wandb import WandbLogger

from .. import utils
from ..data.cfg import SpyroDataConfig
from ..obj import SpyroObjectConfig
from .data import SpyroDetectionDataset
from .model import SpyroDetector


def collate_fn(batch):
    imgs, bboxes, targets = zip(*batch)
    imgs = torch.from_numpy(np.stack(imgs))
    bboxes = [torch.from_numpy(v) for v in bboxes]
    targets = [torch.from_numpy(v) for v in targets]
    return imgs, bboxes, targets


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(SpyroObjectConfig, "obj")
    parser.add_class_arguments(SpyroDetector, "model")
    parser.add_class_arguments(SpyroDataConfig, "data_train")
    parser.add_class_arguments(SpyroDataConfig, "data_valid")
    parser.add_class_arguments(
        pl.Trainer,
        "trainer",
        default=dict(max_steps=20_000, precision="16-mixed"),
        skip={"logger", "callbacks"},
        instantiate=False,
    )

    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--wandb_project", type=str, default="spyropose_detector")
    parser.add_argument("--debug", action="store_true")

    for key in "min_visib_fract", "min_px_count_visib":
        parser.link_arguments(f"data_train.{key}", f"data_valid.{key}")

    for key in "model", "data_train", "data_valid":
        parser.link_arguments("obj", f"{key}.obj", apply_on="instantiate")

    args = parser.parse_args()
    cfg = parser.instantiate_classes(args)

    data_train = SpyroDetectionDataset(cfg.data_train)
    data_valid = SpyroDetectionDataset(cfg.data_valid)
    model: SpyroDetector = cfg.model

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        worker_init_fn=utils.worker_init_fn,
        collate_fn=collate_fn,
    )
    loader_train = torch.utils.data.DataLoader(data_train, shuffle=True, **loader_kwargs)
    loader_valid = torch.utils.data.DataLoader(data_valid, **loader_kwargs)

    if cfg.debug:
        logger = False
        callbacks = []
    else:
        logger = WandbLogger(project=cfg.wandb_project, save_dir="./data")
        logger.log_hyperparams(args.as_flat())
        callbacks: list[pl.Callback] = [LearningRateMonitor()]

    torch.set_float32_matmul_precision("high")
    trainer = pl.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model=model, train_dataloaders=loader_train, val_dataloaders=loader_valid)

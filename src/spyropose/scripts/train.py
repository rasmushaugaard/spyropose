import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import torch.utils.data
from jsonargparse import ArgumentParser
from pytorch_lightning.loggers.wandb import WandbLogger

from .. import utils
from ..data.cfg import SpyroDataConfig
from ..data.dataset import SpyroDataset
from ..model import SpyroModelConfig, SpyroPoseModel
from ..obj import SpyroObjectConfig


def cli_train():
    parser = ArgumentParser()
    parser.add_class_arguments(SpyroObjectConfig, "obj")
    parser.add_class_arguments(SpyroModelConfig, "model")
    parser.add_class_arguments(SpyroDataConfig, "data_train")
    parser.add_class_arguments(SpyroDataConfig, "data_valid", default={"img_aug.enabled": False})
    parser.add_class_arguments(
        pl.Trainer,
        "trainer",
        default=dict(max_steps=50_000),
        skip={"logger", "callbacks"},
        instantiate=False,
    )

    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--wandb_project", type=str, default="spyropose")
    parser.add_argument("--debug", action="store_true")

    for key in "model", "data_train", "data_valid":
        parser.link_arguments("obj", f"{key}.obj", apply_on="instantiate")
    parser.link_arguments("debug", "trainer.enable_checkpointing", lambda debug: not debug)

    args = parser.parse_args()
    cfg = parser.instantiate_classes(args)

    data_train_cfg: SpyroDataConfig = cfg.data_train
    data_valid_cfg: SpyroDataConfig = cfg.data_valid
    assert not data_valid_cfg.img_aug.enabled
    if data_train_cfg.split_dir == data_valid_cfg.split_dir:
        assert not set(data_train_cfg.scene_ids) & set(data_valid_cfg.scene_ids), (
            "train and valid data comes from same split and shares scenes"
        )

    model = SpyroPoseModel(cfg.model)
    data_train = SpyroDataset(data_train_cfg)
    data_valid = SpyroDataset(data_valid_cfg)

    # dataloader
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        worker_init_fn=utils.worker_init_fn,
    )
    loader_train = torch.utils.data.DataLoader(data_train, shuffle=True, **loader_kwargs)
    loader_valid = torch.utils.data.DataLoader(data_valid, **loader_kwargs)

    if cfg.debug:
        logger = False
        callbacks = []
    else:
        logger = WandbLogger(project=cfg.wandb_project, save_dir="./data")
        logger.log_hyperparams(args.as_flat())
        callbacks: list[pl.Callback] = [cb.LearningRateMonitor()]

    torch.set_float32_matmul_precision("high")
    trainer = pl.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model=model, train_dataloaders=loader_train, val_dataloaders=loader_valid)


if __name__ == "__main__":
    cli_train()

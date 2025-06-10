import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import torch.utils.data
from jsonargparse import ArgumentParser
from pytorch_lightning.loggers.wandb import WandbLogger

from .. import utils
from ..data.data_cfg import DatasetConfig
from ..data.dataset import BopInstanceDataset
from ..model import SpyroPoseModel, SpyroPoseModelConfig


def cli_train():
    parser = ArgumentParser()
    parser.add_class_arguments(SpyroPoseModelConfig, "model")
    parser.add_class_arguments(DatasetConfig, "data")
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

    parser.link_arguments("data.obj", "model.obj", apply_on="instantiate")
    parser.link_arguments("debug", "trainer.enable_checkpointing", lambda debug: not debug)

    cfg = parser.parse_args()
    cfg = parser.instantiate_classes(cfg)

    model = SpyroPoseModel(cfg.model)
    data = BopInstanceDataset(cfg.data)

    # dataloader
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        worker_init_fn=utils.worker_init_fn,
    )
    dataloader = torch.utils.data.DataLoader(data, shuffle=True, **loader_kwargs)

    if cfg.debug:
        logger = False
        callbacks = []
    else:
        logger = WandbLogger(project=cfg.wandb_project, save_dir="./data")
        logger.log_hyperparams(cfg.as_flat())
        callbacks: list[pl.Callback] = [cb.LearningRateMonitor()]

    torch.set_float32_matmul_precision("high")
    trainer = pl.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model=model, train_dataloaders=dataloader)


if __name__ == "__main__":
    cli_train()

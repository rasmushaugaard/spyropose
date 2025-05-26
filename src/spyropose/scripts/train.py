import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import torch.utils.data
from jsonargparse import ArgumentParser
from pytorch_lightning.loggers.wandb import WandbLogger

from .. import utils
from ..data.data_cfg import DatasetConfig, ImgAugConfig
from ..data.dataset import BopInstanceDataset
from ..model import SpyroPoseModel, SpyroPoseModelConfig


def cli_train():
    """
    Some things don't belong
    - its weird to pass dataset name and model name to the model as well
    - data config should not have recursion depth

    It could be more clear that the frame of the object is changed!
    Maybe make it a parameter where the est. frame is and save the offset as part of the model.

    Frame and keypoints are automatically determined based on the mesh.
    We don't want to add that responsibility to the spyroposemodel, so we move the instantiation out of the class. This makes even more sense, since the data also depends on the choice (and size) of frame, but not on keypoints.

    Data -> crop_res, recursion_depth, radius, frame offset
    Model -> crop_res, recursion_depth, radius, keypoints (frame offset for inference, and dataset/model name for sanity)

    """

    parser = ArgumentParser()
    parser.add_class_arguments(DatasetConfig, "data")
    parser.add_class_arguments(SpyroPoseModelConfig, "model")
    parser.link_arguments("model.recursion_depth", "data.recursion_depth")
    parser.link_arguments("model.crop_res", "data.crop_res")
    parser.link_arguments("data.obj", "model.obj_name")
    parser.link_arguments("data.root_dir", "model.dataset_name", lambda p: p.name)
    parser.add_class_arguments(
        pl.Trainer,
        "trainer",
        default=dict(max_steps=50_000, accelerator="gpu", devices="auto"),
        skip={"logger", "callbacks"},
    )
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--wandb_project", type=str, default="spyropose")
    parser.add_argument("--debug", action="store_true")
    parser.link_arguments(
        "debug", "trainer.enable_checkpointing", lambda debug: not debug
    )

    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")

    data_args = args.data.clone()
    data_args.img_aug_cfg = ImgAugConfig(**data_args.img_aug_cfg)
    data_cfg = DatasetConfig(**data_args)
    data = BopInstanceDataset(data_cfg)

    model_cfg = SpyroPoseModelConfig(**args.model)
    model_cfg.instantiate_from_mesh(data.mesh)
    model = SpyroPoseModel(
        model_cfg, keypoints=model_cfg.keypoints_from_mesh(data.mesh)
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=utils.worker_init_fn,
    )
    dataloader = torch.utils.data.DataLoader(data, shuffle=True, **loader_kwargs)

    if args.debug:
        logger = False
        callbacks = []
    else:
        logger = WandbLogger(project=args.wandb_project, save_dir="./data")
        logger.log_hyperparams(args.as_flat())
        callbacks = [
            cb.LearningRateMonitor(),
        ]

    trainer = pl.Trainer(
        **args.trainer,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model=model, train_dataloaders=dataloader)


if __name__ == "__main__":
    cli_train()

import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import torch.utils.data
import trimesh
from jsonargparse import ArgumentParser
from pytorch_lightning.loggers.wandb import WandbLogger

from .. import utils
from ..data.data_cfg import DatasetConfig, ImgAugConfig
from ..frame import SpyroFrame
from ..model import SpyroPoseModel, SpyroPoseModelConfig


def get_frame():
    return SpyroFrame()


def cli_train():
    """
    It could be more clear that the frame of the object is changed!
    Maybe make it a parameter where the est. frame is and save the offset as part of the model.

    Frame and keypoints are automatically determined based on the mesh.
    We don't want to add that responsibility to the spyroposemodel, so we move that out of the class. This makes even more sense, since the data also depends on that.
    However, the frame offset should be saved with the model!

    The same frame offset should be used when training a detector / point estimator.

    FrameConfig (radius, center offset, padding ratio)
    FrameConfig -> Mesh
    keypoints -> Mesh, FrameConfig
    PoseModel -> FrameConfig, keypoints
    DetectionModel -> FrameConfig
    Data -> FrameConfig
    """

    parser = ArgumentParser()
    parser.add_class_arguments(SpyroPoseModelConfig, "model", skip={"frame"})
    parser.add_class_arguments(DatasetConfig, "data", skip={"frame"})
    parser.add_argument("--frame", type=SpyroFrame | None)
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

    for name in "dataset", "obj", "recursion_depth", "crop_res":
        parser.link_arguments(f"model.{name}", f"data.{name}")
    parser.link_arguments(
        "debug", "trainer.enable_checkpointing", lambda debug: not debug
    )

    args = parser.parse_args()

    data_args = args.data.clone()
    data_args.img_aug_cfg = ImgAugConfig(**data_args.img_aug_cfg)
    data_cfg = DatasetConfig(**data_args)
    mesh = trimesh.load_mesh(data_cfg.mesh_path)

    if args.frame is not None:
        frame = SpyroFrame(**args.frame)
    else:
        print("Frame not provided. Using frame based on from mesh bounding sphere.")
        frame = SpyroFrame.from_mesh_bounding_sphere(mesh)
    data_cfg.frame = frame

    model_cfg = SpyroPoseModelConfig(**args.model, frame=frame)

    model_cfg.init_keypoints_from_mesh(mesh)
    model = SpyroPoseModel(
        model_cfg, keypoints=model_cfg.init_keypoints_from_mesh(data.mesh)
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

    torch.set_float32_matmul_precision("high")
    trainer = pl.Trainer(
        **args.trainer,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model=model, train_dataloaders=dataloader)


if __name__ == "__main__":
    cli_train()

import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import torch.utils.data
import trimesh
from jsonargparse import ArgumentParser
from pytorch_lightning.loggers.wandb import WandbLogger

from .. import utils
from ..data.data_cfg import DatasetConfig, ImgAugConfig, ObjectConfig
from ..data.dataset import BopInstanceDataset
from ..frame import SpyroFrame
from ..model import SpyroPoseModel, SpyroPoseModelConfig


def init_keypoints_from_mesh(
    mesh: trimesh.Trimesh, frame: SpyroFrame, n_keypoints: int, tol=1.001
):
    frame_vertices = mesh.vertices - np.asarray(frame.obj_t_frame)
    mask = np.linalg.norm(frame_vertices, axis=1) < frame.radius * tol
    frame_vertices = frame_vertices[mask]
    return utils.farthest_point_sampling(frame_vertices, n_keypoints)


def cli_train():
    """
    The same frame offset should be used when training a detector / point estimator.

    FrameConfig (radius, center offset, padding ratio)
    FrameConfig -> Mesh
    keypoints -> Mesh, FrameConfig
    PoseModel -> FrameConfig, keypoints
    DetectionModel -> FrameConfig
    Data -> FrameConfig
    """

    parser = ArgumentParser()

    parser.add_class_arguments(ObjectConfig, "object")
    parser.add_argument("--frame", type=SpyroFrame | None)
    parser.add_argument(
        "--keypoints",
        type=int | tuple[tuple[float, float, float], ...],
        default=16,
        # TODO: potentially also allow box
    )

    parser.add_class_arguments(
        SpyroPoseModelConfig, "model", skip={"obj_cfg", "frame", "keypoints"}
    )
    parser.add_class_arguments(DatasetConfig, "data", skip={"obj_cfg", "frame"})

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

    for name in "recursion_depth", "crop_res":
        parser.link_arguments(f"model.{name}", f"data.{name}")
    parser.link_arguments("debug", "trainer.enable_checkpointing", lambda debug: not debug)

    args = parser.parse_args()

    obj_cfg = ObjectConfig(**args.object)
    mesh = trimesh.load_mesh(obj_cfg.mesh_path)

    # frame
    if args.frame is not None:
        frame = SpyroFrame(**args.frame)
    else:
        print("Frame not provided. Using frame based on from mesh bounding sphere.")
        frame = SpyroFrame.from_mesh_bounding_sphere(mesh)

    # keypoints
    if isinstance(args.keypoints, int):
        keypoints = init_keypoints_from_mesh(
            mesh=mesh, frame=frame, n_keypoints=args.keypoints
        ).tolist()
    else:
        keypoints = args.keypoints

    # model
    model = SpyroPoseModel(
        cfg=SpyroPoseModelConfig(obj_cfg=obj_cfg, frame=frame, keypoints=keypoints, **args.model)
    )

    # dataset
    args.data.img_aug_cfg = ImgAugConfig(**args.data.img_aug_cfg)
    data = BopInstanceDataset(cfg=DatasetConfig(obj_cfg=obj_cfg, frame=frame, **args.data))

    # dataloader
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
        callbacks: list[pl.Callback] = [cb.LearningRateMonitor()]

    torch.set_float32_matmul_precision("high")
    trainer = pl.Trainer(
        **args.trainer,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model=model, train_dataloaders=dataloader)


if __name__ == "__main__":
    cli_train()

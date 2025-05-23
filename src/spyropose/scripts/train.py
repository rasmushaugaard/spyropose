import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import torch.utils.data
from pytorch_lightning.loggers.wandb import WandbLogger

from .. import utils
from ..data.dataset import BopInstanceDataset, DatasetConfig
from ..model import SpyroPoseModel


def main(
    data_cfg: DatasetConfig,
    dropout=0.1,
    num_workers=10,
    lr=1e-4,
    weight_decay=0.0,
    batch_size=4,
    max_steps=50_000,
    debug=False,
    batchnorm=False,
    n_samples=32,
    embed_dim=64,
    vis_model="unet18",
    n_pts=16,
    point_dropout=0.1,
    n_layers=3,
    d_ff=256,
    project="spyropose",
    accelerator="gpu",
    devices="auto",
):
    data = BopInstanceDataset(data_cfg)
    position_scale = 1e-3  # mm to m

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=utils.worker_init_fn,
    )
    dataloader = torch.utils.data.DataLoader(data, shuffle=True, **loader_kwargs)

    pts = utils.sample_keypoints_from_mesh(mesh=data.mesh, n_pts=n_pts)
    pts = pts - data.mesh.bounding_sphere.primitive.center
    pts = pts.T

    model = SpyroPoseModel(
        dataset_name=data_cfg.root_dir.name,
        obj_name=data_cfg.obj,
        obj_radius=data.obj_radius,
        vis_model=vis_model,
        lr=lr,
        weight_decay=weight_decay,
        n_samples=n_samples,
        embed_dim=embed_dim,
        n_layers=n_layers,
        d_ff=d_ff,
        pts=pts,
        n_pts=n_pts,
        recursion_depth=data_cfg.recursion_depth,
        dropout=dropout,
        point_dropout=point_dropout,
        batchnorm=batchnorm,
        position_scale=position_scale,
        debug=debug,
    )

    if debug:
        logger = False
        callbacks = []
    else:
        logger = WandbLogger(project=project, save_dir="./data")
        # logger.log_hyperparams(args)
        callbacks = [
            cb.LearningRateMonitor(),
        ]

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        enable_checkpointing=not debug,
        max_steps=max_steps,
        callbacks=callbacks,
    )
    trainer.fit(model=model, train_dataloaders=dataloader)

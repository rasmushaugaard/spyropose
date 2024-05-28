import argparse

import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import torch.utils.data
from pytorch_lightning.loggers.wandb import WandbLogger

from .. import helpers, utils
from ..data import symsol, symsol_ours
from ..data.bop import config
from ..data.bop.dataset import get_bop_dataset
from ..model import SpyroPoseModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("object")
    parser.add_argument("--gpu-idx", type=int, required=True)
    # model parameters
    parser.add_argument("--batchnorm", action="store_true")
    parser.add_argument("--no-is", dest="importance_sampling", action="store_false")
    parser.add_argument("--kpts", choices=["fps", "box", "none"], default="fps")
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--n-pts", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--vis-model", default="unet18")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--point-dropout", type=float, default=0.1)
    parser.add_argument("--recursion-depth", type=int, default=7)
    parser.add_argument("--number-fourier-components", type=int, default=3)
    parser.add_argument("--scene-rng-train", type=int, nargs=2, default=(0, 49))
    parser.add_argument("--scene-rng-valid", type=int, nargs=2, default=(49, 50))
    # data parameters
    parser.add_argument("--crop-res", type=int, default=224)
    parser.add_argument(
        "--no-translation-offset",
        dest="random_offset_translation",
        action="store_false",
    )
    parser.add_argument(
        "--no-rotation-offset", dest="random_offset_rotation", action="store_false"
    )
    parser.add_argument(
        "--no-image-augmentations", dest="image_augmentations", action="store_false"
    )
    parser.add_argument("--cj-p", type=float, default=1.0)
    parser.add_argument("--cj-brightness", type=float, default=0.5)
    parser.add_argument("--cj-contrast", type=float, default=0.5)
    parser.add_argument("--cj-saturation", type=float, default=0.5)
    parser.add_argument("--cj-hue", type=float, default=0.1)
    parser.add_argument("--low-data-regime", action="store_true")
    # training parameters
    parser.add_argument("--max-steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    dataset_name = args.dataset
    object_name = args.object

    data_kwargs = dict(name=object_name, recursion_depth=args.recursion_depth)
    mesh = helpers.load_mesh(dataset_name, object_name)
    if args.low_data_regime:
        assert dataset_name == "symsol"

    train_se3 = False
    position_scale = None
    if dataset_name == "symsol":
        if args.low_data_regime:
            data_train = symsol.SymsolDataset(**data_kwargs, data_slice=slice(9_500))
            data_valid = symsol.SymsolDataset(
                **data_kwargs, data_slice=slice(9_500, 10_000), random_offset=False
            )
        else:
            data_train = symsol.SymsolDataset(**data_kwargs)
            data_valid = None
        data_test = symsol.SymsolDataset(**data_kwargs, split="test")
    elif dataset_name == "symsol_ours":
        data_train = symsol_ours.SymsolDataset(**data_kwargs)
        data_valid = None
        data_test = symsol_ours.SymsolDataset(**data_kwargs, split="test")
    else:  # bop dataset
        train_se3 = True
        position_scale = 1e-3  # mm to m

        obj_id = int(object_name)
        cfg = config.config[dataset_name]

        data_kwargs = dict(
            **data_kwargs,
            dataset_name=dataset_name,
            crop_res=args.crop_res,
            image_augmentations=args.image_augmentations,
            random_offset_rotation=args.random_offset_rotation,
            cj_brightness=args.cj_brightness,
            cj_contrast=args.cj_contrast,
            cj_saturation=args.cj_saturation,
            cj_hue=args.cj_hue,
            cj_p=args.cj_p,
            scene_ids_train=list(range(*args.scene_rng_train)),
            scene_ids_valid=list(range(*args.scene_rng_valid)),
        )
        data_train, data_valid, data_test = [
            get_bop_dataset(**data_kwargs, split=split)
            for split in ("train", "valid", "test")
        ]

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=utils.worker_init_fn,
    )
    loader_train = torch.utils.data.DataLoader(
        data_train, shuffle=True, **loader_kwargs
    )
    if data_valid is not None:
        loader_valid = torch.utils.data.DataLoader(data_valid, **loader_kwargs)
    else:
        loader_valid = None
    loader_test = torch.utils.data.DataLoader(data_test, **loader_kwargs)

    if args.kpts == "box":
        pts = "box"
    elif args.kpts == "fps":
        pts = utils.sample_keypoints_from_mesh(mesh=mesh, n_pts=args.n_pts)
        pts = pts - mesh.bounding_sphere.primitive.center
        pts = pts.T
    elif args.kpts == "none":
        pts = None
    else:
        raise ValueError()

    obj_radius = mesh.bounding_sphere.primitive.radius if mesh is not None else np.nan
    model = SpyroPoseModel(
        dataset_name=dataset_name,
        obj_name=object_name,
        obj_radius=obj_radius,
        train_se3=train_se3,
        vis_model=args.vis_model,
        lr=args.lr,
        n_samples=args.n_samples,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        pts=pts,
        n_pts=args.n_pts,
        recursion_depth=args.recursion_depth,
        importance_sampling=args.importance_sampling,
        dropout=args.dropout,
        point_dropout=args.point_dropout,
        random_offset_translation=args.random_offset_translation,
        random_offset_rotation=args.random_offset_rotation,
        batchnorm=args.batchnorm,
        position_scale=position_scale,
        debug=args.debug,
    )

    if args.debug:
        logger = False
        callbacks = []
    else:
        logger = WandbLogger(project="spyropose", save_dir="./data")
        logger.log_hyperparams(args)
        callbacks = [
            cb.LearningRateMonitor(),
        ]

    if args.low_data_regime:
        callbacks.append(
            cb.EarlyStopping(
                monitor="val/ll_6",
                patience=5,
                mode="max",
            )
        )
        callbacks.append(
            cb.ModelCheckpoint(monitor="val/ll_6", mode="max", save_last=False)
        )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[args.gpu_idx],
        logger=logger,
        enable_checkpointing=not args.debug,
        max_steps=args.max_steps,
        callbacks=callbacks,
    )
    trainer.fit(
        model=model,
        train_dataloaders=loader_train,
        val_dataloaders=[loader_valid, loader_test],
    )

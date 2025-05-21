from functools import partial

import trimesh

from .data.bop.config import config
from .data.bop.dataset import get_bop_dataset


def get_dataset(name, **kwargs):
    return partial(get_bop_dataset, dataset_name=name, **kwargs)


def dataset_from_model(model, **kwargs):
    return get_dataset(model.dataset_name)(
        name=model.obj_name,
        crop_res=model.crop_res,
        recursion_depth=model.recursion_depth,
        **kwargs,
    )


def load_mesh(dataset_name, object_name) -> trimesh.Trimesh:
    cfg = config[dataset_name]
    return trimesh.load_mesh(
        f"data/bop/{dataset_name}/{cfg.model_folder}/obj_{int(object_name):06d}.ply"
    )

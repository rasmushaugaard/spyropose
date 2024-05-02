from functools import partial

import trimesh

from .data import symsol, symsol_objects, symsol_ours
from .data.bop.config import config
from .data.bop.dataset import get_bop_dataset

datasets = {}
datasets["symsol"] = symsol.SymsolDataset
datasets["symsol_ours"] = symsol_ours.SymsolDataset
for name in "tless", "hb":
    datasets[name] = partial(get_bop_dataset, dataset_name=name)


def dataset_from_model(model, **kwargs):
    return datasets[model.dataset_name](
        name=model.obj_name,
        crop_res=model.crop_res,
        recursion_depth=model.recursion_depth,
        **kwargs,
    )


def load_mesh(dataset_name, object_name) -> trimesh.Trimesh:
    if dataset_name == "symsol":
        return None
    if dataset_name == "symsol_ours":
        return symsol_objects.obj_gen_dict[object_name]()[0]

    cfg = config[dataset_name]
    return trimesh.load_mesh(
        f"data/bop/{dataset_name}/{cfg.model_folder}/obj_{int(object_name):06d}.ply"
    )

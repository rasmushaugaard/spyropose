"""
Use same sets of multiview images as cosypose.
Code is copied from https://github.com/ylabbe/cosypose and modified
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_frame_index(base_dir: Path):
    """modified version of: cosypose/datasets/bop.py:build_index"""
    scene_ids, view_ids, cam_t_worlds, Ks = [], [], [], []
    for scene_dir in base_dir.iterdir():
        for view_id, view in json.load(
            (scene_dir / "scene_camera.json").open()
        ).items():
            cam_t_world = np.eye(4)
            cam_t_world[:3, :3] = np.array(view["cam_R_w2c"]).reshape(3, 3)
            cam_t_world[:3, 3] = view["cam_t_w2c"]
            K = np.array(view["cam_K"]).reshape(3, 3)
            scene_ids.append(int(scene_dir.name))
            view_ids.append(int(view_id))
            cam_t_worlds.append(cam_t_world)
            Ks.append(K)
    return pd.DataFrame(
        dict(scene_id=scene_ids, view_id=view_ids, cam_t_world=cam_t_worlds, K=Ks)
    )


def get_multiview_frame_index(frame_index: pd.DataFrame, n_views: int):
    """modified version of:
    cosypose/datasets/wrappers/multiview_wrapper.py:MultiViewWrapper:__init__"""
    frame_index = frame_index.copy().reset_index(drop=True)
    groups = frame_index.groupby(["scene_id"]).groups

    random_state = np.random.RandomState(0)
    multiview_frame_index = []
    for scene_id, group_ids in groups.items():
        group_ids = random_state.permutation(group_ids)
        len_group = len(group_ids)
        for k, m in enumerate(np.arange(len_group)[::n_views]):
            ids_k = np.arange(len(group_ids))[m : m + n_views].tolist()
            ds_ids = group_ids[ids_k]
            df_group = frame_index.loc[ds_ids]
            multiview_frame_index.append(
                dict(
                    scene_id=scene_id,
                    view_ids=df_group["view_id"].values.tolist(),
                    cams_t_world=np.stack(df_group["cam_t_world"].values),
                    Ks=np.stack(df_group["K"].values),
                    n_views=len(df_group),
                    scene_ds_ids=ds_ids,
                )
            )

    multiview_frame_index = pd.DataFrame(multiview_frame_index)
    multiview_frame_index["group_id"] = np.arange(len(multiview_frame_index))
    return multiview_frame_index

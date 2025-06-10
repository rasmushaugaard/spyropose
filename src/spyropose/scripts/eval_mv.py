import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from .. import helpers, se3_grid
from ..data.bop.config import config
from ..data.bop.cosy_multiview import build_frame_index, get_multiview_frame_index
from ..data.bop.multiview_cropper import MultiViewCropper
from ..model import SpyroPoseModel

parser = argparse.ArgumentParser()
parser.add_argument("run_id")
parser.add_argument("device")
parser.add_argument("--top-k", type=int, default=512)
parser.add_argument("--split", default="test")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

debug = args.debug
device = args.device

model = SpyroPoseModel.load_from_run_id(args.run_id)
model.eval()
model.freeze()
model.to(args.device)
assert model.dataset_name == "tless"

n_views = 4
cfg = config[model.dataset_name]
obj_id = int(model.obj_name)
mesh = helpers.load_mesh(model.dataset_name, model.obj_name)
test_folder = Path(f"data/bop/{model.dataset_name}/{cfg.test_folder}")

data = helpers.datasets[model.dataset_name](
    name=model.obj_name,
    split=args.split,
    recursion_depth=model.recursion_depth,
    regular_grid=True,
)
insts_by_scene_img = defaultdict(lambda: [])
for i, inst in enumerate(data.instances):
    inst["data_idx"] = i
    insts_by_scene_img[(inst["scene_id"], inst["img_id"])].append(inst)
multi_view_cropper = MultiViewCropper(
    dataset=data, obj_radius=mesh.bounding_sphere.primitive.radius, crop_res=224
)

frame_index = get_multiview_frame_index(build_frame_index(test_folder), n_views=n_views)


def to_dev(a):
    return torch.from_numpy(a).to(device)


inst_count = []
lls = []

pbar = tqdm(total=len(data))
for group_idx, group in frame_index.iterrows():
    scene_id = group.scene_id
    insts_by_id = defaultdict(lambda: [])
    for view_id in group.view_ids:
        for inst in insts_by_scene_img[(scene_id, view_id)]:
            insts_by_id[inst["pose_idx"]].append(inst)
    for pose_idx, insts in insts_by_id.items():
        n_views = len(insts)
        pbar.update(n_views)

        inst_count.append(n_views)
        view_idxs = [group.view_ids.index(inst["img_id"]) for inst in insts]
        cams_t_world = group.cams_t_world[view_idxs]  # (n_views, 4, 4)
        # set the world frame to cam0's frame
        cams_t_world = cams_t_world @ np.linalg.inv(cams_t_world[0])
        world_t_cams = np.linalg.inv(cams_t_world)

        Ks = group.Ks[view_idxs]
        view_ids = np.array(group.view_ids)[view_idxs]

        inst = insts[0]

        d = data[inst["data_idx"]]
        t_est = d["t_est"]  # world because cam0 is world frame
        assert t_est.shape == (3, 1)
        t_est = to_dev(t_est).unsqueeze(0)
        t_grid_frame = d["t_grid_frame"]  # grid frame and rotation index is also correct
        assert t_grid_frame.shape == (3, 3)
        t_grid_frame = to_dev(t_grid_frame).unsqueeze(0)

        # (v, r, r, 3), (v, 3, 3)
        imgs, Ks = multi_view_cropper.get(
            scene_id=scene_id,
            world_p_obj_est=t_est.cpu().numpy()[0],
            img_ids=view_ids,
            cams_t_world=cams_t_world,
            Ks=Ks,
        )

        if debug:
            _, axs = plt.subplots(1, n_views + 1)
            for img, ax in zip([d["img"].transpose(1, 2, 0), *imgs], axs):
                ax.imshow(img)
            plt.show()
            quit()

        out = model.forward_infer(
            img=to_dev(imgs.transpose(0, 3, 1, 2)).unsqueeze(0),  # (b, n_cams, c, h, w)
            K=to_dev(Ks).unsqueeze(0),  # (b, n_cams, 3, 3)
            world_t_obj_est=t_est,
            world_R_cam=to_dev(world_t_cams[:, :3, :3]).unsqueeze(0).float(),
            world_t_cam=to_dev(world_t_cams[:, :3, 3:]).unsqueeze(0).float(),
            pos_grid_frame=t_grid_frame,
            top_k=args.top_k,
            pose_bs=None,
        )

        _, ll = se3_grid.locate_poses_in_pyramid(
            q_rot_idx_rlast=to_dev(d[f"rot_idx_target_{model.recursion_depth - 1}"]).unsqueeze(
                0
            ),  # (b, 1)
            log_probs=out["log_probs"],
            rot_idxs=out["rot_idxs"],
            t_est=t_est,
            pos_grid_frame=t_grid_frame,
            q_pos=to_dev(d["t"]).view(1, 1, 3, 1),
            pos_idxs=out["pos_idxs"],
            position_scale=1e-3,
        )

        ll = ll.view(-1)[-1].item()
        for _ in range(n_views):
            lls.append(ll)

print(np.mean(lls))

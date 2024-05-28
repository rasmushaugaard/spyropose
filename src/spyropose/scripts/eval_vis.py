import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from tqdm import tqdm

from .. import helpers, se3_grid, translation_grid, utils, vis
from ..data.bop.config import config
from ..data.bop.dataset import get_bop_dataset
from ..data.renderer import SimpleRenderer
from ..model import SpyroPoseModel

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("device")
parser.add_argument("--dataset")
parser.add_argument("--object")
parser.add_argument("--n", type=int, default=2_000_000)
parser.add_argument("--i", type=int, nargs="*")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--topk", type=int, default=2048)
parser.add_argument("--r", type=int, required=True)
parser.add_argument("--r-grid", type=int)
parser.add_argument("--timer", action="store_true")
parser.add_argument("--scene-id", type=int, default=50)
parser.add_argument("--render-prob-target", type=float, default=0.95)
parser.add_argument("--max-renders", type=int, default=1_000)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--split", default="valid")
parser.add_argument("--box", type=int, nargs=4)
parser.add_argument("--long-offset", type=float, default=0.0)
parser.add_argument("--lat-offset", type=float, default=0.0)
parser.add_argument("--regular-grid", action="store_true")
parser.add_argument("--scene-rng-train", type=int, nargs=2, default=(0, 49))
parser.add_argument("--scene-rng-valid", type=int, nargs=2, default=(49, 50))
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument("--show-x", action='store_true')
args = parser.parse_args()
device = args.device

model = SpyroPoseModel.load_from_checkpoint(args.model_path)
model.freeze()
model.eval()
model.to(device)

dataset_name = model.dataset_name
obj_name = model.obj_name
dataset = helpers.get_dataset(dataset_name)(
    name=model.obj_name,
    recursion_depth=model.recursion_depth,
    regular_grid=args.regular_grid,
    split=args.split,
    scene_ids_train=list(range(*args.scene_rng_train)),
    scene_ids_valid=list(range(*args.scene_rng_valid)),
    random_offset_rotation=False,
)
cfg = config[dataset_name]
mesh = trimesh.load_mesh(
    f"data/bop/{dataset_name}/{cfg.model_folder}/obj_{int(obj_name):06d}.ply"
)
mesh.apply_translation(-mesh.bounding_sphere.primitive.center)
renderer = SimpleRenderer(mesh, near=10.0, far=10_000.0, w=224)

r_grid = args.r if args.r_grid is None else args.r_grid
grid_np = getattr(model, f"grid_{r_grid}").cpu().numpy()

np.random.seed(args.seed)

i_queue = args.i
while True:
    if i_queue:
        i = i_queue.pop(0)
    else:
        i = np.random.randint(len(dataset))
    print(f"\n-------- {i} ----------")
    img_data = {
        k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.tensor(v)).to(
            device
        )[None]
        for k, v in dataset[i].items()
    }
    if args.box is not None:
        l, t, r, b = args.box
        img_data["img"][:, :, t:b, l:r] = 0

    assert args.topk is not None
    for _ in range(2 if args.timer else 1):
        with utils.timer("forward"):
            out = model.forward_infer(
                img=img_data["img"][None],
                K=img_data["K"][None],
                world_t_obj_est=img_data["t_est"],
                pos_grid_frame=img_data["t_grid_frame"],
                pose_bs=10_000,
                top_k=args.topk,
            )
            out["log_probs"][-1].view(-1)[-1].item()
    probs = out["log_probs"][args.r][0].exp()
    print(f"sum prob at r={args.r}: {probs.sum():.3f}")
    idx = out["rot_idxs"][args.r][0]
    pos_grid = out["pos_idxs"][args.r]
    t_est = img_data["t_est"]
    grid_frame = img_data["t_grid_frame"]
    print(f"det(grid_frame): {torch.linalg.det(grid_frame * 1e-3).item():.2e}")

    # quit()
    t_np = (
        translation_grid.grid2pos(
            grid=pos_grid, t_est=t_est, grid_frame=grid_frame, r=args.r + 1
        )
        .cpu()
        .numpy()[0]
    )

    grid_np_out = grid_np[idx.cpu().numpy()]
    with utils.timer("pyramid search"):
        match_idx, ll = se3_grid.locate_poses_in_pyramid(
            q_rot_idx_rlast=img_data[f"rot_idx_target_{model.recursion_depth - 1}"],
            q_pos=img_data["t"].unsqueeze(1),
            t_est=t_est,
            pos_grid_frame=grid_frame,
            rot_idxs=out["rot_idxs"],
            pos_idxs=out["pos_idxs"],
            log_probs=out["log_probs"],
            position_scale=1e-3,
        )
    print("ll:", ll.view(-1).cpu().numpy())

    size = 4
    fig = plt.figure(figsize=(2 * size, 2 * size))
    gs = fig.add_gridspec(3, 3)
    ax0 = fig.add_subplot(gs[1, 0])

    img_np = img_data["img"][0].permute(1, 2, 0).cpu().numpy()
    img_gray = img_np.mean(axis=2, keepdims=True)  # (h, w, 1)
    ax0.imshow(img_np)
    ax0.set_title("input")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, :], projection="mollweide")

    rot_vis = vis.visualize_so3_probabilities(
        grid_np_out,
        probs.cpu(),
        rotations_gt=img_data["R"][:1].cpu(),
        fig=fig,
        ax=ax1,
        display_threshold_probability=0,
        visualize_prob_by_alpha=True,
        show_color_wheel=False,
        s=7 * 1024 / 4**args.r,
        long_offset=args.long_offset,
        lat_offset=args.lat_offset,
        gamma=args.gamma,
    )

    ax2 = fig.add_subplot(gs[1, 1])
    img2 = ax2.imshow(np.zeros_like(img_np))
    ax2.set_title("x")
    ax2.axis("off")

    if args.show_x:
        set_marker_rot = rot_vis["show_marker"](marker="X", s=400)
        scatter_rot_idx = np.argwhere(rot_vis["rotation_mask"])[0]

        last_idx = None

        def cb(event):
            global last_idx
            if event.inaxes is not ax1:
                return
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return
            idx = rot_vis["scatter_tree"].query((x, y))[1]
            idx = scatter_rot_idx[idx]
            if idx == last_idx:
                return
            R = grid_np_out[idx]
            t = t_np[idx]
            set_marker_rot(R)

            img2.set_data(overlay(img_gray, renderer.render(K=K, R=R, t=t) * blue_tint))
            fig.canvas.draw()

        fig.canvas.mpl_connect("motion_notify_event", cb)

    # images
    blue_tint = (0.3, 0.6, 1.0, 1.0)
    green_tint = (0.5, 1.0, 0.5, 1.0)

    def overlay(original, render, alpha=0.6, alpha_original=0.7, normalize_alpha=True):
        # render has pre-multiplied alpha
        if normalize_alpha:
            render = render / render[..., 3].max(initial=1e-3)
        render = render * alpha
        return render[..., :3] + (1 - render[..., 3:]) * alpha_original * original

    # distribution render
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.axis("off")
    K = img_data["K"][0].cpu().numpy()
    probs_sorted, sort_idx = torch.sort(probs, descending=True)
    max_renders = min(
        torch.searchsorted(
            probs_sorted.cumsum(dim=0), args.render_prob_target, right=True
        ).item(),
        args.max_renders,
    )
    idx = sort_idx[:max_renders].cpu().numpy()
    dist_render = np.zeros((224, 224, 4))
    probs_sparse = probs.cpu().numpy()[idx]
    t_sparse = t_np[idx]  # (n, 3, 1)
    grid_sparse = grid_np_out[idx]
    for p_, t_, R_ in tqdm(
        zip(probs_sparse / probs_sparse.sum(), t_sparse, grid_sparse)
    ):
        dist_render += p_ * renderer.render(K=K, R=R_, t=t_)
    ax3.imshow(overlay(img_gray, dist_render * blue_tint))
    ax3.axis("off")
    ax3.set_title(f"p={probs_sparse.sum():.3f}")

    # point estimate (arg max)
    ax6 = fig.add_subplot(gs[2, 2])
    render = renderer.render(K=K, R=grid_sparse[0], t=t_sparse[0])
    ax6.imshow(overlay(img_gray, render * blue_tint))
    ax6.set_title("arg max")
    ax6.axis("off")

    # ground truth
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.imshow(
        overlay(
            img_gray,
            renderer.render(
                K=K,
                R=img_data["R"][0].cpu().numpy(),
                t=img_data["t"][0].cpu().numpy(),
            )
            * (0.5, 1.0, 0.5, 1.0),
        )
    )
    ax5.axis("off")
    ax5.set_title("gt")

    # closest bin
    pos_idx_target_r = translation_grid.pos2grid(
        pos=img_data["t"],
        t_est=img_data["t_est"],
        grid_frame=img_data["t_grid_frame"],
        r=args.r + 1,
    )
    pos_target = (
        translation_grid.grid2pos(
            grid=pos_idx_target_r,
            t_est=img_data["t_est"],
            grid_frame=img_data["t_grid_frame"],
            r=args.r + 1,
        )[0, 0]
        .cpu()
        .numpy()
    )  # (3, 1)
    rlast = model.recursion_depth - 1
    R = grid_np[
        img_data[f"rot_idx_target_{rlast}"].cpu().numpy()[0, 0]
        // (8 ** (rlast - args.r))
    ]
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.imshow(
        overlay(img_gray, renderer.render(K=K, R=R, t=pos_target) * green_tint)
    )  # green tint
    ax4.set_title("closest bin to gt")
    ax4.axis("off")

    plt.show()

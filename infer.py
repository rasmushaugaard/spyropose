import json

import cv2
import einops
import jsonargparse
import numpy as np
import torch
from matplotlib import pyplot as plt

from spyropose.data.auxs import calculate_crop_matrix
from spyropose.detection.model import SpyroDetector
from spyropose.model import SpyroPoseModel
from spyropose.translation_grid import get_translation_grid_frame
from spyropose.vis import visualize_so3_probabilities


def infer(
    device="cuda:0",
    show_det=False,
    show_crop=False,
    recursion_level: int | None = None,
    top_k: int = 512,
):
    # load models
    detector = SpyroDetector.load_from_checkpoint(
        "data/spyropose_detector/2kewoepx/checkpoints/epoch=42-step=20000.ckpt", device
    ).eval()
    detector.freeze()

    spyro = SpyroPoseModel.load_from_checkpoint(
        "data/spyropose/0qkibyrp/checkpoints/epoch=15-step=50000.ckpt", device
    ).eval()
    spyro.freeze()

    # load data
    with open("data/bop/lego_foo/train_pbr/000019/scene_camera.json") as f:
        K = np.asarray(json.load(f)["0"]["cam_K"]).reshape(3, 3)
    img = cv2.imread("data/bop/lego_foo/train_pbr/000019/rgb/000000.jpg", cv2.IMREAD_COLOR_RGB)
    x = img.astype(np.float32) / 255.0  # (h, w, 3) [0, 1]
    x = einops.rearrange(x, "h w d -> 1 d h w")
    x = torch.from_numpy(x).to(device)
    out = detector.model(x)[0]

    if show_det:
        img_det = img.copy()
        for box, score in zip(out["boxes"], out["scores"]):
            l, t, r, b = box.round().long().tolist()
            cv2.rectangle(img_det, (l, t), (r, b), (0, 0, 255))
            cv2.putText(
                img_det, f"{score:.3f}", (l, b), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255)
            )
        cv2.imshow("", cv2.cvtColor(img_det, cv2.COLOR_RGB2BGR))
        cv2.waitKey()

    # pass most confident detection to spyropose
    idx = out["scores"].argmax()
    box = out["boxes"][idx]
    # - estimate position from bbox
    lt, rb = box.reshape(2, 2)
    wh = (rb - lt).abs()
    f = abs(np.linalg.det(K[:2, :2])) ** 0.5
    z = 2 * detector.obj.frame.radius * f / wh.mean().item()
    p = np.linalg.inv(K) @ (*box.reshape(2, 2).mean(dim=0).tolist(), 1)
    p = p / p[2] * z
    # - estimate crop
    crop_res = spyro.cfg.obj.crop_res
    crop_matrices = calculate_crop_matrix(
        K=K,
        t_frame_est=p.reshape(3, 1),
        crop_res=spyro.cfg.obj.crop_res,
        frame_radius=spyro.cfg.obj.frame.radius,
        padding_ratio=spyro.cfg.obj.frame.padding_ratio,
    )
    K_crop: np.ndarray = crop_matrices["K_crop"]
    M_crop: np.ndarray = crop_matrices["M_crop"]
    crop = cv2.warpAffine(img, M_crop[:2], (crop_res, crop_res))

    if show_crop:
        cv2.imshow("", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        cv2.waitKey()

    x = crop.astype(np.float32) / 255.0
    x = einops.rearrange(x, "h w d -> 1 1 d h w")
    x = torch.from_numpy(x).to(device)

    pos_grid_frame = get_translation_grid_frame(
        frame_radius=spyro.cfg.obj.frame.radius,
        t_frame_est=p.reshape(3, 1),
        random_rotation=False,
    )

    out = spyro.forward_infer(
        img=x,
        K=torch.from_numpy(einops.rearrange(K_crop, "n m -> 1 1 n m")).float().to(device),
        world_t_obj_est=torch.from_numpy(p).reshape(1, 3, 1).float().to(device),
        top_k=top_k,
        pos_grid_frame=torch.from_numpy(pos_grid_frame)[None].to(device),
    )

    if recursion_level is None:
        recursion_level = spyro.cfg.r_last
    rot_idxs = out["rot_idxs"][recursion_level][0]
    probs = out["log_probs"][recursion_level][0].exp().cpu().numpy()
    rot = getattr(spyro, f"grid_{recursion_level}")[rot_idxs].cpu().numpy()  # (n, 3, 3)

    print(f"sum(probs) at recursion level {recursion_level} = {probs.sum():.3f}")

    fig = plt.figure()
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2, projection="mollweide")
    ax0.imshow(crop)
    visualize_so3_probabilities(rotations=rot, probabilities=probs, ax=ax1)
    plt.show()


if __name__ == "__main__":
    jsonargparse.auto_cli(infer)

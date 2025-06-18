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
    recursion_level: int | None = None,
    top_k: int = 512,
    scene_id=19,
    img_id=0,
):
    # load models in eval mode and freeze weights
    detector = SpyroDetector.load_eval_freeze("data/spyropose_detector/2kewoepx", device)
    spyro = SpyroPoseModel.load_eval_freeze("data/spyropose/dwq4lb0a", device)

    # load data
    with open(f"data/bop/lego_foo/train_pbr/{scene_id:06d}/scene_camera.json") as f:
        K = np.asarray(json.load(f)[str(img_id)]["cam_K"], dtype=np.float32).reshape(3, 3)
    img = cv2.imread(
        f"data/bop/lego_foo/train_pbr/{scene_id:06d}/rgb/{img_id:06d}.jpg", cv2.IMREAD_COLOR_RGB
    )

    # normalize img and run detection
    x = img.astype(np.float32) / 255.0  # (h, w, 3) [0, 1]
    x = einops.rearrange(x, "h w d -> 1 d h w")
    x = torch.from_numpy(x).to(device)
    det_out = detector.model(x)[0]

    if show_det:
        img_det = img.copy()
        for box, score in zip(det_out["boxes"], det_out["scores"]):
            l, t, r, b = box.round().long().tolist()
            cv2.rectangle(img_det, (l, t), (r, b), (0, 0, 255))
            cv2.putText(
                img_det, f"{score:.3f}", (l, b), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255)
            )
        cv2.imshow("", cv2.cvtColor(img_det, cv2.COLOR_RGB2BGR))
        cv2.waitKey()

    for idx in torch.argsort(det_out["scores"], descending=True):
        box = det_out["boxes"][idx].cpu().numpy()
        cam_t_spyro_frame = detector.estimate_position_from_bbox(box=box, K=K)
        # estimate crop
        crop_res = spyro.cfg.obj.crop_res
        crop_matrices = calculate_crop_matrix(
            K=K,
            t_frame_est=cam_t_spyro_frame.reshape(3, 1),
            crop_res=crop_res,
            frame_radius=spyro.cfg.obj.frame.radius,
            padding_ratio=spyro.cfg.obj.frame.padding_ratio,
        )
        K_crop: np.ndarray = crop_matrices["K_crop"]
        M_crop: np.ndarray = crop_matrices["M_crop"]
        crop = cv2.warpAffine(img, M_crop[:2], (crop_res, crop_res))

        x = crop.astype(np.float32) / 255.0
        x = einops.rearrange(x, "h w d -> 1 1 d h w")
        x = torch.from_numpy(x).to(device)

        # TODO: default in forward infer (regular if multiview)
        pos_grid_frame = get_translation_grid_frame(
            frame_radius=spyro.cfg.obj.frame.radius,
            t_frame_est=cam_t_spyro_frame.reshape(3, 1),
            random_rotation=False,
        )

        spyro_out = spyro.forward_infer(
            img=x,
            K=torch.from_numpy(einops.rearrange(K_crop, "n m -> 1 1 n m")).float().to(device),
            world_t_obj_est=torch.from_numpy(cam_t_spyro_frame)
            .reshape(1, 3, 1)
            .float()
            .to(device),
            top_k=top_k,
            pos_grid_frame=torch.from_numpy(pos_grid_frame)[None].to(device),
        )

        if recursion_level is None:
            recursion_level = spyro.cfg.r_last
        rot_idxs = spyro_out["rot_idxs"][recursion_level][0]
        probs = spyro_out["log_probs"][recursion_level][0].exp().cpu().numpy()
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

import json
from pathlib import Path

import cv2
import jsonargparse
import numpy as np
import torch
from matplotlib import pyplot as plt

from spyropose.detection.model import SpyroDetector
from spyropose.model import SpyroPoseModel
from spyropose.utils import if_none
from spyropose.vis import visualize_so3_probabilities


def infer(
    detector_path: Path,
    spyro_path: Path,
    device="cuda:0",
    r_idx: int | None = None,
    top_k: int = 512,
    split="train_pbr",
    scene_id=19,
    img_id=0,
    fig_scale=3.0,
):
    # load models in eval mode and freeze weights
    detector = SpyroDetector.load_eval_freeze(detector_path, device)
    spyro = SpyroPoseModel.load_eval_freeze(spyro_path, device)
    assert detector.obj == spyro.cfg.obj
    split_dir = spyro.cfg.obj.root_dir / split

    r_idx = if_none(r_idx, spyro.cfg.r_last)

    # load data
    with open(split_dir / f"{scene_id:06d}/scene_camera.json") as f:
        K = np.asarray(json.load(f)[str(img_id)]["cam_K"]).reshape(3, 3)
    img = cv2.imread(
        str(split_dir / f"{scene_id:06d}/rgb/{img_id:06d}.jpg"),
        cv2.IMREAD_COLOR_RGB,
    )

    # run detection on single-image batch
    detections = detector.infer(img[None])[0]

    # visualize detections
    img_det = img.copy()
    for box, score in zip(detections.boxes, detections.scores):
        l, t, r, b = box.round().long().tolist()
        cv2.rectangle(img_det, (l, t), (r, b), (0, 0, 255))
        cv2.putText(img_det, f"{score:.3f}", (l, b), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))

    # run spyropose on crops
    for idx in torch.argsort(detections.scores, descending=True):
        box = detections.boxes[idx].cpu().numpy()
        cam_t_frame = detector.estimate_translation_from_bbox(box=box, K=K)
        crop, K_crop = spyro.crop_from_translation_est(img=img, K=K, cam_t_frame=cam_t_frame)
        spyro_out = spyro.infer(
            # None below adds batch and view dimension to crop image and intrinsics
            img=crop[None, None],
            K=K_crop[None, None],
            world_t_frame_est=cam_t_frame.reshape(1, 3, 1),
            top_k=top_k,
        )

        # TODO make output more digestible
        rot_idxs = spyro_out.rot_idxs[r_idx][0]
        probs = spyro_out.log_probs[r_idx][0].exp().cpu().numpy()
        rot = getattr(spyro, f"grid_{r_idx}")[rot_idxs].cpu().numpy()  # (n, 3, 3)

        print(f"sum(probs) at recursion level {r_idx} = {probs.sum():.3f}")

        # vis
        n_rows, n_cols = 2, 3 + 2
        fig = plt.figure(figsize=(n_cols * fig_scale, n_rows * fig_scale))
        spec = fig.add_gridspec(nrows=n_rows, ncols=n_cols)
        # - detections
        ax = fig.add_subplot(spec[:, :3])
        ax.imshow(img_det)
        ax.set_axis_off()
        # - crop
        ax = fig.add_subplot(spec[0, 3:5])
        ax.imshow(crop)
        ax.set_axis_off()
        # - so3 dist
        ax = fig.add_subplot(spec[1, 3:5], projection="mollweide")
        visualize_so3_probabilities(rotations=rot, probabilities=probs, ax=ax)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    jsonargparse.auto_cli(infer)

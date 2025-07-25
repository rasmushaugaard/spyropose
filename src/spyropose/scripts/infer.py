import json
from pathlib import Path

import cv2
import jsonargparse
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from spyropose.data.renderer import SimpleRenderer
from spyropose.detection.model import SpyroDetector
from spyropose.model import SpyroPoseModel
from spyropose.utils import if_none
from spyropose.vis import visualize_so3_probabilities


def infer(
    detector_path: Path,
    spyro_path: Path,
    scene_id: int,
    img_id: int,
    device="cuda:0",
    r_idx: int | None = None,
    top_k: int = 512,
    split="train_pbr",
    fig_scale=3.0,
    max_dist_renders=10_000,
    dist_render_prob_target=0.95,
    gamma=1.0,
    start_detection_idx: int = 0,
):
    # load models in eval mode and freeze weights
    detector = SpyroDetector.load_eval_freeze(detector_path, device)
    spyro = SpyroPoseModel.load_eval_freeze(spyro_path, device)
    assert detector.obj == spyro.cfg.obj
    obj = detector.obj
    split_dir = spyro.cfg.obj.root_dir / split

    r_idx = if_none(r_idx, spyro.cfg.r_last)

    # load data
    with open(split_dir / f"{scene_id:06d}/scene_camera.json") as f:
        K = np.asarray(json.load(f)[str(img_id)]["cam_K"]).reshape(3, 3)
    img = cv2.imread(
        str(split_dir / f"{scene_id:06d}/rgb/{img_id:06d}.jpg"),
        cv2.IMREAD_COLOR_RGB,
    )
    with open(split_dir / f"{scene_id:06d}/scene_gt.json") as f:
        instances_gt = [inst for inst in json.load(f)[str(img_id)] if inst["obj_id"] == obj.obj_id]
    world_R_objs_gt = np.array([inst["cam_R_m2c"] for inst in instances_gt]).reshape(-1, 3, 3)
    world_t_objs_gt = np.array([inst["cam_t_m2c"] for inst in instances_gt]).reshape(-1, 3, 1)
    world_t_frames_gt = world_t_objs_gt + world_R_objs_gt @ np.asarray(
        obj.frame.obj_t_frame
    ).reshape(3, 1)

    # run detection on single-image batch
    detections = detector.infer(img[None])[0]

    # detection visualization
    img_det = img.copy()
    for box, score in zip(detections.boxes, detections.scores):
        l, t, r, b = box.round().long().tolist()
        cv2.rectangle(img_det, (l, t), (r, b), (0, 0, 255))
        cv2.putText(img_det, f"{score:.3f}", (l, b), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))

    renderer = SimpleRenderer(
        obj.mesh, near=obj.frame.radius, far=obj.frame.radius * 1000, w=obj.crop_res
    )

    # run spyropose on crops
    for i, detection_idx in list(enumerate(torch.argsort(detections.scores, descending=True)))[
        start_detection_idx:
    ]:
        print(f"detection #{i}")
        box = detections.boxes[detection_idx].cpu().numpy()
        cam_t_frame = detector.estimate_translation_from_bbox(box=box, K=K)
        crop, K_crop = spyro.crop_from_translation_est(img=img, K=K, cam_t_frame=cam_t_frame)
        spyro_out = spyro.infer(
            # None below adds batch and view dimension to crop image and intrinsics
            img=crop[None, None],
            K=K_crop[None, None],
            world_t_frame_est=cam_t_frame.reshape(1, 3, 1),
            top_k=top_k,
        )

        gt_dists = np.linalg.norm(cam_t_frame[:, None] - world_t_frames_gt, axis=-2)
        gt_idx_guess = np.argmin(gt_dists)

        # visualization
        def overlay(render, alpha=0.5, color=(0, 0, 1)):
            rgb, alpha = render[..., :3] * color, render[..., 3:] * alpha
            return rgb * alpha + crop / 255.0 * (1 - alpha)

        def draw_frame_inplace(img, R, t):
            cv2.drawFrameAxes(img, K_crop, None, cv2.Rodrigues(R)[0], t, obj.frame.radius)
            np.clip(img, 0, 1, out=img)

        leaf_world_R_obj = spyro_out.leaf_world_R_obj[0].cpu().numpy()
        leaf_world_t_obj = spyro_out.leaf_world_t_obj[0].cpu().numpy()
        leaf_probabilities = spyro_out.leaf_probs[0].cpu().numpy()
        leaf_log_densities = spyro_out.leaf_log_density[0].cpu().numpy()
        print(f"leaf probability sum = {leaf_probabilities.sum().item():.4f}")

        # p(x) distribution render
        dist_render = np.zeros((obj.crop_res, obj.crop_res, 4))
        idx = np.argsort(-leaf_probabilities)
        n = min(
            max_dist_renders,
            np.searchsorted(np.cumsum(leaf_probabilities[idx]), dist_render_prob_target).item(),
        )
        for i in tqdm(idx[:n], desc="rendering distribution image, p(x)"):
            dist_render += leaf_probabilities[i] * renderer.render(
                K_crop, leaf_world_R_obj[i], leaf_world_t_obj[i]
            )

        # max_ll
        max_ll_left_idx = np.argmax(leaf_log_densities)
        max_ll_world_R_obj = leaf_world_R_obj[max_ll_left_idx]
        max_ll_world_t_obj = leaf_world_t_obj[max_ll_left_idx]
        render_max_ll = overlay(
            renderer.render(K=K_crop, R=max_ll_world_R_obj, t=max_ll_world_t_obj)
        )
        draw_frame_inplace(render_max_ll, max_ll_world_R_obj, max_ll_world_t_obj)

        # expected
        expected_world_R_obj = spyro_out.expected_world_R_obj[0].cpu().numpy()
        expected_world_t_obj = spyro_out.expected_world_t_obj[0].cpu().numpy()
        render_expected = overlay(
            renderer.render(K=K_crop, R=expected_world_R_obj, t=expected_world_t_obj)
        )
        draw_frame_inplace(render_expected, expected_world_R_obj, expected_world_t_obj)

        print(
            f"sqrt(angle variance) = {np.rad2deg(spyro_out.angle_variance[0].sqrt().item()):.2f} deg"
        )
        confidence_levels = torch.tensor([0.95, 0.99], device=device)
        R_conf_intervals = spyro_out.expected_R_confidence_interval(confidence_levels[None])[0]
        t_conf_intervals = spyro_out.expected_t_confidence_interval(confidence_levels[None])[0]
        for cl, R_ci, t_ci in zip(confidence_levels, R_conf_intervals, t_conf_intervals):
            print(
                f"Confidence interval {cl.item():.2f}: {np.rad2deg(R_ci.item()):.2f} deg, {t_ci.item():.1f} mm"
            )

        n_rows, n_cols = 2, 3 + 4
        fig = plt.figure(figsize=(n_cols * fig_scale, n_rows * fig_scale))
        spec = fig.add_gridspec(nrows=n_rows, ncols=n_cols)
        # - detections
        ax = fig.add_subplot(spec[:, :3])
        img_det_ = img_det.copy()
        l, t, r, b = box.round().astype(int).tolist()
        cv2.rectangle(img_det_, (l, t), (r, b), (0, 0, 255), 3)
        ax.imshow(img_det_)
        ax.set_axis_off()
        # - se3 dist
        ax = fig.add_subplot(spec[0, 3])
        ax.imshow(overlay(dist_render))
        ax.set_title("p(x)")
        ax.set_axis_off()
        # - max ll
        ax = fig.add_subplot(spec[0, 4])
        ax.imshow(render_max_ll)
        ax.set_title("argmax_x p(x)")
        ax.set_axis_off()
        # - expected
        ax = fig.add_subplot(spec[0, 5])
        ax.imshow(render_expected)
        ax.set_title("E_p x")
        ax.set_axis_off()
        # maybe ground truth
        ax = fig.add_subplot(spec[0, 6])
        R_guess, t_guess = world_R_objs_gt[gt_idx_guess], world_t_objs_gt[gt_idx_guess]
        gt_render = overlay(renderer.render(K=K_crop, R=R_guess, t=t_guess))
        draw_frame_inplace(gt_render, R_guess, t_guess)
        ax.imshow(gt_render)
        ax.set_title("maybe ground truth")
        ax.set_axis_off()
        # - so3 dist
        visualize_so3_probabilities(
            rotations=leaf_world_R_obj,
            rotations_gt=world_R_objs_gt[gt_idx_guess : gt_idx_guess + 1],
            probabilities=leaf_probabilities,
            ax=fig.add_subplot(spec[1, 3:], projection="mollweide"),
            gamma=gamma,
        )
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    jsonargparse.auto_cli(infer)

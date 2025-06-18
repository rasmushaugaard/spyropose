from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import einops
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import rotation_grid, se3_grid, translation_grid, unet, utils
from .obj import SpyroObjectConfig


@dataclass
class SpyroModelConfig:
    obj: SpyroObjectConfig

    embed_dim: int = 64
    n_layers: int = 3
    d_ff: int = 256
    val_top_k: int = 512
    vis_model: str = "unet18"
    position_scale: float = 1e-3

    # training
    lr: float = 1e-4
    weight_decay: float = 0.0
    n_samples: int = 32
    dropout: float = 0.1
    point_dropout: float = 0.1

    @property
    def n_keypoints(self):
        return len(self.obj.keypoints)

    @property
    def r_last(self):  # last recursion depth index
        return self.obj.recursion_depth - 1


class SpyroPoseModel(pl.LightningModule):
    keypoints: Tensor

    def __init__(self, cfg: SpyroModelConfig):
        super().__init__()
        self.cfg = cfg
        # stores hyperparams for future instantiation
        self.save_hyperparameters(logger=False)

        self.register_buffer("keypoints", torch.tensor(cfg.obj.keypoints, dtype=torch.float).mT)

        self.point_dropout = nn.Dropout2d(cfg.point_dropout)

        assert cfg.vis_model == "unet18"  # could experiment with more recent backbones
        self.vis_model = unet.ResNetUNet(feat_preultimate=128, n_class=128)
        vis_channels = 128

        self.n_pts = cfg.n_keypoints
        self.out_of_img_embedding = nn.Parameter(torch.randn(cfg.embed_dim))

        for r in range(cfg.obj.recursion_depth):
            # separate heads for each recursion (could experiment with shared heads)
            self.add_module(f"vis_lin_{r}", nn.Conv2d(vis_channels, cfg.embed_dim, 1))
            layers = [
                nn.Linear(cfg.n_keypoints * cfg.embed_dim, cfg.d_ff),
                nn.GELU(),
                nn.Dropout(p=cfg.dropout),
            ]
            for _ in range(cfg.n_layers - 1):
                layers += [
                    nn.Linear(cfg.d_ff, cfg.d_ff),
                    nn.GELU(),
                    nn.Dropout(p=cfg.dropout),
                ]
            layers.append(nn.Linear(cfg.d_ff, 1))
            setattr(self, f"mlp_{r}", nn.Sequential(*layers))

        for r in range(cfg.obj.recursion_depth):
            # Currently, the full rotation grids are generated on cpu and moved to the model's
            # device. The rotations could be computed on demand to allow deeper pyramids.
            grid = rotation_grid.generate_rotation_grid(recursion_level=r)
            self.register_buffer(
                f"grid_{r}",
                torch.from_numpy(grid),
                persistent=False,
            )

        # we backprop the loss independently per MLP to the output of the vision model, where the
        # grad is summed and then backpropped, to reduce memory usage.
        self.automatic_optimization = False

    def configure_optimizers(self):  # type: ignore
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

        def lr(step, warmup_steps=1000):
            warmup_factor = min(step, warmup_steps) / warmup_steps
            decay_step = max(step - warmup_steps, 0) / (
                self.trainer.estimated_stepping_batches - warmup_steps
            )
            return warmup_factor * (1 + np.cos(decay_step * np.pi)) / 2

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr)
        return [opt], [dict(scheduler=sched, interval="step")]

    def optimizer(self):
        opt = self.optimizers()
        assert isinstance(opt, torch.optim.AdamW)
        return opt

    def forward_vis(self, img) -> Tensor:
        img = (img - 0.5) / 0.2  # approx pre-training normalization
        out = self.vis_model(img)
        out = out["out"] if isinstance(out, dict) else out
        return out  # (b, c, h, w)

    @staticmethod
    def opencv2torch(K: Tensor, h: int, w: int):
        """
        from opencv coordinates [-0.5, res - 0.5]
        to torch coordinates [-1., 1.]
        """
        sx, sy = 2 / w, 2 / h
        return (
            torch.tensor([
                [sx, 0, 0.5 * sx - 1],
                [0, sy, 0.5 * sy - 1],
                [0, 0, 1],
            ]).to(K.device)
            @ K
        )

    def forward_head(
        self,
        K: Tensor,
        vis_feats: Tensor,
        rot: Tensor,
        pos: Tensor,
        r: int,
        batch_size: Optional[int] = None,
    ):
        b, n = rot.shape[:2]
        assert K.shape == (b, 3, 3)
        assert rot.shape == (b, n, 3, 3)
        assert pos.shape == (b, n, 3, 1)
        if self.training:
            assert batch_size is None  # to backprop, all activations need to be kept
        if batch_size is None:
            batch_size = n

        vis_feats = getattr(self, f"vis_lin_{r}")(vis_feats)

        lgts = []
        for i in range(0, n, batch_size):
            rot_batch = rot[:, i : i + batch_size]  # (b, bs, 3, 3)
            t_batch = pos[:, i : i + batch_size]  # (b, bs, 3)
            bs = rot_batch.shape[1]

            # project object pts
            # (b, bs, 3, n_pts)
            pts_cam = rot_batch @ self.keypoints + t_batch
            pts_img = K[:, None] @ pts_cam  # (b, bs, 3, n_pts)
            pts_img = pts_img[:, :, :2] / pts_img[:, :, 2:]  # [-1, 1]
            # (b, bs, 2, n_pts)

            # sample image features
            x = F.grid_sample(
                input=vis_feats,
                grid=einops.rearrange(pts_img, "b bs xy pts -> b bs pts xy"),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )  # (b, e, bs, pts)

            x = einops.rearrange(x, "b e bs pts -> b pts e bs")
            x = self.point_dropout(x)
            x = einops.rearrange(x, "b pts e bs -> pts (b bs) e")

            # assign out-of-image feature
            out_of_img_mask = einops.rearrange(
                (pts_img**2 > 1).any(dim=2),
                "b bs pts -> pts (b bs)",
            )
            x[out_of_img_mask] = self.out_of_img_embedding

            x = einops.rearrange(x, "pts bbs e -> bbs (pts e)")
            x = getattr(self, f"mlp_{r}")(x)  # (bbs,)
            lgts.append(x.view(b, bs))
        return torch.cat(lgts, dim=1)  # (b, n)

    @torch.no_grad()
    def forward_infer(
        self,
        img: Tensor,
        K: Tensor,
        world_t_obj_est: Tensor,  # TODO: frame!
        top_k: int,
        pos_grid_frame: Tensor,  # TODO: rename pos_grid_frame to basis
        world_R_cam: Optional[Tensor] = None,
        world_t_cam: Optional[Tensor] = None,
        pose_bs: Optional[int] = None,
    ):
        """Can be used for both single and multi-view inference"""
        b, c, _, h, w = img.shape
        device = self.device
        assert K.shape == (b, c, 3, 3)
        assert world_t_obj_est.shape == (b, 3, 1)
        assert pos_grid_frame.shape == (b, 3, 3)
        if world_R_cam is None or world_t_cam is None:
            world_R_cam = torch.eye(3, device=device, dtype=torch.float).repeat(b, 1, 1, 1)
            world_t_cam = torch.zeros(b, 1, 3, 1, device=device, dtype=torch.float)
        assert world_R_cam.shape == (b, c, 3, 3), world_R_cam.shape
        assert world_t_cam.shape == (b, c, 3, 1), world_t_cam.shape
        bc = b * c

        cam_R_world = world_R_cam.mT
        cam_t_world = -cam_R_world @ world_t_cam

        K = self.opencv2torch(K=K, h=h, w=w)

        vis_feats = self.forward_vis(img.view(bc, 3, h, w))

        rot_idxs: list[Tensor] = []
        pos_idxs: list[Tensor] = []
        expand_idxs: list[Tensor] = []
        log_probs: list[Tensor] = []
        log_prob = torch.tensor([])

        # recursion 0
        rot_idx, pos_idx = se3_grid.get_idx_recursion_0(b=b, device=device)
        n = rot_idx.shape[1]
        # we cover all bins at recursion 0, log(sum_prob = 1) = 0
        log_sum_prob_expanded = torch.zeros(b, 1, device=device)

        for r in range(self.cfg.obj.recursion_depth):
            if r > 0:
                # take top k bins
                k = min(top_k, log_prob.shape[1])
                log_prob_expanded, expand_idx = torch.topk(log_prob, k=k, dim=1)  # 2 x (b, k)
                expand_idxs.append(expand_idx)
                log_sum_prob_expanded = torch.logsumexp(
                    log_prob_expanded, dim=1, keepdim=True
                )  # (b, 1)
                rot_idx = rot_idx.gather(1, expand_idx)
                pos_idx = pos_idx[  # TODO: gather here as well?
                    torch.arange(b, device=device).view(b, 1),
                    expand_idx,
                ]
                # and expand them
                rot_idx, pos_idx = se3_grid.expand(rot_idx=rot_idx, pos_idx=pos_idx, flat=True)
                n = rot_idx.shape[1]
            assert rot_idx.shape == (b, n)
            assert pos_idx.shape == (b, n, 3)

            world_R_obj: Tensor = getattr(self, f"grid_{r}")[rot_idx]  # (b, n, 3, 3)
            world_t_obj = translation_grid.grid2pos(
                grid=pos_idx, t_est=world_t_obj_est, grid_frame=pos_grid_frame, r=r + 1
            )  # (b, n, 3, 1)

            cam_R_obj = cam_R_world.unsqueeze(2) @ world_R_obj.unsqueeze(1)  # (b, c, n, 3, 3)
            cam_t_obj = cam_R_world.unsqueeze(2) @ world_t_obj.unsqueeze(
                1
            ) + cam_t_world.unsqueeze(2)  # (b, c, n, 3, 1)

            lgts = self.forward_head(
                K=K.view(bc, 3, 3),
                vis_feats=vis_feats,
                rot=cam_R_obj.view(bc, n, 3, 3),
                pos=cam_t_obj.view(bc, n, 3, 1),
                r=r,
                batch_size=pose_bs,
            ).view(b, c, n)

            # evaluate expanded poses
            log_prob = (
                torch.log_softmax(
                    lgts.mean(dim=1),  # TODO: sum with temp param instead
                    dim=1,
                )
                + log_sum_prob_expanded
            )  # (b, n)

            rot_idxs.append(rot_idx)
            pos_idxs.append(pos_idx)
            log_probs.append(log_prob)

        # Indicate that no idxs from the last recursion are expanded
        expand_idxs.append(torch.empty(b, 0, dtype=torch.long, device=device))

        return dict(
            rot_idxs=rot_idxs,
            pos_idxs=pos_idxs,
            log_probs=log_probs,
            expand_idxs=expand_idxs,
        )

    def forward_train(
        self,
        img: Tensor,
        K: Tensor,
        t_est: torch.Tensor,
        t_target: Tensor,
        t_grid_frame: Tensor,
        rot_idx_target_rlast: Tensor,
        R_offset: Tensor,
    ):
        b, _, h, w = img.shape
        device = self.device
        assert K.shape == (b, 3, 3)
        assert t_est.shape == (b, 3, 1)
        assert R_offset.shape == (b, 3, 3)
        assert rot_idx_target_rlast.shape == (b,), rot_idx_target_rlast.shape
        assert t_grid_frame.shape == (b, 3, 3)
        bf = 64  # branch factor

        K = self.opencv2torch(K=K, h=h, w=w)

        vis_feats = self.forward_vis(img)
        # accumulate gradients at vis_feats_detached before backprop'ing them to vision model
        vis_feats_ = vis_feats.detach()

        if self.training:
            vis_feats_.requires_grad = True
            self.optimizer().zero_grad()

        # subtract up to one se3 recursion 0 (pos rec. 1) grid spacing
        # TODO: why 0.5?
        t_est = t_est - t_grid_frame @ torch.rand(b, 3, 1, device=device) * 0.5

        rot_idx, pos_idx = se3_grid.get_idx_recursion_0(b=b, device=device, extended=True)
        n = rot_idx.shape[1]

        pos_idx_target_rlast = translation_grid.pos2grid(
            pos=t_target, t_est=t_est, grid_frame=t_grid_frame, r=self.cfg.r_last + 1
        )

        losses: list[Tensor] = []
        s = 0
        log_q = torch.tensor([])
        lgts = torch.tensor([])

        for r in range(self.cfg.obj.recursion_depth):
            assert rot_idx.shape == (
                b,
                n,
            ), f"{b=}, {n=})"
            assert pos_idx.shape == (b, n, 3)

            # Concatenate the true (nearest) grid point to the sampled points,
            # to process all of them in parallel.
            # Nearest rotation is found in the dataloader because it's cpu-bound.
            rot_idx_target_r = rot_idx_target_rlast.div(
                8 ** (self.cfg.r_last - r), rounding_mode="trunc"
            )
            rot_idx = torch.cat(
                (
                    rot_idx_target_r.view(b, 1),
                    rot_idx,
                ),
                dim=1,
            )  # (b, 1+n)

            pos_idx_target_r = pos_idx_target_rlast.div(
                2 ** (self.cfg.r_last - r), rounding_mode="trunc"
            )
            pos_idx = torch.cat(
                (pos_idx_target_r.view(b, 1, 3), pos_idx.view(b, n, 3)), dim=1
            )  # (b, 1+n, 3)
            pos = translation_grid.grid2pos(
                grid=pos_idx, t_est=t_est, grid_frame=t_grid_frame, r=r + 1
            )

            # forward
            lgts = self.forward_head(
                K=K,
                vis_feats=vis_feats_,
                r=r,
                rot=R_offset.unsqueeze(1).mT
                @ getattr(self, f"grid_{r}")[rot_idx],  # (b, 1+n, 3, 3),
                pos=pos,
            )  # (b, 1+n)

            # retain lgts for sampling
            lgts_q = lgts[:, 1:].detach()  # (b, n)

            # loss
            if r == 0:
                lgts[:, 1:].sub_(np.log(bf / n))
            else:
                lgts[:, 1:].view(b, s, bf).sub_(log_q.view(b, s, 1))

            loss = F.cross_entropy(
                lgts,
                torch.zeros(b, device=device, dtype=int),
            )
            losses.append(loss.detach())
            if self.training:
                loss.backward()  # backwards to vis_feats_detached

            # samples for next recursion
            if r == 0:
                # sample across all
                log_q, sample_idx = utils.sample_from_lgts(lgts=lgts_q, n=self.cfg.n_samples)
                rot_idx = rot_idx[:, 1:].gather(1, sample_idx)
                pos_idx = pos_idx[:, 1:][
                    torch.arange(b, device=device).view(b, 1), sample_idx
                ]  # (b, s, 3)
            else:
                # sample one per expanded element
                assert s == self.cfg.n_samples
                log_q_, sample_idx = utils.sample_from_lgts(lgts=lgts_q.view(b, s, bf), n=1)
                # sampling probability is recursive
                log_q = log_q + log_q_.view(b, s)
                rot_idx = rot_idx[:, 1:].view(b, s, bf).gather(2, sample_idx).squeeze(2)  # (b, s)
                pos_idx = pos_idx[:, 1:].view(b, s, bf, 3)[
                    torch.arange(b, device=device).view(b, 1),
                    torch.arange(s, device=device).view(1, s),
                    sample_idx.view(b, s),
                ]  # (b, s, 3)

            # expand next recursion
            s = rot_idx.shape[1]
            n = s * bf
            rot_idx, pos_idx = se3_grid.expand(
                rot_idx=rot_idx, pos_idx=pos_idx, flat=True
            )  # (b, s * bf), (b, s * bf, 3)

        if self.training:
            vis_feats.backward(gradient=vis_feats_.grad)
            self.optimizer().step()
            self.lr_schedulers().step()  # type: ignore

        return losses

    def step(self, batch, log_prefix):
        losses = self.forward_train(
            img=batch["img"],
            K=batch["K"],
            t_est=batch["t_est"],
            t_grid_frame=batch["t_grid_frame"],
            t_target=batch["t"],
            # only provide the same modality / symmetry during training
            # assuming no knowledge about symmetries during training:
            rot_idx_target_rlast=batch[f"rot_idx_target_{self.cfg.obj.recursion_depth - 1}"][:, 0],
            R_offset=batch["R_offset"],
        )
        for r in range(self.cfg.obj.recursion_depth):
            self.log(f"{log_prefix}/loss_{r}", losses[r], add_dataloader_idx=False)
        self.log(
            f"{log_prefix}/loss",
            torch.stack(losses).mean(),
            add_dataloader_idx=False,
        )

    def training_step(self, batch, *_):
        self.step(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        prefix = f"val_{dataloader_idx}"
        self.step(batch, prefix)
        ll = self.eval_step(batch=batch, pose_bs=10_000, top_k=self.cfg.val_top_k)
        for r in range(self.cfg.obj.recursion_depth):
            self.log(f"{prefix}/ll_{r}", ll[..., r].mean(), add_dataloader_idx=False)

    def forward_infer_batch(
        self, batch, top_k: Optional[int] = None, pose_bs: Optional[int] = None
    ):
        if top_k is None:
            top_k = self.cfg.val_top_k

        return self.forward_infer(
            img=batch["img"].unsqueeze(1),
            K=batch["K"].unsqueeze(1),
            world_t_obj_est=batch["t_est"],
            pos_grid_frame=batch["t_grid_frame"],
            top_k=top_k,
            pose_bs=pose_bs,
        )

    def eval_step(self, batch, pose_bs=10_000, top_k=512):
        out = self.forward_infer_batch(batch, top_k=top_k, pose_bs=pose_bs)
        _, ll = se3_grid.locate_poses_in_pyramid(
            q_rot_idx_rlast=batch[f"rot_idx_target_{self.cfg.r_last}"],
            log_probs=out["log_probs"],
            rot_idxs=out["rot_idxs"],
            t_est=batch["t_est"],
            pos_grid_frame=batch["t_grid_frame"],
            q_pos=batch["t"].unsqueeze(1),
            pos_idxs=out["pos_idxs"],
            position_scale=self.cfg.position_scale,
        )
        return ll

    @staticmethod
    def load_from_run_id(run_id, return_fp=False):
        # TODO
        ckpt_path = Path("data") / "spyropose" / run_id / "checkpoints"
        ckpt_path = list(ckpt_path.glob("*.ckpt"))
        assert len(ckpt_path) == 1
        ckpt_path = ckpt_path[0]
        model = SpyroPoseModel.load_from_checkpoint(ckpt_path)
        if return_fp:
            return model, ckpt_path
        return model

import warnings
from pathlib import Path
from typing import Union

import einops
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torchvision.ops

from . import rotation_grid, se3_grid, translation_grid, unet, utils

Tensor = torch.Tensor


class SpyroPoseModel(pl.LightningModule):
    def __init__(
        self,
        dataset_name: str,
        obj_name: str,
        obj_radius: float,
        pts: Union[str, np.ndarray],
        train_se3=True,  # so3 otherwise
        importance_sampling=True,
        batchnorm=False,
        lr=1e-4,
        weight_decay=0.0,
        embed_dim=64,
        n_pts=16,
        n_layers=3,
        d_ff=256,
        vis_model="unet18",
        crop_res=224,
        recursion_depth=6,
        n_samples=32,
        random_offset_translation=True,
        random_offset_rotation=True,
        dropout=0.0,
        point_dropout=0.0,
        val_top_k=512,
        number_fourier_components=3,
        position_scale: float = None,
        debug=False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        if kwargs:
            warnings.warn(f"kwargs ignored: {kwargs}")

        self.dataset_name = dataset_name
        self.obj_name = obj_name
        self.train_se3 = train_se3
        self.obj_radius = obj_radius
        self.pts = pts
        self.n_pts = n_pts
        self.crop_res = crop_res
        self.importance_sampling = importance_sampling
        self.embed_dim = embed_dim
        self.d_ff = d_ff
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.point_dropout = nn.Dropout2d(point_dropout) if point_dropout > 0 else False
        self.lr = lr
        self.weight_decay = weight_decay
        self.recursion_depth = recursion_depth
        self.n_samples = n_samples
        self.random_offset_translation = random_offset_translation
        self.random_offset_rotation = random_offset_rotation
        self.val_top_k = val_top_k
        self.number_fourier_components = number_fourier_components
        self.debug = debug
        self.position_scale = position_scale

        if batchnorm:
            # Note: normalizing over L in (N, C, L) with importance sampling could be
            # problematic since batchnorm doesn't use the IS weights. We haven't
            # experienced problems with this, but using the IS weights in batchnorm or
            # using another normalization that does not rely on normalizing over L might
            # help.
            def NormLayer(f):
                return [nn.BatchNorm1d(f)]
        else:

            def NormLayer(f):
                return []

        if self.pts is None:  # model without keypoints, similar to ImplicitPDF
            assert not train_se3
            assert vis_model == "resnet50"
            self.vis_model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
            self.vis_model.fc = nn.Linear(self.vis_model.fc.in_features, d_ff)
            if number_fourier_components == 0:
                d_pos = 9
            else:
                d_pos = 9 * 2 * number_fourier_components
            self.encoder = nn.Conv1d(d_pos, self.d_ff, 1)
            for r in range(self.recursion_depth):
                setattr(
                    self,
                    f"head_{r}",
                    nn.Sequential(
                        *NormLayer(self.d_ff),
                        nn.GELU(),
                        nn.Conv1d(self.d_ff, self.d_ff, 1),
                        *NormLayer(self.d_ff),
                        nn.GELU(),
                        nn.Conv1d(self.d_ff, self.d_ff, 1),
                        *NormLayer(self.d_ff),
                        nn.GELU(),
                        nn.Conv1d(self.d_ff, 1, 1),
                    ),
                )
        else:  # model with keypoints
            assert vis_model == "unet18"
            self.vis_model = unet.ResNetUNet(feat_preultimate=128, n_class=128)
            vis_channels = 128
            self.out_of_img_embedding = nn.Parameter(torch.randn(embed_dim))

            for r in range(self.recursion_depth):
                # separate heads for each recursion (could experiment with shared heads)
                setattr(self, f"vis_lin_{r}", nn.Conv2d(vis_channels, embed_dim, 1))

                # init position of points
                if isinstance(self.pts, str) and self.pts == "box":
                    assert n_pts == 16
                    pts = torch.stack(
                        torch.meshgrid(
                            *([torch.tensor([-1.0, 1.0])] * 3), indexing="ij"
                        ),
                        dim=0,
                    )
                    pts = pts.flatten(1)  # (3, 2, 2, 2) -> (3, 8)
                    pts = torch.cat([pts, pts * 0.5], dim=1) * obj_radius
                    assert pts.shape == (3, 16), pts.shape
                elif isinstance(self.pts, np.ndarray):
                    pts = torch.from_numpy(self.pts).float()
                else:
                    raise ValueError()
                setattr(self, f"pts_pos_{r}", nn.Parameter(pts, requires_grad=False))

                drop_layer = [nn.Dropout(p=dropout)] if dropout > 0 else []
                layers = [
                    nn.Linear(n_pts * embed_dim, d_ff),
                    *NormLayer(d_ff),
                    nn.GELU(),
                    *drop_layer,
                ]
                for _ in range(n_layers - 1):
                    layers += [
                        nn.Linear(d_ff, d_ff),
                        *NormLayer(d_ff),
                        nn.GELU(),
                        *drop_layer,
                    ]
                layers.append(nn.Linear(d_ff, 1))
                setattr(self, f"mlp_{r}", nn.Sequential(*layers))

        for r in range(self.recursion_depth):
            """
            Currently, the full rotation grids are generarted on cpu and moved to the 
            model's device. 
            The rotations could be computed on demand to allow deeper pyramids.
            """
            grid = rotation_grid.generate_rotation_grid(recursion_level=r)
            self.register_buffer(
                f"grid_{r}",
                torch.from_numpy(grid),
                persistent=False,
            )

        # we backprop the loss independently per MLP to the output of the vision model,
        # where the grad is summed and then backpropped, to reduce memory usage.
        self.automatic_optimization = False

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        def lr(step, warmup_steps=1000):
            warmup_factor = min(step, warmup_steps) / warmup_steps
            decay_step = max(step - warmup_steps, 0) / (
                self.trainer.estimated_stepping_batches - warmup_steps
            )
            return warmup_factor * (1 + np.cos(decay_step * np.pi)) / 2

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr)
        return [opt], [dict(scheduler=sched)]

    def forward_vis(self, img) -> Tensor:
        img = (img - 0.5) / 0.2  # approx pre-training normalization
        out = self.vis_model(img)
        out = out["out"] if isinstance(out, dict) else out
        return out  # (b, c, h, w)

    @staticmethod
    def opencv2torch(K: Tensor, h, w):
        """
        from opencv coordinates [-0.5, res - 0.5]
        to torch coordinates [-1., 1.]
        """
        sx, sy = 2 / w, 2 / h
        return (
            torch.tensor(
                [
                    [sx, 0, 0.5 * sx - 1],
                    [0, sy, 0.5 * sy - 1],
                    [0, 0, 1],
                ]
            ).to(K.device)
            @ K
        )

    @staticmethod
    def pos_enc(x, nf):
        # (..., d) -> (..., d * nf * 2)
        if nf == 0:
            return x
        t = x[..., None] * 2 ** torch.arange(nf, device=x.device)
        return torch.cat((t.sin(), t.cos()), dim=-1).flatten(-2)

    def forward_head_no_kp(
        self,
        K: Tensor,
        vis_feats: Tensor,
        rot: Tensor,
        r: int,
        batch_size: int = None,
    ):
        b, n = rot.shape[:2]
        assert K.shape == (b, 3, 3)
        assert rot.shape == (b, n, 3, 3)
        if self.training:
            assert batch_size is None  # to backprop, all activations need to be kept
        if batch_size is None:
            batch_size = n
        assert vis_feats.shape == (b, self.d_ff)

        lgts = []
        for i in range(0, n, batch_size):
            rot_batch = rot[:, i : i + batch_size]  # (b, bs, 3, 3)
            bs = rot_batch.shape[1]
            x = einops.rearrange(
                self.pos_enc(
                    x=rot_batch.view(b, bs, 9),
                    nf=self.number_fourier_components,
                ),
                "b bs e -> b e bs",
            )
            x = vis_feats.unsqueeze(2) + self.encoder(x)
            x = getattr(self, f"head_{r}")(x)  # (b, bs, 1)

            lgts.append(x.view(b, bs))
        return torch.cat(lgts, dim=1)  # (b, n)

    def forward_head(
        self,
        K: Tensor,
        vis_feats: Tensor,
        rot: Tensor,
        pos: Tensor,
        r: int,
        batch_size: int = None,
    ):
        if self.pts is None:
            return self.forward_head_no_kp(
                K=K, vis_feats=vis_feats, rot=rot, r=r, batch_size=batch_size
            )

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
            pts_cam = rot_batch @ getattr(self, f"pts_pos_{r}") + t_batch
            pts_img = K[:, None] @ pts_cam  # (b, bs, 3, n_pts)
            pts_img = pts_img[:, :, :2] / pts_img[:, :, 2:]  # [-1, 1]
            # (b, bs, 2, n_pts)

            # sample image features
            feat_sampled = F.grid_sample(
                input=vis_feats,
                grid=einops.rearrange(pts_img, "b bs xy pts -> b bs pts xy"),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )  # (b, e, bs, pts)

            if self.point_dropout:
                feat_sampled = einops.rearrange(
                    feat_sampled, "b e bs pts -> b pts e bs"
                )
                feat_sampled = self.point_dropout(feat_sampled)
                feat_sampled = einops.rearrange(
                    feat_sampled, "b pts e bs -> b e bs pts"
                )

            feat_sampled = einops.rearrange(feat_sampled, "b e bs pts -> pts (b bs) e")

            # assign out of image feature
            feat_sampled[
                einops.rearrange(
                    ((pts_img < -1) | (1 < pts_img)).any(dim=2),
                    "b bs pts -> pts (b bs)",
                )
            ] = self.out_of_img_embedding

            mlp = getattr(self, f"mlp_{r}")
            x = einops.rearrange(feat_sampled, "pts bbs e -> bbs (pts e)")
            x = mlp[0](x)  # (bbs, d_ff)
            for layer in mlp[1:]:
                x = layer(x)  # (bbs, 1)
            lgts.append(x.view(b, bs))
        return torch.cat(lgts, dim=1)  # (b, n)

    @torch.no_grad()
    def forward_infer(
        self,
        img: Tensor,
        K: Tensor,
        world_t_obj_est: Tensor,
        top_k: int,
        pos_grid_frame: torch.Tensor = None,
        world_R_cam: Tensor = None,
        world_t_cam: Tensor = None,
        pose_bs: int = None,
    ):
        """Can be used for both single and multi-view inference"""
        b, c, _, h, w = img.shape
        device = self.device
        assert K.shape == (b, c, 3, 3)
        assert world_t_obj_est.shape == (b, 3, 1)
        if pos_grid_frame is None:
            se3 = False
        else:
            assert pos_grid_frame.shape == (b, 3, 3)
            se3 = True
        if se3 != self.train_se3:
            warnings.warn(f"train_se3 = {self.train_se3}, se3: {se3}")
        if world_R_cam is None or world_t_cam is None:
            assert c == 1
            world_R_cam = torch.eye(3, device=device, dtype=torch.float).view(
                1, 1, 3, 3
            )
            world_t_cam = torch.zeros(1, 1, 3, 1, device=device, dtype=torch.float)
        else:
            assert world_R_cam.shape == (b, c, 3, 3)
            assert world_t_cam.shape == (b, c, 3, 1)
        bc = b * c

        cam_R_world = world_R_cam.mT
        cam_t_world = -cam_R_world @ world_t_cam

        K = self.opencv2torch(K=K, h=h, w=w)

        vis_feats = self.forward_vis(img.view(bc, 3, h, w))

        rot_idxs, pos_idxs, expand_idxs = [], [], []
        log_probs = []
        log_prob = None

        for r in range(self.recursion_depth):
            if r == 0:
                rot_idx, pos_idx = se3_grid.get_idx_recursion_0(
                    b=b, device=device, se3=se3
                )
                n = rot_idx.shape[1]
                # we cover all bins at recursion 0, log(sum_prob = 1) = 0
                log_sum_prob_expanded = torch.zeros(b, 1, device=device)
            else:
                # take top k bins
                k = min(top_k, log_prob.shape[1])
                log_prob_expanded, expand_idx = torch.topk(
                    log_prob, k=k, dim=1
                )  # 2 x (b, k)
                expand_idxs.append(expand_idx)
                log_sum_prob_expanded = torch.logsumexp(
                    log_prob_expanded, dim=1, keepdim=True
                )  # (b, 1)
                rot_idx = rot_idx.gather(1, expand_idx)
                if se3:
                    pos_idx = pos_idx[
                        torch.arange(b, device=device).view(b, 1),
                        expand_idx,
                    ]
                # and expand them
                rot_idx, pos_idx = se3_grid.expand(
                    rot_idx=rot_idx,
                    pos_idx=pos_idx,
                    flat=True,
                )
                n = rot_idx.shape[1]
            assert rot_idx.shape == (b, n)
            if se3:
                assert pos_idx.shape == (b, n, 3)

            world_R_obj = getattr(self, f"grid_{r}")[rot_idx]  # (b, n, 3, 3)
            if se3:
                world_t_obj = translation_grid.grid2pos(
                    grid=pos_idx,
                    t_est=world_t_obj_est,
                    grid_frame=pos_grid_frame,
                    r=r + 1,
                )  # (b, n, 3, 1)
            else:
                world_t_obj = einops.repeat(
                    world_t_obj_est, "b d 1 -> b n d 1", n=n, d=3
                )

            cam_R_obj = cam_R_world.unsqueeze(2) @ world_R_obj.unsqueeze(
                1
            )  # (b, c, n, 3, 3)
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
                    lgts.mean(dim=1),  # mean lgts over cams
                    dim=1,
                )
                + log_sum_prob_expanded
            )  # (b, n)

            rot_idxs.append(rot_idx)
            pos_idxs.append(pos_idx)
            log_probs.append(log_prob)

        # Indicate that none from the last recursion are expanded
        expand_idxs.append(torch.empty(b, 0, dtype=int, device=device))

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
        if self.train_se3:
            assert t_grid_frame.shape == (b, 3, 3)
            bf = 64  # branch factor
        else:
            bf = 8

        K = self.opencv2torch(K=K, h=h, w=w)

        vis_feats = self.forward_vis(img)
        # accumulate gradients at vis_feats_detached before backwarding them to vision model
        vis_feats_ = vis_feats.detach()

        if self.training:
            vis_feats_.requires_grad = True
            opt = self.optimizers()
            opt.zero_grad()

        if self.train_se3 and self.random_offset_translation:
            # subtract up to one se3 recursion 0 (pos rec. 1) grid spacing
            t_est = t_est - t_grid_frame @ torch.rand(b, 3, 1, device=device) * 0.5

        rot_idx, pos_idx = se3_grid.get_idx_recursion_0(
            b=b,
            device=device,
            extended=self.random_offset_translation,
            se3=self.train_se3,
        )
        n = rot_idx.shape[1]

        rlast = self.recursion_depth - 1
        if self.train_se3:
            pos_idx_target_rlast = translation_grid.pos2grid(
                pos=t_target, t_est=t_est, grid_frame=t_grid_frame, r=rlast + 1
            )

        losses = []
        s = None
        log_q = None

        for r in range(self.recursion_depth):
            if not self.importance_sampling:
                # sample negatives uniformly at each recursion,
                # unless n < n_samples, in which case the whole grid is evaluated for lower variance
                if self.train_se3:
                    sidelen_0 = 3 if self.random_offset_translation else 2
                    sidelen_r = sidelen_0 * 2**r
                    n_pos = sidelen_r**3
                    n_rot = 72 * 8**r
                    n = n_pos * n_rot
                    assert n == (72 * sidelen_0**3) * 64**r
                    if n < self.n_samples:
                        rot_idx = torch.arange(72 * 8**r, device=device)  # (n_rot,)
                        side_idx = torch.arange(sidelen_r, device=device)
                        pos_idx = torch.stack(
                            torch.meshgrid([side_idx] * 3, indexing="ij"), dim=-1
                        )
                        assert pos_idx.shape == (sidelen_r, sidelen_r, sidelen_r, 3)
                        pos_idx = pos_idx.view(n_pos, 3)

                        rot_idx = einops.repeat(
                            rot_idx, "nr -> b (nr np)", b=b, np=n_pos
                        )
                        pos_idx = einops.repeat(
                            pos_idx, "np d -> b (nr np) d", b=b, nr=n_rot
                        )
                    else:
                        rot_idx = torch.randint(
                            n_rot, size=(b, self.n_samples), device=device
                        )
                        pos_idx = torch.randint(
                            sidelen_r, size=(b, self.n_samples, 3), device=device
                        )
                        n = self.n_samples
                else:
                    n = 72 * 8**r
                    if n <= self.n_samples:
                        rot_idx = einops.repeat(
                            torch.arange(n, device=device), "n -> b n", b=b
                        )
                    else:
                        rot_idx = torch.randint(
                            n, size=(b, self.n_samples), device=device
                        )
                        n = self.n_samples

            assert rot_idx.shape == (
                b,
                n,
            ), f"expected ({b}, {n}) but got {rot_idx.shape}"
            if self.train_se3:
                assert pos_idx.shape == (b, n, 3)

            # Concatenate the true (nearest) grid point to the sampled points,
            # to process all of them in parallel.
            # Nearest rotation is found in the dataloader because it's using a cpu-bound library
            rot_idx_target_r = rot_idx_target_rlast.div(
                8 ** (rlast - r), rounding_mode="trunc"
            )
            rot_idx = torch.cat(
                (
                    rot_idx_target_r.view(b, 1),
                    rot_idx,
                ),
                dim=1,
            )  # (b, 1+n)

            if self.train_se3:
                pos_idx_target_r = pos_idx_target_rlast.div(
                    2 ** (rlast - r), rounding_mode="trunc"
                )
                pos_idx = torch.cat(
                    (pos_idx_target_r.view(b, 1, 3), pos_idx.view(b, n, 3)), dim=1
                )  # (b, 1+n, 3)
                pos = translation_grid.grid2pos(
                    grid=pos_idx, t_est=t_est, grid_frame=t_grid_frame, r=r + 1
                )
            else:
                pos = einops.repeat(t_est, "b d 1 -> b np1 d 1", np1=n + 1)

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
            if self.importance_sampling:
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
            if self.importance_sampling:
                if r == 0:
                    # sample across all
                    log_q, sample_idx = utils.sample_from_lgts(
                        lgts=lgts_q, n=self.n_samples
                    )
                    rot_idx = rot_idx[:, 1:].gather(1, sample_idx)
                    if self.train_se3:
                        pos_idx = pos_idx[:, 1:][
                            torch.arange(b, device=device).view(b, 1), sample_idx
                        ]  # (b, s, 3)
                else:
                    # sample one per expanded element
                    assert s == self.n_samples
                    log_q_, sample_idx = utils.sample_from_lgts(
                        lgts=lgts_q.view(b, s, bf), n=1
                    )
                    # sampling probability is recursive
                    log_q = log_q + log_q_.view(b, s)
                    rot_idx = (
                        rot_idx[:, 1:].view(b, s, bf).gather(2, sample_idx).squeeze(2)
                    )  # (b, s)
                    if self.train_se3:
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
            opt.step()
            self.lr_schedulers().step()
        return dict(lgts=lgts, idxs=rot_idx, losses=losses)

    def step(self, batch, log_prefix):
        res = self.forward_train(
            img=batch["img"],
            K=batch["K"],
            t_est=batch["t_est"],
            t_grid_frame=batch["t_grid_frame"] if self.train_se3 else None,
            t_target=batch["t"] if self.train_se3 else None,
            # only provide the same modality / symmetry during training
            # assuming no knowledge about symmetries during training:
            rot_idx_target_rlast=batch[f"rot_idx_target_{self.recursion_depth - 1}"][
                :, 0
            ],
            R_offset=batch["R_offset"],
        )
        for r in range(self.recursion_depth):
            self.log(
                f"{log_prefix}/loss_{r}", res["losses"][r], add_dataloader_idx=False
            )
        self.log(
            f"{log_prefix}/loss",
            torch.stack(res["losses"]).mean(),
            add_dataloader_idx=False,
        )

    def training_step(self, batch, *_):
        self.step(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=1):
        prefix = ["val", "test"][dataloader_idx]
        self.step(batch, prefix)
        ll = self.eval_step(
            batch=batch,
            se3=self.train_se3,
            pose_bs=10_000,
            top_k=self.val_top_k,
            position_scale=self.position_scale,
        )
        for r in range(self.recursion_depth):
            self.log(f"{prefix}/ll_{r}", ll[..., r].mean(), add_dataloader_idx=False)

    def forward_infer_batch(
        self, batch, se3: bool = None, top_k: int = None, pose_bs: int = None
    ):
        if se3 is None:
            se3 = self.train_se3
        if top_k is None:
            top_k = self.val_top_k

        return self.forward_infer(
            img=batch["img"].unsqueeze(1),
            K=batch["K"].unsqueeze(1),
            world_t_obj_est=batch["t_est"],
            pos_grid_frame=batch["t_grid_frame"] if se3 else None,
            top_k=top_k,
            pose_bs=pose_bs,
        )

    def eval_step(
        self, batch, se3: bool, pose_bs=10_000, top_k=512, position_scale=None
    ):
        out = self.forward_infer_batch(batch, se3=se3, top_k=top_k, pose_bs=pose_bs)

        _, ll = se3_grid.locate_poses_in_pyramid(
            q_rot_idx_rlast=batch[f"rot_idx_target_{self.recursion_depth - 1}"],
            log_probs=out["log_probs"],
            rot_idxs=out["rot_idxs"],
            **(
                dict(
                    t_est=batch["t_est"],
                    pos_grid_frame=batch["t_grid_frame"],
                    q_pos=batch["t"].unsqueeze(1),
                    pos_idxs=out["pos_idxs"],
                    position_scale=position_scale,
                )
                if se3
                else {}
            ),
        )

        return ll

    @staticmethod
    def load_from_run_id(run_id, return_fp=False):
        ckpt_path = Path("data") / "spyropose" / run_id / "checkpoints"
        ckpt_path = list(ckpt_path.glob("*.ckpt"))
        assert len(ckpt_path) == 1
        ckpt_path = ckpt_path[0]
        model = SpyroPoseModel.load_from_checkpoint(ckpt_path)
        if return_fp:
            return model, ckpt_path
        return model

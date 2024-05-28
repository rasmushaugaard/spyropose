"""
Images are rendered in the dataloaders,
but fixed by seed to simulate a finite dataset.
"""

from functools import cached_property

import numpy as np
import torch
import torch.utils.data
from scipy.spatial.transform import Rotation

from .. import rotation_grid
from . import symsol_objects
from .renderer import SimpleRenderer


class SymsolDataset(torch.utils.data.Dataset):
    train_len = 45_000
    test_len = 5_000

    def __init__(
        self,
        name: str,
        split="train",
        crop_res=224,
        z=1.5,
        padding=1.3,
        random_rotate_grid=True,
        recursion_depth=6,
        opengl_device_index=0,
        **kwargs,
    ):
        super().__init__()
        assert split in {"train", "test"}
        self.mesh, self.syms = symsol_objects.obj_gen_dict[name]()
        self.center = self.mesh.bounding_sphere.primitive.center
        self.z = z
        assert np.allclose(self.center, 0.0)
        self.r = self.mesh.bounding_sphere.primitive.radius
        assert np.allclose(self.r, 0.5), self.r

        self.random_rotate_grid = random_rotate_grid
        self.device_index = opengl_device_index
        self.test = split == "test"
        self.crop_res = crop_res
        self.r_last = recursion_depth - 1

        self.t = np.array([0, 0, 1.5]).reshape(3, 1).astype(np.float32)
        self.R = (
            Rotation.random(num=len(self), random_state=1 if self.test else 0)
            .as_matrix()
            .astype(np.float32)
        )

        # get the intrinsics of the crop around the estimated position
        #   scale: f * d * pad / z = res
        f = self.crop_res * z / (self.r * 2 * padding)
        #   center: fx/z + cx = res / 2 - 0.5
        c = self.crop_res / 2 - 0.5
        self.K = np.array(
            (
                (f, 0, c),
                (0, f, c),
                (0, 0, 1),
            )
        ).astype(np.float32)

    @cached_property
    def renderer(self):
        # lazy instantiation of renderer, such that all workers get their own
        # opengl context
        # TODO: it would be more foolproof to detect if ctx is not owned by this process
        return SimpleRenderer(
            mesh=self.mesh,
            w=self.crop_res,
            device_idx=self.device_index,
            components=3,
            dtype="f4",
            near=self.z - self.r,
            far=self.z + self.r,
        )

    def __len__(self):
        return self.test_len if self.test else self.train_len

    def __getitem__(self, i):
        R = self.R[i]
        t = self.t

        img = self.renderer.render(K=self.K, R=R, t=t)
        img = img.transpose(2, 0, 1).copy()

        Rt = np.eye(4)
        Rt[:3, :3] = R
        Rt[:3, 3:] = t
        Rt = Rt @ self.syms  # (n_syms, 4, 4)

        inst = dict(
            img=img,
            K=self.K,
            R=R,
            t=t,
            Rt=Rt,
            t_est=self.t,
            # TODO: t_grid_frame not used. can we remove?
            t_grid_frame=np.zeros((3, 3), dtype=np.float32),
        )
        # find rotation target indices
        if self.test:
            R_offset = np.eye(3).astype(np.float32)
        else:
            np.random.seed()
            R_offset = Rotation.random().as_matrix().astype(np.float32)
        inst["R_offset"] = R_offset
        inst[f"rot_idx_target_{self.r_last}"] = rotation_grid.get_closest_pix(
            R_offset @ Rt[:, :3, :3], self.r_last
        )
        return inst

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data
from scipy.spatial.transform import Rotation

from .. import rotation_grid

name_dict = dict(
    # symsol 1
    cone="cone",
    cyl="cyl",
    tet="tet",
    cube="cube",
    ico="icosa",
    # symsol 2
    sphX="sphereX",
    cylO="cylO",
    tetX="tetX",
)


class SymsolDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name: str,
        path="data/symsol",
        split="train",
        recursion_depth=6,
        crop_res=224,
        data_slice=slice(None),
        random_offset=True,
        **kwargs,
    ):
        super().__init__()
        assert crop_res == 224
        assert split in {"train", "test"}
        self.test = split == "test"
        self.name = name = name_dict[name]
        self.path = Path(path) / split
        self.rotations = np.load(self.path / "rotations.npz")[name]  # (n, m_sym, 3, 3)
        self.idxs = np.arange(len(self.rotations))[data_slice]
        self.rotations = self.rotations[data_slice]
        self.m_sym = self.rotations.shape[1]
        self.r_last = recursion_depth - 1
        self.random_offset = (split == "train") and random_offset

    def __len__(self):
        return len(self.rotations)

    def __getitem__(self, i):
        img = cv2.imread(
            str(self.path / "images" / f"{self.name}_{self.idxs[i]:05d}.png")
        )
        assert img is not None
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # (c, h, w) [0, 1]
        rots = self.rotations[i]  # (m_sym, 3, 3)
        R = rots[0]
        inst = dict(
            img=img,
            R=R,
            K=np.zeros((3, 3), dtype=np.float32),
            t=np.zeros((3, 1), dtype=np.float32),
            t_grid_frame=np.zeros((3, 3), dtype=np.float32),
            t_est=np.zeros((3, 1), dtype=np.float32),
        )
        # find rotation target indices
        if self.random_offset:
            R_offset = Rotation.random().as_matrix().astype(np.float32)
        else:
            R_offset = np.eye(3, dtype=np.float32)
        inst["R_offset"] = R_offset
        inst[f"rot_idx_target_{self.r_last}"] = rotation_grid.get_closest_pix(
            R_offset @ rots, self.r_last
        )
        return inst

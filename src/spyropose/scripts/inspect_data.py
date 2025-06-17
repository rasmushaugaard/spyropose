import cv2
import jsonargparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import trimesh
from matplotlib.axes import Axes

from ..data.cfg import SpyroDataConfig
from ..data.dataset import SpyroDataset


def main(
    data: SpyroDataConfig,
    draw_frame=True,
    draw_kpts=True,
    draw_verts=False,
    marker_size=10,
    marker_thickness=2,
    same_inst=False,
    seed=0,
):
    dataset = SpyroDataset(data)
    verts = trimesh.load_mesh(data.obj.mesh_path).vertices - data.obj.frame.obj_t_frame
    kpts = data.obj.keypoints
    np.random.seed(seed)

    axs = plt.subplots(4, 4, figsize=(10, 10))[1]
    i = np.random.randint(len(dataset))
    for ax in axs.reshape(-1):
        ax: Axes

        i_ = i if same_inst else np.random.randint(len(dataset))
        d = dataset[i_]
        im: np.ndarray = d["img"].transpose(1, 2, 0)
        K = d["K"]
        R = d["R"]
        t = d["t"]

        for p, do_draw in (verts, draw_verts), (kpts, draw_kpts):
            if not do_draw:
                continue
            p = K @ (R @ p.T + t)
            p = p[:2] / p[2:]
            cmap = plt.get_cmap("tab20")
            for ci, p in enumerate(np.round(p).astype(int).T):
                c = cmap(ci % cmap.N)
                cv2.drawMarker(im, p, c, cv2.MARKER_TILTED_CROSS, marker_size, marker_thickness)

        if draw_frame:
            # cv2 assumes K to be upper triangular with positive diagonals
            # due to image augmentations, this is not necessarily true.
            # One could have changed the image augmentations to rotate the camera instead of the intrinsics. Anyways:
            # u = K (Rp + t) = K_(R_Rp + R_t)
            K_, R_ = scipy.linalg.rq(K)
            K_: np.ndarray
            sign = np.sign(K_.diagonal())
            K_ = K_ * sign.reshape(1, 3)
            R_ = R_ * sign.reshape(3, 1)
            assert np.allclose(K_ @ R_, K), np.linalg.norm(K_ @ R_ - K)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.drawFrameAxes(
                image=im,
                cameraMatrix=K_,
                rvec=cv2.Rodrigues(R_ @ R)[0],
                tvec=R_ @ t,
                length=data.obj.frame.radius,
                distCoeffs=np.zeros(5),
            )
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        ax.imshow(im.clip(0, 1))
        ax.axis("off")
        ax.set_title(str(i_))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    jsonargparse.auto_cli(main)

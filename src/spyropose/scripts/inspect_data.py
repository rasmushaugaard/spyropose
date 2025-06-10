import cv2
import jsonargparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import trimesh
from matplotlib.axes import Axes

from ..data.data_cfg import DatasetConfig
from ..data.dataset import BopInstanceDataset


def main(
    data: DatasetConfig,
    pts_alpha=0.7,
    draw_frame=True,
    same_inst=False,
    seed=0,
):
    dataset = BopInstanceDataset(data)
    pts = trimesh.load_mesh(data.obj.mesh_path).vertices - data.obj.frame.obj_t_frame

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

        vts = R @ pts.T + t
        p = K @ vts
        p = p[:2] / p[2:]
        u, v = np.round(p).astype(int).clip(0, data.obj.crop_res - 1)
        im[v, u] = (1 - pts_alpha) * im[v, u] + pts_alpha * np.array((1, 0, 0))

        if draw_frame:
            # cv2 assumes K to be upper triangular with positive diagonals
            # u = K (Rp + t) = K_(R_Rp + R_t)
            K_, R_ = scipy.linalg.rq(K)
            sign = np.sign(K_.diagonal())
            K_ = K_ * sign.reshape(1, 3)
            R_ = R_ * sign.reshape(3, 1)
            assert np.allclose(K_ @ R_, K), np.linalg.norm(K_ @ R_ - K)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.drawFrameAxes(  # type: ignore
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

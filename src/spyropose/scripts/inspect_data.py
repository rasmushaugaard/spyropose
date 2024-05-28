import argparse

import matplotlib.pyplot as plt
import numpy as np

from .. import helpers

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
parser.add_argument("name")
parser.add_argument("--split", default="train")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--pts-alpha", type=float, default=0.5)
parser.add_argument("--frame-alpha", type=float, default=0.8)
parser.add_argument("--same-inst", action="store_true")
args = parser.parse_args()
np.random.seed(args.seed)

dataset = helpers.datasets[args.dataset](
    name=args.name, split=args.split, recursion_depth=6
)
mesh = helpers.load_mesh(args.dataset, args.name)
if mesh is not None:
    mesh.apply_translation(-mesh.bounding_sphere.primitive.center)
    pts = mesh.vertices

axs = plt.subplots(4, 4, figsize=(10, 10))[1]
i = np.random.randint(len(dataset))
for ax in axs.reshape(-1):
    i_ = i if args.same_inst else np.random.randint(len(dataset))
    d = dataset[i_]
    im = d["img"].transpose(1, 2, 0)
    K = d["K"]

    if mesh is not None:
        vts = d["R"] @ pts.T + d["t"]
        p = K @ vts
        p = p[:2] / p[2:]
        u, v = np.round(p).astype(int).clip(0, 223)
        im[v, u] = (1 - args.pts_alpha) * im[v, u] + args.pts_alpha * np.eye(3)[0]
    view_R_obj = d["R"]

    c = K @ d["t"]
    c = c[:2, 0] / c[2, 0]
    for j in range(3):
        z = view_R_obj[:2, j]
        z = np.array([[0, z[0]], [0, z[1]]]) * 100 + c[:, None]
        ax.plot(*z, c="rgb"[j], alpha=args.frame_alpha)
    ax.imshow(im)
    ax.axis("off")
    ax.set_title(str(i_))
plt.tight_layout()
plt.show()

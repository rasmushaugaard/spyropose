import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn
import trimesh
from scipy.spatial.transform import Rotation
from scipy.stats import chi2

Tensor = torch.Tensor


def get_random_rotations(n):
    return Rotation.random(n).as_matrix().astype(np.float32)  # (n, 3, 3)


def normalize(x, axis=-1):
    return x / np.linalg.norm(x, axis=axis, keepdims=True)


class Lambda(torch.nn.Module):
    def __init__(self, fun):
        super().__init__()
        self.fun = fun

    def forward(self, x):
        return self.fun(x)


def to_device(d, device):
    return {k: v.to(device, non_blocking=device != "cpu") for k, v in d.items()}


def to_tensor_batch(d):
    return {
        k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.tensor(v))[None]
        for k, v in d.items()
    }


def worker_init_fn(*_):
    # each worker should only use one os thread
    # numpy/cv2 takes advantage of multithreading by default
    import os

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    import cv2

    cv2.setNumThreads(0)

    # random seed
    import numpy as np

    np.random.seed(None)


@contextmanager
def timer(text):
    start = time.time()
    yield
    print(text, time.time() - start)


def sample_uniform_unit_sphere_surface(n):
    x = np.random.randn(n, 3)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def sample_truncated_normal(n, std, trunc):
    if std == 0:
        return np.zeros((n, 3))
    # find boundary on chi2 cdf
    q = chi2.cdf((trunc / std) ** 2, df=3)
    # then sample uniformly up until the boundary
    q = np.random.rand(n) * q
    r = chi2.ppf(q, df=3) ** (1 / 2)
    x = sample_uniform_unit_sphere_surface(n) * r[:, None]
    return x * std


def sample_from_lgts(lgts: Tensor, n: int):
    """Samples from last dimension"""
    log_p = torch.log_softmax(lgts, dim=-1)
    cdf = log_p.exp().cumsum(dim=-1)
    sample_idx = torch.searchsorted(
        cdf,
        # last element of cdf should be one, but is not necessarily, due to
        # floating point imprecision. Multiplying rand with last element of cdf
        # avoids binary search errors from last element being less than one.
        torch.rand(*cdf.shape[:-1], n, device=lgts.device) * cdf[..., -1:],
    )
    return log_p.gather(-1, sample_idx), sample_idx


def farthest_point_sampling(pc, n):
    m = len(pc)
    assert pc.shape == (m, 3)

    p = pc.mean(axis=0, keepdims=True)
    for i in range(n):
        dists = np.linalg.norm(p[:, None] - pc[None], axis=2)
        # choose the point in pc which is furthest away from all points in p
        idx = dists.min(axis=0).argmax()
        p = np.concatenate([p, pc[idx : idx + 1]])
        if i == 0:  # discard mean point
            p = p[1:]

    return p


@contextmanager
def np_random_seed(seed: int):
    random_state = np.random.get_state()
    np.random.seed(seed)
    yield
    np.random.set_state(random_state)


def sample_keypoints_from_mesh(mesh: trimesh.Trimesh, n_pts: int):
    """
    first samples a large amount of uniform samples with a fixed seed,
    followed by farthest point sampling to get a small set of close to
    evenly sampled surface points
    """
    with np_random_seed(0):
        samples = mesh.sample(10_000)
    return farthest_point_sampling(samples, n_pts)


def rotation_between_vectors(a, b):
    """
    Returns a rotation matrix, satisfying normalize(b) = R normalize(a)
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    """
    a, b = normalize(a), normalize(b)
    v = np.cross(a, b)
    c = a @ b
    vx = np.cross(np.eye(3), v)
    R = np.eye(3) + vx + vx @ vx / (1 + c)
    return R


def to_alpha_img(img):
    img = img.transpose(1, 2, 0)
    img = np.concatenate(
        (
            img,
            (img > 0).all(axis=2, keepdims=True),
        ),
        axis=2,
    )  # (h, w, 4)
    return img

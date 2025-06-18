import time
from contextlib import contextmanager
from pathlib import Path
from typing import Type, TypeVar

import einops
import numpy as np
import pytorch_lightning as pl
import torch
from scipy.stats import chi2
from torch import Tensor

type Array = np.ndarray | Tensor


def get_array_namespace(x: Array):
    if isinstance(x, Tensor):
        return torch
    elif isinstance(x, np.ndarray):
        return np
    raise ValueError()


def as_tensor(x: Array):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x


def as_ndarray(x: Array):
    if isinstance(x, Tensor):
        x = x.cpu().numpy()
    return x


def if_none(x, other):
    if x is None:
        return other
    return x


def normalize_vectors(x, axis=-1):
    return x / np.linalg.norm(x, axis=axis, keepdims=True)


def normalize_images(x: Array):
    x = as_tensor(x)
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    assert x.dtype == torch.float32
    return einops.rearrange(x, "... h w d -> ... d h w", d=3)


def worker_init_fn(*_):
    # each worker should only use one os thread
    # numpy/cv2 takes advantage of multithreading by default
    # I'm not sure how much of this is still necessary but shared numpy seeds and worker multiprocessing has led to problems previously
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


def farthest_point_sampling(pc: np.ndarray, n: int):
    m = len(pc)
    assert pc.shape == (m, 3)

    p: np.ndarray = pc.mean(axis=0, keepdims=True)
    for i in range(n):
        dists = np.linalg.norm(p[:, None] - pc[None], axis=2)
        # choose the point in pc which is furthest away from all points in p
        idx = dists.min(axis=0).argmax()
        p = np.concatenate([p, pc[idx : idx + 1]])
        if i == 0:  # discard mean point
            p = p[1:]

    return p


ModelType = TypeVar("ModelType", bound=pl.LightningModule)


def load_eval_freeze(cls: Type[ModelType], path: str | Path, device):
    path = Path(path)
    if path.suffix != ".ckpt":
        path = path / "checkpoints"
        model_fps = list((path).glob("*.ckpt"))
        assert len(model_fps) == 1, (
            f"One and only one model should be in {path}."
            f" Found {model_fps}."
            f" If there are multiple checkpoints, specify the full path to the desired checkpoint."
        )
        path = model_fps[0]
    model = cls.load_from_checkpoint(path, device)
    model.eval()
    model.freeze()
    return model

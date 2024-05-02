import numpy as np
from tqdm import tqdm

from . import helpers


def _main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("name")
    parser.add_argument("split")
    args = parser.parse_args()

    data = helpers.datasets[args.dataset](
        name=args.name,
        split=args.split,
        recursion_depth=5,
    )

    log_pu = []

    log_R = np.log(np.pi**2)
    for d in tqdm(data):
        # p = 1 / V
        # log p = -log V = - log Vp - log VR
        log_Vp = np.linalg.slogdet(d["t_grid_frame"] * 1e-3)[1]
        log_pu.append(-log_R - log_Vp)

    print(np.mean(log_pu))


if __name__ == "__main__":
    _main()

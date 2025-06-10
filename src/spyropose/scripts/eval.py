import warnings

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from .. import helpers, utils
from ..model import SpyroPoseModel

_bs = 10
_pose_bs = None
_topk = 512


def evaluate(
    run_id,
    device,
    bs=_bs,
    pose_bs=_pose_bs,
    topk=_topk,
    cache=True,
    overwrite=False,
    runtime_test=False,
    split="test",
    regular_grid=False,
):
    if runtime_test:
        cache = False
        if bs > 1:
            warnings.warn(f"bs is {bs}!!")

    model, model_path = SpyroPoseModel.load_from_run_id(run_id, return_fp=True)
    folder = model_path.parent

    ckpt_name = model_path.with_suffix("").name
    split_suffix = f"_{split}" if split != "test" else ""
    regular_grid_suffix = "_regulargrid" if regular_grid else ""
    result_fp = folder / f"{ckpt_name}_eval_topk={topk}{split_suffix}{regular_grid_suffix}.npy"
    if result_fp.exists() and cache and not overwrite:
        # warnings.warn(f'{id}, topk={topk} already evaluated')
        return np.load(result_fp)

    model = SpyroPoseModel.load_from_checkpoint(model_path)
    model.freeze()
    model.eval()
    model.to(device)

    dataset = helpers.datasets[model.dataset_name](
        name=model.obj_name,
        split=split,
        recursion_depth=model.recursion_depth,
        regular_grid=regular_grid,
    )
    print(len(dataset))
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=bs,
        num_workers=10,
        worker_init_fn=utils.worker_init_fn,
        persistent_workers=runtime_test,
    )

    if runtime_test:
        for _, batch in zip(range(20), loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            model.forward_infer_batch(
                batch=batch, se3=model.train_se3, top_k=topk, pose_bs=pose_bs
            )

    all_log_prob_gt = []
    for batch in tqdm(loader, smoothing=0):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if runtime_test:
            model.forward_infer_batch(
                batch=batch, se3=model.train_se3, top_k=topk, pose_bs=pose_bs
            )
        else:
            ll = model.eval_step(
                batch=batch,
                se3=model.train_se3,
                top_k=topk,
                pose_bs=pose_bs,
                # position_scale is only used for se3 lls (i.e. on bop).
                # bop is in mm, and we calculate ll wrt. si-units: m
                position_scale=1e-3 if model.train_se3 else None,
            )
            all_log_prob_gt.append(ll)

    all_log_prob_gt = torch.cat(all_log_prob_gt).cpu().numpy()
    if cache:
        np.save(str(result_fp), all_log_prob_gt)
    return all_log_prob_gt


def _main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("id")
    parser.add_argument("device")
    parser.add_argument("--bs", type=int, default=_bs)
    parser.add_argument("--pose-bs", type=int, default=_pose_bs)
    parser.add_argument("--topk", type=int, default=_topk)
    parser.add_argument("--no-cache", dest="cache", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--perf", action="store_true")
    parser.add_argument("--regular-grid", action="store_true")
    args = parser.parse_args()

    ll = evaluate(
        run_id=args.id,
        device=args.device,
        bs=args.bs,
        pose_bs=args.pose_bs,
        topk=args.topk,
        cache=args.cache,
        overwrite=args.overwrite,
        runtime_test=args.perf,
        regular_grid=args.regular_grid,
    )
    print(ll.shape)
    print(ll.mean(axis=0))


if __name__ == "__main__":
    _main()

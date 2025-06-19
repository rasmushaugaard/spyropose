# SpyroPose

Official code for Spyropose: SE(3) pyramids for object pose distribution estimation, ICCVW 2023.  
[Project page](https://spyropose.github.io/).

## Installation

We recommend using [uv](https://docs.astral.sh/uv/) for dependency management.
In a project, add spyropose as a dependency, for example from a git repo as below

```bash
uv add REPO_URL
```

## Training Spyro

Spyropose trains a model per object.

Store your training data in the [BOP format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md) in `./data/bop/DATASET`

```bash
uv run -m spyropose.scripts.inspect_data \
  --data.obj.dataset=DATASET \
  --data.obj.obj=OBJECT_ID_OR_NAME
```

If the data looks fine, you can train a spyropose model with the below command.
This would use scenes with index 0 through 18 for training and scene 19 for validation.

```bash
uv run -m spyropose.scripts.train \
  --obj.dataset=DATASET \
  --obj.obj=OBJECT_ID_OR_NAME \
  --data_train.scene_id_range=[0,19] \
  --data_valid.scene_id_range=[19,20]
```

## A detector

A script to train a simple detector is included and can be trained with a similar script:

```bash
uv run -m spyropose.detection.train \
  --obj.dataset=DATASET \
  --obj.obj=OBJECT_ID_OR_NAME \
  --data_train.scene_id_range=[0,19] \
  --data_valid.scene_id_range=[19,20]
```

## Inference

See [./src/spyropose/scripts/infer.py](./src/spyropose/scripts/infer.py).

Can be run like so:

```bash
uv run -m spyropose.scripts.infer \
  ./data/spyropose_detector/2kewoepx ./data/spyropose/dwq4lb0a 19 0
```

## Reproducing results from paper

Check out initial commit.

## Citation

```bibtex
@inproceedings{haugaard2023spyropose,
  title={Spyropose: Se (3) pyramids for object pose distribution estimation},
  author={Haugaard, Rasmus Laurvig and Hagelskj{\ae}r, Frederik and Iversen, Thorbj{\o}rn Mosekj{\ae}r},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2082--2091},
  year={2023}
}
```

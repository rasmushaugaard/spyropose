# SpyroPose

Official code for Spyropose: SE(3) pyramids for object pose distribution estimation, ICCVW 2023.  
[Project page](https://spyropose.github.io/).

## Installation

We recommend using [uv](https://docs.astral.sh/uv/) for dependency management.
In a project, add spyropose as a dependency

```bash
uv add git+REPO_URL
```

## Training Spyro

Spyropose trains a model per object.

Store your training data in the [BOP format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md).

```bash
uv run -m spyropose.scripts.inspect_data DATASET_PATH OBJECT_ID_OR_NAME
```

If the data looks fine, you can train a spyropose model like so:

```bash
uv run -m spyropose.scripts.train DATASET_PATH OBJECT_ID_OR_NAME
```

This will save a model checkpoint to be loaded during inference.

## Inference

```python
from spyropose.model import SpyroPose
model = SpyroPose.from_ckpt(PATH_TO_TRAINED_MODEL)
...
model()
# TODO
```

## Detector

```bash
# inspect detection data
uv run -m spyropose.detection.data --obj.dataset=DATASET --obj.obj=OBJECT --data_train.scene_rng="[0,19]" --data_valid.scene_rng="[19,20]"
# to train a simple detector
uv run -m spyropose.detection.train  # similar args as above
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

## TODO

- [x] allow frame / keyboard configuration
- [x] save all relevant hyperparameters to make inference easier with the model. model name, frame config, etc.
- [x] log hyper parameters
- [x] reintroduce validation set
- [x] make sure eval_vis is working
- [x] add script to train simple detector
- [x] add depth estimation to detector inference
- [x] make full inference example
- [ ] make output dataclass (leaf)
- [ ] update readme

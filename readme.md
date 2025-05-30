# SpyroPose

Official code for Spyropose: SE(3) pyramids for object pose distribution estimation, ICCVW 2023.  
[Project page](https://spyropose.github.io/).

## Installation

We recommend using [uv](https://docs.astral.sh/uv/) for dependency management.
In a project, add spyropose as a dependency

```bash
uv add git+REPO_URL
```

## Training

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

- refactor FrameConfig (offset + radius)
  - Model & Data -> FrameConfig
- save all relevant hyperparameters to make inference easier with the model
  - model name, frame config, etc.
- reintroduce validation set
- make sure inference scripts is running
- note about detection format / assumptions

## Frame argument problem

- I want to be able to init it during training by default with bounding sphere
  - this is then passed to both data and model
- It should also be saved along with the model to avoid having to store something multiple places (strong coupling between model and frame indeed)
- The frame would also be necessary when training a detector.

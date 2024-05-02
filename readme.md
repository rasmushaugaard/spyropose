# SpyroPose
Official code for Spyropose: SE(3) pyramids for object pose distribution estimation, ICCVW 2023.  
[Project page](https://spyropose.github.io/).

We wanted to clean up the code a bit before releasing it, but haven't yet found time for it. Instead of waiting to have that time, here is the "raw" code.
Pull requests are welcome.

Tested on Ubuntu 22.

## Dependencies

### SYMSOL
Download, extract and symlink the symsol dataset to `./data/symsol`:
```bash
$ wget https://storage.googleapis.com/gresearch/implicit-pdf/symsol_dataset.zip
$ unzip symsol_dataset.zip -d [data_folder]/symsol
$ ln -s [data_folder]/symsol ./data/symsol
```

### BOP
For training and evaluating on TLESS and HB,
download and extract the datasets from [https://bop.felk.cvut.cz/datasets/](https://bop.felk.cvut.cz/datasets/) 
and symlink the root folder to `./data/bop`.

### Environment
```bash
$ conda env create -f spyropose.yml
$ conda activate spyropose
```

# Reproducing results

## Training models

### SO(3): SYMSOL I & SYMSOL II

Ours (w/o KP & IS)
```bash
$ python -m spyropose.scripts.train symsol [object] --kpts none --vis-model resnet50 --batch-size 16 --batchnorm --no-is --n-samples 1024 --gpu-idx [gpu_idx]
```

Ours (w/o KP)
Importance sampling expands all samples by the branch factor (8 for SO3), so n_samples=128 results in 1024 function evaluations per recursion.
```bash
$ python -m spyropose.scripts.train [symsol, symsol_ours] [object] --kpts none --vis-model resnet50 --batch-size 16 --batchnorm --n-samples 128 --gpu-idx [gpu_idx]
```

Ours (w/ box KP)
```bash
$ python -m spyropose.scripts.train symsol_ours [object] --kpts box --n-samples 128 --gpu-idx [gpu_idx]
```

Ours (w/o IS)
```bash
$ python -m spyropose.scripts.train symsol_ours [object] --no-is --n-samples 1024 --gpu-idx [gpu_idx]
```

Ours
```bash
$ python -m spyropose.scripts.train symsol_ours [object] --n-samples 128 --gpu-idx [gpu_idx]
```

### SO(3) low data regime

Ours (w/o KP & IS)
```bash
$ python -m spyropose.scripts.train symsol [object] --kpts none --vis-model resnet50 --batch-size 16 --batchnorm --no-is --n-samples 1024 --gpu-idx [gpu_idx] --low-data-regime
```

Ours (w/o KP)
```bash
$ python -m spyropose.scripts.train symsol [object] --kpts none --vis-model resnet50 --batch-size 16 --batchnorm --n-samples 128 --gpu-idx [gpu_idx] --low-data-regime
```


### SE(3): BOP

Ours (w/o IS)
```bash
$ python -m spyropose.scripts.train [dataset] [object] --max-steps 100_000 --no-is --n-samples 2048 --gpu-idx [gpu_idx]
```

Ours
```bash
python -m spyropose.scripts.train [dataset] [object] --max-steps 100_000 --n-samples 32 --gpu-idx [gpu_idx]
```


## Evaluation

Each model is assigned a unique `run_id` at the beginning of training.
To evaluate a trained model:

```bash
$ python -m spyropose.scripts.eval [run_id] [device]
```

And for multi-view on TLESS models:
```bash
$ python -m spyropose.scripts.eval_mv [run_id] [device]
```


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
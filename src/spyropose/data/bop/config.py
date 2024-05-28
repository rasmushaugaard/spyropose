from collections import defaultdict


class DatasetConfig:
    model_folder = "models"
    train_folder = "train"
    test_folder = "test"
    img_folder = "rgb"
    depth_folder = "depth"
    img_ext = "png"
    depth_ext = "png"


config = defaultdict(lambda *_: DatasetConfig())

config["tless"] = tless = DatasetConfig()
tless.model_folder = "models_cad"
tless.test_folder = "test_primesense"
tless.train_folder = "train_primesense"

config["hb"] = hb = DatasetConfig()
# test gt are not provided
hb.test_folder = "val_primesense"

config["itodd"] = itodd = DatasetConfig()
itodd.depth_ext = "tif"
itodd.img_folder = "gray"
itodd.img_ext = "tif"
# test gt are not provided
itodd.test_folder = "val"
# ITODD only had a few validation annotations per object,
# so we chose hb instead.

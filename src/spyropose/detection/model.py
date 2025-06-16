import pytorch_lightning as pl
import torch
import torchvision.models.detection

from ..obj import SpyroObjectConfig


class SpyroDetector(pl.LightningModule):
    def __init__(self, obj: SpyroObjectConfig, lr=1e-4, one_cycle=False):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.obj = obj
        self.lr = lr
        self.one_cycle = one_cycle
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=2)

    def step(self, x, prefix):
        imgs, all_boxes, all_labels = x
        batch_size = len(imgs)
        targets = []
        for boxes, labels in zip(all_boxes, all_labels):
            targets.append(dict(boxes=boxes, labels=labels))
        loss_dict = self.model(imgs, targets)
        for k, v in loss_dict.items():
            self.log(f"{prefix}/{k}", v, batch_size=batch_size)
        loss: torch.Tensor = sum(loss_dict.values())  # type: ignore
        self.log(f"{prefix}/loss", loss, batch_size=batch_size)
        return loss

    def training_step(self, x, _):
        return self.step(x, "train")

    def validation_step(self, x, *_):
        self.model.train()
        return self.step(x, "valid")

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.one_cycle:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                pct_start=0.1,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches,  # type: ignore
            )
            return [optimizer], [dict(scheduler=scheduler, interval="step")]
        else:
            return optimizer

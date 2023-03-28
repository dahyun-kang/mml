import math
import clip
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision
import numpy as np

from torch import nn
from einops import rearrange, repeat
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

from model.transformer import TransformerEncoderLayer

import pdb

class Decoupled_learner(LightningModule):
    def __init__(self, args, dm):
        super().__init__()

        self.args = args
        self.dm = dm
        self.num_classes = dm.num_classes
        self.backbone = self._init_backbone()
        self.dim = 2048

        self.classifier = nn.Linear(self.dim, self.num_classes)

        self.train_class_count = self._init_LT_setting()

    def _init_backbone(self):
        if 'imagenet' in self.args.dataset:
            if self.args.backbone == 'resnext50':
                model = torchvision.models.resnext50_32x4d()
                model.fc = nn.Identity()
        if 'places' in self.args.dataset:
            if self.args.backbone == 'resnet152':
                model = torchvision.models.resnet152(weights='IMAGENET1K_V1')
                model.fc = nn.Identity()

        if self.args.backbone == 'clipvitb':
            model, preprocess = clip.load("ViT-B/32")
            model.forward = model.encode_image
        elif self.args.backbone == 'clipRN50':
            model, preprocess = clip.load("RN50")
            model.forward = model.encode_image

        return model

    def _init_LT_setting(self):
        self.dm.setup(stage='init')
        train_labels = np.array(self.dm.dataset_train.targets).astype(int)
        train_class_count = []
        for c in range(self.dm.num_classes):
            train_class_count.append(len(train_labels[train_labels == c]))
        return train_class_count
    
    def forward(self, x):
        out = self.backbone(x)
        out = self.classifier(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)

        return {'loss': loss}

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y) * 100.

        self.count_correct += (preds == y).int().sum()
        self.count_valimgs += int(y.shape[0])

        correct = (preds == y).int()

        for ans, lbl in zip(correct.tolist(), y.tolist()):
            self.count_class_correct[lbl] += ans
            self.count_class_valimgs[lbl] += 1

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
        return {f'{stage}_loss': loss}

    def on_test_epoch_start(self):
        self.count_correct = 0
        self.count_valimgs = 0

        self.count_class_correct = [0 for c in range(self.num_classes)]
        self.count_class_valimgs = [0 for c in range(self.num_classes)]

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def on_validation_epoch_start(self):
        self.count_correct = 0
        self.count_valimgs = 0

        self.count_class_correct = [0 for c in range(self.num_classes)]
        self.count_class_valimgs = [0 for c in range(self.num_classes)]

    def validation_epoch_end(self, outputs):
        epoch = self.trainer.current_epoch
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # a bit inaccurate; drop_last=False
        epoch_acc = self.count_correct / self.count_valimgs * 100.

        result = f'Epoch {epoch}: | val_loss: {epoch_loss:.4f} | val_acc: {epoch_acc:.2f}'

        many_shot = []
        medium_shot = []
        few_shot = []

        for c in range(self.dm.num_classes):
            if self.train_class_count[c] > self.args.many_shot_thr:
                many_shot.append((self.count_class_correct[c] / self.count_class_valimgs[c]))
            elif self.train_class_count[c] < self.args.low_shot_thr:
                few_shot.append((self.count_class_correct[c] / self.count_class_valimgs[c]))
            else:
                medium_shot.append((self.count_class_correct[c] / self.count_class_valimgs[c]))

        if len(many_shot) == 0: many_shot.append(0)
        if len(medium_shot) == 0: medium_shot.append(0)
        if len(few_shot) == 0: few_shot.append(0)

        result += f" | val_many: {np.mean(many_shot)*100.:.2f} | val_medium: {np.mean(medium_shot)*100.:.2f} | val_few: {np.mean(few_shot)*100.:.2f}"

        result = "\n\n\n" + result + "\n"
        print(result)

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.args.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.maxepochs, eta_min=0.0)

        return {"optimizer": optimizer, "lr_scheduler": lr_schedule}
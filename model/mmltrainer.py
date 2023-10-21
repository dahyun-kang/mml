""" PL wrapper """

import torch
import numpy as np

from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

from model.mmlearner import MemoryModularLearner

import pdb


class MemoryModularLearnerTrainer(LightningModule):
    def __init__(self, args, dm):
        super().__init__()

        self.args = args
        self.dm = dm

        self.count_correct = {'trn': 0.0, 'val': 0.0, 'tst': 0.0}
        self.count_all = {'trn': 0.0, 'val': 0.0, 'tst': 0.0}
        self.loss_all = {'trn': [], 'val': [], 'tst': []}

        self.modeldtype = torch.float16 if 'clip' in args.backbone else torch.float32
        factory_kwargs = {'device': self.device, 'dtype': self.modeldtype}

        self.learner = MemoryModularLearner(args, dm, **factory_kwargs)

    def on_fit_start(self):
        # torch.nn.init.trunc_normal_(self.learner.generic_tokens, mean=0.0, std=0.02)
        self.learner._load_memory_and_prototype(splits=['trn', 'val'])
        if self.args.runfree == 'zsclip':  # or RAC
            self.learner._load_cls_label(splits=['trn', 'val'])

    def on_test_start(self):
        if self.args.episodiceval:
            self.learner._load_episodic_test_memory_and_prototype()
        else:
            self.learner._load_memory_and_prototype(splits=['tst'])
        if self.args.runfree == 'zsclip':  # or RAC
            self.learner._load_cls_label(splits=['tst'])

    def record_metrics(self, count_correct_batch, count_all_batch, loss, stage):
        self.count_correct[stage] += count_correct_batch
        self.count_all[stage] += count_all_batch
        self.loss_all[stage].append(loss)

    def each_step(self, batch, stage=None):
        self.learner.backbone.eval()
        x, y = batch
        logits = self.learner(x, y, stage=stage)
        loss = self.learner.loss_fn(logits, y)

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = accuracy(preds, y) * 100.

            count_correct = (preds == y).int().sum()
            batchsize = int(y.shape[0])  # batchsize may vary as drop_last=False
            self.record_metrics(count_correct, batchsize, loss, stage)

        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        return self.each_step(batch, stage='trn')

    def validation_step(self, batch, batch_idx):
        return self.each_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.each_step(batch, "tst")

    def each_epoch_end(self, stage):
        epoch = self.trainer.current_epoch
        epoch_loss = torch.stack(self.loss_all[stage]).mean()  # a bit inaccurate; drop_last=False
        epoch_acc = self.count_correct[stage] / self.count_all[stage] * 100.

        self.log(f'{stage}/loss', epoch_loss, on_epoch=True)
        self.log(f'{stage}/acc', epoch_acc, on_epoch=True)

        result = f'Epoch {epoch}: | {stage}/loss: {epoch_loss:.4f} | {stage}/acc: {epoch_acc:.2f}'

        # re-initialize metric cache
        self.count_correct[stage] = 0.
        self.count_all[stage] = 0.
        self.loss_all[stage] = []

        result = "\n\n\n" + result + "\n"
        print(result)

    '''
    def on_train_epoch_start(self):
        if True:
            self.trainer.optimizers[0].param_groups[0]['capturable'] = True
    '''

    def on_train_epoch_end(self):
        self.each_epoch_end(stage='trn')

    def on_validation_epoch_end(self):
        self.each_epoch_end(stage='val')

    def on_test_epoch_end(self):
        self.each_epoch_end(stage='tst')

    def configure_optimizers(self):
        param_list = []
        for k, v in self.learner.named_parameters():
            # if 'backbone.transformer.resblocks.11' in k or 'ln_final' in k:  # RAC
            if not 'backbone' in k:  # or 'ln' in k:
                param_list.append(v)
        optimizer = torch.optim.Adam(
            param_list,
            lr=self.args.lr,
            # momentum=0.9,
            weight_decay=self.args.wd,
            eps=1e-6,
        )

        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 70], gamma=0.1)
        # return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        return {"optimizer": optimizer}

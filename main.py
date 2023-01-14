""" Query-Adaptive Memory Referencing Classification """
import argparse

import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F

import torchvision

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from torchmetrics.functional import accuracy

from einops import rearrange
from datamodule import return_datamodule


class LitResnet(LightningModule):
    def __init__(self, args, dm):
        super().__init__()

        self.save_hyperparameters()
        self.args = self.hparams.args
        self.num_classes = dm.num_classes
        self.model = self._init_model()
        # self.fc = nn.Linear(512, self.num_classes)
        self.memory_list, self.memid_list = self._init_memory_list()

        '''
        self.memory_list = nn.Linear(*list(reversed(memory_list.shape)), bias=False)
        with torch.no_grad():
            self.memory_list.weight.copy_(memory_list)
        '''

    def _init_model(self):
        model = torchvision.models.resnet18(weights=None, num_classes=1000)
        model.fc = nn.Identity()
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        return model

    def _init_memory_list(self):
        train_loader = self.hparams.dm.unshuffled_train_dataloader()

        model = self.model.cuda()
        model.eval()
        memory_list = [None] * self.num_classes
        assert self.args.dataset in ['cifar10', 'cifar100']
        memid_list = [None] * (self.num_classes * 5000)  # num samples
        count = 0
        with torch.inference_mode():
            for dataid, x, y in train_loader:
                x = x.cuda()
                out = model(x)
                for data_i, out_i, y_i in zip(dataid, out, y):
                    # dataid <- iteration id
                    memid_list[data_i] = count
                    count += 1

                    out_i = out_i.unsqueeze(0)
                    if memory_list[y_i] is None:
                        memory_list[y_i] = [out_i]
                    else:
                        memory_list[y_i].append(out_i) # + torch.cat([memory_list[y_i], out_i], dim=0)

        for c in range(self.num_classes):
            memory_list[c] = torch.cat(memory_list[c], dim=0)

        # num classes, num images, dim
        memory_list = torch.stack(memory_list, dim=0)

        # num classes * num images
        memid_list = torch.tensor(memid_list).cuda()

        # num classes * num_images, dim
        # memory_list = torch.cat(memory_list, dim=0)

        memory_list = memory_list.detach()
        memory_list.requires_grad = False
        memid_list = memid_list.detach()
        memid_list.requires_grad = False

        return memory_list, memid_list

    def Xforward(self, x):
        out = self.model(x)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)

    def forward(self, x):
        out = self.model(x)
        classwise_sim = torch.einsum('b d, c n d -> b c n', out, self.memory_list)

        '''
        classwise_sim = self.memory_list(out)
        classwise_sim = rearrange(classwise_sim, 'b (c n) -> b c n', c=self.num_classes)
        '''

        # B, C, N -> B, C, K
        topk_sim, indices = classwise_sim.topk(k=self.args.k, dim=-1, largest=True, sorted=False)

        # B, C, K -> B, C
        topk_sim = topk_sim.mean(dim=-1)

        return F.log_softmax(topk_sim, dim=1)

    def training_step(self, batch, batch_idx):
        id, x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return {'id': id, 'x': x, 'loss': loss}

    # def training_step_end(self, batch_parts): # ???????? 이걸로 하니 in-place anormality 탐지되더니 밑에걸로 하니까 사라짐
    # training_step_end는 optim step 도중이고
    # train_batch_end는 step 계산 끝난 이후인듯
    def train_batch_end(self, outputs, batch, batch_idx):

        # track data id and replace the old embedding with the new one
        with torch.no_grad():
            id, x, _ = batch
            out = self.model(x)
            newmem = rearrange(self.memory_list, 'c n d -> (c n) d')
            mem_id = torch.gather(input=self.memid_list, dim=0, index=id)
            newmem.scatter_(dim=0, index=mem_id.unsqueeze(1), src=out)
            self.memory_list = rearrange(newmem, '(c n) d -> c n d', c=self.num_classes).clone()

    def training_epoch_end(self, outputs):
        # print((self.memory_list_oldw == self.memory_list.weight).sum())
        # self.memory_list_oldw.copy_(self.memory_list.weight)

        '''
        with torch.no_grad():
            e = 0.1
            new_memory_list = self._init_memory_list()
            old_memory_list = self.memory_list.weight
            self.memory_list.weight.copy_((1 - e) * old_memory_list + e * new_memory_list)
        '''

    def evaluate(self, batch, stage=None):
        id, x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y) * 100.

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
        return {f'{stage}_loss': loss, f'{stage}_acc': acc}

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def validation_epoch_end(self, outputs):
        epoch = self.trainer.current_epoch
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs =  [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        print(f'Epoch {epoch}: | val_loss: {epoch_loss:.4f} | val_acc: {epoch_acc:.2f}\n')
        # return {'val_loss': epoch_loss, 'val_acc': epoch_acc}

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            # self.memory_list.parameters(),
            # self.fc.parameters(),
            lr=self.args.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // self.args.bsz
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


if __name__ == '__main__':
    seed_everything(7)

    parser = argparse.ArgumentParser(description='Query-Adaptive Memory Referencing Classification')
    parser.add_argument('--datapath', type=str, default='/ssd1t/datasets', help='Dataset path containing the root dir of pascal & coco')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'], help='Experiment dataset')
    parser.add_argument('--logpath', type=str, default='', help='Checkpoint saving dir identifier')
    parser.add_argument('--bsz', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--k', type=int, default=10, help='K KNN')
    parser.add_argument('--maxepochs', type=int, default=1000, help='Max iterations')
    parser.add_argument('--nowandb', action='store_true', help='Flag not to log at wandb')
    args = parser.parse_args()

    dm = return_datamodule(args.datapath, args.dataset, args.bsz)
    model = LitResnet(args, dm=dm)

    trainer = Trainer(
        max_epochs=args.maxepochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=CSVLogger(save_dir='logs') if args.nowandb else WandbLogger(name=args.logpath, save_dir='logs', project=f'qbmr-{args.dataset}'),
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)

""" Query-Adaptive Memory Referencing Classification """
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F

import torchvision

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

from torchmetrics.functional import accuracy

from einops import rearrange


class CIFAR100DataModule(LightningDataModule):
    def __init__(self, data_dir='data', batch_size=256, num_workers=0):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str):
        train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

        val_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

        self.dataset_train = torchvision.datasets.CIFAR100(root=self.hparams.data_dir, train=True, transform=train_transforms, download=True)
        self.dataset_val = torchvision.datasets.CIFAR100(root=self.hparams.data_dir, train=False, transform=val_transforms, download=True)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()


class LitResnet(LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters()
        self.args = self.hparams.args
        self.model = self._init_model()
        # self.fc = nn.Linear(512, NUM_CLASSES)
        memory_list = self._init_memory_list()

        self.memory_list = nn.Linear(*list(reversed(memory_list.shape)), bias=False)
        with torch.no_grad():
            self.memory_list.weight.copy_(memory_list)

    def _init_model(self):
        model = torchvision.models.resnet18(weights=None, num_classes=1000)
        model.fc = nn.Identity()
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        return model

    def _init_memory_list(self):
        from torchvision import datasets, transforms

        transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
                )
        ])
        train_dataset = datasets.CIFAR100(root=self.args.datapath, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.args.bsz, shuffle=False)

        model = self.model.cuda()
        model.eval()
        memory_list = [None] * len(train_dataset.classes)
        with torch.inference_mode():
            for x, y in train_loader:
                x = x.cuda()
                out = model(x)
                for out_i, y_i in zip(out, y):
                    out_i = out_i.unsqueeze(0)
                    if memory_list[y_i] is None:
                        memory_list[y_i] = out_i
                    else:
                        memory_list[y_i] = torch.cat([memory_list[y_i], out_i], dim=0)

        # num classes, num images, dim
        # memory_list = torch.stack(memory_list, dim=0)

        # num classes * num_images, dim
        memory_list = torch.cat(memory_list, dim=0)
        memory_list = memory_list.detach()
        memory_list.requires_grad = False
        return memory_list

    def Xforward(self, x):
        out = self.model(x)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)

    def forward(self, x):
        out = self.model(x)
        # classwise_sim = torch.einsum('b d, c n d -> b c n', out, self.memory_list)

        classwise_sim = self.memory_list(out)
        classwise_sim = rearrange(classwise_sim, 'b (c n) -> b c n', c=100)  # TODO: remove hardcoded numclasses

        # B, C, N -> B, C, K
        topk_sim, indices = classwise_sim.topk(k=self.args.k, dim=-1, largest=True, sorted=False)

        # B, C, K -> B, C
        topk_sim = topk_sim.mean(dim=-1)

        return F.log_softmax(topk_sim, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

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
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

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
        print(f'Epoch {epoch}: | val_loss: {epoch_loss:.4f} | val_acc: {epoch_acc * 100.:.2f}\n')
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
    parser.add_argument('--maxepochs', type=int, default=2000, help='Max iterations')

    parser.add_argument('--nowandb', action='store_true', help='Flag not to log at wandb')
    args = parser.parse_args()

    PATH_DATASETS = os.environ.get("PATH_DATASETS", "data")

    dataset = 'cifar100'


    cifar10_dm = CIFAR10DataModule(
        data_dir=args.datapath,
        batch_size=args.bsz,
        num_workers=8,
        # train_transforms=train_transforms,
        # test_transforms=test_transforms,
        # val_transforms=test_transforms,
    )

    cifar100_dm = CIFAR100DataModule(
        data_dir=args.datapath,
        batch_size=args.bsz,
        num_workers=8,
    )

    model = LitResnet(args)

    trainer = Trainer(
        max_epochs=args.maxepochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=CSVLogger(save_dir='logs') if args.nowandb else WandbLogger(name=args.logpath, save_dir='logs', project=f'qbmr-{args.dataset}'),
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
    )

    trainer.fit(model, cifar100_dm)
    trainer.test(model, datamodule=cifar100_dm)

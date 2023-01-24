""" Query-Adaptive Memory Referencing Classification """
import argparse
from tqdm import tqdm

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
        self.backbone = self._init_backbone()
        self.fc = nn.Linear(512, self.num_classes)
        self.memory_list = None
        self.knnformer = nn.TransformerEncoderLayer(d_model=512,
                                                    nhead=8,
                                                    dim_feedforward=512,
                                                    dropout=0.0,
                                                    # activation=F.relu,
                                                    layer_norm_eps=1e-05,
                                                    batch_first=True,
                                                    norm_first=True,
                                                    device=self.device)

    def _init_backbone(self):
        """
        Init pretrained backbone
        """
        # TODO: Set backbone as args
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1', num_classes=1000)
        model.fc = nn.Identity()
        # model.avgpool = nn.Identity()

        # Retain img size at the shallowest layer
        if 'cifar' in self.args.dataset:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()

        model.requires_grad_(requires_grad=False)
        return model

    def on_fit_start(self):
        self.memory_list = self._init_memory_list()

    def _init_memory_list(self):
        train_loader = self.hparams.dm.unshuffled_train_dataloader()

        if self.args.dataset == 'cifar10':
            num_samples = 5000
        elif self.args.dataset == 'cifar100':
            num_samples = 500
        elif self.args.dataset == 'places365':
            max_num_samples = 5000
            # num_samples = 3000
            num_samples = 1000
        elif self.args.dataset == 'caltech101':
            max_num_samples = 800
            # num_samples = 3000
            num_samples = 40
        elif self.args.dataset == 'country211':
            max_num_samples = 800
            num_samples = 40
        elif self.args.dataset == 'fgvcaircraft':
            max_num_samples = 100
            num_samples = 100
        elif self.args.dataset == 'food101':
            max_num_samples = 750
            num_samples = 750
        else:
            raise NotImplementedError

        model = self.backbone.cuda()
        model.eval()
        memory_list = [None] * self.num_classes
        class_count = [0] * self.num_classes
        count = 0

        with torch.inference_mode():
            for x, y in tqdm(train_loader):
                x = x.cuda()
                out = model(x)
                for out_i, y_i in zip(out, y):
                    if class_count[y_i] >= num_samples:
                        continue
                    class_count[y_i] += 1

                    out_i = out_i.unsqueeze(0)
                    if memory_list[y_i] is None:
                        memory_list[y_i] = [out_i]
                    else:
                        memory_list[y_i].append(out_i) # + torch.cat([memory_list[y_i], out_i], dim=0)

        for c in range(self.num_classes):
            memory_list[c] = torch.cat(memory_list[c], dim=0)

        # num classes, num images, dim
        memory_list = torch.stack(memory_list, dim=0)

        # num classes * num_images, dim
        # memory_list = torch.cat(memory_list, dim=0)

        memory_list = memory_list.detach()
        memory_list.requires_grad = False

        return memory_list

    def forward(self, x):
        out = self.backbone(x)

        with torch.no_grad():
            classwise_sim = torch.einsum('b d, c n d -> b c n', out, self.memory_list)
            # B, C, N -> B, C, K
            topk_sim, indices = classwise_sim.topk(k=self.args.k, dim=-1, largest=True, sorted=False)
            # 1, C, 1
            class_idx = torch.arange(self.num_classes).unsqueeze(0).unsqueeze(-1)
            # C, N, D [[1, C, 1], [B, C, K]] -> B, C, K, D
            knnemb = self.memory_list[class_idx, indices]
            # B, C, K, D -> B, C, D
            knnemb_avg = torch.mean(knnemb, dim=2)
            # (B, 1, D), (B, C, D) -> B, (1 + C), D
            context_emb = torch.cat([out.unsqueeze(1), knnemb_avg], dim=1)

        out = self.knnformer(context_emb)
        out = self.fc(out[:, 0])

        return F.log_softmax(out, dim=1)

    def forward_naiveaddedformer(self, x):
        # this model flattens the output without pooling
        # B (D H W)
        out = self.backbone(x)

        out = rearrange(out, 'b (d l) -> b l d', l=49, d=512)  # TODO: remove hardcode
        out = self.knnformer(out)
        out = self.fc(out.mean(dim=1))
        return F.log_softmax(out, dim=1)

    def forward_directknnmatching(self, x):
        out = self.backbone(x)

        '''
        classwise_sim = torch.einsum('b d, c n d -> b c n', out, self.memory_list)

        classwise_sim = self.memory_list(out)
        classwise_sim = rearrange(classwise_sim, 'b (c n) -> b c n', c=self.num_classes)

        # B, C, N -> B, C, K
        topk_sim, indices = classwise_sim.topk(k=self.args.k, dim=-1, largest=True, sorted=False)
        '''

        with torch.no_grad():
            classwise_sim = torch.einsum('b d, c n d -> b c n', out, self.memory_list)

            # B, C, N -> B, C, K
            topk_sim, indices = classwise_sim.topk(k=self.args.k, dim=-1, largest=True, sorted=False)

            # 1, C, 1
            class_idx = torch.arange(self.num_classes).unsqueeze(0).unsqueeze(-1)

            # C, N, D [[1, C, 1], [B, C, K]] -> B, C, K, D
            knnemb = self.memory_list[class_idx, indices]

        topk_sim = torch.einsum('b d, b c k d -> b c k', out, knnemb)
        # B, C, K -> B, C
        topk_sim = topk_sim.mean(dim=-1)

        return F.log_softmax(topk_sim, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)

        return {'loss': loss}

    '''
    def on_train_batch_end(self, outputs, batch, batch_idx):

        # track data id and replace the old embedding with the new one
        with torch.no_grad():
            id, x, _ = batch
            emb = self.backbone(x)
            mem_id = torch.gather(input=self.memid_list, dim=0, index=id)

            if torch.all(mem_id == -1):
                return

            valid_mem_id = mem_id[mem_id != -1]
            valid_emb = emb[mem_id != -1]

            newmem = rearrange(self.memory_list, 'c n d -> (c n) d')
            newmem.scatter_(dim=0, index=valid_mem_id.unsqueeze(1), src=valid_emb)
            self.memory_list = rearrange(newmem, '(c n) d -> c n d', c=self.num_classes).clone()
    '''

    '''
    # 이걸 쓰면 왠진 몰라도 memory leak 이 생김 빡침
    def training_epoch_end(self, outputs):
        # print((self.memory_list_oldw == self.memory_list.weight).sum())
        # self.memory_list_oldw.copy_(self.memory_list.weight)

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
        param_list = []
        for k, v in self.named_parameters():
            if not 'backbone' in k:
                param_list.append(v)
        optimizer = torch.optim.SGD(
            param_list,
            # list(self.knnformer.parameters()) + list(self.fc.parameters()),
            lr=self.args.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

        return {"optimizer": optimizer}


if __name__ == '__main__':
    seed_everything(7)

    parser = argparse.ArgumentParser(description='Query-Adaptive Memory Referencing Classification')
    parser.add_argument('--datapath', type=str, default='/ssd1t/datasets', help='Dataset path containing the root dir of pascal & coco')
    parser.add_argument('--dataset', type=str, default='cifar100', help='Experiment dataset')
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
        num_sanity_val_steps=0,
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)

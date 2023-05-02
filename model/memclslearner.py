""" Memory-based Classification Learner """

import os
import os.path as osp
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


class MemClsLearner(LightningModule):
    def __init__(self, args, dm):
        super().__init__()

        self.args = args
        self.dm = dm
        # self.num_classes = dm.num_classes # please use self.dm.num_classes directly.
        self.backbone = self._init_backbone()
        self.dim = 512
        self.nhead = 8
        self.cachedir = osp.join(os.getcwd(), 'cache', args.dataset, args.backbone)

        self.memory_list = None
        self.modeldtype = torch.float16 if 'clip' in args.backbone else torch.float32
        factory_kwargs = {'device': self.device, 'dtype': self.modeldtype}
        '''
        self.knnformer = TransformerEncoderLayer(d_model=self.dim,
                                                 nhead=self.nhead,
                                                 dim_feedforward=self.dim,
                                                 dropout=0.0,
                                                 # activation=F.relu,
                                                 layer_norm_eps=1e-05,
                                                 batch_first=True,
                                                 norm_first=True,
                                                 **factory_kwargs,
                                                 )
        '''
        self.fc = nn.Sequential(
                          nn.Linear(self.dim, self.dim, **factory_kwargs),
                          nn.ReLU(inplace=True),
                          nn.Linear(self.dim, self.dim, **factory_kwargs),
                      )
        def eye(submodule):
            if isinstance(submodule, nn.Linear):
                torch.nn.init.eye_(submodule.weight)
                submodule.bias.data.fill_(0.00)
        self.fc.apply(eye)

        # self.generic_tokens = self._init_generic_tokens()

    def _init_backbone(self):
        """ Init pretrained backbone """
        if 'resnet' in self.args.backbone:
            if self.args.backbone == 'resnet18':
                model = torchvision.models.resnet18(weights='IMAGENET1K_V1', num_classes=1000)
            elif self.args.backbone == 'resnet50':
                model = torchvision.models.resnet50(weights='IMAGENET1K_V1', num_classes=1000)
            else:
                raise NotImplementedError

            model.fc = nn.Identity()
            # model.avgpool = nn.Identity()

            '''
            # use this for training from scratch
            # Retain img size at the shallowest layer
            if 'cifar' in self.args.dataset and not args.nakata22:
                model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                model.maxpool = nn.Identity()
            '''
        elif self.args.backbone == 'clipvitb':
            model, preprocess = clip.load("ViT-B/32")
            model.forward = model.encode_image  # TODO: function overriding; refrain this type of coding
        elif self.args.backbone == 'clipRN50':
            model, preprocess = clip.load("RN50")
            model.forward = model.encode_image

        model.requires_grad_(requires_grad=False)
        return model

    def _init_generic_tokens(self):
        _generic_tokens = torch.empty(self.args.ntokens, 512, dtype=self.modeldtype, requires_grad=True)
        generic_tokens = nn.Parameter(_generic_tokens.clone(), requires_grad=True)
        # moved to self.on_fit_start; should be called after params being loaded to cuda
        # nn.init.trunc_normal_(self.generic_tokens, mean=0.0, std=0.02)
        return generic_tokens

    def _load_memory_and_prototype(self):
        with torch.no_grad():
            memory_dict = dict()
            for split, loader in zip(['trn', 'val', 'tst'],
                                     [self.dm.unshuffled_train_dataloader,
                                      self.dm.val_dataloader,
                                      self.dm.test_dataloader]):
                img_embed_path = osp.join(self.cachedir, f'{split}_img_embed.pth')
                img_label_path = osp.join(self.cachedir, f'{split}_img_label.pth')

                if osp.exists(img_embed_path):
                    print(f'\n *** [{split}] loading embed/label checkpoints from {self.cachedir}. *** \n')
                    img_embed = torch.load(img_embed_path)
                    img_label = torch.load(img_label_path)
                else:
                    print(f'\n *** [{split}] embed/label not found. Generating embed/label checkpoints and saving them at {self.cachedir}. *** \n')
                    if not osp.exists(self.cachedir): os.makedirs(self.cachedir)
                    img_embed, img_label = self._init_memory(loader, split)
                    torch.save(img_embed, img_embed_path)
                    torch.save(img_label, img_label_path)

                img_proto = [img_embed[img_label == c].mean(dim=0) for c in range(self.dm.num_classes)]
                img_proto = torch.stack(img_proto, dim=0)  # C, D

                # same as self.trn_img_embed = something
                self.register_buffer(f'{split}_img_embed', img_embed, persistent=False)
                self.register_buffer(f'{split}_img_label', img_label, persistent=False)
                self.register_buffer(f'{split}_img_proto', img_proto, persistent=False)

            self.train_class_count = [torch.sum(self.trn_img_label == c) for c in range(self.dm.num_classes)]

    def _init_memory(self, loader, split):
        '''
        Return an irregular memory list of image features with different number for each class
        '''
        backbone = self.backbone.to(self.device)
        backbone.eval()
        embed_list = []
        label_list = []

        with torch.inference_mode():
            for x, y in tqdm(loader(), desc=f'Generating {split} emb'):
                x = x.to(self.device)
                out = backbone(x)
                embed_list.append(out)
                label_list.append(y)

        embed_1d = torch.cat(embed_list, dim=0).detach()  # N, D
        label_1d = torch.cat(label_list, dim=0).detach()  # N
        embed_1d.requires_grad = False
        label_1d.requires_grad = False
        return embed_1d, label_1d

    def on_fit_start(self):
        self._load_memory_and_prototype()

    def on_test_start(self):
        self._load_memory_and_prototype()

    # def forward_protofcmatching(self, x, y):
    def forward(self, x, y):
        out = self.backbone(x)
        out = self.fc(out)

        proto_ = self.trn_img_proto.to(x.device)
        out_ = out
        # proto_ = F.normalize(self.trn_img_proto.to(x.device), dim=-1, p=2)
        # out_ = F.normalize(out, dim=-1, p=2)
        sim = torch.einsum('c d, b d -> b d', proto_, out_) # * 0.001
        return sim

    def forward_m8_1(self, x, y):
        '''
        <m8.1>
        parellel transformers with global (class-agnostic) kNN

        - parallel input update (knnformer1, knnformer2)
        - avg probs
        - no l2normalization
        '''

        out = self.backbone(x)
        batchsize = out.shape[0]

        with torch.no_grad():
            # irregular memory
            if isinstance(self.memory_list, list):
                self.memory_list = torch.cat(self.memory_list, dim=0)
            # regular but 3-dimensional memory
            if len(self.memory_list.shape) == 3:
                self.memory_list = rearrange(self.memory_list, 'c n d -> (c n) d')

            classwise_sim = torch.einsum('b d, n d -> b n', out, self.memory_list)
            if self.training:  # to ignore self-voting
                # B, C, N -> B, C, K
                topk_sim, indices = classwise_sim.topk(k=self.args.k + 1, dim=-1, largest=True, sorted=True)
                indices = indices[:, 1:]
            else:
                topk_sim, indices = classwise_sim.topk(k=self.args.k, dim=-1, largest=True, sorted=True)

            # N, D [[B, K] -> B, K, D
            knnemb = self.memory_list[indices]

            # B, 1, D
            tr_q = out.unsqueeze(1)
            # (B, 1, D), (B, K, D) -> B, (1 + K), D
            tr_knn_cat = torch.cat([tr_q, knnemb], dim=1)

        qout = self.linear(out)
        nout = self.knnformer(tr_q, tr_knn_cat, tr_knn_cat)

        qout = torch.einsum('b d, c d -> b c', qout, self.global_proto)
        nout = torch.einsum('b d, c d -> b c', nout[:, 0], self.global_proto)

        avgprob = 0.5 * (F.softmax(qout, dim=1) + F.softmax(nout, dim=1))
        avgprob += 1e-6  # to prevent numerical unstability
        return torch.log(avgprob)

    def forward_m8_1_1(self, x, y):
        '''
        <m8.1.1 -> m29>
        parellel transformers with global (class-agnostic) kNN + text emb
        - parallel input update (linear, knnformer)
        - avg probs
        - no l2normalization
        - concat text embedding
        '''
        out = self.backbone(x)
        batchsize = out.shape[0]
        with torch.no_grad():
            # irregular memory
            if isinstance(self.memory_list, list):
                self.memory_list = torch.cat(self.memory_list, dim=0)
            # regular but 3-dimensional memory
            if len(self.memory_list.shape) == 3:
                self.memory_list = rearrange(self.memory_list, 'c n d -> (c n) d')
            classwise_sim = torch.einsum('b d, n d -> b n', out, self.memory_list)
            if self.training:  # to ignore self-voting
                # B, C, N -> B, C, K
                topk_sim, indices = classwise_sim.topk(k=self.args.k + 1, dim=-1, largest=True, sorted=True)
                indices = indices[:, 1:]
            else:
                topk_sim, indices = classwise_sim.topk(k=self.args.k, dim=-1, largest=True, sorted=True)
            # N, D [[B, K] -> B, K, D
            knnemb = self.memory_list[indices]
            # B, 1, D
            tr_q = out.unsqueeze(1)
            # text emb
            # N [[B, K] -> B, K
            knnclass = self.label_list[indices]
            knnemb = torch.cat([knnemb, self.textlabel_list[knnclass]], dim=-1)
            # (B, 1, D), (B, K, D) -> B, (1 + K), D
            tr_knn_cat = torch.cat([tr_q.repeat(1, 1, 2), knnemb], dim=1)
        qout = self.linear(out)
        # text emb
        tr_knn_cat = self.knnencoder(tr_knn_cat)
        nout = self.knnformer(tr_q, tr_knn_cat, tr_knn_cat)
        qout = torch.einsum('b d, c d -> b c', qout, self.global_proto)
        nout = torch.einsum('b d, c d -> b c', nout[:, 0], self.global_proto)
        avgprob = 0.5 * (F.softmax(qout, dim=1) + F.softmax(nout, dim=1))
        avgprob = torch.clamp(avgprob, 1e-6)  # to prevent numerical unstability
        return torch.log(avgprob)

    def training_step(self, batch, batch_idx):
        self.backbone.eval()
        x, y = batch
        logits = self(x, y)
        '''
        qout, nout = logits
        avgprob = 0.5 * (F.softmax(qout + torch.log(self.base_prob.to(qout.device)), dim=1) + F.softmax(nout + torch.log(self.base_prob.to(nout.device)), dim=1))
        avgprob = torch.clamp(avgprob, 1e-6)  # to prevent numerical unstability
        loss = F.cross_entropy(avgprob, y)
        # loss = F.cross_entropy(logits + torch.log(self.base_prob.to(logits.device)), y)
        '''
        loss = F.cross_entropy(logits, y)
        self.log("train/loss", loss)
        return {'loss': loss}

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x, y)
        '''
        qout, nout = logits
        logits = 0.5 * (F.softmax(qout, dim=1) + F.softmax(nout, dim=1))
        avgprob = 0.5 * (F.softmax(qout + torch.log(self.base_prob.to(logits.device)), dim=1) + F.softmax(nout + torch.log(self.base_prob.to(logits.device)), dim=1))
        avgprob = torch.clamp(avgprob, 1e-6)  # to prevent numerical unstability
        loss = F.cross_entropy(avgprob, y)
        # loss = F.cross_entropy(logits + torch.log(self.base_prob.to(logits.device)), y)
        '''
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y) * 100.

        self.count_correct += (preds == y).int().sum()
        self.count_valimgs += int(y.shape[0])

        if self.args.LT:
            correct = (preds == y).int()

            for ans, lbl in zip(correct.tolist(), y.tolist()):
                self.count_class_correct[lbl] += ans
                self.count_class_valimgs[lbl] += 1

        if stage:
            self.log(f"{stage}/loss", loss, prog_bar=True)
            self.log(f"{stage}/acc", acc, prog_bar=True)
        return {f'{stage}/loss': loss}

    def on_test_epoch_start(self):
        self.count_correct = 0
        self.count_valimgs = 0

        if self.args.LT:
            self.count_class_correct = [0 for c in range(self.dm.num_classes)]
            self.count_class_valimgs = [0 for c in range(self.dm.num_classes)]

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def on_validation_epoch_start(self):
        self.count_correct = 0
        self.count_valimgs = 0

        if self.args.LT:
            self.count_class_correct = [0 for c in range(self.dm.num_classes)]
            self.count_class_valimgs = [0 for c in range(self.dm.num_classes)]

    def validation_epoch_end(self, outputs):
        epoch = self.trainer.current_epoch
        batch_losses = [x["val/loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # a bit inaccurate; drop_last=False
        epoch_acc = self.count_correct / self.count_valimgs * 100.

        result = f'Epoch {epoch}: | val_loss: {epoch_loss:.4f} | val_acc: {epoch_acc:.2f}'
        if self.args.LT:
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
        param_list = []
        for k, v in self.named_parameters():
            if not 'backbone' in k:
                param_list.append(v)
        optimizer = torch.optim.SGD(
            param_list,
            lr=self.args.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

        return {"optimizer": optimizer}

    def standardize(self, x, dim=1, eps=1e-6):
        out = x - x.mean(dim=dim, keepdim=True)
        out = out / (out.std(dim=dim, keepdim=True) + eps)
        return out


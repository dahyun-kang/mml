""" Memory-based Classification Learner """

import math
import clip
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision

from torch import nn
from einops import rearrange
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

from model.transformer import TransformerEncoderLayer


class MemClsLearner(LightningModule):
    def __init__(self, args, dm):
        super().__init__()

        self.args = args
        self.dm = dm
        self.num_classes = dm.num_classes
        self.backbone = self._init_backbone()
        self.dim = 512

        self.memory_list = None
        _dtype = torch.float16 if 'clip' in args.backbone else torch.float32
        self.qinformer = TransformerEncoderLayer(d_model=self.dim,
                                                 nhead=8,
                                                 dim_feedforward=self.dim,
                                                 dropout=0.0,
                                                 # activation=F.relu,
                                                 layer_norm_eps=1e-05,
                                                 batch_first=True,
                                                 norm_first=True,
                                                 device=self.device,
                                                 dtype=_dtype,
                                                 )
        self.knnformer = TransformerEncoderLayer(d_model=self.dim,
                                                 nhead=8,
                                                 dim_feedforward=self.dim,
                                                 dropout=0.0,
                                                 # activation=F.relu,
                                                 layer_norm_eps=1e-05,
                                                 batch_first=True,
                                                 norm_first=True,
                                                 device=self.device,
                                                 dtype=_dtype,
                                                 )
        if args.backbone == 'resnet50':
            self.knnformer = nn.Sequential(
                nn.Linear(2048, self.dim),
                self.knnformer,
            )

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

    def on_fit_start(self):
        with torch.no_grad():
            self.memory_list = self._init_memory_list()

            self.global_proto = self.memory_list.mean(dim=1)
            self.global_proto = self.global_proto.detach()
            self.global_proto.requires_grad = False

            self.memory_list = rearrange(self.memory_list, 'c n d -> (c n) d')

    def on_test_start(self):
        if self.memory_list == None:
            self.memory_list = self._init_memory_list()

    def _init_memory_list(self):
        train_loader = self.dm.unshuffled_train_dataloader()
        max_num_samples = self.dm.max_num_samples

        backbone = self.backbone.to(self.device)
        backbone.eval()
        memory_list = [None] * self.num_classes
        count = 0

        with torch.inference_mode():
            for x, y in tqdm(train_loader):
                x = x.to(self.device)
                out = backbone(x)
                for out_i, y_i in zip(out, y):
                    if memory_list[y_i] is not None and len(memory_list[y_i]) == max_num_samples:
                        continue

                    out_i = out_i.unsqueeze(0)
                    if memory_list[y_i] is None:
                        memory_list[y_i] = [out_i]
                    else:
                        memory_list[y_i].append(out_i)

        for c in range(self.num_classes):
            memory_list[c] = torch.cat(memory_list[c], dim=0)

        # num classes, num images, dim
        memory_list = torch.stack(memory_list, dim=0)

        memory_list = memory_list.detach()
        memory_list.requires_grad = False

        return memory_list

    def forward(self, x):
        out = self.backbone(x)

        with torch.no_grad():
            classwise_sim = torch.einsum('b d, n d -> b n', out, self.memory_list)
            # B, N -> B, K
            topk_sim, indices = classwise_sim.topk(k=self.args.k, dim=-1, largest=True, sorted=False)

            # C, N, D [[B, K]] -> B, K, D
            knnemb = self.memory_list[indices]

            # corresponding_proto = self.global_proto[class_ids]  # self.global_proto_learned(class_ids)

            # B, 1, D
            tr_q = out.unsqueeze(1)
            # (B, 1, D), (B, C, D) -> B, (1 + C), D
            tr_knn_cat = torch.cat([tr_q, knnemb], dim=1)

        qout = self.qinformer(tr_q, tr_q, tr_q)
        nout = self.knnformer(tr_q, tr_knn_cat, tr_knn_cat)

        qout = torch.einsum('b d, c d -> b c', qout[:, 0], self.global_proto)
        nout = torch.einsum('b d, c d -> b c', nout[:, 0], self.global_proto)

        # return torch.log(0.5 * (F.softmax(qout, dim=1) + F.softmax(nout, dim=1)))
        avgprob = 0.5 * (F.softmax(qout, dim=1) + F.softmax(nout, dim=1))
        avgprob = torch.clamp(avgprob, 1e-6)  # to prevent numerical unstability
        return torch.log(avgprob)

    def forward_classwiseknn(self, x):
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

        tr_q = out.unsqueeze(1)
        out = self.knnformer(tr_q, context_emb, context_emb)

        sim = torch.einsum('b d, c d -> b c', out[:, 0], self.global_proto)

        return F.log_softmax(out, dim=1)

    def forward_nakata22(self, x):
        out = self.backbone(x)

        def majority_vote(input):
            stack = []
            count = [0]*self.num_classes
            for item in input:
                count[item.cpu().item()] += 1
                if not stack or stack[-1] == item:
                    stack.append(item)
                else:
                    stack.pop()

            # onehot = (input[0] if not stack else stack[0]).cpu().item() # real majority vote
            onehot = torch.argmax(torch.tensor(count)).item() # just vote
            result = torch.tensor([0.]*self.num_classes)
            result[onehot] = 1.0

            return result.to(self.device)

        with torch.no_grad():
            num_samples = self.memory_list.shape[1]
            all_features = self.memory_list.view([-1, self.memory_list.shape[2]])

            similarity_mat = torch.einsum('b d, n d -> b n', F.normalize(out, dim=-1), F.normalize(all_features, dim=-1))

            topk_sim, indices = similarity_mat.topk(k=self.args.k, dim=-1, largest=True, sorted=False)

            indices = torch.div(indices, num_samples, rounding_mode='trunc')
            voting_result = torch.stack(list(map(majority_vote, indices)))

        return voting_result

    def forward_prototypematching(self, x):
        out = self.backbone(x)
        tr_q = out.unsqueeze(1)

        out = self.qinformer(tr_q, tr_q, tr_q)
        out = torch.einsum('b d, c d -> b c', out[:, 0], self.global_proto)
        return F.log_softmax(out, dim=1)

    def forward_naiveaddedformer(self, x):
        # this model flattens the output without pooling
        # B (D H W)
        out = self.backbone(x)

        out = rearrange(out, 'b (d l) -> b l d', l=49, d=self.dim)  # TODO: remove hardcode
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
        self.backbone.eval()
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)

        return {'loss': loss}

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

        self.count_correct += (preds == y).int().sum()
        self.count_valimgs += int(y.shape[0])

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
        return {f'{stage}_loss': loss}

    def on_test_epoch_start(self):
        self.count_correct = 0
        self.count_valimgs = 0

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def on_validation_epoch_start(self):
        self.count_correct = 0
        self.count_valimgs = 0

    def validation_epoch_end(self, outputs):
        epoch = self.trainer.current_epoch
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # a bit inaccurate; drop_last=False
        epoch_acc = self.count_correct / self.count_valimgs * 100.

        print(f'Epoch {epoch}: | val_loss: {epoch_loss:.4f} | val_acc: {epoch_acc:.2f}\n')

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


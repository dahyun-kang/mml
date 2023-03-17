""" Memory-based Classification Learner """

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

class MemClsLearner(LightningModule):
    def __init__(self, args, dm):
        super().__init__()

        self.args = args
        self.dm = dm
        self.num_classes = dm.num_classes
        self.backbone = self._init_backbone()
        self.dim = 512
        self.nhead = 8

        self.memory_list = None
        self.modeldtype = torch.float16 if 'clip' in args.backbone else torch.float32
        self.knnformer2 = TransformerEncoderLayer(d_model=self.dim,
                                                 nhead=self.nhead,
                                                 dim_feedforward=self.dim,
                                                 dropout=0.0,
                                                 # activation=F.relu,
                                                 layer_norm_eps=1e-05,
                                                 batch_first=True,
                                                 norm_first=True,
                                                 device=self.device,
                                                 dtype=self.modeldtype,
                                                 )
        self.knnformer = TransformerEncoderLayer(d_model=self.dim,
                                                 nhead=self.nhead,
                                                 dim_feedforward=self.dim,
                                                 dropout=0.0,
                                                 # activation=F.relu,
                                                 layer_norm_eps=1e-05,
                                                 batch_first=True,
                                                 norm_first=True,
                                                 device=self.device,
                                                 dtype=self.modeldtype,
                                                 )

        self.generic_tokens = self._init_generic_tokens()

        if args.backbone == 'resnet50':
            self.knnformer = nn.Sequential(
                nn.Linear(2048, self.dim),
                self.knnformer,
            )

        if args.LT:
            self.train_class_count = self._init_LT_setting()

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

    def _init_LT_setting(self):
        self.dm.setup(stage='init')
        train_labels = np.array(self.dm.dataset_train.targets).astype(int)
        train_class_count = []
        for c in range(self.dm.num_classes):
            train_class_count.append(len(train_labels[train_labels == c]))
        return train_class_count

    def on_fit_start(self):
        with torch.no_grad():
            self.memory_list = self._init_memory_list()

            self.global_proto = self.memory_list.mean(dim=1)
            self.global_proto = self.global_proto.detach()
            self.global_proto.requires_grad = False

            # normalize
            self.memory_list = F.normalize(self.memory_list, p=2, dim=-1)
            self.global_proto = F.normalize(self.global_proto, p=2, dim=-1)

            # self.memory_list = rearrange(self.memory_list, 'c n d -> (c n) d')

            self.generic_tokens = nn.init.trunc_normal_(self.generic_tokens, mean=0.0, std=0.02)
            self.old_generic_tokens = self.generic_tokens.clone()

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
        """
        0301 m18
        - unnormalized generic tokens as input
        - standardized prototypes after update
        - l2normalized features used
        """

        out = self.backbone(x)
        # normalize
        out = F.normalize(out, p=2, dim=-1)
        batchsize = out.shape[0]

        with torch.no_grad():
            classwise_sim = torch.einsum('b d, c n d -> b c n', out, self.memory_list)
            if self.training:  # to ignore self-voting
                # B, C, N -> B, C, K
                topk_sim, indices = classwise_sim.topk(k=self.args.k + 1, dim=-1, largest=True, sorted=True)
                # indices = indices[:, :, 1:]
                top1_indices = indices[:, :, 0]
                max_class_indices = top1_indices.argmax(dim=1)  # highly likely the self (or another twin in the feature space)
                indices[range(batchsize), max_class_indices, :-1] = indices[range(batchsize), max_class_indices, 1:]
                indices = indices[:, :, :-1]
            else:
                topk_sim, indices = classwise_sim.topk(k=self.args.k, dim=-1, largest=True, sorted=True)

            # 1, C, 1
            class_idx = torch.arange(self.num_classes).unsqueeze(0).unsqueeze(-1)
            # C, N, D [[1, C, 1], [B, C, K]] -> B, C, K, D
            knnemb = self.memory_list[class_idx, indices]

            knnemb = torch.cat([repeat(self.global_proto, 'c d -> b c 1 d', b=batchsize), knnemb], dim=2)
            # knnemb = rearrange(knnemb, 'b c k d -> (b c) k d')
            knnemb = rearrange(knnemb, 'b c k d -> b (c k) d')

        # (M, D), (B, CK, D) -> B, M, D
        updated_tokens = self.knnformer(repeat(self.generic_tokens, 'm d -> b m d', b=batchsize), knnemb, knnemb)
        # updated_tokens = F.normalize(updated_tokens, p=2, eps=1e-6, dim=-1)

        # no kNN baseline; no prototype update at all!
        # gtokens = self.generic_tokens.unsqueeze(0)
        # _updated_proto = self.knnformer2(self.global_proto.unsqueeze(0), gtokens, gtokens)

        # (B, C, D), (B, M, D) -> B, C, D
        _updated_proto = self.knnformer2(repeat(self.global_proto, 'c d -> b c d', b=batchsize), updated_tokens, updated_tokens)

        # standization & l2 normalization
        _updated_proto = self.standardize(_updated_proto)

        # output becomes NaN if it's commented!!
        updated_proto = F.normalize(_updated_proto, p=2, eps=1e-6, dim=-1)

        sim = torch.einsum('b d, b c d -> b c', out, updated_proto)

        # no kNN baseline; no prototype update at all!
        # sim = torch.einsum('b d, c d -> b c', out, updated_proto.squeeze(0))
        return F.log_softmax(sim / 0.01, dim=1)

    def forward_old_m151617(self, x):
        idx = idx % self.dm.max_num_samples
        out = self.backbone(x)
        # normalize
        out = F.normalize(out, p=2, dim=-1)
        batchsize = out.shape[0]

        with torch.no_grad():
            classwise_sim = torch.einsum('b d, c n d -> b c n', out, self.memory_list)
            '''
            if self.training:  # to ignore self-voting
                # B, C, N -> B, C, K
                topk_sim, indices = classwise_sim.topk(k=self.args.k + 1, dim=-1, largest=True, sorted=True)
                # indices = indices[:, :, 1:]
                top1_indices = indices[:, :, 0]
                max_class_indices = top1_indices.argmax(dim=1)  # highly likely the self (or another twin in the feature space)
                indices[range(batchsize), max_class_indices, :-1] = indices[range(batchsize), max_class_indices, 1:]
                indices = indices[:, :, :-1]
            else:
                topk_sim, indices = classwise_sim.topk(k=self.args.k, dim=-1, largest=True, sorted=True)
            '''


            ''' M15 noisy kNN; define a buffered pool and select k among them
            topk_sim, indices = classwise_sim.topk(k=self.dm.max_num_samples // 4, dim=-1, largest=True, sorted=True)
            rand_indices = torch.randint(low=0, high=self.dm.max_num_samples // 4, size=[batchsize, self.num_classes, self.args.k])
            rand_indices = rand_indices.to(indices.device)
            indices = torch.gather(input=indices, dim=-1, index=rand_indices)
            '''

            # 1, C, 1
            class_idx = torch.arange(self.num_classes).unsqueeze(0).unsqueeze(-1)
            # C, N, D [[1, C, 1], [B, C, K]] -> B, C, K, D
            knnemb = self.memory_list[class_idx, indices]


            ''' M16 convex-combinated kNN
            cvx_coeff = torch.randn(1, 1, self.args.k, 1).to(knnemb.device).type(knnemb.dtype)
            cvx_coeff = torch.softmax(cvx_coeff, dim=2)
            knnemb = torch.sum(cvx_coeff * knnemb, dim=2)
            knnemb = F.normalize(knnemb, p=2, dim=-1)
            knnemb = torch.cat([repeat(self.global_proto, 'c d -> b c d', b=batchsize), knnemb], dim=1)
            '''


            ''' M17 fixed random pairs
            class_idx = torch.arange(self.num_classes).unsqueeze(0).unsqueeze(-1)

            pairs = []
            for i in idx:
                pair = self.memory_list[:, i + 1:i+self.args.k + 1]
                pairs.append(pair)

            pairs = torch.stack(pairs, dim=0).to(out.device).type(out.dtype)
            knnemb = pairs
            '''

            knnemb = torch.cat([repeat(self.global_proto, 'c d -> b c 1 d', b=batchsize), knnemb], dim=2)
            knnemb = rearrange(knnemb, 'b c k d -> (b c) k d')

        # (B, C, D), (B, CK, D) -> B, C, D
        updated_proto = self.knnformer(repeat(self.global_proto, 'c d -> (repeat c) 1 d', repeat=batchsize), knnemb, knnemb)
        updated_proto = F.normalize(updated_proto, p=2, eps=1e-6, dim=-1)
        sim = torch.einsum('b d, b c d -> b c', out, rearrange(updated_proto, '(b c) 1 d -> b c d', c=self.num_classes))

        return F.log_softmax(sim / 0.01, dim=1)

    def forward_m8_parallel_input_update_include_nearest_nol2normalization(self, x):
        out = self.backbone(x)
        batchsize = out.shape[0]

        with torch.no_grad():
            classwise_sim = torch.einsum('b d, c n d -> b c n', out, self.memory_list)
            if self.training:  # to ignore self-voting
                # B, C, N -> B, C, K
                topk_sim, indices = classwise_sim.topk(k=self.args.k + 1, dim=-1, largest=True, sorted=True)
                # indices = indices[:, :, 1:]
                top1_indices = indices[:, :, 0]
                max_class_indices = top1_indices.argmax(dim=1)  # highly likely the self (or another twin in the feature space)
                indices[range(batchsize), max_class_indices, :-1] = indices[range(batchsize), max_class_indices, 1:]
                indices = indices[:, :, :-1]
            else:
                topk_sim, indices = classwise_sim.topk(k=self.args.k, dim=-1, largest=True, sorted=True)

            # 1, C, 1
            class_idx = torch.arange(self.num_classes).unsqueeze(0).unsqueeze(-1)
            # C, N, D [[1, C, 1], [B, C, K]] -> B, C, K, D
            knnemb = self.memory_list[class_idx, indices]
            knnemb = rearrange(knnemb, 'b c k d -> b (c k) d')
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

        if self.args.LT:
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

        with torch.no_grad():
            # torch.set_printoptions(precision=2, edgeitems=50, linewidth=240)
            diff = (self.old_generic_tokens - self.generic_tokens).abs()
            # print(diff.mean(), diff.mean().max())
            # print(diff)
            # print(self.generic_tokens)
            # print()
            self.old_generic_tokens = self.generic_tokens.clone()

    def validation_epoch_end(self, outputs):
        epoch = self.trainer.current_epoch
        batch_losses = [x["val_loss"] for x in outputs]
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

        result = "\n\n" + result + "\n\n"
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
            # list(self.knnformer.parameters()) + list(self.fc.parameters()),
            lr=self.args.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

        return {"optimizer": optimizer}

    def standardize(self, x, dim=1, eps=1e-6):
        out = x - x.mean(dim=dim, keepdim=True)
        out = out / (out.std(dim=dim, keepdim=True) + eps)
        return out


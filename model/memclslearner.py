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

from model.transformer import TransformerEncoderLayer, ResidualAttentionBlock

import pdb


class MemClsLearner(LightningModule):
    def __init__(self, args, dm):
        super().__init__()

        self.args = args
        self.dm = dm
        self.backbone = self._init_backbone()
        self.dim = 512
        self.cachedir = osp.join(os.getcwd(), 'cache', args.dataset, args.backbone)

        self.count_correct = {'trn': 0.0, 'val': 0.0, 'tst': 0.0}
        self.count_all = {'trn': 0.0, 'val': 0.0, 'tst': 0.0}
        self.loss_all = {'trn': [], 'val': [], 'tst': []}

        self.modeldtype = torch.float16 if 'clip' in args.backbone else torch.float32
        factory_kwargs = {'device': self.device, 'dtype': self.modeldtype}

        self.attn = ResidualAttentionBlock(d_model=self.dim, n_head=1, **factory_kwargs)

        # self.generic_tokens = self._init_generic_tokens()

    def _init_backbone(self):
        """ Init pretrained backbone """
        if 'resnet' in self.args.backbone:
            if self.args.backbone == 'resnet18':
                model = torchvision.models.resnet18(weights='IMAGENET1K_V1', num_classes=1000)
            elif self.args.backbone == 'resnet50':
                model = torchvision.models.resnet50(weights='IMAGENET1K_V1', num_classes=1000)
            elif self.args.backbone == 'resnet50coco':
                model = torchvision.models.segmentation.deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
                model.classifier = nn.Identity()
                model.aux_classifier = nn.Identity()
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
        elif self.args.backbone == 'mobilenetcoco':
            model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights='COCO_WITH_VOC_LABELS_V1')
            model.classifier = nn.Identity()
            model.aux_classifier = nn.Identity()

        model.requires_grad_(requires_grad=False)
        '''
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                model.requires_grad_(requires_grad=True)
        '''
        return model

    def _init_generic_tokens(self):
        _generic_tokens = torch.empty(self.args.ntokens, 512, dtype=self.modeldtype, requires_grad=True)
        generic_tokens = nn.Parameter(_generic_tokens.clone(), requires_grad=True)
        # moved to self.on_fit_start; should be called after params being loaded to cuda
        # nn.init.trunc_normal_(self.generic_tokens, mean=0.0, std=0.02)
        return generic_tokens

    def _load_memory_and_prototype(self):
        self.img_embed = {'trn': None, 'val': None, 'tst': None}
        self.img_label = {'trn': None, 'val': None, 'tst': None}
        self.img_proto = {'trn': None, 'val': None, 'tst': None}
        self.txt_embed = {'trn': None, 'val': None, 'tst': None}
        self.txt_label = {'trn': None, 'val': None, 'tst': None}

        for split, img_loader, txt_loader in \
                zip(['trn', 'val', 'tst'],
                    [self.dm.train_memory_dataloader,
                        self.dm.val_memory_dataloader,
                        self.dm.test_memory_dataloader],
                    [self.dm.train_text_dataloader,
                        self.dm.val_text_dataloader,
                        self.dm.test_text_dataloader]):
            img_embed_path = osp.join(self.cachedir, f'{split}_img_embed.pth')
            img_label_path = osp.join(self.cachedir, f'{split}_img_label.pth')
            txt_embed_path = osp.join(self.cachedir, f'{split}_txt_embed.pth')
            txt_label_path = osp.join(self.cachedir, f'{split}_txt_label.pth')

            if osp.exists(img_embed_path):
                print(f'\n *** [{split}] loading embed/label checkpoints from {self.cachedir}. *** \n')
                img_embed = torch.load(img_embed_path) ; img_label = torch.load(img_label_path)
                txt_embed = torch.load(txt_embed_path) ; txt_label = torch.load(txt_label_path)
            else:
                print(f'\n *** [{split}] embed/label not found. Generating embed/label checkpoints and saving them at {self.cachedir}. *** \n')
                if not osp.exists(self.cachedir): os.makedirs(self.cachedir)
                img_embed, img_label = self._init_memory(img_loader, split, modality='img')
                txt_embed, txt_label = self._init_memory(txt_loader, split, modality='txt')
                torch.save(img_embed, img_embed_path) ; torch.save(img_label, img_label_path)
                torch.save(txt_embed, txt_embed_path) ; torch.save(txt_label, txt_label_path)

            img_label_idx = img_label.unique().sort()[0]
            img_proto = [img_embed[img_label == c].mean(dim=0) for c in img_label_idx]
            img_proto = torch.stack(img_proto, dim=0)  # C, D

            self.img_embed[split] = img_embed ; self.img_label[split] = img_label
            self.txt_embed[split] = txt_embed ; self.txt_label[split] = txt_label
            self.img_proto[split] = img_proto

            num_samples = int(img_embed.shape[0]) // (int(max(img_label)) + 1)
            print(f"\nLoaded memory info: #_of_samples = {img_embed.shape[0]} ({max(img_label)+1}x{num_samples}), dim_of_samples = {img_embed.shape[1]}")

        trn_img_label_idx = self.img_label['trn'].unique().sort()[0]
        self.train_class_count = [torch.sum(self.img_label['trn'] == c) for c in trn_img_label_idx]

    def _init_memory(self, loader, split, modality):
        '''
        Return an irregular memory list of image/text features with different number for each class
        '''
        backbone = self.backbone.to(self.device)
        backbone.eval()
        embed_list = []
        label_list = []

        with torch.inference_mode():
            for x, y in tqdm(loader(), desc=f'Generating {modality} {split} emb'):
                x = x.to(self.device)
                if modality == 'img':
                    out = backbone(x)  # = backbone.encode_image(x)
                elif modality == 'txt':
                    out = backbone.encode_text(x)
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

    # python main.py --datapath /home/eunchan/datasets/ --backbone clipvitb --dataset imagenet100 --logpath log --runfree naiveproto --eval --nowandb
    def forward_naive_protomatching(self, x, y, stage=None):
        out = self.backbone(x)

        assert self.args.eval, "This method can't be learned"

        proto = self.img_proto['tst'].to(x.device)

        out_ = out
        proto_ = proto

        # l2_norm
        # out_ = F.normalize(out, dim=-1, p=2)
        # proto_ = F.normalize(proto, dim=-1, p=2)

        sim = torch.einsum('c d, b d -> b c', proto_, out_) # * 0.001
        return sim

    # python main.py --datapath /home/eunchan/datasets/ --backbone clipvitb --dataset imagenet100 --logpath log --runfree nakata22 --k 1 --eval --nowandb
    def forward_nakata22(self, x, y, stage=None):
        out = self.backbone(x)

        assert self.args.eval, "This method can't be learned"

        memory = self.img_embed['tst'].to(x.device)
        labels = self.img_label['tst']
        num_cls = max(labels)+1

        out_ = out
        memory_ = memory

        # l2_norm
        # out_ = F.normalize(out, dim=-1, p=2)
        # memory_ = F.normalize(memory, dim=-1, p=2)

        globalsim = torch.einsum('b d, n d -> b n', out_, memory_)
        _, indices = globalsim.topk(k=self.args.k, dim=-1, largest=True, sorted=True)
        globalcls = labels[indices]

        def majority_vote(input):
            count = [0]*num_cls
            for item in input:
                count[item.cpu().item()] += 1

            onehot = torch.argmax(torch.tensor(count)).item() # just vote
            result = torch.tensor([0.]*num_cls)
            result[onehot] = 1.0

            return result.to(x.device)

        sim = torch.stack(list(map(majority_vote, globalcls)))

        return sim

    # main.py --datapath /home/dahyun/datasets --dataset imagenet130samples --backbone clipvitb --logpath yourlog --lr 5e-5 --wd 1e-2 --k 0 --multemp 32
    def forward_pb2_protomatching(self, x, y, stage):
        out = self.backbone(x)
        out = self.attn(out.unsqueeze(1), out.unsqueeze(1), out.unsqueeze(1)).squeeze(1)

        proto = self.img_proto[stage]
        proto_ = F.normalize(proto.to(x.device), dim=-1, p=2)
        out_ = F.normalize(out, dim=-1, p=2)

        sim = torch.einsum('c d, b d -> b c', proto_, out_) * self.args.multemp
        return sim

    # main.py --datapath /home/dahyun/datasets --dataset imagenet130samples --backbone clipvitb --logpath yourlog --lr 5e-5 --wd 1e-2 --k 16 --multemp 32
    def forward_p1_textknn_featupdate(self, x, y, stage):
        out = self.backbone(x)
        with torch.no_grad():
            classwise_sim = torch.einsum('b d, n d -> b n', out, self.txt_embed[stage])
            _, indices = classwise_sim.topk(k=self.args.k, dim=-1, largest=True, sorted=True)

            # N, D [[B, K] -> B, K, D
            knnemb = self.txt_embed[stage][indices]

            # B, 1, D
            tr_q = out.unsqueeze(1)

            # (B, 1, D), (B, K, D) -> B, (1 + K), D
            kv = torch.cat([tr_q, knnemb], dim=1)

        out = self.attn(tr_q, kv, kv).squeeze(1)

        proto = self.img_proto[stage]
        proto_ = F.normalize(proto.to(x.device), dim=-1, p=2)
        out_ = F.normalize(out, dim=-1, p=2)

        sim = torch.einsum('c d, b d -> b c', proto_, out_) * self.args.multemp
        return sim

    # main.py --datapath /home/dahyun/datasets --dataset imagenet130samples --backbone clipvitb --logpath yourlog --lr 5e-5 --wd 1e-2 --k 16 --multemp 32
    def forward_p2_textknn_parrellel_logitupdate(self, x, y, stage):
        clipfeat = self.backbone(x)

        with torch.no_grad():
            classwise_sim = torch.einsum('b d, n d -> b n', clipfeat, self.txt_embed[stage])
            _, indices = classwise_sim.topk(k=self.args.k, dim=-1, largest=True, sorted=True)

            # N, D [[B, K] -> B, K, D
            knnemb = self.txt_embed[stage][indices]

            # B, 1, D
            tr_q = clipfeat.unsqueeze(1)

            # (B, 1, D), (B, K, D) -> B, (1 + K), D
            kv = torch.cat([tr_q, knnemb], dim=1)

        out = self.attn(tr_q, kv, kv).squeeze(1)
        proto = self.img_proto[stage]

        proto_ = F.normalize(proto.to(x.device), dim=-1, p=2)
        out_ = F.normalize(out, dim=-1, p=2)
        clipfeat_ = F.normalize(clipfeat, dim=-1, p=2)

        sim_clip = torch.einsum('c d, b d -> b c', proto_, clipfeat_)
        sim_text = torch.einsum('c d, b d -> b c', proto_, out_)

        # logit fusion
        sim = 0.5 * (sim_clip + sim_text) * self.args.multemp

        # pb5: wrong
        # sim = 0.5 * (F.softmax(sim_clip, dim=-1) + F.softmax(sim_text, dim=-1)) * self.args.multemp + 1e-6
        # pb5: crprobfusion : wrong
        # sim = 0.5 * (F.softmax(sim_clip * self.args.multemp, dim=-1) + F.softmax(sim_text * self.args.multemp, dim=-1)) + 1e-6
        # pb5: crprobfusion
        # sim = torch.log((0.5 * F.softmax(sim_clip * self.args.multemp, dim=-1) + 0.5 * F.softmax(sim_text * self.args.multemp, dim=-1)) + 1e-6)
        return sim

    def record_metrics(self, count_correct_batch, count_all_batch, loss, stage):
        self.count_correct[stage] += count_correct_batch
        self.count_all[stage] += count_all_batch
        self.loss_all[stage].append(loss)

        if self.args.LT:
            raise NotImplementedError('trn/val/tst split not considered')
            correct = (preds == y).int()

            for ans, lbl in zip(correct.tolist(), y.tolist()):
                self.count_class_correct[lbl] += ans
                self.count_class_valimgs[lbl] += 1

    def each_step(self, batch, stage=None):
        self.backbone.eval()
        x, y = batch
        logits = self(x, y, stage=stage)
        loss = F.cross_entropy(logits, y)

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

        if self.args.LT:
            # You will come across self.count_class_correct undefined error. You can define it in the
            # class initializer and re-initialize it in this function after use
            raise NotImplementedError('trn/val acc compuation not implemented')
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

    def on_train_epoch_end(self):
        self.each_epoch_end(stage='trn')

    def on_validation_epoch_end(self):
        self.each_epoch_end(stage='val')

    def on_test_epoch_end(self):
        self.each_epoch_end(stage='tst')

    def configure_optimizers(self):
        param_list = []
        for k, v in self.named_parameters():
            if not 'backbone' in k:  # or 'ln' in k:
                param_list.append(v)
        optimizer = torch.optim.Adam(
            param_list,
            lr=self.args.lr,
            # momentum=0.9,
            weight_decay=self.args.wd,
            eps=1e-6,
        )

        return {"optimizer": optimizer}

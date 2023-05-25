""" Memory-Modular Learner """

import os
import os.path as osp
import clip
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from model.transformer import TransformerEncoderLayer, ResidualAttentionBlock

import pdb


class MemoryModularLearner(nn.Module):
    def __init__(self, args, dm, **factory_kwargs):
        super().__init__()

        self.args = args
        self.dm = dm
        self.backbone = self._init_backbone()
        self.dim = 512
        self.cachedir = osp.join(os.getcwd(), 'cache', args.dataset, args.backbone)

        self.modeldtype = torch.float16 if 'clip' in args.backbone else torch.float32

        # self.attn = ResidualAttentionBlock(d_model=self.dim, n_head=1, **factory_kwargs)
        self.attn_txt = ResidualAttentionBlock(d_model=self.dim, n_head=1, **factory_kwargs)
        self.attn_img = ResidualAttentionBlock(d_model=self.dim, n_head=1, **factory_kwargs)
        self.loss_fn = nn.CrossEntropyLoss()

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
        self.txt_proto = {'trn': None, 'val': None, 'tst': None}

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

            txt_label_idx = txt_label.unique().sort()[0]
            txt_proto = [txt_embed[txt_label == c].mean(dim=0) for c in txt_label_idx]
            txt_proto = torch.stack(txt_proto, dim=0)  # C, D

            self.img_embed[split] = img_embed ; self.txt_embed[split] = txt_embed
            self.img_label[split] = img_label ; self.txt_label[split] = txt_label
            self.img_proto[split] = img_proto ; self.txt_proto[split] = txt_proto

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

    # python main.py --datapath /home/eunchan/datasets/ --backbone clipvitb --dataset imagenet100 --logpath log --runfree naiveproto --eval --nowandb
    def forward_naive_protomatching(self, x, y, stage=None):
        with torch.no_grad():
            out = self.backbone(x)

        assert self.args.eval, "This method can't be learned"

        proto = self.img_proto['tst'].to(x.device)

        # l2_norm
        out_ = F.normalize(out, dim=-1, p=2)
        proto_ = F.normalize(proto, dim=-1, p=2)

        sim = torch.einsum('c d, b d -> b c', proto_, out_) # * 0.001
        return sim

    # python main.py --datapath /home/eunchan/datasets/ --backbone clipvitb --dataset imagenet100 --logpath log --runfree nakata22 --k 1 --eval --nowandb
    def forward_nakata22(self, x, y, stage=None):
        with torch.no_grad():
            out = self.backbone(x)

        assert self.args.eval, "This method can't be learned"

        memory = self.img_embed['tst'].to(x.device)
        labels = self.img_label['tst']
        num_cls = max(labels)+1

        # l2_norm
        out_ = F.normalize(out, dim=-1, p=2)
        memory_ = F.normalize(memory, dim=-1, p=2)

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

    def forward_p15_1_crossmodal_self_imgknn_textknn_probfusion_nokvinput(self, x, y, stage):
        def retrieve_knn(x, mem, k):
            with torch.no_grad():
                classwise_sim = torch.einsum('b d, n d -> b n', x, mem)
                _, indices = classwise_sim.topk(k=k, dim=-1, largest=True, sorted=True)

                # N, D [[B, K] -> B, K, D
                knnemb = mem[indices]
                return knnemb

        with torch.no_grad():
            clipfeat = self.backbone(x)

        kv_txt = retrieve_knn(x=clipfeat, mem=self.txt_embed[stage], k=self.args.k)
        kv_img = retrieve_knn(x=clipfeat, mem=self.img_embed[stage], k=self.args.k)

        out_txt = self.attn_txt(clipfeat.unsqueeze(1), kv_txt, kv_txt).squeeze(1)
        out_img = self.attn_img(clipfeat.unsqueeze(1), kv_img, kv_img).squeeze(1)

        out_txt_ = F.normalize(out_txt, dim=-1, p=2, eps=1e-8)
        out_img_ = F.normalize(out_img, dim=-1, p=2, eps=1e-8)
        clipfeat_ = F.normalize(clipfeat, dim=-1, p=2, eps=1e-8)
        proto_txt_ = F.normalize(self.txt_proto[stage].to(x.device), dim=-1, p=2, eps=1e-8)
        proto_img_ = F.normalize(self.img_proto[stage].to(x.device), dim=-1, p=2, eps=1e-8)

        sim_clip = torch.einsum('c d, b d -> b c', proto_txt_, clipfeat_) * 32.
        sim_txt = torch.einsum('c d, b d -> b c', proto_img_, out_txt_) * self.args.multemp
        sim_img = torch.einsum('c d, b d -> b c', proto_txt_, out_img_) * self.args.multemp

        # p2: logit fusion
        # sim = 0.5 * (sim_clip + sim_text) * self.args.multemp

        # p3: prob fusion
        # Be AWARE of using either
        #     1) F.cross_entropy(unnormalized_tensor)
        #     2) F.nll_loss(torch.log(F.softmax(unnormalized_tensor)))
        sim_avg = (F.softmax(sim_clip, dim=-1) + F.softmax(sim_txt, dim=-1) + F.softmax(sim_img, dim=-1)) / 3.
        sim = torch.log(sim_avg + 1e-6)
        self.loss_fn = nn.NLLLoss()  # only comes with log_softmax

        return sim

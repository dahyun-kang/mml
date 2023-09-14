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
        return generic_tokens

    def _load_memory_and_prototype(self):
        self.img_embed = {'trn': None, 'val': None, 'tst': None}
        self.img_label = {'trn': None, 'val': None, 'tst': None}
        self.img_proto = {'trn': None, 'val': None, 'tst': None}
        self.txt_embed = {'trn': None, 'val': None, 'tst': None}
        self.txt_label = {'trn': None, 'val': None, 'tst': None}
        self.txt_proto = {'trn': None, 'val': None, 'tst': None}

        # load or save memory
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

            txt_label_idx = txt_label.unique().sort()[0]
            txt_proto = [txt_embed[txt_label == c].mean(dim=0) for c in txt_label_idx]
            txt_proto = torch.stack(txt_proto, dim=0)  # C, D

            self.img_embed[split] = img_embed ; self.txt_embed[split] = txt_embed
            self.img_label[split] = img_label ; self.txt_label[split] = txt_label  # actually not used in current forward
            self.txt_proto[split] = txt_proto

            num_samples = int(img_embed.shape[0]) // (int(max(img_label)) + 1)
            print(f"\nLoaded memory info: #_of_samples = {img_embed.shape[0]} ({max(img_label)+1}x{num_samples}), dim_of_samples = {img_embed.shape[1]}")

        # load few-shot prototype
        for split, img_shot_loader in \
                zip(['trn', 'val', 'tst'],
                    [self.dm.train_shot_dataloader,
                        self.dm.val_shot_dataloader,
                        self.dm.test_shot_dataloader]):

            img_proto_path = osp.join(self.cachedir, f'{split}_img_proto.pth')

            # clean few-shot image samples
            '''
            if osp.exists(img_proto_path):
                print(f'\n *** [{split}] loading class prototype checkpoints from {self.cachedir}. *** \n')
                img_proto = torch.load(img_proto_path)
            else:
                print(f'\n *** [{split}] embed/label not found. Generating class prototype checkpoints and saving them at {self.cachedir}. *** \n')
                img_embed, img_label = self._init_memory(img_shot_loader, split, modality='img')

                img_label_idx = img_label.unique().sort()[0]
                img_proto = [img_embed[img_label == c].mean(dim=0) for c in img_label_idx]
                img_proto = torch.stack(img_proto, dim=0)  # C, D
                # torch.save(img_proto, img_proto_path)
            '''

            img_embed = self.img_embed[split] ; txt_embed = self.txt_embed[split]
            img_label = self.img_label[split] ; txt_label = self.txt_label[split]

            img_proto_list = []
            for c in torch.unique(img_label):
                img_embed_c = img_embed[img_label == c]
                txt_embed_c = txt_embed[txt_label == c]
                img_embed_c_l2 = F.normalize(img_embed_c, dim=-1, p=2)
                txt_embed_c_l2 = F.normalize(txt_embed_c, dim=-1, p=2)
                sim = torch.einsum('n d, m d -> n m', img_embed_c_l2, txt_embed_c_l2)
                sim = sim.sum(dim=-1)
                _, indices = sim.topk(k=16, dim=-1, largest=True, sorted=True)  # TODO: make it args
                img_proto_list.append(img_embed_c[indices].mean(dim=0))

            img_proto = torch.stack(img_proto_list, dim=0)

            self.img_proto[split] = img_proto

            print(f"\n{split} class prototype info: dim_of_samples = {img_proto.shape[0]}x{img_proto.shape[1]}")

        trn_img_label_idx = self.img_label['trn'].unique().sort()[0]
        self.train_class_count = [torch.sum(self.img_label['trn'] == c) for c in trn_img_label_idx]

    def _load_episodic_test_memory_and_prototype(self):
        split = 'tst'  # by default
        self.img_embed = {split: None}
        self.img_label = {split: None}
        self.img_proto = {split: None}
        self.txt_embed = {split: None}
        self.txt_label = {split: None}
        self.txt_proto = {split: None}

        classset = self.dm.classset  # renewed everytime with random seeds
        img_loader = self.dm.test_memory_dataloader
        img_embed, img_label = self._init_memory(img_loader, split, modality='img')
        self.img_embed[split] = img_embed
        self.img_label[split] = img_label  # 0, 1, 2, ...

        img_label_idx = img_label.unique().sort()[0]
        img_proto = [img_embed[img_label == c].mean(dim=0) for c in img_label_idx]
        self.img_proto[split] = torch.stack(img_proto, dim=0)  # C, D

        if not hasattr(self, 'txt_embed_tst_all'):
            txt_embed_path = osp.join(self.cachedir, f'{split}_txt_embed.pth')
            self.txt_embed_tst_all = torch.load(txt_embed_path)
        if not hasattr(self, 'txt_label_tst_all'):
            txt_label_path = osp.join(self.cachedir, f'{split}_txt_label.pth')
            self.txt_label_tst_all = torch.load(txt_label_path)

        txt_embed = []
        txt_label = []
        for c in classset:
            idx_c = torch.nonzero(self.txt_label_tst_all == c).squeeze()
            txt_embed.append(self.txt_embed_tst_all[idx_c])
            txt_label.append(self.txt_label_tst_all[idx_c])

        self.txt_embed[split] = torch.cat(txt_embed, dim=0).detach()  # N, D
        # actually this one is not used in forward
        self.txt_label[split] = torch.cat(txt_label, dim=0).detach()  # original class index

        txt_label_idx = self.txt_label[split].unique().sort()[0]
        txt_proto = [self.txt_embed[split][self.txt_label[split] == c].mean(dim=0) for c in txt_label_idx]
        self.txt_proto[split] = torch.stack(txt_proto, dim=0)  # C, D

    def _init_memory(self, loader, split, modality):
        '''
        Return an irregular memory list of image/text features with different number for each class
        '''
        backbone = self.backbone.cuda()
        backbone.eval()
        embed_list = []
        label_list = []

        with torch.inference_mode():
            for x, y in tqdm(loader(), desc=f'Generating {modality} {split} emb'):
                x = x.cuda()
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

        # assert self.args.eval, "This method can't be learned"  # TODO: add episodiceval

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

        # p15.1
        kv_txt = retrieve_knn(x=clipfeat, mem=self.txt_embed[stage], k=self.args.k)
        kv_img = retrieve_knn(x=clipfeat, mem=self.img_embed[stage], k=self.args.k)

        out_txt = self.attn_txt(clipfeat.unsqueeze(1), kv_txt, kv_txt).squeeze(1)
        out_img = self.attn_img(clipfeat.unsqueeze(1), kv_img, kv_img).squeeze(1)

        # p15.2 self attn
        # out_txt = self.attn_txt(clipfeat.unsqueeze(1), clipfeat.unsqueeze(1), clipfeat.unsqueeze(1)).squeeze(1)
        # out_img = self.attn_img(clipfeat.unsqueeze(1), clipfeat.unsqueeze(1), clipfeat.unsqueeze(1)).squeeze(1)

        out_txt_ = F.normalize(out_txt, dim=-1, p=2)
        out_img_ = F.normalize(out_img, dim=-1, p=2)
        clipfeat_ = F.normalize(clipfeat, dim=-1, p=2)
        proto_txt_ = F.normalize(self.txt_proto[stage].to(x.device), dim=-1, p=2)
        proto_img_ = F.normalize(self.img_proto[stage].to(x.device), dim=-1, p=2)

        sim_clip = torch.einsum('c d, b d -> b c', proto_txt_, clipfeat_) * 32 # * self.temp1
        sim_txt = torch.einsum('c d, b d -> b c', proto_img_, out_txt_) * self.args.multemp # * self.temp2
        sim_img = torch.einsum('c d, b d -> b c', proto_txt_, out_img_) * self.args.multemp # * self.temp3

        # p2: logit fusion
        # sim = 0.5 * (sim_clip + sim_text) * self.args.multemp

        # p3: prob fusion
        # Be AWARE of using either
        #     1) F.cross_entropy(unnormalized_tensor)
        #     2) F.nll_loss(torch.log(F.softmax(unnormalized_tensor)))
        sim_avg = (F.softmax(sim_clip, dim=-1) + F.softmax(sim_txt, dim=-1) + F.softmax(sim_img, dim=-1)) / 3.
        # sim_avg = (F.softmax(sim_clip, dim=-1) + F.softmax(sim_txt, dim=-1)) / 2.

        sim = torch.log(sim_avg + 1e-6)
        self.loss_fn = nn.NLLLoss()  # only comes with log_softmax

        return sim

    # def forward_0804_p20_trainqry_mem_1nnexclude_l2search_crossmodal_noclipbase_imgknn_textknn_logitfusion_nokvinput(self, x, y, stage):
    # def forward_0909_p21_addclipfeat_txtembedl2simlargest16shotproto(self, x, y, stage):
    def forward(self, x, y, stage):
        def retrieve_knn(x, mem, k):
            with torch.no_grad():
                x_ = F.normalize(x, p=2, dim=-1)
                mem_ = F.normalize(mem, p=2, dim=-1)  # fix at the beginning
                sim = torch.einsum('b d, n d -> b n', x_, mem_)
                _, indices = sim.topk(k=k, dim=-1, largest=True, sorted=True)

                # N, D [[B, K] -> B, K, D
                knnemb = mem[indices]
                return knnemb

        with torch.no_grad():
            clipfeat = self.backbone(x)

        kv_txt = retrieve_knn(x=clipfeat, mem=self.txt_embed[stage], k=self.args.tk)
        kv_img = retrieve_knn(x=clipfeat, mem=self.img_embed[stage], k=self.args.ik)

        out_txt = self.attn_txt(clipfeat.unsqueeze(1), kv_txt, kv_txt).squeeze(1) + clipfeat
        out_img = self.attn_img(clipfeat.unsqueeze(1), kv_img, kv_img).squeeze(1) + clipfeat

        '''
        out_txt = self.attn_txt(clipfeat.unsqueeze(1), clipfeat.unsqueeze(1), clipfeat.unsqueeze(1)).squeeze(1)
        out_img = self.attn_img(clipfeat.unsqueeze(1), clipfeat.unsqueeze(1), clipfeat.unsqueeze(1)).squeeze(1)
        '''

        # clipfeat_ = F.normalize(clipfeat, dim=-1, p=2)
        out_txt_ = F.normalize(out_txt, dim=-1, p=2)
        out_img_ = F.normalize(out_img, dim=-1, p=2)
        proto_txt_ = F.normalize(self.txt_proto[stage].to(x.device), dim=-1, p=2)
        proto_img_ = F.normalize(self.img_proto[stage].to(x.device), dim=-1, p=2)

        # sim_clip = torch.einsum('c d, b d -> b c', proto_txt_, clipfeat_) * 8.  # 32 with probfusion
        sim_txt = torch.einsum('c d, b d -> b c', proto_img_, out_txt_) * self.args.multemp
        sim_img = torch.einsum('c d, b d -> b c', proto_txt_, out_img_) * self.args.multemp

        sim = sim_txt + sim_img

        return sim

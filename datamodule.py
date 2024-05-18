import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule

from dataclasses import dataclass
from dataset import DatasetSplitLoader, TextTokenMemoryDataset, WebvisionMemoryDataset, FGMemoryDataset, ImageNet1K


@dataclass
class DisjointClassDataModule(LightningDataModule):
    datasetsroot: str = '/ssd2t/datasets/'
    datasetdir: str = 'imagenetunseen'
    imgsize: int = 224
    batchsize: int = 256
    nworkers: int = 8
    trnsplit: str = 'train'
    valsplit: str = 'val'
    tstsplit: str = 'test'
    ntrainsamples: int = 0  # args input
    nvalsamples: int = 200

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.root = os.path.join(self.datasetsroot, self.datasetdir)

        self.query = {}
        self.imgmem = {}
        self.txtmem = {}
        self.shot = {}

    def setup(self, stage: str):
        # if episodiceval: return self.setup_episodic_eval()

        trn_split = DatasetSplitLoader(self.root, self.trnsplit, 'trn_label.json')
        val_split = DatasetSplitLoader(self.root, self.valsplit, 'val_label.json')
        tst_split = DatasetSplitLoader(self.root, self.tstsplit, 'tst_label.json')

        self.query['trn'], _ = trn_split.split(nsamples=self.ntrainsamples, nshot=0)
        self.shot['trn'] = self.query['trn']
        # self.add_label_noise(self.query['trn'].targets, ratio=0.4)
        self.query['val'], self.shot['val'] = val_split.split(nsamples=self.nvalsamples, nshot=self.nshot)  # nsamples=self.nsamples, is_shot=True, nshot=self.nshot)  # full
        self.query['tst'], self.shot['tst'] = tst_split.split(nsamples=self.nvalsamples, nshot=self.nshot)  # nsamples=self.nsamples, is_shot=True, nshot=self.nshot)  # full)

        # self.query['trn'] = self.shot['tst']  # linear on test, RAC
        # self.query['val'] = self.query['tst']  # linear on test, RAC

        imgmemroot = os.path.join(self.datasetsroot, 'webvisionv1')
        self.imgmem['trn'] = WebvisionMemoryDataset(imgmemroot=imgmemroot, queryroot=self.root, classid2target=trn_split.classid2target)
        self.imgmem['val'] = WebvisionMemoryDataset(imgmemroot=imgmemroot, queryroot=self.root, classid2target=val_split.classid2target)
        self.imgmem['tst'] = WebvisionMemoryDataset(imgmemroot=imgmemroot, queryroot=self.root, classid2target=tst_split.classid2target)

        # self.imgmem['trn'] = ImageNet1K(datasetsroot=self.datasetsroot, label_file=f'{self.datasetdir}/trn_label.json', split='train')
        # self.imgmem['val'] = ImageNet1K(datasetsroot=self.datasetsroot, label_file=f'{self.datasetdir}/val_label.json', split='train')
        # self.imgmem['tst'] = ImageNet1K(datasetsroot=self.datasetsroot, label_file=f'{self.datasetdir}/tst_label.json', split='train')

        self.txtmem['trn'] = TextTokenMemoryDataset(self.root, classid2target=trn_split.classid2target)
        self.txtmem['val'] = TextTokenMemoryDataset(self.root, classid2target=val_split.classid2target)
        self.txtmem['tst'] = TextTokenMemoryDataset(self.root, classid2target=tst_split.classid2target)

        if self.nshot > 0:
            assert len(set(self.query['val'].data).intersection(set(self.shot['val'].data))) == 0
            assert len(set(self.query['tst'].data).intersection(set(self.shot['tst'].data))) == 0

    def train_dataloader(self):
        return DataLoader(self.query['trn'], batch_size=self.batchsize, num_workers=self.nworkers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.query['val'], batch_size=self.batchsize, num_workers=self.nworkers)

    def test_dataloader(self):
        return DataLoader(self.query['tst'], batch_size=self.batchsize, num_workers=self.nworkers)

    def txtmem_dataloader(self, split):
        return DataLoader(self.txtmem[split], batch_size=self.batchsize, num_workers=self.nworkers)

    def shot_dataloader(self, split):
        return DataLoader(self.shot[split], batch_size=self.batchsize, num_workers=self.nworkers)

    def imgmem_dataloader(self, split):
        return DataLoader(self.imgmem[split], batch_size=self.batchsize, num_workers=self.nworkers)
    
    def target2txtlabel(self, split):
        return self.txtmem[split].target2txtlabel

    '''
    def setup_episodic_eval(self, nclass=5, nshot=5, nquery=15):
        root = os.path.join(self.datasetsroot, self.datasetdir)

        if not hasattr(self, 'dataset_test_all'):
            self.dataset_test_all = self.dataset(root, splitdir=self.tstsplit, label_file='tst_label.json', nsamples=None, is_shot=True, nshot=None)
            imgmemroot = os.path.join(self.datasetsroot, 'webvisionv1')
            self.dataset_test_memory_all = WebvisionMemoryDataset(root=imgmemroot, label_file=os.path.join(root, 'tst_label.json'))

        classset = torch.randperm(self.dataset_test_all.num_classes)[:nclass]
        classset = classset.sort()[0]
        self.classset = classset

        shot_img_path = []
        shot_targets = []
        query_img_path = []
        query_targets = []
        mem_img_path = []
        mem_targets = []

        # i: 0, 1, 2, ... | c: original class indices
        for i, c in enumerate(classset):
            # collect c-th class samples
            idx_c = torch.nonzero(torch.tensor(self.dataset_test_all.targets) == c).squeeze().tolist()
            img_path_c = np.array(self.dataset_test_all.img_path)[idx_c].tolist()
            # targets_c = np.array(self.targets)[idx_c].tolist()

            # randomly sample n samples
            randidx = torch.randperm(len(img_path_c))[:nshot + nquery]
            shot_idx = randidx[:nshot]
            query_idx = randidx[nshot:]

            shot_img_path += [img_path_c[shot_idx]] if nshot == 1 else np.array(img_path_c)[shot_idx].tolist()
            query_img_path += np.array(img_path_c)[query_idx].tolist()
            shot_targets += [i] * nshot
            query_targets += [i] * nquery

            # collect c-th class "memory"
            mem_idx_c = torch.nonzero(torch.tensor(self.dataset_test_memory_all.targets) == c).squeeze().tolist()
            mem_img_path_c = np.array(self.dataset_test_memory_all.img_path)[mem_idx_c].tolist()
            mem_img_path += mem_img_path_c
            mem_targets += [i] * len(mem_img_path_c)

        self.query['tst'] = SubsetDataset(data=query_img_path, targets=query_targets)
        self.shot['tst'] = SubsetDataset(data=shot_img_path, targets=shot_targets)
        self.imgmem['tst'] = SubsetDataset(data=mem_img_path, targets=mem_targets)
        self.txtmem['tst'] = TextTokenMemoryDataset(root, classids=self.query['tst'].classids)
    '''

    def add_label_noise(self, targets, ratio=0.4):
        datalen = len(targets)
        numclass = max(targets) + 1
        numnoise = int(datalen * ratio)
        noiseidx = torch.randperm(datalen)[:numnoise]
        targets_tensor = torch.tensor(targets)
        targets_tensor[noiseidx] = torch.randint(low=0, high=numclass, size=[numnoise])
        targets = targets_tensor.tolist()

    @property
    def num_classes(self) -> int:
        return self.query['trn'].num_classes


class ImageNet1KDataModule(DisjointClassDataModule):
    def setup(self, stage: str):
        # self.query['trn'] = self.dataset(self.root, splitdir=self.trnsplit, label_file='standard_label.json', nsamples=None, is_shot=False, nshot=0)
        trn_split = DatasetSplitLoader(self.root, self.trnsplit, 'standard_label.json')
        self.query['trn'], _ = trn_split.split(nsamples=self.ntrainsamples, nshot=0)
        self.query['val'] = ImageNet1K(datasetsroot=self.datasetsroot, classsplit_file=f'{self.datasetdir}/standard_label.json', split='val')
        self.query['tst'] = self.query['val']

        self.shot['trn'] = self.query['trn']  # equivalent to dataset_train and shared with all splits
        self.shot['val'] = self.shot['trn']
        self.shot['tst'] = self.shot['trn']

        # self.imgmem['trn'] = ImageNet1K(datasetsroot=self.datasetsroot, label_file=f'{self.datasetdir}/standard_label.json', split='train')
        imgmemroot = os.path.join(self.datasetsroot, 'webvisionv1')
        self.imgmem['trn'] = WebvisionMemoryDataset(imgmemroot=imgmemroot, queryroot=self.root, classid2target=trn_split.classid2target)
        self.imgmem['val'] = self.imgmem['trn']
        self.imgmem['tst'] = self.imgmem['trn']

        self.txtmem['trn'] = TextTokenMemoryDataset(self.root, classid2target=trn_split.classid2target)
        self.txtmem['val'] = self.txtmem['trn']
        self.txtmem['tst'] = self.txtmem['trn']


class SameClassFGDataModule(DisjointClassDataModule):
    def __init__(self, imgmemdataset, **kwargs):
        super().__init__(**kwargs)
        self.root = os.path.join(self.datasetsroot, self.datasetdir)
        self.imgmemdataset = imgmemdataset

    def setup(self, stage: str):
        trn_split = DatasetSplitLoader(self.root, self.trnsplit, 'standard_label.json')
        val_split = DatasetSplitLoader(self.root, self.valsplit, 'standard_label.json')

        self.query['trn'], _ = trn_split.split(nsamples=self.ntrainsamples, nshot=0)
        # self.add_label_noise(self.query['trn'].targets, ratio=0.4)
        self.query['val'], _ = val_split.split(nsamples=None, nshot=0)
        self.query['tst'] = self.query['val']

        self.shot['trn'] = self.query['trn']
        self.shot['val'] = self.shot['trn']
        self.shot['tst'] = self.shot['val']

        self.imgmem['trn'] = self.imgmemdataset(root=self.root, memory_dir='memory', classid2target=trn_split.classid2target)
        self.imgmem['val'] = self.imgmem['trn']
        self.imgmem['tst'] = self.imgmem['trn']

        self.txtmem['trn'] = TextTokenMemoryDataset(self.root, classid2target=trn_split.classid2target)
        self.txtmem['val'] = self.txtmem['trn']
        self.txtmem['tst'] = self.txtmem['trn']


class DisjointClassFGDataModule(DisjointClassDataModule):
    def __init__(self, imgmemdataset, **kwargs):
        super().__init__(**kwargs)
        self.imgmemdataset = imgmemdataset

    def setup(self, stage: str):
        trn_split = DatasetSplitLoader(self.root, self.trnsplit, 'trn_label.json')
        val_split = DatasetSplitLoader(self.root, self.valsplit, 'val_label.json')
        tst_split = DatasetSplitLoader(self.root, self.tstsplit, 'tst_label.json')

        self.query['trn'], _ = trn_split.split(nsamples=self.ntrainsamples, nshot=0)
        self.query['val'], self.shot['val'] = val_split.split(nsamples=self.nvalsamples, nshot=self.nshot)
        self.query['tst'], self.shot['tst'] = tst_split.split(nsamples=self.nvalsamples, nshot=self.nshot)

        self.imgmem['trn'] = self.imgmemdataset(root=self.root, memory_dir='memory', classid2target=trn_split.classid2target)
        self.imgmem['val'] = self.imgmemdataset(root=self.root, memory_dir='memory', classid2target=val_split.classid2target)
        self.imgmem['tst'] = self.imgmemdataset(root=self.root, memory_dir='memory', classid2target=tst_split.classid2target)

        self.txtmem['trn'] = TextTokenMemoryDataset(self.root, classid2target=trn_split.classid2target)
        self.txtmem['val'] = TextTokenMemoryDataset(self.root, classid2target=val_split.classid2target)
        self.txtmem['tst'] = TextTokenMemoryDataset(self.root, classid2target=tst_split.classid2target)

        if self.nshot > 0:
            assert len(set(self.query['val'].data).intersection(set(self.shot['val'].data))) == 0
            assert len(set(self.query['tst'].data).intersection(set(self.shot['tst'].data))) == 0


def return_datamodule(datapath, dataset, batchsize, ntrainsamples, shot):
    if 'imagenetunseen' in dataset:
        dm = DisjointClassDataModule(datasetsroot=datapath, datasetdir=dataset, batchsize=batchsize, ntrainsamples=ntrainsamples, nshot=shot)
    elif 'imagenetseen' in dataset:
        dm = ImageNet1KDataModule(datasetsroot=datapath, datasetdir=dataset, batchsize=batchsize, ntrainsamples=ntrainsamples, nshot=shot)
    elif dataset == 'cub200':
        dm = DisjointClassFGDataModule(datasetsroot=datapath, datasetdir='CUB_200_2011', imgmemdataset=FGMemoryDataset, batchsize=batchsize, trnsplit='Train_150', valsplit='Val_50', tstsplit='Val_50', nsamples=ntrainsamples, nshot=shot)
    elif dataset in ['caltech-101', 'oxford_pets', 'oxford_flowers', 'dtd', 'fgvc_aircraft', 'eurosat', 'sun397', 'ucf101', 'stanford_cars', 'food-101']:
        dm = SameClassFGDataModule(datasetsroot=datapath, datasetdir=dataset, imgmemdataset=FGMemoryDataset, batchsize=batchsize, ntrainsamples=ntrainsamples, nshot=shot)
    else:
        raise NotImplementedError
    '''
    elif 'imagenethalf' in dataset:
        dm = DisjointClassDataModule(datasetsroot=datapath, datasetdir=dataset, batchsize=batchsize, trnsplit='train', valsplit='val', tstsplit='val', nshot=shot)
    elif 'miniimagenet' in datasetkey:
        dm = DisjointClassDataModule(datasetsroot=datapath, datasetdir=dataset, batchsize=batchsize, trnsplit='train', valsplit='val', tstsplit='val', nshot=shot, nsamples=595)
    elif datasetkey == 'cub200seen':
        dm = SameClassFGDataModule(datasetsroot=datapath, datasetdir='CUB_200_2011', imgmemdataset=CUBMemoryDataset, batchsize=batchsize, trnsplit='Train', valsplit='Val', tstsplit='Val', nshot=shot)
    elif datasetkey == 'food101':
        dm = DisjointClassFGDataModule(datasetsroot=datapath, datasetdir='food-101', imgmemdataset=FoodMemoryDataset, batchsize=batchsize, trnsplit='Train_class', valsplit='Val_class', tstsplit='Val_class', nshot=shot)
    '''

    return dm

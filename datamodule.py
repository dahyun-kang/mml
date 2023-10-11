import os
import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule

from dataclasses import dataclass
from dataset import DisjointClassDataset, TextTokenMemoryDataset, WebvisionMemoryDataset, CUBMemoryDataset, FoodMemoryDataset, ImageNet1K, SubsetDataset


@dataclass
class DisjointClassDataModule(LightningDataModule):
    datasetsroot: str = '/ssd1t/datasets/'
    datasetdir: str = 'imagenetunseen16shots_seed7'
    imgsize: int = 224
    batchsize: int = 256
    nworkers: int = 8
    trainsplit: str = 'train'
    valsplit: str = 'val'
    testsplit: str = 'test'
    nsamples: int = 200
    nshot: int = 16

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.dataset = DisjointClassDataset

    def setup(self, stage: str):
        # if episodiceval: return self.setup_episodic_eval()
        queryroot = os.path.join(self.datasetsroot, self.datasetdir)

        self.query = {}
        self.imgmem = {}
        self.txtmem = {}
        self.shot = {}

        self.query['trn'] = self.dataset(queryroot, splitdir=self.trainsplit, label_file='trn_label.json', nsamples=None, is_shot=False, nshot=0)
        '''
        datalen = len(self.dataset_train.targets)
        numclass = max(self.dataset_train.targets) + 1
        numnoise = int(datalen * 0.05)
        noiseidx = torch.randperm(datalen)[:numnoise]
        targets = torch.tensor(self.dataset_train.targets)
        targets[noiseidx] = torch.randint(low=0, high=numclass, size=[numnoise])
        self.dataset_train.targets = targets.tolist()
        '''

        self.query['val'] = self.dataset(queryroot, splitdir=self.valsplit, label_file='val_label.json', nsamples=self.nsamples, is_shot=False, nshot=self.nshot)
        self.query['tst'] = self.dataset(queryroot, splitdir=self.testsplit, label_file='tst_label.json', nsamples=self.nsamples, is_shot=False, nshot=self.nshot)

        self.shot['trn'] = self.query['trn']
        self.shot['val'] = self.dataset(queryroot, splitdir=self.valsplit, label_file='val_label.json', nsamples=None, is_shot=True, nshot=self.nshot)  # nsamples=self.nsamples, is_shot=True, nshot=self.nshot)  # full
        self.shot['tst'] = self.dataset(queryroot, splitdir=self.testsplit, label_file='tst_label.json', nsamples=None, is_shot=True, nshot=self.nshot) # nsamples=self.nsamples, is_shot=True, nshot=self.nshot)  # full

        # self.query['trn'] = self.shot['tst']  # linear on test, RAC
        # self.query['val'] = self.query['tst']  # linear on test, RAC

        imgmemroot = os.path.join(self.datasetsroot, 'webvisionv1')
        self.imgmem['trn'] = WebvisionMemoryDataset(imgmemroot=imgmemroot, queryroot=queryroot, label_file='trn_label.json')
        self.imgmem['val'] = WebvisionMemoryDataset(imgmemroot=imgmemroot, queryroot=queryroot, label_file='val_label.json')
        self.imgmem['tst'] = WebvisionMemoryDataset(imgmemroot=imgmemroot, queryroot=queryroot, label_file='tst_label.json')

        # self.imgmem['trn'] = ImageNet1K(datasetsroot=self.datasetsroot, label_file=f'{self.datasetdir}/trn_label.json', split='train')
        # self.imgmem['val'] = ImageNet1K(datasetsroot=self.datasetsroot, label_file=f'{self.datasetdir}/val_label.json', split='train')
        # self.imgmem['tst'] = ImageNet1K(datasetsroot=self.datasetsroot, label_file=f'{self.datasetdir}/tst_label.json', split='train')

        self.txtmem['trn'] = TextTokenMemoryDataset(queryroot, classids=self.query['trn'].classids)
        self.txtmem['val'] = TextTokenMemoryDataset(queryroot, classids=self.query['val'].classids)
        self.txtmem['tst'] = TextTokenMemoryDataset(queryroot, classids=self.query['tst'].classids)

        assert len(set(self.query['val'].img_path).intersection(set(self.shot['val'].img_path))) == 0
        assert len(set(self.query['tst'].img_path).intersection(set(self.shot['tst'].img_path))) == 0

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

    def setup_episodic_eval(self, nclass=5, nshot=5, nquery=15):
        root = os.path.join(self.datasetsroot, self.datasetdir)

        if not hasattr(self, 'dataset_test_all'):
            self.dataset_test_all = self.dataset(root, splitdir=self.testsplit, label_file='tst_label.json', nsamples=None, is_shot=True, nshot=None)
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

        self.query = {}
        self.shot = {}
        self.imgmem = {}
        self.txtmem = {}

        self.query['tst'] = SubsetDataset(data=query_img_path, targets=query_targets)
        self.shot['tst'] = SubsetDataset(data=shot_img_path, targets=shot_targets)
        self.imgmem['tst'] = SubsetDataset(data=mem_img_path, targets=mem_targets)
        self.txtmem['tst'] = TextTokenMemoryDataset(root, classids=self.query['tst'].classids)

    @property
    def num_classes(self) -> int:
        return self.query['trn'].num_classes


class SameClassDataModule(DisjointClassDataModule):
    def setup(self, stage: str):
        queryroot = os.path.join(self.datasetsroot, self.datasetdir)

        self.query = {}
        self.imgmem = {}
        self.txtmem = {}
        self.shot = {}

        self.query['trn'] = self.dataset(queryroot, splitdir=self.trainsplit, label_file='standard_label.json', nsamples=None, is_shot=False, nshot=0)
        self.query['val'] = ImageNet1K(datasetsroot=self.datasetsroot, label_file=f'{self.datasetdir}/standard_label.json', split='val')
        self.query['tst'] = self.query['val']

        self.shot['trn'] = self.query['trn']  # equivalent to dataset_train and shared with all splits
        self.shot['val'] = self.shot['trn']
        self.shot['tst'] = self.shot['trn']

        # self.imgmem['trn'] = ImageNet1K(datasetsroot=self.datasetsroot, label_file=f'{self.datasetdir}/standard_label.json', split='train')
        imgmemroot = os.path.join(self.datasetsroot, 'webvisionv1')
        self.imgmem['trn'] = WebvisionMemoryDataset(imgmemroot=imgmemroot, queryroot=queryroot, label_file='standard_label.json')
        self.imgmem['val'] = self.imgmem['trn']
        self.imgmem['tst'] = self.imgmem['trn']

        self.txtmem['trn'] = TextTokenMemoryDataset(queryroot, classids=self.query['trn'].classids)
        self.txtmem['val'] = self.txtmem['trn']
        self.txtmem['tst'] = self.txtmem['trn']


class Food101DataModule_Standard(SameClassDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datasetsroot = 'food-101'
        self.nsamples = None  # val/test queries
        self.nshot = None

        self.trainsplit = 'Train'
        self.valsplit = 'Val'
        self.testsplit = 'Val'

    def setup(self, stage: str):
        root = os.path.join(self.datasetsroot, self.datasetsroot)

        self.query = {}
        self.imgmem = {}
        self.txtmem = {}
        self.shot = {}

        self.query['trn'] = self.dataset(root, splitdir=self.trainsplit, label_file='standard_label.json', nsamples=None, is_shot=False, nshot=0)
        self.query['val'] = self.dataset(root, splitdir=self.valsplit, label_file='standard_label.json', nsamples=self.nsamples, is_shot=False, nshot=0)
        self.query['tst'] = self.query['val']

        self.shot['trn'] = None
        self.shot['val'] = None
        self.shot['tst'] = None

        self.imgmem['trn'] = FoodMemoryDataset(root=root, memory_dir='memory', label_file='standard_label.json')
        self.imgmem['val'] = self.imgmem['trn']
        self.imgmem['tst'] = self.imgmem['trn']

        self.txtmem['trn'] = TextTokenMemoryDataset(root, classids=self.query['trn'].classids)
        self.txtmem['val'] = self.txtmem['trn']
        self.txtmem['tst'] = self.txtmem['trn']


class Cub2011DataModule(DisjointClassDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nsamples = 60  # val/test queries # train: 41 ~ 60 images per class, val: 49 ~ 60 images per class
        self.datasetsroot = 'CUB_200_2011'
        self.nshot = 5
        self.trainsplit = 'Train_class'
        self.valsplit = 'Val_class'
        self.testsplit = 'Val_class'

    def setup(self, stage: str):
        root = os.path.join(self.datasetsroot, self.datasetsroot)

        self.query = {}
        self.imgmem = {}
        self.txtmem = {}
        self.shot = {}

        self.query['trn'] = self.dataset(root, splitdir=self.trainsplit, label_file='trn_label.json', nsamples=5, is_shot=False, nshot=0)
        self.query['val'] = self.dataset(root, splitdir=self.valsplit, label_file='val_label.json', nsamples=self.nsamples, is_shot=False, nshot=self.nshot)
        self.query['tst'] = self.dataset(root, splitdir=self.testsplit, label_file='tst_label.json', nsamples=self.nsamples, is_shot=False, nshot=self.nshot)
        self.shot['trn'] = self.query['trn']
        self.shot['val'] = self.dataset(root, splitdir=self.valsplit, label_file='val_label.json', nsamples=None, is_shot=True, nshot=self.nshot)
                                          # nsamples=self.nsamples, is_shot=True, nshot=self.nshot)  # full
        self.shot['tst'] = self.dataset(root, splitdir=self.testsplit, label_file='tst_label.json',
                                          nsamples=None, is_shot=True, nshot=self.nshot)
                                          # nsamples=self.nsamples, is_shot=True, nshot=self.nshot)  # full

        assert len(set(self.query['val'].img_path).intersection(set(self.shot['val'].img_path))) == 0
        assert len(set(self.query['tst'].img_path).intersection(set(self.shot['tst'].img_path))) == 0

        # Remove the code duplicates
        self.imgmem['trn'] = CUBMemoryDataset(root=root, memory_dir='memory', label_file='trn_label.json')
        self.imgmem['val'] = CUBMemoryDataset(root=root, memory_dir='memory', label_file='val_label.json')
        self.imgmem['tst'] = CUBMemoryDataset(root=root, memory_dir='memory', label_file='tst_label.json')

        self.txtmem['trn'] = TextTokenMemoryDataset(root, classids=self.query['trn'].classids)
        self.txtmem['val'] = TextTokenMemoryDataset(root, classids=self.query['val'].classids)
        self.txtmem['tst'] = TextTokenMemoryDataset(root, classids=self.query['tst'].classids)


class Cub2011DataModule_Standard(SameClassDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datasetsroot = 'CUB_200_2011'
        self.nsamples = None  # val/test queries
        self.nshot = None

        self.trainsplit = 'Train'
        self.valsplit = 'Val'
        self.testsplit = 'Val'

    def setup(self, stage: str):
        root = os.path.join(self.datasetsroot, self.datasetsroot)

        self.query = {}
        self.imgmem = {}
        self.txtmem = {}
        self.shot = {}

        self.query['trn'] = self.dataset(root, splitdir=self.trainsplit, label_file='standard_label.json', nsamples=5, is_shot=False, nshot=0)
        self.query['val'] = self.dataset(root, splitdir=self.valsplit, label_file='standard_label.json', nsamples=self.nsamples, is_shot=False, nshot=0)
        self.query['tst'] = self.query['val']

        self.shot['trn'] = self.query['trn']
        self.shot['val'] = self.shot['trn']
        self.shot['tst'] = self.shot['trn']

        self.imgmem['trn'] = CUBMemoryDataset(root=root, memory_dir='memory', label_file='standard_label.json')
        self.imgmem['val'] = self.imgmem['trn']
        self.imgmem['tst'] = self.imgmem['trn']

        self.txtmem['trn'] = TextTokenMemoryDataset(root, classids=self.query['trn'].classids)
        self.txtmem['val'] = self.txtmem['trn']
        self.txtmem['tst'] = self.txtmem['trn']


@dataclass
class ImageNet1KDataModule(LightningDataModule):
    datasetsroot: str
    imgsize: int
    batchsize: int
    nworkers: int

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    def imgmem_dataloader(self, split):
        assert split == 'tst'
        dataset = ImageNet1K(datasetsroot=self.datasetsroot, split='train')
        loader = DataLoader(dataset, batch_size=self.batchsize, num_workers=self.nworkers, shuffle=False, sampler=None)
        return loader

    def test_dataloader(self):
        dataset = ImageNet1K(datasetsroot=self.datasetsroot, split='val')
        loader = DataLoader(dataset, batch_size=self.batchsize, num_workers=self.nworkers, shuffle=False, sampler=None)
        return loader


def return_datamodule(datapath, dataset, batchsize, shot):
    datasetkey = dataset if 'seed' not in dataset else dataset.split('_seed')[0]
    if 'imagenetunseen' in datasetkey:
        dm = DisjointClassDataModule(datasetsroot=datapath, datasetdir=dataset, batchsize=batchsize, trainsplit='train', valsplit='val', testsplit='test', nshot=shot)
    elif 'imagenethalf' in datasetkey:
        dm = DisjointClassDataModule(datasetsroot=datapath, datasetdir=dataset, batchsize=batchsize, trainsplit='train', valsplit='val', testsplit='val', nshot=shot)
    elif 'miniimagenet' in datasetkey:
        dm = DisjointClassDataModule(datasetsroot=datapath, datasetdir=dataset, batchsize=batchsize, trainsplit='train', valsplit='val', testsplit='val', nshot=shot, nsamples=595)
    elif datasetkey == 'cub2011':
        dm = DisjointClassDataModule(datasetsroot=datapath, datasetdir=dataset, batchsize=batchsize, trainsplit='Train', valsplit='Val', testsplit='Val', nshot=shot)
    elif 'imagenetseen' in datasetkey:
        dm = SameClassDataModule(datasetsroot=datapath, datasetdir=dataset, batchsize=batchsize, trainsplit='train', valsplit='val', testsplit='test', nshot=shot)
    elif datasetkey == 'cub2011standard' or datasetkey == 'food101standard' :
        dm = SameClassDataModule(datasetsroot=datapath, datasetdir=dataset, batchsize=batchsize, trainsplit='Train', valsplit='Val', testsplit='Val', nshot=shot)
    elif datasetkey == 'imagenet1K':
        dm = ImageNet1KDataModule(datasetsroot=datapath, batchsize=batchsize)

    return dm

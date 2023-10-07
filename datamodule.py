import os
import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule

from dataset import ClassDisjointAbstractDataset, TextTokenMemoryDataset, WebvisionMemoryDataset, CUBMemoryDataset, FoodMemoryDataset, ImageNet1K


class Transforms:
    @staticmethod
    def clip_transform(n_px):
        '''
        https://github.com/openai/CLIP/blob/main/clip/clip.py
        '''
        try:
            from torchvision.transforms import InterpolationMode
            BICUBIC = InterpolationMode.BICUBIC
        except ImportError:
            from PIL import Image
            BICUBIC = Image.BICUBIC

        def _convert_image_to_rgb(image):
            return image.convert("RGB")

        return torchvision.transforms.Compose([
            torchvision.transforms.Resize(n_px, interpolation=BICUBIC),
            torchvision.transforms.CenterCrop(n_px),
            _convert_image_to_rgb,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


class AbstractDataModule(LightningDataModule):
    def __init__(self, datadir='data', dataset='', imgsize=224, batchsize=256, num_workers=0, transform_type=None, train_split=None, val_split=None):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = ClassDisjointAbstractDataset
        self.dataset_query = {'trn': None, 'val': None}
        if transform_type == 'CLIP':
            self.train_transform = Transforms.clip_transform(imgsize)
            self.val_transform = Transforms.clip_transform(imgsize)
        else:
            self.train_transform = Transforms.train_transform(imgsize)
            self.val_transform = Transforms.val_transform(imgsize)

    @property
    def num_classes(self) -> int:
        return None

    def setup(self, stage: str):
        self.dataset_query['trn'] = self.dataset(root=self.hparams.datadir, split=self.hparams.train_split, transform=self.train_transform, download=True)
        self.dataset_query['val'] = self.dataset(root=self.hparams.datadir, split=self.hparams.val_split, transform=self.val_transform, download=True)

    def train_dataloader(self):
        return DataLoader(self.dataset_query['trn'], batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers, shuffle=True)

    def unshuffled_train_dataloader(self):
        if self.dataset_query['trn'] is None:
            self.setup(stage='init')
        return DataLoader(self.dataset_query['trn'], batch_size=512, num_workers=self.hparams.num_workers, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.dataset_query['val'], batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()


class ImageNet100DataModule(AbstractDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_query_num_samples = 1100
        self.dataset_root = 'imagenet100'
        self.len_shot = 200
        self.train_subdirs = ['train.X1', 'train.X3', 'train.X4']
        self.val_subdirs = ['train.X2']
        self.test_subdirs = ['train.X2']

    def setup(self, stage: str):
        # if episodiceval: return self.setup_episodic_eval()
        root = os.path.join(self.hparams.datadir, self.dataset_root)
        label_mapping_file = 'labels.txt'

        synset2txtlabel = self.imagenetsynset2txtlabel()

        self.dataset_query = {}
        self.dataset_imgmem = {}
        self.dataset_txtmem = {}
        self.dataset_shot = {}

        self.dataset_query['trn'] = self.dataset(root, sub_dirs=self.train_subdirs, label_file='trn_label.json', label_mapping_file=label_mapping_file,
                                          max_samples=None, transform=self.train_transform, is_shot=False, len_shot=0)
        '''
        datalen = len(self.dataset_train.targets)
        numclass = max(self.dataset_train.targets) + 1
        numnoise = int(datalen * 0.05)
        noiseidx = torch.randperm(datalen)[:numnoise]
        targets = torch.tensor(self.dataset_train.targets)
        targets[noiseidx] = torch.randint(low=0, high=numclass, size=[numnoise])
        self.dataset_train.targets = targets.tolist()
        '''

        self.dataset_query['val'] = self.dataset(root, sub_dirs=self.val_subdirs, label_file='val_label.json', label_mapping_file=label_mapping_file,
                                          max_samples=self.max_query_num_samples, transform=self.val_transform, is_shot=False, len_shot=self.len_shot)
        self.dataset_query['tst'] = self.dataset(root, sub_dirs=self.test_subdirs, label_file='tst_label.json', label_mapping_file=label_mapping_file,
                                          max_samples=self.max_query_num_samples, transform=self.val_transform, is_shot=False, len_shot=self.len_shot)
        self.dataset_shot['trn'] = self.dataset_query['trn']
        self.dataset_shot['val'] = self.dataset(root, sub_dirs=self.val_subdirs, label_file='val_label.json', label_mapping_file=label_mapping_file,
                                          max_samples=None, transform=self.val_transform, is_shot=True, len_shot=self.len_shot)
                                          # max_samples=self.max_query_num_samples, transform=self.val_transform, is_shot=True, len_shot=self.len_shot)  # full
        self.dataset_shot['tst'] = self.dataset(root, sub_dirs=self.test_subdirs, label_file='tst_label.json', label_mapping_file=label_mapping_file,
                                          max_samples=None, transform=self.val_transform, is_shot=True, len_shot=self.len_shot)
                                          # max_samples=self.max_query_num_samples, transform=self.val_transform, is_shot=True, len_shot=self.len_shot)  # full

        # self.dataset_query['trn'] = self.dataset_shot['tst']  # linear on test, RAC
        # self.dataset_query['val'] = self.dataset_query['tst']  # linear on test, RAC


        assert len(set(self.dataset_query['val'].img_path).intersection(set(self.dataset_shot['val'].img_path))) == 0
        assert len(set(self.dataset_query['tst'].img_path).intersection(set(self.dataset_shot['tst'].img_path))) == 0

        # Remove the code duplicates
        self.dataset_imgmem['trn'] = WebvisionMemoryDataset(root=os.path.join(self.hparams.datadir, 'webvisionv1'),
                                                      label_file=os.path.join(root, 'trn_label.json'),
                                                      synset2txtlabel=synset2txtlabel,
                                                      transform=self.train_transform,
                                                      len_memory=1000)
        self.dataset_imgmem['val'] = WebvisionMemoryDataset(root=os.path.join(self.hparams.datadir, 'webvisionv1'),
                                                      label_file=os.path.join(root, 'val_label.json'),
                                                      synset2txtlabel=synset2txtlabel,
                                                      transform=self.val_transform,
                                                      len_memory=1000)
        self.dataset_imgmem['tst'] = WebvisionMemoryDataset(root=os.path.join(self.hparams.datadir, 'webvisionv1'),
                                                      label_file=os.path.join(root, 'tst_label.json'),
                                                      synset2txtlabel=synset2txtlabel,
                                                      transform=self.val_transform,
                                                      len_memory=1000)
        # self.dataset_imgmem['trn'] = ImageNet1K(split='train', datadir=self.hparams.datadir, transform=self.train_transform, label_file=f'{self.dataset_root}/trn_label.json', synset2txtlabel=synset2txtlabel)
        # self.dataset_imgmem['val'] = ImageNet1K(split='train', datadir=self.hparams.datadir, transform=self.val_transform, label_file=f'{self.dataset_root}/val_label.json', synset2txtlabel=synset2txtlabel)
        # self.dataset_imgmem['tst'] = ImageNet1K(split='train', datadir=self.hparams.datadir, transform=self.val_transform, label_file=f'{self.dataset_root}/tst_label.json', synset2txtlabel=synset2txtlabel)

        self.dataset_txtmem['trn'] = TextTokenMemoryDataset(self.dataset_query['trn'].text_tokens, self.dataset_query['trn'].num_sents)
        self.dataset_txtmem['val'] = TextTokenMemoryDataset(self.dataset_query['val'].text_tokens, self.dataset_query['val'].num_sents)
        self.dataset_txtmem['tst'] = TextTokenMemoryDataset(self.dataset_query['tst'].text_tokens, self.dataset_query['tst'].num_sents)

    def test_dataloader(self):
        return DataLoader(self.dataset_query['tst'], batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers)

    def txtmem_dataloader(self, split):
        return DataLoader(self.dataset_txtmem[split], batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers)

    def shot_dataloader(self, split):
        return DataLoader(self.dataset_shot[split], batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers)

    def imgmem_dataloader(self, split):
        return DataLoader(self.dataset_imgmem[split], batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers)

    def setup_episodic_eval(self, nclass=5, nshot=5, nquery=15):
        root = os.path.join(self.hparams.datadir, self.dataset_root)
        label_mapping_file = 'labels.txt'

        if not hasattr(self, 'dataset_test_all'):
            self.dataset_test_all = self.dataset(root, sub_dirs=self.test_subdirs, label_file='tst_label.json', label_mapping_file=label_mapping_file,
                                                max_samples=None, transform=self.val_transform, is_shot=True, len_shot=None)
            self.dataset_test_memory_all = WebvisionMemoryDataset(root=os.path.join(self.hparams.datadir, 'webvisionv1'),
                                                         label_file=os.path.join(root, 'tst_label.json'),
                                                         transform=self.val_transform, len_memory=1000)

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

        self.dataset_query = {}
        self.dataset_shot = {}
        self.dataset_imgmem = {}
        self.dataset_txtmem = {}

        self.dataset_query['tst'] = SubsetDataset(data=query_img_path, targets=query_targets, transform=self.val_transform)
        self.dataset_shot['tst'] = SubsetDataset(data=shot_img_path, targets=shot_targets, transform=self.val_transform)
        self.dataset_imgmem['tst'] = SubsetDataset(data=mem_img_path, targets=mem_targets, transform=self.val_transform)
        self.dataset_txtmem['tst'] = TextTokenMemoryDataset(self.dataset_test_all.text_tokens, self.dataset_test_all.num_sents)

    def imagenetsynset2txtlabel(self):
        synset2txtlabel = {}
        with open(os.path.join(self.hparams.datadir, self.dataset_root, 'cliplabels.txt'), 'r') as f:
            lines = list(f.read().splitlines())
            for line in lines:
                synset, target, txtlabel = line.split(',')
                synset2txtlabel[synset] = txtlabel
        return synset2txtlabel

    @property
    def num_classes(self) -> int:
        return self.dataset_query['trn'].num_classes


class MiniImagenetDataModule(ImageNet100DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_query_num_samples = 595
        self.dataset_root = 'miniimagenet'
        self.len_shot = 5
        self.train_subdirs = ['train']
        self.val_subdirs = ['val']
        self.test_subdirs = ['test']


class ImageNetHalf16shotDataModule(ImageNet100DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_query_num_samples = 200  # val/test queries
        self.dataset_root = kwargs['dataset']
        self.len_shot = 16
        self.train_subdirs = ['train']
        self.val_subdirs = ['val']
        self.test_subdirs = ['val']


class ImageNetUnseen4shotDataModule(ImageNet100DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_query_num_samples = 200  # val/test queries
        self.dataset_root = kwargs['dataset']
        self.len_shot = 4
        self.train_subdirs = ['train']
        self.val_subdirs = ['val']
        self.test_subdirs = ['test']


class ImageNetUnseen16shotDataModule(ImageNet100DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_query_num_samples = 200  # val/test queries
        self.dataset_root = kwargs['dataset']
        self.len_shot = 16
        self.train_subdirs = ['train']
        self.val_subdirs = ['val']
        self.test_subdirs = ['test']


class ImageNetUnseenfullshotDataModule(ImageNet100DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_query_num_samples = 200  # val/test queries
        self.dataset_root = 'ILSVRC_unseenfullshots'
        self.len_shot = 1100
        self.train_subdirs = ['train']
        self.val_subdirs = ['val']
        self.test_subdirs = ['test']


class ImageNetUnseen64shotDataModule(ImageNet100DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_query_num_samples = 200  # val/test queries
        self.dataset_root = kwargs['dataset']
        self.len_shot = 64
        self.train_subdirs = ['train']
        self.val_subdirs = ['val']
        self.test_subdirs = ['test']


class ImageNetUnseen256shotDataModule(ImageNet100DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_query_num_samples = 200  # val/test queries
        self.dataset_root = kwargs['dataset']
        self.len_shot = 256
        self.train_subdirs = ['train']
        self.val_subdirs = ['val']
        self.test_subdirs = ['test']


class ImageNetUnseen512shotDataModule(ImageNet100DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_query_num_samples = 200  # val/test queries
        self.dataset_root = kwargs['dataset']
        self.len_shot = 512
        self.train_subdirs = ['train']
        self.val_subdirs = ['val']
        self.test_subdirs = ['test']


class ImageNet100DataModule_Standard(ImageNet100DataModule):
    ''' seen classes '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_subdirs = ['train.X1', 'train.X2', 'train.X3', 'train.X4']
        self.val_subdirs = ['val.X']
        self.test_subdirs = ['val.X']

    def setup(self, stage: str):
        root = os.path.join(self.hparams.datadir, self.dataset_root)
        label_mapping_file = 'labels.txt'

        synset2txtlabel = self.imagenetsynset2txtlabel()

        self.dataset_query = {}
        self.dataset_imgmem = {}
        self.dataset_txtmem = {}
        self.dataset_shot = {}

        self.dataset_query['trn'] = self.dataset(root, sub_dirs=self.train_subdirs, label_file='standard_label.json', label_mapping_file=label_mapping_file,
                                          max_samples=None, transform=self.train_transform, is_shot=False, len_shot=0)
        self.dataset_query['val'] = ImageNet1K(split='val', datadir=self.hparams.datadir, transform=self.val_transform, label_file = f'{self.dataset_root}/standard_label.json', synset2txtlabel=synset2txtlabel)
        self.dataset_query['tst'] = self.dataset_query['val']

        self.dataset_shot['trn'] = self.dataset_query['trn']  # equivalent to dataset_train and shared with all splits
        self.dataset_shot['val'] = self.dataset_shot['trn']
        self.dataset_shot['tst'] = self.dataset_shot['trn']


        # self.dataset_imgmem['trn'] = ImageNet1K(split='train', datadir=self.hparams.datadir, transform=self.val_transform, label_file = f'{self.dataset_root}/standard_label.json', synset2txtlabel=synset2txtlabel)
        self.dataset_imgmem['trn'] = WebvisionMemoryDataset(root=os.path.join(self.hparams.datadir, 'webvisionv1'),
                                                      label_file=os.path.join(root, 'standard_label.json'),
                                                      synset2txtlabel=synset2txtlabel,
                                                      transform=self.train_transform,
                                                      len_memory=1000)
        self.dataset_imgmem['val'] = self.dataset_imgmem['trn']
        self.dataset_imgmem['tst'] = self.dataset_imgmem['trn']

        self.dataset_txtmem['trn'] = TextTokenMemoryDataset(self.dataset_query['trn'].text_tokens, self.dataset_query['trn'].num_sents)
        self.dataset_txtmem['val'] = self.dataset_txtmem['trn']
        self.dataset_txtmem['tst'] = self.dataset_txtmem['trn']


class ImageNetSeen16shotDataModule(ImageNet100DataModule_Standard):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_query_num_samples = 200  # val/test queries
        self.dataset_root = kwargs['dataset']
        self.len_shot = 16
        self.train_subdirs = ['train']
        self.val_subdirs = ['val']
        self.test_subdirs = ['test']


class Food101DataModule_Standard(ImageNet100DataModule_Standard):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_root = 'food-101'
        self.max_query_num_samples = None  # val/test queries
        self.len_shot = None

        self.train_subdirs = ['Train']
        self.val_subdirs = ['Val']
        self.test_subdirs = ['Val']

    def setup(self, stage: str):
        root = os.path.join(self.hparams.datadir, self.dataset_root)
        label_mapping_file = 'labels.txt'

        self.dataset_query = {}
        self.dataset_imgmem = {}
        self.dataset_txtmem = {}
        self.dataset_shot = {}

        self.dataset_query['trn'] = self.dataset(root, sub_dirs=self.train_subdirs, label_file='standard_label.json', label_mapping_file=label_mapping_file,
                                          max_samples=None, transform=self.train_transform, is_shot=False, len_shot=0)
        self.dataset_query['val'] = self.dataset(root, train=False, sub_dirs=self.val_subdirs, label_file='standard_label.json', label_mapping_file=label_mapping_file,
                                          max_samples=self.max_query_num_samples, transform=self.val_transform, is_shot=False, len_shot=0)
        self.dataset_query['tst'] = self.dataset_query['val']

        self.dataset_shot['trn'] = None
        self.dataset_shot['val'] = None
        self.dataset_shot['tst'] = None

        self.dataset_imgmem['trn'] = FoodMemoryDataset(root=root, memory_dir='memory', label_file='standard_label.json', transform=self.train_transform)
        self.dataset_imgmem['val'] = self.dataset_imgmem['trn']
        self.dataset_imgmem['tst'] = self.dataset_imgmem['trn']

        self.dataset_txtmem['trn'] = TextTokenMemoryDataset(self.dataset_query['trn'].text_tokens, self.dataset_query['trn'].num_sents)
        self.dataset_txtmem['val'] = self.dataset_txtmem['trn']
        self.dataset_txtmem['tst'] = self.dataset_txtmem['trn']


class Cub2011DataModule(ImageNet100DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_query_num_samples = 60  # val/test queries # train: 41 ~ 60 images per class, val: 49 ~ 60 images per class
        self.dataset_root = 'CUB_200_2011'
        self.len_shot = 5
        self.train_subdirs = ['Train_class']
        self.val_subdirs = ['Val_class']
        self.test_subdirs = ['Val_class']

    def setup(self, stage: str):
        # if episodiceval: return self.setup_episodic_eval()
        root = os.path.join(self.hparams.datadir, self.dataset_root)
        label_mapping_file = 'labels.txt'

        self.dataset_query = {}
        self.dataset_imgmem = {}
        self.dataset_txtmem = {}
        self.dataset_shot = {}

        self.dataset_query['trn'] = self.dataset(root, sub_dirs=self.train_subdirs, label_file='trn_label.json', label_mapping_file=label_mapping_file,
                                          max_samples=5, transform=self.train_transform, is_shot=False, len_shot=0)

        self.dataset_query['val'] = self.dataset(root, sub_dirs=self.val_subdirs, label_file='val_label.json', label_mapping_file=label_mapping_file,
                                          max_samples=self.max_query_num_samples, transform=self.val_transform, is_shot=False, len_shot=self.len_shot)
        self.dataset_query['tst'] = self.dataset(root, sub_dirs=self.test_subdirs, label_file='tst_label.json', label_mapping_file=label_mapping_file,
                                          max_samples=self.max_query_num_samples, transform=self.val_transform, is_shot=False, len_shot=self.len_shot)
        self.dataset_shot['trn'] = self.dataset_query['trn']
        self.dataset_shot['val'] = self.dataset(root, sub_dirs=self.val_subdirs, label_file='val_label.json', label_mapping_file=label_mapping_file,
                                          max_samples=None, transform=self.val_transform, is_shot=True, len_shot=self.len_shot)
                                          # max_samples=self.max_query_num_samples, transform=self.val_transform, is_shot=True, len_shot=self.len_shot)  # full
        self.dataset_shot['tst'] = self.dataset(root, sub_dirs=self.test_subdirs, label_file='tst_label.json', label_mapping_file=label_mapping_file,
                                          max_samples=None, transform=self.val_transform, is_shot=True, len_shot=self.len_shot)
                                          # max_samples=self.max_query_num_samples, transform=self.val_transform, is_shot=True, len_shot=self.len_shot)  # full

        assert len(set(self.dataset_query['val'].img_path).intersection(set(self.dataset_shot['val'].img_path))) == 0
        assert len(set(self.dataset_query['tst'].img_path).intersection(set(self.dataset_shot['tst'].img_path))) == 0

        # Remove the code duplicates
        self.dataset_imgmem['trn'] = CUBMemoryDataset(root=root, memory_dir='memory', label_file='trn_label.json', transform=self.train_transform)
        self.dataset_imgmem['val'] = CUBMemoryDataset(root=root, memory_dir='memory', label_file='val_label.json', transform=self.train_transform)
        self.dataset_imgmem['tst'] = CUBMemoryDataset(root=root, memory_dir='memory', label_file='tst_label.json', transform=self.train_transform)

        self.dataset_txtmem['trn'] = TextTokenMemoryDataset(self.dataset_query['trn'].text_tokens, self.dataset_query['trn'].num_sents)
        self.dataset_txtmem['val'] = TextTokenMemoryDataset(self.dataset_query['val'].text_tokens, self.dataset_query['val'].num_sents)
        self.dataset_txtmem['tst'] = TextTokenMemoryDataset(self.dataset_query['tst'].text_tokens, self.dataset_query['tst'].num_sents)


class Cub2011DataModule_Standard(ImageNet100DataModule_Standard):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_root = 'CUB_200_2011'
        self.max_query_num_samples = None  # val/test queries
        self.len_shot = None

        self.train_subdirs = ['Train']
        self.val_subdirs = ['Val']
        self.test_subdirs = ['Val']

    def setup(self, stage: str):
        root = os.path.join(self.hparams.datadir, self.dataset_root)
        label_mapping_file = 'labels.txt'

        self.dataset_query = {}
        self.dataset_imgmem = {}
        self.dataset_txtmem = {}
        self.dataset_shot = {}

        self.dataset_query['trn'] = self.dataset(root, sub_dirs=self.train_subdirs, label_file='standard_label.json', label_mapping_file=label_mapping_file,
                                          max_samples=5, transform=self.train_transform, is_shot=False, len_shot=0)
        self.dataset_query['val'] = self.dataset(root, train=False, sub_dirs=self.val_subdirs, label_file='standard_label.json', label_mapping_file=label_mapping_file,
                                          max_samples=self.max_query_num_samples, transform=self.val_transform, is_shot=False, len_shot=0)
        self.dataset_query['tst'] = self.dataset_query['val']

        self.dataset_shot['trn'] = self.dataset_query['trn']
        self.dataset_shot['val'] = self.dataset_shot['trn']
        self.dataset_shot['tst'] = self.dataset_shot['trn']

        self.dataset_imgmem['trn'] = CUBMemoryDataset(root=root, memory_dir='memory', label_file='standard_label.json', transform=self.train_transform)
        self.dataset_imgmem['val'] = self.dataset_imgmem['trn']
        self.dataset_imgmem['tst'] = self.dataset_imgmem['trn']

        self.dataset_txtmem['trn'] = TextTokenMemoryDataset(self.dataset_query['trn'].text_tokens, self.dataset_query['trn'].num_sents)
        self.dataset_txtmem['val'] = self.dataset_txtmem['trn']
        self.dataset_txtmem['tst'] = self.dataset_txtmem['trn']


class ImageNet1KDataModule(LightningDataModule):
    def __init__(self, datadir, imgsize=224, batchsize=256, num_workers=0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.transform = Transforms.clip_transform(imgsize)

    def imgmem_dataloader(self, split):
        assert split == 'tst'
        dataset = ImageNet1K(datadir=self.hparams.datadir, split='train', transform=self.transform)
        loader = DataLoader(dataset, batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers, shuffle=False, sampler=None)
        return loader

    def test_dataloader(self):
        dataset = ImageNet1K(datadir=self.hparams.datadir, split='val', transform=self.transform)
        loader = DataLoader(dataset, batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers, shuffle=False, sampler=None)
        return loader


def return_datamodule(datapath, dataset, batchsize, backbone):
    datasetkey = dataset if 'seed' not in dataset else dataset.split('_seed')[0]
    dataset_dict = {
                    'food101standard': Food101DataModule_Standard,
                    'cub2011' : Cub2011DataModule,
                    'cub2011standard' : Cub2011DataModule_Standard,
                    'imagenetseen16shots' : ImageNetSeen16shotDataModule,
                    'imagenethalf16shots' : ImageNetHalf16shotDataModule,
                    'imagenetunseen4shots' : ImageNetUnseen4shotDataModule,
                    'imagenetunseen16shots' : ImageNetUnseen16shotDataModule,
                    'imagenetunseen64shots' : ImageNetUnseen64shotDataModule,
                    'imagenetunseen256shots' : ImageNetUnseen256shotDataModule,
                    'imagenetunseen512shots' : ImageNetUnseen512shotDataModule,
                    'imagenetunseenfullshots' : ImageNetUnseenfullshotDataModule,
                    'imagenet1K' : ImageNet1KDataModule,
                    }
    transform_type = 'CLIP' if 'clip' in backbone else None
    datamodule = dataset_dict[datasetkey](
        datadir=datapath,
        dataset=dataset,
        imgsize=224,
        batchsize=batchsize,
        num_workers=8,
        transform_type=transform_type,
    )

    return datamodule

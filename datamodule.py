import torch
import torch.distributed as dist
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule
from PIL import Image

import os
import json

from pytorch_lightning import seed_everything
from sampler.ClassAwareSampler import ClassAwareSampler
from sampler.WeightedSampler import WeightedDistributedSampler
from text_data.preprocess import SentPreProcessor

class Transforms:
    @staticmethod
    def train_transform(imgsize):
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((imgsize, imgsize)),  # TODO: randomcrop?
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @staticmethod
    def val_transform(imgsize):
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((imgsize, imgsize)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

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

    @staticmethod
    def LT_train_transform(imgsize):
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(imgsize),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @staticmethod
    def LT_val_transform(imgsize):
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(imgsize),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )


    '''
    # use these for training from scratch
    from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

    @staticmethod
    def cifar_train_transform(imgsize):
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.Resize((imgsize, imgsize)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

    @staticmethod
    def cifar_val_transform(imgsize):
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((imgsize, imgsize)),
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )
    '''


class AbstractDataModule(LightningDataModule):
    def __init__(self, datadir='data', imgsize=224, batchsize=256, num_workers=0, transform_type=None, train_split=None, val_split=None, sampler=None):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = None
        self.dataset_train = self.dataset_val = None
        self.sampler = None
        if transform_type == 'CLIP':
            self.train_transform = Transforms.clip_transform(imgsize)
            self.val_transform = Transforms.clip_transform(imgsize)
        elif transform_type == 'LT':
            self.train_transform = Transforms.LT_train_transform(imgsize)
            self.val_transform = Transforms.LT_val_transform(imgsize)
        else:
            self.train_transform = Transforms.train_transform(imgsize)
            self.val_transform = Transforms.val_transform(imgsize)

    @property
    def num_classes(self) -> int:
        return None

    def setup(self, stage: str):
        self.dataset_train = self.dataset(root=self.hparams.datadir, split=self.hparams.train_split, transform=self.train_transform, download=True)
        self.dataset_val = self.dataset(root=self.hparams.datadir, split=self.hparams.val_split, transform=self.val_transform, download=True)


    def sampler_loader(self):

        def is_dist_avail_and_initialized():
            if not dist.is_available():
                return False
            if not dist.is_initialized():
                return False
            return True

        def get_world_size():
            if not is_dist_avail_and_initialized():
                return 1
            return dist.get_world_size()


        def get_rank():
            if not is_dist_avail_and_initialized():
                return 0
            return dist.get_rank()

        num_tasks = get_world_size()
        global_rank = get_rank()
        if self.hparams.sampler == 'ClassAware':
            self.sampler = ClassAwareSampler(self.dataset_train, num_samples_cls=4)
        elif self.hparams.sampler == 'SquareRoot':

            training_labels = np.array(self.dataset_train.targets).astype(int)
            train_class_counts = [len(training_labels[training_labels == l]) for l in range(self.num_classes)]
            weights = 1. / torch.tensor(train_class_counts, dtype=torch.float)
            weights.sqrt_()
            samples_weights = weights[list(self.dataset_train.targets)]
            self.sampler = WeightedDistributedSampler(
                dataset=self.dataset_train, weights=samples_weights, replacement=True,
                num_replicas=num_tasks, rank=global_rank, deterministic=True
            )

    def train_dataloader(self):
        self.sampler_loader()
        return DataLoader(self.dataset_train, batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers, shuffle=False if self.sampler else True, sampler=self.sampler)

    def unshuffled_train_dataloader(self):
        if self.dataset_train is None:
            self.setup(stage='init')
        return DataLoader(self.dataset_train, batch_size=512, num_workers=self.hparams.num_workers, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()

class CIFAR10DataModule(AbstractDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.dataset = torchvision.datasets.CIFAR10
        self.max_num_samples = 5000

    def setup(self, stage: str):
        self.dataset_train = self.dataset(root=self.hparams.datadir, train=True, transform=self.train_transform, download=True)
        self.dataset_val = self.dataset(root=self.hparams.datadir, train=False, transform=self.val_transform, download=True)

    @property
    def num_classes(self) -> int:
        return 10

class CIFAR100DataModule(AbstractDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.dataset = torchvision.datasets.CIFAR100
        self.max_num_samples = 500

    def setup(self, stage: str):
        self.dataset_train = self.dataset(root=self.hparams.datadir, train=True, transform=self.train_transform, download=True)
        self.dataset_val = self.dataset(root=self.hparams.datadir, train=False, transform=self.val_transform, download=True)

    @property
    def num_classes(self) -> int:
        return 100


class Food101DataModule(AbstractDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(train_split='train', val_split='test', *args, **kwargs)
        self.dataset = torchvision.datasets.Food101
        self.max_num_samples = 750

    @property
    def num_classes(self) -> int:
        return 101


class Places365DataModule(AbstractDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(train_split='train-standard', val_split='val', *args, **kwargs)
        self.dataset = torchvision.datasets.Places365
        self.max_num_samples = 500  # max num sample is actually ~4K

    def setup(self, stage: str):
        # small=True for small-image-size dataset
        self.dataset_train = self.dataset(root=self.hparams.datadir, split=self.hparams.train_split, transform=self.train_transform, download=False, small=True)
        self.dataset_val = self.dataset(root=self.hparams.datadir, split=self.hparams.val_split, transform=self.val_transform, download=False, small=True)

    @property
    def num_classes(self) -> int:
        return 365


class FGVCAircraftDataModule(AbstractDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(train_split='train', val_split='val', *args, **kwargs)
        self.dataset = torchvision.datasets.FGVCAircraft
        self.max_num_samples = 32

    @property
    def num_classes(self) -> int:
        return 100


class STL10DataModule(AbstractDataModule):  # STL images are 96x96 pixels
    def __init__(self, *args, **kwargs):
        super().__init__(train_split='train', val_split='test', *args, **kwargs)
        self.dataset = torchvision.datasets.STL10
        self.max_num_samples = 500

    @property
    def num_classes(self) -> int:
        return 10

class LT_Dataset(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

class ImageNet_LT_DataModule(AbstractDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = LT_Dataset
        self.max_num_samples = 4

    def setup(self, stage: str):
        root = os.path.join(self.hparams.datadir, 'ImageNet')
        txt = './LT_txt/ImageNet_LT'

        self.dataset_train = self.dataset(root=root, txt=os.path.join(txt, 'ImageNet_LT_train.txt'), transform=self.train_transform)
        self.dataset_val = self.dataset(root=root, txt=os.path.join(txt, 'ImageNet_LT_val.txt'), transform=self.val_transform)
        self.dataset_test = self.dataset(root=root, txt=os.path.join(txt, 'ImageNet_LT_test.txt'), transform=self.val_transform)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers)

    @property
    def num_classes(self) -> int:
        return 1000

class Places_LT_DataModule(AbstractDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = LT_Dataset
        self.max_num_samples = 5

    def setup(self, stage: str):
        root = os.path.join(self.hparams.datadir, 'Places-LT')
        txt = './LT_txt/Places_LT'

        self.dataset_train = self.dataset(root=root, txt=os.path.join(txt, 'Places_LT_train.txt'), transform=self.train_transform)
        self.dataset_val = self.dataset(root=root, txt=os.path.join(txt, 'Places_LT_val.txt'), transform=self.val_transform)
        self.dataset_test = self.dataset(root=root, txt=os.path.join(txt, 'Places_LT_test.txt'), transform=self.val_transform)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers)

    @property
    def num_classes(self) -> int:
        return 365

class CarsDataModule(AbstractDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(train_split='train', val_split='test', *args, **kwargs)
        self.dataset = torchvision.datasets.StanfordCars
        self.max_num_samples = 24  # max is 40~50

    @property
    def num_classes(self) -> int:
        return 196

"""
ImageNet100 Dataset

Parameters
----------
    * root : str
        root directory of ImageNet100 dataset
    * train : bool
        True if load train dataset. False otherwise (validation, test dataset)
    * sub_dirs : list ['train.X1', 'train.X2', 'train.X3', 'train.X4', 'val.X']
        sub directories that you want to include in
        if you want to make validation dataset, just use ['val.X']
    * label_file : str
        label file of dataset
        if you remain it empty, it will make for you automatically
        But, if you make a validation(or test) dataset, you must specify the label file what you use for it
        Just use same label file of train dataset that has same setting with your validation dataset
    * label_mapping_file : str
        it is "labels.txt" file. you can find it in this link
        https://github.com/ChangyaoTian/VL-LTR/releases/download/text-corpus/imagenet.zip
    * wiki_dir : str
        it is "wiki" directory. you can find it in upper link
    * max_classes : None, int
        maximum number of classes.
        you can control the number of classes with this argument.
        It will cut out classes like this. [0 ~ C] -> [0 ~ max_classes-1]
    * max_samples : None, int
        maximum number of samples of classes.
        you can control the number of smaples of classes with this argument.
    * transform
        just transform

Attribute (may someone needed....)
-------
    * loaded_idxs : 1 x max_classes
        The indecies that dataset actually consist of. It look like ['n01440764', 'n01484850', ....]
    * num_classes
        The number of classes
    * num_samples
        number of samples of classes.
    * text_tokens : list[torch.tensor x num_classes]
        tokens of wikipedia descriptions generated by clip.simple_tokenizer
        Each tokens has shape like [#_sentences x (context length + 2)] (2 for start, end token)
    * num_sents
        number of sentences of each classes. same as #_sentences
"""
class ImageNet100_Dataset(Dataset):
    def __init__(self, root, train, sub_dirs = [], label_file = '', label_mapping_file = 'labels.txt', wiki_dir = 'wiki', max_classes = None, max_samples = None, transform=None, is_memory=False, len_memory=1000):
        self.root = root

        # memory_split cant handle variable query length. Let's fix the len of the memory and set the remainder as queries
        self.len_memory = len_memory  # query : [0 ~ (-len_memory - 1 or max_samples)], memory : [len_memory ~ to the last elem]

        self.sub_dirs = sub_dirs # choose in ['train.X1', 'train.X2', 'train.X3', 'train.X4', 'val.X']
        self.sub_idxs = [sorted(os.listdir(os.path.join(self.root, subdir))) for subdir in self.sub_dirs]

        if not train: assert (label_file != '' and os.path.exists(os.path.join(root,label_file))), "label file needed for non-train dataset"
        idxs_cls = self._label_generator(root, label_file)
        loaded_idxs = idxs_cls.keys()
        self.loaded_idxs = sorted(loaded_idxs, key = lambda item: idxs_cls[item]) # sort idxs with 0 ~ (# of classes - 1) order
        if max_classes:
            self.loaded_idxs = self.loaded_idxs[:max_classes]

        self.num_classes = len(self.loaded_idxs)
        # deprecated to handle variable num of queries
        # self.num_samples = max_samples if max_samples else self.total_samples - self.memory_split if is_memory else self.memory_split

        self.img_path = []
        self.targets = []
        self.transform = transform

        num_samples_count = [0] * self.num_classes

        for i, idxs in enumerate(self.sub_idxs):
            for idx in idxs:
                if idx not in self.loaded_idxs: continue

                imgdirs = sorted(os.listdir(os.path.join(root, self.sub_dirs[i], idx)))
                if is_memory:
                    num_samples_i = self.len_memory if self.len_memory else len(imgdirs)
                    imgdirs = imgdirs[-num_samples_i:]
                else:
                    num_samples_i = min(max_samples, len(imgdirs) - self.len_memory) if max_samples else len(imgdirs) - self.len_memory
                    imgdirs = imgdirs[:num_samples_i]

                for imgdir in imgdirs:
                    target = idxs_cls[idx]
                    if num_samples_count[target] >= num_samples_i: continue

                    self.img_path.append(os.path.join(root, self.sub_dirs[i], idx, imgdir))
                    self.targets.append(target)

                    num_samples_count[target] += 1

        # assert sum(num_samples_count) == self.num_samples * self.num_classes
        assert self.num_classes == max(self.targets) + 1

        # sentence token generator
        self.text_tokens = self._make_sentence_tokens(label_mapping_file, wiki_dir)
        self.num_sents = [token.shape[0] for token in self.text_tokens]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def _label_generator(self, root, txt):
        if txt != '' and os.path.exists(os.path.join(root, txt)):
            print(f'Label file exist : {os.path.join(root, txt)}')
            with open(os.path.join(root, txt), 'r') as jsonfile:
                idx_classes = json.load(jsonfile)
        else:
            txt = 'label.json' if txt == '' else txt
            print(f"No label file found, make new one on {os.path.join(root, txt)}")
            flatten_idxs = []
            for s in self.sub_idxs: flatten_idxs += s

            idx_classes = {}
            for i, idx in enumerate(flatten_idxs):
                idx_classes[idx] = i
            with open(os.path.join(root, txt), 'w') as jsonfile:
                json.dump(idx_classes, jsonfile, indent='')

        return idx_classes

    def _make_sentence_tokens(self, label_mapping_file, wiki_dir):
        preprocessor = SentPreProcessor(self.root, self.loaded_idxs, label_mapping_file, wiki_dir, context_length=75)
        return preprocessor.make_sentence_tokens()


class SubsetDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):

        path = self.data[index]
        label = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def __len__(self):
        return len(self.targets)


class TextToken_Dataset(Dataset):
    def __init__(self, text_tokens: list, num_sents: list):
        self.data = torch.cat(text_tokens)
        self.targets = []

        targets = [[idx]*nsents for idx, nsents in enumerate(num_sents)]
        for t in targets: self.targets += t

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.targets[index]

        return sample, label

class ImageNet100DataModule(AbstractDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = ImageNet100_Dataset

        self.max_classes = None
        self.max_qeury_num_samples = 1100
        self.dataset_root = 'imagenet100'
        # self.memory_split = 1000
        self.len_memory = 200
        # self.total_samples = 1300
        self.train_subdirs = ['train.X1', 'train.X3', 'train.X4']
        self.val_subdirs = ['train.X2']
        self.test_subdirs = ['train.X2']

    def setup(self, stage: str):
        # if episodiceval: return self.setup_episodic_eval()
        root = os.path.join(self.hparams.datadir, self.dataset_root)
        label_mapping_file = 'labels.txt'
        wiki_dir = 'wiki'

        self.dataset_train = self.dataset(root, train=True, sub_dirs=self.train_subdirs, label_file='trn_label.json', label_mapping_file=label_mapping_file, wiki_dir=wiki_dir,
                                          max_classes=self.max_classes, max_samples=None, transform=self.train_transform, is_memory=False, len_memory=0)
        '''
        datalen = len(self.dataset_train.targets)
        numclass = max(self.dataset_train.targets) + 1
        numnoise = int(datalen * 0.05)
        noiseidx = torch.randperm(datalen)[:numnoise]
        targets = torch.tensor(self.dataset_train.targets)
        targets[noiseidx] = torch.randint(low=0, high=numclass, size=[numnoise])
        self.dataset_train.targets = targets.tolist()
        '''

        self.dataset_val = self.dataset(root, train=True, sub_dirs=self.val_subdirs, label_file='val_label.json', label_mapping_file=label_mapping_file, wiki_dir=wiki_dir,
                                          max_classes=None, max_samples=None, transform=self.val_transform, is_memory=False, len_memory=self.len_memory)
        self.dataset_test = self.dataset(root, train=True, sub_dirs=self.test_subdirs, label_file='tst_label.json', label_mapping_file=label_mapping_file, wiki_dir=wiki_dir,
                                          max_classes=None, max_samples=None, transform=self.val_transform, is_memory=False, len_memory=self.len_memory)

        self.dataset_train_shot = self.dataset(root, train=True, sub_dirs=self.train_subdirs, label_file='trn_label.json', label_mapping_file=label_mapping_file, wiki_dir=wiki_dir,
                                          max_classes=self.max_classes, max_samples=0, transform=self.train_transform, is_memory=True, len_memory=None)  # becomes equivalent to dataset_train
        self.dataset_val_shot = self.dataset(root, train=True, sub_dirs=self.val_subdirs, label_file='val_label.json', label_mapping_file=label_mapping_file, wiki_dir=wiki_dir,
                                          max_classes=None, max_samples=None, transform=self.val_transform, is_memory=True, len_memory=self.len_memory)
        self.dataset_test_shot = self.dataset(root, train=True, sub_dirs=self.test_subdirs, label_file='tst_label.json', label_mapping_file=label_mapping_file, wiki_dir=wiki_dir,
                                          max_classes=None, max_samples=None, transform=self.val_transform, is_memory=True, len_memory=self.len_memory)

        self.dataset_train_memory = Webvision_dataset(root=os.path.join(self.hparams.datadir, 'webvisionv1'),
                                                      label_file = os.path.join(root, 'trn_label.json'),
                                                      transform=self.train_transform,
                                                      len_memory=1000)
        self.dataset_val_memory = Webvision_dataset(root=os.path.join(self.hparams.datadir, 'webvisionv1'),
                                                      label_file = os.path.join(root, 'val_label.json'),
                                                      transform=self.val_transform,
                                                      len_memory=1000)
        self.dataset_test_memory = Webvision_dataset(root=os.path.join(self.hparams.datadir, 'webvisionv1'),
                                                      label_file = os.path.join(root, 'tst_label.json'),
                                                      transform=self.val_transform,
                                                      len_memory=1000)

        self.dataset_train_text = TextToken_Dataset(self.dataset_train.text_tokens, self.dataset_train.num_sents)
        self.dataset_val_text = TextToken_Dataset(self.dataset_val.text_tokens, self.dataset_val.num_sents)
        self.dataset_test_text = TextToken_Dataset(self.dataset_test.text_tokens, self.dataset_test.num_sents)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers)

    def train_text_dataloader(self):
        return DataLoader(self.dataset_train_text, batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers)

    def val_text_dataloader(self):
        return DataLoader(self.dataset_val_text, batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers)

    def test_text_dataloader(self):
        return DataLoader(self.dataset_test_text, batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers)

    def train_shot_dataloader(self):
        return DataLoader(self.dataset_train_shot, batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers)
    def val_shot_dataloader(self):
        return DataLoader(self.dataset_val_shot, batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers)
    def test_shot_dataloader(self):
        return DataLoader(self.dataset_test_shot, batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers)

    def train_memory_dataloader(self):
        return DataLoader(self.dataset_train_memory, batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers)
    def val_memory_dataloader(self):
        return DataLoader(self.dataset_val_memory, batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers)
    def test_memory_dataloader(self):
        return DataLoader(self.dataset_test_memory, batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers)

    def setup_episodic_eval(self, nclass=5, nshot=5, nquery=15):
        root = os.path.join(self.hparams.datadir, self.dataset_root)
        label_mapping_file = 'labels.txt'
        wiki_dir = 'wiki'

        if not hasattr(self, 'dataset_test_all'):
            self.dataset_test_all = self.dataset(root, train=True, sub_dirs=self.test_subdirs, label_file='tst_label.json', label_mapping_file=label_mapping_file, wiki_dir=wiki_dir,
                                                max_classes=None, max_samples=None, transform=self.val_transform, is_memory=True, len_memory=None)
            self.dataset_test_memory_all = Webvision_dataset(root=os.path.join(self.hparams.datadir, 'webvisionv1'),
                                                         label_file = os.path.join(root, 'tst_label.json'),
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

        self.dataset_test = SubsetDataset(data=query_img_path, targets=query_targets, transform=self.val_transform)
        self.dataset_test_shot = SubsetDataset(data=shot_img_path, targets=shot_targets, transform=self.val_transform)
        self.dataset_test_memory = SubsetDataset(data=mem_img_path, targets=mem_targets, transform=self.val_transform)

        self.dataset_test_text = TextToken_Dataset(self.dataset_test_all.text_tokens, self.dataset_test_all.num_sents)

    @property
    def num_classes(self) -> int:
        return self.dataset_train.num_classes


class ImageNet1000DataModule(ImageNet100DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = ImageNet100_Dataset

        self.max_qeury_num_samples = 700
        self.dataset_root = 'imagenet-mini'
        self.memory_split = 700
        self.total_samples = 1000


class ImageNet40samplesDataModule(ImageNet100DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = ImageNet100_Dataset

        self.max_qeury_num_samples = 10
        self.dataset_root = 'ILSVRC_40samples'
        self.len_memory = 30
        self.train_subdirs = ['train']
        self.val_subdirs = ['val']
        self.test_subdirs = ['val']


class MiniImagenetDataModule(ImageNet100DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = ImageNet100_Dataset

        self.max_qeury_num_samples = 595
        self.dataset_root = 'miniimagenet'
        self.len_memory = 5
        self.train_subdirs = ['train']
        self.val_subdirs = ['val']
        self.test_subdirs = ['test']


class ImageNet130samplesDataModule(ImageNet100DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = ImageNet100_Dataset

        self.max_qeury_num_samples = 100
        self.dataset_root = 'ILSVRC_130samples'
        # self.memory_split = 100
        self.len_memory = 30
        # self.total_samples = 130  # to make it length-agnostic
        self.train_subdirs = ['train']
        self.val_subdirs = ['val']
        self.test_subdirs = ['val']


class ImageNet500samplesDataModule(ImageNet100DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = ImageNet100_Dataset

        self.max_qeury_num_samples = 300
        self.dataset_root = 'ILSVRC_500samples'
        # self.memory_split = 400
        self.len_memory = 200
        # self.total_samples = 500  # to make it length-agnostic
        self.train_subdirs = ['train']
        self.val_subdirs = ['val']
        self.test_subdirs = ['val']


class ImageNetFullsamplesDataModule(ImageNet100DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = ImageNet100_Dataset

        self.max_qeury_num_samples = 1100
        self.dataset_root = 'ILSVRC_1300samples'
        self.len_memory = 200
        # self.total_samples = 1300  # to make it length-agnostic
        self.train_subdirs = ['train']
        self.val_subdirs = ['val']
        self.test_subdirs = ['val']


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
        wiki_dir = 'wiki'

        self.dataset_train = self.dataset(root, train=True, sub_dirs=self.train_subdirs, label_file='standard_label.json', label_mapping_file=label_mapping_file, wiki_dir=wiki_dir,
                                          max_classes=self.max_classes, max_samples=self.max_qeury_num_samples, transform=self.train_transform, is_memory=False, len_memory=self.len_memory)
        self.dataset_val = self.dataset(root, train=False, sub_dirs=self.val_subdirs, label_file='standard_label.json', label_mapping_file=label_mapping_file, wiki_dir=wiki_dir,
                                          max_classes=None, max_samples=50, transform=self.val_transform, is_memory=False, len_memory=0)
        self.dataset_test = self.dataset_val

        self.dataset_train_memory = self.dataset(root, train=True, sub_dirs=self.train_subdirs, label_file='standard_label.json', label_mapping_file=label_mapping_file, wiki_dir=wiki_dir,
                                          max_classes=self.max_classes, max_samples=None, transform=self.train_transform, is_memory=True, len_memory=self.len_memory)
        self.dataset_val_memory = self.dataset_train_memory
        self.dataset_test_memory = self.dataset_train_memory

        self.dataset_train_text = TextToken_Dataset(self.dataset_train.text_tokens, self.dataset_train.num_sents)
        self.dataset_val_text = self.dataset_train_text
        self.dataset_test_text = self.dataset_train_text


class Cub2011DataModule_Standard(ImageNet100DataModule_Standard):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_root = 'CUB_200_2011'
        self.len_memory = 5 # each classes in dataset have 29~30 images each

        self.train_subdirs = ['Train']
        self.val_subdirs = ['Val']
        self.test_subdirs = ['Val']


class ImageNet1Kclasses160samples(ImageNet100DataModule):
    ''' seen classes '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset = ImageNet100_Dataset

        self.max_classes = None
        self.max_qeury_num_samples = 30  # 130
        self.dataset_root = 'ILSVRC_1Kclasses160samples'
        self.len_memory = 30  # shot
        self.train_subdirs = ['train']
        self.val_subdirs = ['val']
        self.test_subdirs = ['val']

    def setup(self, stage: str):
        root = os.path.join(self.hparams.datadir, self.dataset_root)
        label_mapping_file = 'labels.txt'
        wiki_dir = 'wiki'

        self.dataset_train = self.dataset(root, train=True, sub_dirs=self.train_subdirs, label_file='standard_label.json', label_mapping_file=label_mapping_file, wiki_dir=wiki_dir,
                                          max_classes=self.max_classes, max_samples=self.max_qeury_num_samples, transform=self.train_transform, is_memory=False, len_memory=self.len_memory)
        self.dataset_val = self.dataset(root, train=False, sub_dirs=self.val_subdirs, label_file='standard_label.json', label_mapping_file=label_mapping_file, wiki_dir=wiki_dir,
                                          max_classes=None, max_samples=30, transform=self.val_transform, is_memory=False, len_memory=0)
        self.dataset_test = self.dataset_val

        self.dataset_train_shot = self.dataset_train  # becomes equivalent to dataset_train
        self.dataset_val_shot = self.dataset_train
        self.dataset_test_shot = self.dataset_train

        self.dataset_train_memory = Webvision_dataset(root=os.path.join(self.hparams.datadir, 'webvisionv1'),
                                                      label_file = os.path.join(root, 'standard_label.json'),
                                                      transform=self.train_transform,
                                                      len_memory=1000)
        self.dataset_val_memory = Webvision_dataset(root=os.path.join(self.hparams.datadir, 'webvisionv1'),
                                                      label_file = os.path.join(root, 'standard_label.json'),
                                                      transform=self.val_transform,
                                                      len_memory=1000)
        self.dataset_test_memory = Webvision_dataset(root=os.path.join(self.hparams.datadir, 'webvisionv1'),
                                                      label_file = os.path.join(root, 'standard_label.json'),
                                                      transform=self.val_transform,
                                                      len_memory=1000)

        self.dataset_train_text = TextToken_Dataset(self.dataset_train.text_tokens, self.dataset_train.num_sents)
        self.dataset_val_text = self.dataset_train_text
        self.dataset_test_text = self.dataset_train_text


class ImageNet100classes160samples(ImageNet100DataModule):
    ''' seen classes '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset = ImageNet100_Dataset

        self.max_classes = None
        self.max_qeury_num_samples = 100
        self.dataset_root = 'ILSVRC_100classes160samples'
        self.len_memory = 30
        self.train_subdirs = ['train']
        self.val_subdirs = ['val']
        self.test_subdirs = ['val']

    def setup(self, stage: str):
        root = os.path.join(self.hparams.datadir, self.dataset_root)
        label_mapping_file = 'labels.txt'
        wiki_dir = 'wiki'

        self.dataset_train = self.dataset(root, train=True, sub_dirs=self.train_subdirs, label_file='standard_label.json', label_mapping_file=label_mapping_file, wiki_dir=wiki_dir,
                                          max_classes=self.max_classes, max_samples=self.max_qeury_num_samples, transform=self.train_transform, is_memory=False, len_memory=self.len_memory)
        self.dataset_val = self.dataset(root, train=False, sub_dirs=self.val_subdirs, label_file='standard_label.json', label_mapping_file=label_mapping_file, wiki_dir=wiki_dir,
                                          max_classes=None, max_samples=30, transform=self.val_transform, is_memory=False, len_memory=0)
        self.dataset_test = self.dataset_val

        self.dataset_train_memory = self.dataset(root, train=True, sub_dirs=self.train_subdirs, label_file='standard_label.json', label_mapping_file=label_mapping_file, wiki_dir=wiki_dir,
                                          max_classes=self.max_classes, max_samples=None, transform=self.train_transform, is_memory=True, len_memory=self.len_memory)
        self.dataset_val_memory = self.dataset_train_memory
        self.dataset_test_memory = self.dataset_train_memory

        self.dataset_train_text = TextToken_Dataset(self.dataset_train.text_tokens, self.dataset_train.num_sents)
        self.dataset_val_text = self.dataset_train_text
        self.dataset_test_text = self.dataset_train_text


class Webvision_dataset(Dataset):
    def __init__(self, root='webvision', label_file = '', max_classes = None, transform=None, len_memory=1000):
        self.transform = transform

        # n0xxxxxxx -> 'web' 0 ~ 999
        synset2webtarget = {}
        webtarget2synset = {}
        with open(os.path.join(root, 'info/synsets.txt')) as f:
            lines = f.readlines()
        for linenum, line in enumerate(lines):
            nxxxxxxxx = line.split()[0]
            synset2webtarget[nxxxxxxxx] = linenum
            webtarget2synset[linenum] = nxxxxxxxx

        with open(label_file) as f:
            idxs_cls = json.load(f)
        synset_set = idxs_cls.keys()  # [nxxxxxxxx, ..., nxxxxxxxx]
        num_classes = len(synset_set)

        with open(os.path.join(root, 'info/train_filelist_google.txt')) as f:
            lines = f.readlines()

        self.img_path = []
        self.targets = []
        num_samples_count = [0] * num_classes

        for line in lines:
            img, webtarget = line.split()
            webtarget = int(webtarget)

            if webtarget2synset[webtarget] in synset_set:
                # webvision is always memory
                synset = webtarget2synset[webtarget]
                target = idxs_cls[synset]
                if num_samples_count[target] == 0:
                    print(webtarget, '->', synset, '->', target)
                if num_samples_count[target] >= len_memory: continue
                self.img_path.append(os.path.join(root, img))
                self.targets.append(target)
                num_samples_count[target] += 1

        '''
        with open(os.path.join(root, 'info/synsets.txt')) as f:
            lines = f.readlines()

        self.txtlabels = {}

        for linenum, line in enumerate(lines):
            nxxxxxxxx = line.split()[0]
            classtxtlabel = line[10:-1]
            if nxxxxxxxx in synset_set:
                target = idxs_cls[nxxxxxxxx]
                self.txtlabels[target] = classtxtlabel
        '''

    def __getitem__(self, index):
        img_path = self.img_path[index]
        target = self.targets[index]
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image)
        return img, target

    def __len__(self):
        return len(self.targets)


def return_datamodule(datapath, dataset, batchsize, backbone, sampler = None):
    dataset_dict = {'cifar10': CIFAR10DataModule,
                    'cifar100': CIFAR100DataModule,
                    'food101': Food101DataModule,
                    'places365': Places365DataModule,
                    'fgvcaircraft': FGVCAircraftDataModule,
                    'cars': CarsDataModule,
                    'stl10': STL10DataModule,
                    'imagenetLT': ImageNet_LT_DataModule,
                    'placesLT': Places_LT_DataModule,
                    'imagenet100' : ImageNet100DataModule,
                    'imagenetmini' : ImageNet1000DataModule,
                    'imagenet100standard' : ImageNet100DataModule_Standard,  # seen
                    'imagenet1Kclasses160samples' : ImageNet1Kclasses160samples,  # seen
                    'imagenet100classes160samples' : ImageNet100classes160samples,  # seen
                    'miniimagenet' : MiniImagenetDataModule,
                    'imagenet40samples' : ImageNet40samplesDataModule,
                    'imagenet130samples' : ImageNet130samplesDataModule,
                    'imagenet500samples' : ImageNet500samplesDataModule,
                    'imagenetfullsamples' : ImageNetFullsamplesDataModule,
                    'cub2011standard' : Cub2011DataModule_Standard,
                    }

    transform_type = 'CLIP'if 'clip' in backbone and not 'LT' in dataset else 'LT' if 'LT' in dataset else None
    datamodule = dataset_dict[dataset](
        datadir=datapath,
        imgsize=224,
        batchsize=batchsize,
        num_workers=8,
        transform_type=transform_type,
        sampler=sampler
    )

    return datamodule


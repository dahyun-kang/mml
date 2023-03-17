import torch
import torch.distributed as dist
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule
from PIL import Image

from sampler.ClassAwareSampler import ClassAwareSampler
from sampler.WeightedSampler import WeightedDistributedSampler

import os

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
    def __init__(self, datadir='data', imgsize=224, batchsize=256, num_workers=0, use_clip_transform=True, train_split=None, val_split=None, sampler=None):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = None
        self.dataset_train = self.dataset_val = None
        self.sampler = None
        if use_clip_transform:
            self.train_transform = Transforms.clip_transform(imgsize)
            self.val_transform = Transforms.clip_transform(imgsize)
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
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(dist.is_available())
        print(dist.is_initialized())

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

def return_datamodule(datapath, dataset, batchsize, backbone, sampler = None):
    dataset_dict = {'cifar10': CIFAR10DataModule,
                    'cifar100': CIFAR100DataModule,
                    'food101': Food101DataModule,
                    'places365': Places365DataModule,
                    'fgvcaircraft': FGVCAircraftDataModule,
                    'cars': CarsDataModule,
                    'stl10': STL10DataModule,
                    'imagenetLT': ImageNet_LT_DataModule,
                    'placesLT': Places_LT_DataModule
                    }

    use_clip_transform = True if 'clip' in backbone and not 'LT' in dataset else False
    datamodule = dataset_dict[dataset](
        datadir=datapath,
        imgsize=224,
        batchsize=batchsize,
        num_workers=8,
        use_clip_transform=use_clip_transform,
        sampler=sampler
    )

    return datamodule

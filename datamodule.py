import torchvision
from torch.utils.data import DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, imagenet_normalization

from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def clip_transform(n_px):
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(n_px, interpolation=BICUBIC),
        torchvision.transforms.CenterCrop(n_px),
        _convert_image_to_rgb,
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class CIFAR10DataModule(LightningDataModule):
    def __init__(self, datadir='data', imgsize=32, batchsize=256, num_workers=0, reproduce=False):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = torchvision.datasets.CIFAR10
        self.dataset_train = self.dataset_val = None
        self.reproduce = reproduce

    def setup(self, stage: str):

        train_transforms = clip_transform(self.hparams.imgsize) if self.reproduce else \
            torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.Resize((self.hparams.imgsize, self.hparams.imgsize)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )
        val_transforms = clip_transform(self.hparams.imgsize) if self.reproduce else \
            torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((self.hparams.imgsize, self.hparams.imgsize)),
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

        self.dataset_train = self.dataset(root=self.hparams.datadir, train=True, transform=train_transforms, download=True)
        self.dataset_val = self.dataset(root=self.hparams.datadir, train=False, transform=val_transforms, download=True)

    @property
    def num_classes(self) -> int:
        return 10

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers, shuffle=True)

    def unshuffled_train_dataloader(self):
        if self.dataset_train is None:
            self.setup(stage='init')
        return DataLoader(self.dataset_train, batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()


class CIFAR100DataModule(CIFAR10DataModule):
    def __init__(self, datadir='data', imgsize=32, batchsize=256, num_workers=0, reproduce=False):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = torchvision.datasets.CIFAR100
        self.reproduce = reproduce

    @property
    def num_classes(self) -> int:
        return 100


class ImgSize224DataModule(LightningDataModule):
    def __init__(self, datadir='data', imgsize=224, batchsize=256, num_workers=0, train_split=None, val_split=None, reproduce=False):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = None  # torchvision.datasets.Food101
        self.dataset_train = self.dataset_val = None

        self.train_transform = clip_transform(self.hparams.imgsize) if reproduce else \
            torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((self.hparams.imgsize, self.hparams.imgsize)),  # TODO: randomcrop?
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                imagenet_normalization(),
            ]
        )
        self.val_transform = clip_transform(self.hparams.imgsize) if reproduce else \
            torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((self.hparams.imgsize, self.hparams.imgsize)),
                torchvision.transforms.ToTensor(),
                imagenet_normalization(),
            ]
        )

    def setup(self, stage: str):
        self.dataset_train = self.dataset(root=self.hparams.datadir, split=self.hparams.train_split, transform=self.train_transform, download=True)
        self.dataset_val = self.dataset(root=self.hparams.datadir, split=self.hparams.val_split, transform=self.val_transform, download=True)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers, shuffle=True)

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


class Food101DataModule(ImgSize224DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(train_split='train', val_split='test', *args, **kwargs)
        self.dataset = torchvision.datasets.Food101

    @property
    def num_classes(self) -> int:
        return 101


class Places365DataModule(ImgSize224DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(train_split='train-standard', val_split='val', *args, **kwargs)
        self.dataset = torchvision.datasets.Places365

    def setup(self, stage: str):
        # small=True for small-image-size dataset
        self.dataset_train = self.dataset(root=self.hparams.datadir, split=self.hparams.train_split, transform=self.train_transform, download=False, small=True)
        self.dataset_val = self.dataset(root=self.hparams.datadir, split=self.hparams.val_split, transform=self.val_transform, download=False, small=True)

    @property
    def num_classes(self) -> int:
        return 365


class FGVCAircraftDataModule(ImgSize224DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(train_split='train', val_split='val', *args, **kwargs)
        self.dataset = torchvision.datasets.FGVCAircraft

    @property
    def num_classes(self) -> int:
        return 100


class STL10DataModule(ImgSize224DataModule):  # STL images are 96x96 pixels
    def __init__(self, *args, **kwargs):
        super().__init__(train_split='train', val_split='test', *args, **kwargs)
        self.dataset = torchvision.datasets.STL10

    @property
    def num_classes(self) -> int:
        return 10


def return_datamodule(datapath, dataset, batchsize, backbone, reproduce = False):
    dataset_dict = {'cifar10': CIFAR10DataModule,
                    'cifar100': CIFAR100DataModule,
                    'food101': Food101DataModule,
                    'places365': Places365DataModule,
                    'fgvcaircraft': FGVCAircraftDataModule,
                    'stl10': STL10DataModule,
                    }
    imgsize = 32 if 'cifar' in dataset and 'resnet' in backbone else 224
    datamodule = dataset_dict[dataset](
        datadir=datapath,
        imgsize=imgsize,
        batchsize=batchsize,
        num_workers=8,
        reproduce = reproduce # TODO change argument reproduce -> clip for future clip based model's works
    )

    return datamodule

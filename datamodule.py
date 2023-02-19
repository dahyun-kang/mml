import torchvision
from torch.utils.data import DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, imagenet_normalization


class Transforms:
    @staticmethod
    def train_transform(imgsize):
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((imgsize, imgsize)),  # TODO: randomcrop?
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                imagenet_normalization(),
            ]
        )

    @staticmethod
    def val_transform(imgsize):
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((imgsize, imgsize)),
                torchvision.transforms.ToTensor(),
                imagenet_normalization(),
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

    '''
    # use these for training from scratch
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


class CIFAR10DataModule(LightningDataModule):
    def __init__(self, datadir='data', imgsize=32, batchsize=256, num_workers=0, use_clip_transform=True):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = torchvision.datasets.CIFAR10
        self.max_num_samples = 5000

        self.dataset_train = self.dataset_val = None
        if use_clip_transform:
            self.train_transform = Transforms.clip_transform(imgsize)
            self.val_transform = Transforms.clip_transform(imgsize)
        else:
            self.train_transform = Transforms.train_transform(imgsize)
            self.val_transform = Transforms.val_transform(imgsize)

    @property
    def num_classes(self) -> int:
        return 10

    def setup(self, stage: str):
        self.dataset_train = self.dataset(root=self.hparams.datadir, train=True, transform=self.train_transform, download=True)
        self.dataset_val = self.dataset(root=self.hparams.datadir, train=False, transform=self.val_transform, download=True)

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.dataset = torchvision.datasets.CIFAR100
        self.max_num_samples = 500

    @property
    def num_classes(self) -> int:
        return 100


class ImgSize224DataModule(LightningDataModule):
    def __init__(self, datadir='data', imgsize=224, batchsize=256, num_workers=0, use_clip_transform=True, train_split=None, val_split=None):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = None
        self.dataset_train = self.dataset_val = None
        if use_clip_transform:
            self.train_transform = Transforms.clip_transform(imgsize)
            self.val_transform = Transforms.clip_transform(imgsize)
        else:
            self.train_transform = Transforms.train_transform(imgsize)
            self.val_transform = Transforms.val_transform(imgsize)

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
        self.max_num_samples = 750

    @property
    def num_classes(self) -> int:
        return 101


class Places365DataModule(ImgSize224DataModule):
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


class FGVCAircraftDataModule(ImgSize224DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(train_split='train', val_split='val', *args, **kwargs)
        self.dataset = torchvision.datasets.FGVCAircraft
        self.max_num_samples = 32

    @property
    def num_classes(self) -> int:
        return 100


class STL10DataModule(ImgSize224DataModule):  # STL images are 96x96 pixels
    def __init__(self, *args, **kwargs):
        super().__init__(train_split='train', val_split='test', *args, **kwargs)
        self.dataset = torchvision.datasets.STL10
        self.max_num_samples = 500

    @property
    def num_classes(self) -> int:
        return 10


def return_datamodule(datapath, dataset, batchsize, backbone):
    dataset_dict = {'cifar10': CIFAR10DataModule,
                    'cifar100': CIFAR100DataModule,
                    'food101': Food101DataModule,
                    'places365': Places365DataModule,
                    'fgvcaircraft': FGVCAircraftDataModule,
                    'stl10': STL10DataModule,
                    }

    use_clip_transform = True if 'clip' in backbone else False
    datamodule = dataset_dict[dataset](
        datadir=datapath,
        imgsize=224,
        batchsize=batchsize,
        num_workers=8,
        use_clip_transform=use_clip_transform,
    )

    return datamodule

import torchvision
from torch.utils.data import DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, imagenet_normalization


class CIFAR10DataModule(LightningDataModule):
    def __init__(self, data_dir='data', batch_size=256, num_workers=0):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = torchvision.datasets.CIFAR10
        self.dataset_train = self.dataset_val = None

    def setup(self, stage: str):
        train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

        val_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

        self.dataset_train = self.dataset(root=self.hparams.data_dir, train=True, transform=train_transforms, download=True)
        self.dataset_val = self.dataset(root=self.hparams.data_dir, train=False, transform=val_transforms, download=True)

    @property
    def num_classes(self) -> int:
        return 10

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)

    def unshuffled_train_dataloader(self):
        if self.dataset_train is None:
            self.setup(stage='init')
        return DataLoader(self.dataset_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()


class CIFAR100DataModule(CIFAR10DataModule):
    def __init__(self, data_dir='data', batch_size=256, num_workers=0):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = torchvision.datasets.CIFAR100

    @property
    def num_classes(self) -> int:
        return 100


class Food101DataModule(LightningDataModule):
    def __init__(self, data_dir='data', batch_size=256, num_workers=0):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = torchvision.datasets.Food101
        self.dataset_train = self.dataset_val = None

    def setup(self, stage: str):
        train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),  # TODO: randomcrop?
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                imagenet_normalization(),
            ]
        )

        val_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                imagenet_normalization(),
            ]
        )

        self.dataset_train = self.dataset(root=self.hparams.data_dir, split='train', transform=train_transforms, download=True)
        self.dataset_val = self.dataset(root=self.hparams.data_dir, split='test', transform=val_transforms, download=True)

    @property
    def num_classes(self) -> int:
        return 101

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)

    def unshuffled_train_dataloader(self):
        if self.dataset_train is None:
            self.setup(stage='init')
        return DataLoader(self.dataset_train, batch_size=512, num_workers=self.hparams.num_workers, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()


def return_datamodule(datapath, dataset, bsz):
    dataset_dict = {'cifar10': CIFAR10DataModule,
                    'cifar100': CIFAR100DataModule,
                    'food101': Food101DataModule,
                    }
    datamodule = dataset_dict[dataset](
        data_dir=datapath,
        batch_size=bsz,
        num_workers=8,
        # train_transforms=train_transforms,
        # test_transforms=test_transforms,
        # val_transforms=test_transforms,
    )

    return datamodule

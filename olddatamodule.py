

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


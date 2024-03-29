import os
import json
import copy
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset

from text_data.preprocess import SentPreProcessor


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


class SubsetDataset(Dataset):
    def __init__(self, data, targets, imgsize=224):
        self.data = data
        self.targets = targets
        self.transform = Transforms.clip_transform(imgsize)

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


"""
Attribute
-------
    * classids : 1 x max_classes
        The indecies that dataset actually consist of. It look like ['n01440764', 'n01484850', ....]
    * class2textlabel_file : str
        it is "labels.txt" file. you can find it in this link
        https://github.com/ChangyaoTian/VL-LTR/releases/download/text-corpus/imagenet.zip
    * text_tokens : list[torch.tensor x num_classes]
        tokens of wikipedia descriptions generated by clip.simple_tokenizer
        Each tokens has shape like [#_sentences x (context length + 2)] (2 for start, end token)
    * num_sents
        number of sentences of each classes. same as #_sentences
"""
class TextTokenMemoryDataset(Dataset):
    def __init__(self, root, class2textlabel_file='labels.txt', wiki_dir='wiki', classids=[]):
        # sentence token generator
        preprocessor = SentPreProcessor(root, classids, class2textlabel_file, wiki_dir, context_length=75)
        text_tokens = preprocessor.make_sentence_tokens()
        num_sents = [token.shape[0] for token in text_tokens]

        self.data = torch.cat(text_tokens)
        self.targets = []
        for idx, nsents in enumerate(num_sents):
            self.targets += [idx] * nsents

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.targets[index]

        return sample, label


"""
DisjointClassDataset
    * splitdirs : str
        split idenfied by dirs e.g. 'train', 'val', 'test'
    * label_file : str
        label file of dataset
        if you remain it empty, it will make for you automatically
        But, if you make a validation(or test) dataset, you must specify the label file what you use for it
        Just use same label file of train dataset that has same setting with your validation dataset
    * classids : list
        The indecies that dataset actually consist of. It look like ['n01440764', 'n01484850', ....]
"""
class DisjointClassDataset(Dataset):
    def __init__(self, root, splitdir='train', label_file='trn_label.json', nsamples=None, is_shot=False, nshot=16, imgsize=224):
        self.transform = Transforms.clip_transform(imgsize)

        # directory-identified classes
        classdirnames = sorted(os.listdir(os.path.join(root, splitdir)))
        classid2target = self._label_generator(root, label_file, classdirnames)
        self.classids = sorted(classdirnames, key = lambda item: classid2target[item])  # sort idxs with 0 ~ (# of classes - 1) order

        self.img_path = []
        self.targets = []

        for classid in self.classids:
            imgpaths_c = sorted(os.listdir(os.path.join(root, splitdir, classid)))

            # query: [0 ~ (-nshot - 1 or nsamples)],
            # shot: [nshot ~ to the last elem]
            if is_shot:
                nsamples_c = nshot if nshot else len(imgpaths_c)
                imgpaths_c = imgpaths_c[-nsamples_c:]
                # imgpaths_c = imgdirs[200:]  # full
            else:
                nsamples_c = min(nsamples, len(imgpaths_c) - nshot) if nsamples else len(imgpaths_c) - nshot
                # nsamples_c = nsamples if nsamples else len(imgpaths_c) - nshot  # full
                imgpaths_c = imgpaths_c[:nsamples_c]

            for imgpath in imgpaths_c:
                imgabspath = os.path.join(root, splitdir, classid, imgpath)
                target = classid2target[classid]
                self.img_path.append(imgabspath)
                self.targets.append(target)

        num_classes = len(self.classids)
        assert num_classes == max(self.targets) + 1

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

    def _label_generator(self, root, txt, classdirnames):
        if txt != '' and os.path.exists(os.path.join(root, txt)):
            print(f'Label file exist : {os.path.join(root, txt)}')
            with open(os.path.join(root, txt), 'r') as jsonfile:
                idx_classes = json.load(jsonfile)
        else:
            txt = 'label.json' if txt == '' else txt
            print(f"No label file found, make new one on {os.path.join(root, txt)}")

            idx_classes = {idx: i for i, idx in enumerate(classdirnames)}
            with open(os.path.join(root, txt), 'w') as jsonfile:
                json.dump(idx_classes, jsonfile, indent='')

        return idx_classes


class CUBMemoryDataset(Dataset):
    def __init__(self, root, memory_dir='memory', label_file='standard_label.json', imgsize=224):
        self.transform = Transforms.clip_transform(imgsize)
        self.label_file = label_file

        self.memory_dir = os.path.join(root, memory_dir)
        with open(os.path.join(root, label_file)) as f:
            mapper = json.load(f)
        self.dir2target, self.txtlabels = self.mapper_refiner(mapper)

        folder_dirs = os.listdir(self.memory_dir)
        self.memory = self.get_all_files(folder_dirs)

    def mapper_refiner(self, mapper):
        txtlabels = []
        new_mapper = copy.deepcopy(mapper)

        for key1 in mapper.keys():
            target = new_mapper[key1]
            txtlabels.append([target, key1])
        ordered_txtlabels = ['' for _ in range(len(txtlabels))]
        for target, key in txtlabels:
            ordered_txtlabels[target] = key

        return new_mapper, txtlabels

    def get_all_files(self, folder_dirs):
        all_files = []

        for folder in folder_dirs:
            folder_key = folder  # '_'.join(folder.split())
            if folder_key not in self.dir2target.keys():
                continue
            folder_dir = os.path.join(self.memory_dir, folder)
            folder_files = os.listdir(folder_dir)
            target = self.dir2target[folder_key]

            for file in folder_files:
                if file.split('.')[-1] != 'jpg':
                    continue
                all_files.append([os.path.join(folder_dir, file), target])

        return all_files

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, index):
        img_path, target = self.memory[index]
        img = Image.open(img_path).convert('RGB')
        if not (self.transform == None):
            img = self.transform(img)
        return img, target


class FoodMemoryDataset(CUBMemoryDataset):
     def mapper_refiner(self, mapper):
        txtlabels = []
        new_mapper = copy.deepcopy(mapper)

        for key1 in mapper.keys():
            # key2 add
            target = new_mapper[key1]
            key2 = key1
            new_mapper[key2] = target

            # get txtlaquerobels
            key = ' '.join(key2.split('_'))
            txtlabels.append([target, key])
        ordered_txtlabels = ['' for _ in range(len(txtlabels))]
        for target, key in txtlabels:
            ordered_txtlabels[target] = key

        return new_mapper, txtlabels


class WebvisionMemoryDataset(Dataset):
    def __init__(self, imgmemroot, queryroot, label_file = '', len_memory=1000, webvisionsource='google', imgsize=224):
        self.transform = Transforms.clip_transform(imgsize)
        self.queryroot = queryroot

        # n0xxxxxxx -> 'web' 0 ~ 999
        synset2webtarget = {}
        webtarget2synset = {}
        with open(os.path.join(imgmemroot, 'info/synsets.txt')) as f:
            lines = f.readlines()
        for linenum, line in enumerate(lines):
            nxxxxxxxx = line.split()[0]
            synset2webtarget[nxxxxxxxx] = linenum
            webtarget2synset[linenum] = nxxxxxxxx

        with open(os.path.join(queryroot, label_file)) as f:
            idxs_cls = json.load(f)
        synset_set = idxs_cls.keys()  # [nxxxxxxxx, ..., nxxxxxxxx]
        num_classes = len(synset_set)

        # webvisionsource = {'google'/ 'flickr'}
        with open(os.path.join(imgmemroot, f'info/train_filelist_{webvisionsource}.txt')) as f:
            lines = f.readlines()

        self.img_path = []
        self.targets = []
        nsamples_count = [0] * num_classes

        for line in lines:
            img, webtarget = line.split()
            webtarget = int(webtarget)

            if webtarget2synset[webtarget] in synset_set:
                # webvision is always memory
                synset = webtarget2synset[webtarget]
                target = idxs_cls[synset]
                # if nsamples_count[target] == 0:
                #     print(webtarget, '->', synset, '->', target)
                if nsamples_count[target] >= len_memory: continue
                self.img_path.append(os.path.join(imgmemroot, img))
                self.targets.append(target)
                nsamples_count[target] += 1

        with open(os.path.join(imgmemroot, 'info/synsets.txt')) as f:
            lines = f.readlines()

        self.txtlabels = {}

        synset2txtlabel = self.imagenetsynset2txtlabel()
        for linenum, line in enumerate(lines):
            nxxxxxxxx = line.split()[0]
            if nxxxxxxxx in synset_set:
                target = idxs_cls[nxxxxxxxx]
                clstxtlabel = synset2txtlabel[nxxxxxxxx]  # clipstyle txtlabel
                self.txtlabels[target] = clstxtlabel

    def imagenetsynset2txtlabel(self):
        synset2txtlabel = {}
        with open(os.path.join(self.queryroot, 'cliplabels.txt'), 'r') as f:
            lines = list(f.read().splitlines())
            for line in lines:
                synset, target, txtlabel = line.split(',')
                synset2txtlabel[synset] = txtlabel
        return synset2txtlabel

    def __getitem__(self, index):
        img_path = self.img_path[index]
        target = self.targets[index]
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image)
        return img, target

    def __len__(self):
        return len(self.targets)


class ImageNet1K(Dataset):
    def __init__(self, datasetsroot, label_file, split='train', imgsize=224):
        super().__init__()
        self.label_file = label_file
        self.datasetsroot = datasetsroot
        self.datasetroot = os.path.join(datasetsroot, 'ILSVRC/Data/CLS-LOC')
        self.imgclassesdir = os.path.join(self.datasetroot, split)
        self.transform = Transforms.clip_transform(imgsize)
        self.txtlabels = {}
        self.synset2txtlabel = self.imagenetsynset2txtlabel()

        with open(os.path.join(self.datasetsroot, self.label_file), 'r') as f:
            self.synset2target = json.load(f)

        if split == 'train':
            self._init_trainsplit()
        elif split == 'val':
            self._init_valsplit()

    def _init_trainsplit(self):
        synsets = os.listdir(self.imgclassesdir)
        synsets.sort()

        self.img_path = []
        self.targets = []

        for synset in synsets:
            if synset not in self.synset2target.keys():
                continue
            target = self.synset2target[synset]
            self.txtlabels[target] = self.synset2txtlabel[synset]

            synset_dir = os.path.join(self.imgclassesdir, synset)
            imgpath_c = sorted(os.listdir(synset_dir))
            imgpath_c = [os.path.join(synset_dir, img) for img in imgpath_c]

            for imgpath_c_i in imgpath_c:
                self.img_path.append(imgpath_c_i)
                self.targets.append(target)

    def _init_valsplit(self):
        self.img_path = [os.path.join(self.imgclassesdir, img) for img in os.listdir(self.imgclassesdir)]
        self.img_path.sort()
        with open(os.path.join(self.datasetsroot, 'ILSVRC/Annotations/CLS-LOC/val/imagenet_2012_validation_synset_labels.txt'), 'r') as f:
            synsets = list(f.read().splitlines())
        self.targets = [self.synset2target[t] for t in synsets]
        for synset in synsets:
            target = self.synset2target[synset]
            self.txtlabels[target] = self.synset2txtlabel[synset]

    def imagenetsynset2txtlabel(self):
        synset2txtlabel = {}
        with open(os.path.join(self.datasetroot, 'cliplabels.txt'), 'r') as f:
            lines = list(f.read().splitlines())
            for line in lines:
                synset, target, txtlabel = line.split(',')
                synset2txtlabel[synset] = txtlabel
        return synset2txtlabel

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

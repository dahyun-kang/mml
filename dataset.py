import os
import json
import copy
from PIL import Image

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

from text_data.preprocess import SentPreProcessor
from clip.clip import _transform as clip_transform


class SubsetDataset(Dataset):
    def __init__(self, data, targets, imgsize=224):
        self.data = data
        self.targets = targets
        self.transform = clip_transform(imgsize)

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


class TextTokenMemoryDataset(Dataset):
    def __init__(self, root, classid2target, wiki_dir='wiki'):
        '''
        * class2textlabel_file : "labels.txt" file. refer the link below
        https://github.com/ChangyaoTian/VL-LTR/releases/download/text-corpus/imagenet.zip
        '''
        # sentence token generator
        preprocessor = SentPreProcessor(root, classid2target, wiki_dir)
        text_tokens = preprocessor.make_sentence_tokens()
        num_sents = [token.shape[0] for token in text_tokens]

        self.data = torch.cat(text_tokens)
        self.targets = []
        for idx, nsents in enumerate(num_sents):
            self.targets += [idx] * nsents

        self.target2txtlabel = preprocessor.get_txtlabel()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.targets[index]

        return sample, label


class DatasetSplitLoader:
    def __init__(self, root, splitdir='train', classsplit_file='trn_label.json'):
        # classsplit_file : class split identifier file. if empty, make a new one
        # directory-identified classes
        classdirnames = sorted(os.listdir(os.path.join(root, splitdir)))
        self.classid2target = self._label_generator(root, classsplit_file, classdirnames)
        self.classids = sorted(classdirnames, key = lambda item: self.classid2target[item])  # sort idxs with 0 ~ (# of classes - 1) order

        self.img_path = {}
        self.classlen = {}

        for classid in self.classids:
            imgpaths_c = sorted(os.listdir(os.path.join(root, splitdir, classid)))
            self.img_path[classid] = []
            self.classlen[classid] = len(imgpaths_c)
            # randominze img indice. seed fixed at main.py
            rand_idx = torch.randperm(len(imgpaths_c)).tolist()
            imgpaths_c = np.array(imgpaths_c)[rand_idx].tolist()

            for imgpath in imgpaths_c:
                imgabspath = os.path.join(root, splitdir, classid, imgpath)
                self.img_path[classid].append(imgabspath)

    def split(self, nsamples=None, nshot=16):
        query_img_path = []
        query_targets = []
        shot_img_path = []
        shot_targets = []

        for classid in self.classids:
            len_c = self.classlen[classid]
            target = self.classid2target[classid]

            query_nsamples_c = min(nsamples, len_c - nshot) if nsamples else len_c - nshot
            # nsamples_c = nsamples if nsamples else len(imgpaths_c) - nshot  # full
            query_imgpaths_c = self.img_path[classid][:query_nsamples_c]
            query_img_path += query_imgpaths_c
            query_targets += [target] * query_nsamples_c

            if nshot > 0:
                shot_nsamples_c = nshot if nshot else len_c
                shot_imgpaths_c = self.img_path[classid][-shot_nsamples_c:]
                # shot_imgpaths_c = shot_imgdirs[200:]  # full
                shot_img_path += shot_imgpaths_c
                shot_targets += [target] * shot_nsamples_c

        query_dataset = SubsetDataset(data=query_img_path, targets=query_targets)

        if nshot > 0:
            shot_dataset = SubsetDataset(data=shot_img_path, targets=shot_targets)
        else:
            shot_dataset = None

        return query_dataset, shot_dataset

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


class FGMemoryDataset(Dataset):
    def __init__(self, root, classid2target, memory_dir='memory', imgsize=224):
        self.transform = clip_transform(imgsize)
        self.data = []
        self.targets = []

        memory_dir = os.path.join(root, memory_dir)
        folder_dirs = os.listdir(memory_dir)

        for dirname in folder_dirs:
            if dirname not in classid2target.keys():
                continue
            folder_dir = os.path.join(memory_dir, dirname)
            folder_files = os.listdir(folder_dir)
            target = classid2target[dirname]

            for file in folder_files:
                if file.split('.')[-1] != 'jpg':
                    continue
                self.data.append(os.path.join(folder_dir, file))
                self.targets.append(target)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img_path = self.data[index]
        target = self.targets[index]
        img = Image.open(img_path).convert('RGB')
        if not (self.transform == None):
            img = self.transform(img)
        return img, target


class WebvisionMemoryDataset(Dataset):
    def __init__(self, imgmemroot, queryroot, classid2target, len_memory=100, webvisionsource='google', imgsize=224):
        self.transform = clip_transform(imgsize)
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

        synset_set = classid2target.keys()  # [nxxxxxxxx, ..., nxxxxxxxx]
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
                target = classid2target[synset]
                # if nsamples_count[target] == 0:
                #     print(webtarget, '->', synset, '->', target)
                if nsamples_count[target] >= len_memory: continue
                self.img_path.append(os.path.join(imgmemroot, img))
                self.targets.append(target)
                nsamples_count[target] += 1

        '''
        with open(os.path.join(imgmemroot, 'info/synsets.txt')) as f:
            lines = f.readlines()

        synset2txtlabel = self.imagenetsynset2txtlabel()
        for linenum, line in enumerate(lines):
            nxxxxxxxx = line.split()[0]
            if nxxxxxxxx in synset_set:
                target = classid2target[nxxxxxxxx]
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
        '''

    def __getitem__(self, index):
        img_path = self.img_path[index]
        target = self.targets[index]
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image)
        return img, target

    def __len__(self):
        return len(self.targets)


class ImageNet1K(Dataset):
    def __init__(self, datasetsroot, classsplit_file, split='train', imgsize=224):
        super().__init__()
        self.classsplit_file = classsplit_file
        self.datasetsroot = datasetsroot
        self.datasetroot = os.path.join(datasetsroot, 'ILSVRC/Data/CLS-LOC')
        self.imgclassesdir = os.path.join(self.datasetroot, split)
        self.transform = clip_transform(imgsize)
        self.txtlabels = {}
        self.synset2txtlabel = self.imagenetsynset2txtlabel()

        with open(os.path.join(self.datasetsroot, self.classsplit_file), 'r') as f:
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

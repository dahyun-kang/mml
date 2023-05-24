'''
imagenet1K (ILSVRC) subset generator

1) mv ILSVRC_subset_generator.py $dataroot
    here $dataroot is the root for ILSVRC such as /ssd1t/datasets
2) adjust num_samples in L23
3) python ILSVRC_subset_generator.py
'''


import os
import os.path as osp
import random
import shutil

seed = 7
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

def force_symlink(file1, file2):
    try:
        os.symlink(file1, file2)
    except FileExistsError:
        os.remove(file2)
        os.symlink(file1, file2)

num_samples = 130
# num_samples = 500

source_split_root = osp.join(os.getcwd(), 'ILSVRC/Data/CLS-LOC')
train_path = osp.join(source_split_root, 'train')
val_path = osp.join(source_split_root, 'val')
test_path = osp.join(source_split_root, 'test')

output_path_root = osp.join(os.getcwd(), f'ILSVRC_{num_samples}samples')
output_train_path = osp.join(output_path_root, 'train')
output_val_path = osp.join(output_path_root, 'val')


## symlinking train
class_dir = os.listdir(train_path)
class_dir.sort()
random.shuffle(class_dir)  # shuffle class dir with a fixed rand seed

min, max = 999999, 0
for i, class_i in enumerate(class_dir):

    class_fulldir = osp.join(train_path, class_i)
    imagelist = os.listdir(class_fulldir)
    if len(imagelist) > max:
        max = len(imagelist)
    if len(imagelist) < min:
        min = len(imagelist)


    imagelist_subset = imagelist[:num_samples]

    if len(imagelist_subset) != num_samples:
        print(f'Warning: {i}th class {class_i} only has max {len(imagelist_subset)} samples')
        import pdb ; pdb.set_trace()

    if i % 4 == 0:
        outputclass_fulldir = osp.join(output_val_path, class_i)
        print(f'linking {i}th class {class_i} as val split')
    else:
        outputclass_fulldir = osp.join(output_train_path, class_i)
        print(f'linking {i}th class {class_i} as train split')

    if not osp.exists(outputclass_fulldir):
        os.makedirs(outputclass_fulldir)

    # print(f'# images: {len(imagelist)} | min # images: {min} | max # images: {max}')

    for img in imagelist_subset:
        force_symlink(os.path.join(class_fulldir, img), os.path.join(outputclass_fulldir, img))

## symlinking labels.txt
print(f'linking labels.txt')
force_symlink(os.path.join(source_split_root, 'labels.txt'), output_path_root)

## symlinking wiki
print(f'linking wiki')
force_symlink(os.path.join(source_split_root, 'wiki'), output_path_root)

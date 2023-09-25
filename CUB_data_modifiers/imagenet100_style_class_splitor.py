import os
import shutil

if __name__ == '__main__':

    all_dir = './images'
    train_dir = './Train_class'
    val_dir = './Val_class'

    idxs_of_images_file = './images.txt'
    classes_of_idxs_file = './image_class_labels.txt'
    name_of_classes_file = './classes.txt'

    train_test_split = './train_test_split.txt'

    class_flag = 100 # 0~99 : train, 100~199 : val

    with open(name_of_classes_file, 'r') as f:
        data = f.readlines()
    name_of_classes = [l.split()[1] for l in data]
    num_classes = len(name_of_classes)

    # make folders of Train and Val
    for ddir in [train_dir, val_dir]:
        if not os.path.exists(ddir):
            os.mkdir(ddir)
            for idx, cname in enumerate(name_of_classes):
                if ddir == train_dir:
                    if idx < class_flag:
                        os.mkdir(os.path.join(ddir, cname))
                else:
                    if idx >= class_flag:
                        os.mkdir(os.path.join(ddir, cname))

    with open(idxs_of_images_file, 'r') as f:
        data = f.readlines()
    images_path = [l.split()[1] for l in data]

    with open(classes_of_idxs_file, 'r') as f:
        data = f.readlines()
    classes = [int(l.split()[1]) for l in data]

    with open(train_test_split, 'r') as f:
        data = f.readlines()
    splits = [int(l.split()[1]) for l in data]


    num_train = 0
    num_val = 0

    num_train_inclass = [0]*num_classes
    num_val_inclass = [0]*num_classes

    for i_path, i_class, i_split in zip(images_path, classes, splits):
        
        if i_class <= class_flag: 
            num_train += 1
            num_train_inclass[i_class-1]+=1
        else: 
            num_val += 1
            num_val_inclass[i_class-1]+=1

        from_dir = os.path.join(all_dir, i_path)
        to_dir = os.path.join(train_dir if i_class <= class_flag else val_dir, i_path)

        shutil.copyfile(from_dir, to_dir)

    print(f"Total number of Train images: {num_train}")
    print(f"Total number of Val images: {num_val}")
    print(f"Max value of number of images in Train classes: {max(num_train_inclass[:class_flag])}")
    print(f"Min value of number of images in Train classes: {min(num_train_inclass[:class_flag])}")
    print(f"Max value of number of images in Val classes: {max(num_val_inclass[class_flag:])}")
    print(f"Min value of number of images in Val classes: {min(num_val_inclass[class_flag:])}")
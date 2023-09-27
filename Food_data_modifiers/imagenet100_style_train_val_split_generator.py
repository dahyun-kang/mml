import os
import shutil

if __name__ == '__main__':

    all_dir = './images'
    train_dir = './Train'
    val_dir = './Val'

    train_file = './meta/meta/train.txt'
    test_file = './meta/meta/text.txt'
    label_idx_file = './meta/meta/classes.txt'

    # get classes name
    cnames = []

    with open(label_idx_file, 'r') as f:
        data = f.readlines()
    for target, line in enumerate(data):
        class_name = line.split()[0]
        cnames.append(class_name)

    # make folders of Train and Val
    for ddir in [train_dir, val_dir]:
        if not os.path.exists(ddir):
            os.mkdir(ddir)
            for cname in cnames:
                os.mkdir(os.path.join(ddir, cname))

    # train
    with open(train_file, 'r') as f:
        data = f.readlines()
    for line in data:
        img_path = line + '.jpg'
        from_dir = os.path.join(all_dir, img_path)
        to_dir = os.path.join(train_dir, img_path)

        shutil.copyfile(from_dir, to_dir)

    # train
    with open(test_file, 'r') as f:
        data = f.readlines()
    for line in data:
        img_path = line + '.jpg'
        from_dir = os.path.join(all_dir, img_path)
        to_dir = os.path.join(val_dir, img_path)

        shutil.copyfile(from_dir, to_dir)

    print("Done")
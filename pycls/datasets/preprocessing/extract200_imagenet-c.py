import os
from distutils.dir_util import copy_tree

path = "/ws/data"
class_path = os.path.join(path, "imagenet-a/README.txt")

with open(class_path, 'r') as f:
    data = f.readlines()
    class_list_ = data[12:]

class_list = []
for row in class_list_:
    class_name = row.split(" ")[0]
    class_list.append(class_name)

dataset_dir = "/ws/data/imagenet-c"
for folder in sorted(os.listdir(dataset_dir)):
    sub_dir = os.path.join(dataset_dir, folder)
    for sub_folder in sorted(os.listdir(sub_dir)):
        sub_dir2 = os.path.join(sub_dir, sub_folder)
        for sub_folder2 in sorted(os.listdir(sub_dir2)):
            sub_dir3 = os.path.join(sub_dir2, sub_folder2)
            for class_name in os.listdir(sub_dir3):
                sub_dir4 = os.path.join(sub_dir3, class_name)
                if class_name in class_list:
                    path = sub_dir4
                    new_path = os.path.join("/ws/data/imagenet200-c", folder, sub_folder, sub_folder2, class_name)
                    copy_tree(path, new_path)





for row in class_list:
    class_name = row.split(" ")[0]
    train_path = "/ws/data/imagenet/train/" + class_name
    new_train_path = "/ws/data/imagenet200/train/" + class_name
    val_path = "/ws/data/imagenet/val/" + class_name
    new_val_path = "/ws/data/imagenet200/val/" + class_name
    copy_tree(train_path, new_train_path)
    copy_tree(val_path, new_val_path)

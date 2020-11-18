import os
from distutils.dir_util import copy_tree

import json

path = "/ws/data"
class_path = os.path.join(path, "imagenet/imagenet_class_index.json")

with open(class_path, 'r') as f:
    data = json.load(f)
    lst = list(data.values())
    folder_list = [i[0] for i in list(data.values())[:200]]
    class_list = [i[1] for i in list(data.values())[:200]]
    print(folder_list)
    print(class_list)

dataset_dir = "/ws/data/imagenet-c"
for folder in sorted(os.listdir(dataset_dir)):
    sub_dir = os.path.join(dataset_dir, folder)
    for sub_folder in sorted(os.listdir(sub_dir)):
        sub_dir2 = os.path.join(sub_dir, sub_folder)
        for sub_folder2 in sorted(os.listdir(sub_dir2)):
            sub_dir3 = os.path.join(sub_dir2, sub_folder2)
            for class_name in os.listdir(sub_dir3):
                sub_dir4 = os.path.join(sub_dir3, class_name)
                if class_name in folder_list:
                    path = sub_dir4
                    new_path = os.path.join("/ws/data/imagenet200-c", folder, sub_folder, sub_folder2, class_name)
                    copy_tree(path, new_path)


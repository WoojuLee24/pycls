import os
from distutils.dir_util import copy_tree
from shutil import copyfile
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

for row in sorted(folder_list):
    class_name = row.split(" ")[0]
    # train_path = "/ws/data/imagenet/train/" + class_name
    # new_train_path = "/ws/data/imagenet200/train/" + class_name
    # copy_tree(train_path, new_train_path)
    if class_name == '\n':
        continue
    val_path = os.path.join("/ws/data/imagenet/train/", class_name)
    new_val_path = os.path.join("/ws/data/imagenet200/train/", class_name)
    if not os.path.exists(new_val_path):
        os.mkdir(new_val_path)
    for png in sorted(os.listdir(val_path)):
        val_png = os.path.join(val_path, png)
        new_val_png = os.path.join("/ws/data/imagenet200/train/", class_name, png)
        copyfile(val_png, new_val_png)



    # if not os.path.exists(new_train_path):
    #     os.mkdir(new_train_path)
    # if not os.path.exits(new_val_path):
    #     os.mkdir(new_val_path)
